"""Fail-closed GRPO group gating for strict outcome supervision.

The reward function emits a binary ``acc`` value for every trajectory.  This
module keeps only prompt groups containing both a strict failure and a strict
success.  Uniform groups are masked out after all reward/KL processing so they
cannot acquire an advantage from process proxies or the KL penalty.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable


def _binary(value: Any) -> int:
    number = float(value)
    if not math.isfinite(number) or number not in (0.0, 1.0):
        raise ValueError(f"strict correctness must be binary, got {value!r}")
    return int(number)


def strict_correctness_group_stats(
    uids: Iterable[Any],
    correctness: Iterable[Any],
    eligibility: Iterable[Any] | None = None,
    policy_versions: Iterable[Any] | None = None,
    expected_policy_version: int | None = None,
) -> tuple[list[bool], dict[str, float]]:
    """Return the active-sample mask and aggregate gate metrics.

    A group is active only when its strict correctness labels contain both 0
    and 1.  The returned metrics intentionally contain counts only; prompt
    identifiers never leave the private training batch.
    """

    uid_list = list(uids)
    labels = [_binary(value) for value in correctness]
    if not uid_list or len(uid_list) != len(labels):
        raise ValueError("uids and strict correctness must have the same non-zero length")
    eligible = (
        [_binary(value) for value in eligibility]
        if eligibility is not None
        else [1] * len(uid_list)
    )
    if len(eligible) != len(uid_list):
        raise ValueError("eligibility must have the same length as uids")
    versions = list(policy_versions) if policy_versions is not None else [None] * len(uid_list)
    if len(versions) != len(uid_list):
        raise ValueError("policy_versions must have the same length as uids")

    grouped: dict[Any, list[tuple[int, int, Any]]] = defaultdict(list)
    for uid, label, allowed, version in zip(uid_list, labels, eligible, versions, strict=True):
        grouped[uid].append((label, allowed, version))

    invalid_uids = {
        uid for uid, values in grouped.items() if not all(allowed for _, allowed, _ in values)
    }
    stale_uids: set[Any] = set()
    if expected_policy_version is not None:
        stale_uids = {
            uid
            for uid, values in grouped.items()
            if any(version is None or int(version) != int(expected_policy_version) for _, _, version in values)
        }
        invalid_uids |= stale_uids
    active_uids = {
        uid
        for uid, values in grouped.items()
        if uid not in invalid_uids and {label for label, _, _ in values} == {0, 1}
    }
    all_wrong = sum(
        uid not in invalid_uids and {label for label, _, _ in values} == {0}
        for uid, values in grouped.items()
    )
    all_correct = sum(
        uid not in invalid_uids and {label for label, _, _ in values} == {1}
        for uid, values in grouped.items()
    )
    mixed = len(active_uids)
    mask = [uid in active_uids for uid in uid_list]
    metrics = {
        "grpo/strict_mixed_groups": float(mixed),
        "grpo/skipped_uniform_groups": float(all_wrong + all_correct),
        "grpo/skipped_all_wrong_groups": float(all_wrong),
        "grpo/skipped_all_correct_groups": float(all_correct),
        "grpo/skipped_hard_gate_groups": float(len(invalid_uids)),
        "grpo/skipped_stale_policy_groups": float(len(stale_uids)),
        "grpo/effective_samples": float(sum(mask)),
        "grpo/skipped_samples": float(len(mask) - sum(mask)),
        "grpo/total_groups": float(len(grouped)),
    }
    return mask, metrics


def apply_strict_correctness_group_gate(batch: Any) -> tuple[Any, dict[str, float]]:
    """Zero advantages/returns for uniform groups and mark empty updates.

    ``batch`` is deliberately duck-typed so the core policy can be unit-tested
    without importing veRL.  At runtime it is a ``DataProto`` with torch
    tensors in ``batch.batch`` and numpy arrays in ``non_tensor_batch``.
    """

    if "uid" not in batch.non_tensor_batch:
        raise KeyError("strict GRPO gate requires non_tensor_batch['uid']")
    if "acc" not in batch.non_tensor_batch:
        raise KeyError("strict GRPO gate requires reward extra field 'acc'")
    if not {"advantages", "returns", "response_mask"}.issubset(batch.batch):
        raise KeyError(
            "strict GRPO gate requires advantages, returns, and response_mask "
            "after reward/KL assembly"
        )

    expected_policy_version = batch.meta_info.get("strict_expected_policy_version")
    version_values = None
    if expected_policy_version is not None:
        minimum = batch.non_tensor_batch.get("min_global_steps")
        maximum = batch.non_tensor_batch.get("max_global_steps")
        if minimum is None or maximum is None:
            version_values = [None] * len(batch.non_tensor_batch["uid"])
        else:
            version_values = [
                int(left) if left is not None and right is not None and int(left) == int(right) else None
                for left, right in zip(minimum, maximum, strict=True)
            ]
    success_values = batch.non_tensor_batch.get("success")
    if success_values is None:
        success_values = batch.non_tensor_batch["acc"]
    train_mask_values = batch.non_tensor_batch.get("train_mask")
    if train_mask_values is None:
        train_mask_values = batch.non_tensor_batch.get("online_eligible")
    mask, metrics = strict_correctness_group_stats(
        batch.non_tensor_batch["uid"],
        success_values,
        train_mask_values,
        version_values,
        expected_policy_version,
    )
    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]
    response_mask = batch.batch["response_mask"]
    try:
        import torch

        sample_mask = torch.as_tensor(mask, device=advantages.device, dtype=advantages.dtype)
        while sample_mask.ndim < advantages.ndim:
            sample_mask = sample_mask.unsqueeze(-1)
        batch.batch["advantages"] = advantages * sample_mask
        batch.batch["returns"] = returns * sample_mask.to(dtype=returns.dtype)
        response_sample_mask = torch.as_tensor(
            mask, device=response_mask.device, dtype=response_mask.dtype
        )
        while response_sample_mask.ndim < response_mask.ndim:
            response_sample_mask = response_sample_mask.unsqueeze(-1)
        batch.batch["response_mask"] = response_mask * response_sample_mask
    except ImportError as exc:  # pragma: no cover - veRL always provides torch
        raise RuntimeError("torch is required to apply the runtime GRPO gate") from exc

    batch.meta_info["strict_group_should_update_actor"] = bool(
        metrics["grpo/strict_mixed_groups"] > 0
    )
    return batch, metrics
