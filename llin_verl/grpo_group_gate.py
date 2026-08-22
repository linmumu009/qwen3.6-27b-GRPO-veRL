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
    uids: Iterable[Any], correctness: Iterable[Any]
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

    grouped: dict[Any, list[int]] = defaultdict(list)
    for uid, label in zip(uid_list, labels, strict=True):
        grouped[uid].append(label)

    active_uids = {uid for uid, values in grouped.items() if set(values) == {0, 1}}
    all_wrong = sum(set(values) == {0} for values in grouped.values())
    all_correct = sum(set(values) == {1} for values in grouped.values())
    mixed = len(active_uids)
    mask = [uid in active_uids for uid in uid_list]
    metrics = {
        "grpo/strict_mixed_groups": float(mixed),
        "grpo/skipped_uniform_groups": float(all_wrong + all_correct),
        "grpo/skipped_all_wrong_groups": float(all_wrong),
        "grpo/skipped_all_correct_groups": float(all_correct),
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
    if "advantages" not in batch.batch or "returns" not in batch.batch:
        raise KeyError("strict GRPO gate must run after advantage computation")

    mask, metrics = strict_correctness_group_stats(
        batch.non_tensor_batch["uid"], batch.non_tensor_batch["acc"]
    )
    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]
    try:
        import torch

        sample_mask = torch.as_tensor(mask, device=advantages.device, dtype=advantages.dtype)
        while sample_mask.ndim < advantages.ndim:
            sample_mask = sample_mask.unsqueeze(-1)
        batch.batch["advantages"] = advantages * sample_mask
        batch.batch["returns"] = returns * sample_mask.to(dtype=returns.dtype)
    except ImportError as exc:  # pragma: no cover - veRL always provides torch
        raise RuntimeError("torch is required to apply the runtime GRPO gate") from exc

    batch.meta_info["strict_group_should_update_actor"] = bool(
        metrics["grpo/strict_mixed_groups"] > 0
    )
    return batch, metrics
