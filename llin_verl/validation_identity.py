"""Identity-safe validation alignment for partial agent rollout batches.

The veRL validation path repeats every prompt ``n`` times before dispatching
the resulting batch across agent-loop workers.  Fastest-K or UNKNOWN handling
may return fewer physical trajectories than were requested.  This module joins
only real returned trajectories to their source rows and emits an explicit
status record for every requested slot; it never pads, duplicates, or truncates
model output to make the batch sizes agree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


IDENTITY_KEY = "__llin_validation_identity__"
PADDING_KEY = "__llin_validation_padding__"
SLOT_KEY = "__llin_validation_sample_slot__"
POLICY_KEY = "__llin_validation_policy_version__"


def _required_text(value: Any, name: str) -> str:
    text = str(value or "")
    if not text:
        raise ValueError(f"validation identity lacks {name}")
    return text


def make_validation_identity(
    task_id: Any,
    prefix_state_id: Any,
    policy_version: Any,
    sample_slot: Any,
) -> str:
    """Return the frozen four-part identity for one requested trajectory."""

    task = _required_text(task_id, "task_id")
    prefix = _required_text(prefix_state_id, "prefix_state_id")
    policy = int(policy_version)
    slot = int(sample_slot)
    if slot < 0:
        raise ValueError("validation sample_slot must be non-negative")
    return f"{task}::{prefix}::policy-{policy}::slot-{slot}"


def build_validation_identities(
    repeated_extra_info: Sequence[Mapping[str, Any]],
    *,
    samples_per_state: int,
    policy_version: int,
) -> tuple[list[str], list[int], list[int]]:
    """Stamp an interleaved repeated batch with stable per-slot identities."""

    samples = int(samples_per_state)
    if samples <= 0:
        raise ValueError("samples_per_state must be positive")
    identities: list[str] = []
    slots: list[int] = []
    policies: list[int] = []
    for row_index, extra in enumerate(repeated_extra_info):
        if not isinstance(extra, Mapping):
            raise ValueError("validation extra_info must be a mapping")
        slot = row_index % samples
        identities.append(
            make_validation_identity(
                extra.get("task_id") or extra.get("instruction_sha256"),
                extra.get("prefix_state_id"),
                policy_version,
                slot,
            )
        )
        slots.append(slot)
        policies.append(int(policy_version))
    if len(set(identities)) != len(identities):
        raise ValueError("expected validation identities are not unique")
    return identities, slots, policies


def mark_padding_identities(
    identities: Sequence[str],
    *,
    expected_count: int,
) -> tuple[list[str], list[bool]]:
    """Give divisor padding rows unique identities instead of positional unpad."""

    expected = int(expected_count)
    if expected < 0 or expected > len(identities):
        raise ValueError("invalid expected validation count")
    output = list(identities)
    padding = [False] * len(output)
    for offset in range(expected, len(output)):
        output[offset] = f"__llin_padding__::{offset - expected}"
        padding[offset] = True
    return output, padding


def align_returned_validation(
    expected_identities: Sequence[str],
    returned_identities: Sequence[str],
    *,
    returned_padding: Sequence[bool] | None = None,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    """Join actual returned rows to expected rows by identity.

    Returns ``(expected_indices, returned_indices, status_rows)``.  The first
    two lists have equal length and preserve actual return order.  Missing
    expected slots appear only in ``status_rows`` as ``NOT_RETURNED`` and are
    therefore available for explicit resampling without becoming fake model
    trajectories.
    """

    expected = [str(value) for value in expected_identities]
    returned = [str(value) for value in returned_identities]
    if len(set(expected)) != len(expected):
        raise ValueError("expected validation identities are not unique")
    if returned_padding is None:
        padding = [value.startswith("__llin_padding__::") for value in returned]
    else:
        padding = [bool(value) for value in returned_padding]
        if len(padding) != len(returned):
            raise ValueError("returned padding mask length mismatch")

    expected_index = {identity: index for index, identity in enumerate(expected)}
    selected_expected: list[int] = []
    selected_returned: list[int] = []
    observed: set[str] = set()
    padding_returned = 0
    for output_index, (identity, is_padding) in enumerate(zip(returned, padding, strict=True)):
        if is_padding or identity.startswith("__llin_padding__::"):
            padding_returned += 1
            continue
        if identity not in expected_index:
            raise ValueError(f"returned validation identity is outside expected set: {identity!r}")
        if identity in observed:
            raise ValueError(f"returned validation identity is duplicated: {identity!r}")
        observed.add(identity)
        selected_expected.append(expected_index[identity])
        selected_returned.append(output_index)

    status_rows = [
        {
            "validation_identity": identity,
            "status": "RETURNED" if identity in observed else "NOT_RETURNED",
            "returned": identity in observed,
            "resample_required": identity not in observed,
            "reason": "" if identity in observed else "fastest_k_cancelled_or_runtime_missing",
        }
        for identity in expected
    ]
    if status_rows:
        status_rows[0]["padding_rows_returned_and_excluded"] = padding_returned
    return selected_expected, selected_returned, status_rows


def apply_judge_states(
    status_rows: Sequence[Mapping[str, Any]],
    returned_identities: Sequence[str],
    judge_states: Sequence[Any] | None,
) -> list[dict[str, Any]]:
    """Mark returned UNKNOWN slots for resampling without changing rewards."""

    rows = [dict(row) for row in status_rows]
    if judge_states is None:
        return rows
    returned = [str(value) for value in returned_identities]
    states = [str(value or "UNKNOWN").upper() for value in judge_states]
    if len(returned) != len(states):
        raise ValueError("judge state count does not match returned validation identities")
    state_by_id = dict(zip(returned, states, strict=True))
    if len(state_by_id) != len(returned):
        raise ValueError("returned validation identities are not unique during judge-state join")
    for row in rows:
        identity = str(row["validation_identity"])
        if identity not in state_by_id:
            continue
        state = state_by_id[identity]
        if state not in {"PASS", "FAIL", "UNKNOWN"}:
            raise ValueError(f"unsupported validation judge state: {state!r}")
        row["status"] = state
        row["judge_state"] = state
        row["resample_required"] = state == "UNKNOWN"
        row["reason"] = "judge_unknown" if state == "UNKNOWN" else ""
    return rows


def write_identity_status(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically persist the private expected/returned slot ledger as 0600."""

    materialized = [dict(row) for row in rows]
    identities = [str(row.get("validation_identity") or "") for row in materialized]
    if any(not identity for identity in identities):
        raise ValueError("validation status row lacks identity")
    if len(set(identities)) != len(identities):
        raise ValueError("validation status ledger contains duplicate identities")
    target = Path(path)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    os.chmod(target, 0o600)
