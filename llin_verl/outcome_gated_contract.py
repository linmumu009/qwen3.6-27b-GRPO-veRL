"""Pure helpers for the Qwen3.8 approved43 outcome-gated contract."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evidence_binding_hash(ground_truth: dict[str, Any]) -> str:
    """Bind the approved task identity to its process-evidence contract."""

    return stable_json_hash(
        {
            "environment_id": ground_truth.get("environment_id"),
            "verification_sql": ground_truth.get("verification_sql"),
            "evidence_plan": ground_truth.get("evidence_plan"),
            "required_tables": ground_truth.get("required_tables", []),
            "must_use_fields": ground_truth.get("must_use_fields", []),
        }
    )


def normalized_group_advantages(rewards: Iterable[float]) -> list[float]:
    values = [float(value) for value in rewards]
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= 0:
        return [0.0] * len(values)
    scale = math.sqrt(variance)
    return [(value - mean) / scale for value in values]


def audit_mixed_group_advantages(
    correctness: Iterable[int | float], rewards: Iterable[float]
) -> dict[str, Any]:
    labels = [int(bool(value)) for value in correctness]
    scores = [float(value) for value in rewards]
    if len(labels) != len(scores) or not labels:
        raise ValueError("correctness and rewards must be non-empty and aligned")
    if not 0 < sum(labels) < len(labels):
        raise ValueError("advantage audit requires one mixed outcome group")
    advantages = normalized_group_advantages(scores)
    wrong_advantages = [value for value, label in zip(advantages, labels, strict=True) if not label]
    return {
        "advantages": advantages,
        "incorrect_count": len(wrong_advantages),
        "incorrect_positive_advantage_count": sum(value > 0 for value in wrong_advantages),
        "incorrect_nonnegative_advantage_count": sum(value >= 0 for value in wrong_advantages),
        "all_incorrect_strictly_negative": all(value < 0 for value in wrong_advantages),
        "maximum_incorrect_advantage": max(wrong_advantages),
    }


@dataclass
class HardGateGroup:
    accepted: list[Any] = field(default_factory=list)
    attempts: int = 0
    closed: bool = False


class TristateResampleBuffer:
    """Accept exactly N PASS/FAIL trajectories; resample UNKNOWN.

    This is deliberately independent of veRL tensor types so the rollout
    scheduler can invoke it before constructing a train batch.
    """

    def __init__(self, *, target_size: int = 8, max_attempts: int = 16) -> None:
        if target_size <= 0 or max_attempts < target_size:
            raise ValueError("max_attempts must be at least target_size > 0")
        self.target_size = target_size
        self.max_attempts = max_attempts
        self._groups: dict[str, HardGateGroup] = defaultdict(HardGateGroup)

    def observe(self, group_id: str, trajectory: Any, *, train_mask: bool) -> str:
        group = self._groups[str(group_id)]
        if group.closed:
            raise RuntimeError("group already closed")
        group.attempts += 1
        if train_mask:
            group.accepted.append(trajectory)
        if len(group.accepted) == self.target_size:
            group.closed = True
            return "ready"
        if group.attempts >= self.max_attempts:
            group.closed = True
            return "skip"
        return "resample"

    def result(self, group_id: str) -> dict[str, Any]:
        group = self._groups[str(group_id)]
        return {
            "accepted": list(group.accepted),
            "accepted_count": len(group.accepted),
            "attempt_count": group.attempts,
            "closed": group.closed,
            "ready": group.closed and len(group.accepted) == self.target_size,
            "skipped": group.closed and len(group.accepted) < self.target_size,
        }


class HardGateResampleBuffer(TristateResampleBuffer):
    """Compatibility adapter for pre-v6 callers; new code uses ``train_mask``."""

    def observe(self, group_id: str, trajectory: Any, *, hard_gate_passed: bool) -> str:
        return super().observe(group_id, trajectory, train_mask=hard_gate_passed)
