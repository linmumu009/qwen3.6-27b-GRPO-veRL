#!/usr/bin/env python3
"""Build a non-sensitive audit summary for tiered Qwen3.8 canary dumps.

The input JSONL files stay on the training server.  The generated report keeps
only numeric aggregates and already-hashed task/trajectory identities; prompt,
gold, model output, SQL, tool arguments, and tool responses are never copied.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Iterable


NUMERIC_FIELDS = (
    "tiered_reward",
    "score",
    "train_mask",
    "success",
    "final_answer_correct",
    "attempted_relevant_readonly_sql",
    "successful_relevant_readonly_sql",
    "guess_correct_blocked",
    "query_attempt_count",
    "tool_response_tokens",
    "irrelevant_query_ratio",
    "duplicate_query_ratio",
    "Eq",
    "Et",
    "Eb",
    "E",
    "tool_event_count",
    "unsafe",
    "budget_exceeded",
    "trajectory_advantage_mean",
    "trajectory_return_mean",
    "trajectory_active_response_tokens",
    "trajectory_total_response_tokens",
    "sampling_policy_version_min",
    "sampling_policy_version_max",
)

FLOAT = r"[-+0-9.eE]+"
TRAIN_STAGE_RE = re.compile(
    rf"\[LLIN_TRAIN_STAGE\] step=(\d+) "
    rf"queue_wait_s=({FLOAT}) deserialize_s=({FLOAT}) assemble_s=({FLOAT}) "
    rf"reward_s=({FLOAT}) old_log_prob_s=({FLOAT}) ref_log_prob_s=({FLOAT}) "
    rf"adv_s=({FLOAT}) update_actor_s=({FLOAT}) step_s=({FLOAT})"
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _reward(row: dict[str, Any]) -> float:
    return _float(row.get("tiered_reward", row.get("score", 0.0)))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _describe(values: Iterable[Any]) -> dict[str, float | int | None]:
    clean = [_float(value, math.nan) for value in values]
    clean = [value for value in clean if math.isfinite(value)]
    return {
        "n": len(clean),
        "min": min(clean) if clean else None,
        "mean": fmean(clean) if clean else None,
        "p25": _percentile(clean, 0.25),
        "p50": _percentile(clean, 0.50),
        "p75": _percentile(clean, 0.75),
        "p90": _percentile(clean, 0.90),
        "max": max(clean) if clean else None,
    }


def _identity(row: dict[str, Any], key: str) -> str:
    value = str(row.get(key) or "")
    # Identities are SHA-256 digests emitted by the frozen reward.  Fail closed
    # instead of copying an unexpected raw identifier into the safe report.
    if len(value) == 64 and all(char in "0123456789abcdef" for char in value.casefold()):
        return value.casefold()
    return ""


def _iter_files(directory: Path) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for path in directory.glob("*.jsonl"):
        try:
            step = int(path.stem)
        except ValueError:
            continue
        result.append((step, path))
    return sorted(result)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object row in {path}")
            rows.append(value)
    return rows


def _row_digest(row: dict[str, Any]) -> str:
    return _identity(row, "trajectory_identity_sha256")


def _group_summary(task_identity: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [_reward(row) for row in rows]
    successes = [_float(row.get("success")) > 0.5 for row in rows]
    masks = [_float(row.get("train_mask")) > 0.5 for row in rows]
    advantages = [_float(row.get("trajectory_advantage_mean")) for row in rows]
    exact_group = len(rows) == 8
    all_observable = all(masks)
    mixed = exact_group and all_observable and 0 < sum(successes) < 8
    uniform_or_unknown = not mixed
    wrong_rewards = [reward for reward, success in zip(rewards, successes) if not success]
    correct_rewards = [reward for reward, success in zip(rewards, successes) if success]
    return {
        "task_identity_sha256": task_identity,
        "size": len(rows),
        "train_mask_count": sum(masks),
        "success_count": sum(successes),
        "strict_mixed": mixed,
        "strict_should_update_actor": mixed,
        "reward_vector": rewards,
        "advantage_vector": advantages,
        "uniform_or_unknown_nonzero_advantage_count": sum(
            uniform_or_unknown and abs(value) > 1e-12 for value in advantages
        ),
        "wrong_reward_max": max(wrong_rewards) if wrong_rewards else None,
        "correct_grounded_reward_min": min(correct_rewards) if correct_rewards else None,
        "wrong_above_correct_count": sum(
            wrong > correct
            for wrong in wrong_rewards
            for correct in correct_rewards
        ),
    }


def _step_summary(step: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_task_identity = 0
    for row in rows:
        identity = _identity(row, "task_identity_sha256")
        if not identity:
            missing_task_identity += 1
            identity = "missing"
        by_task[identity].append(row)
    groups = [_group_summary(identity, group) for identity, group in sorted(by_task.items())]
    rewards = [_reward(row) for row in rows]
    violations = {
        "wrong_reward_gt_0_2": sum(
            _float(row.get("success")) <= 0.5 and _reward(row) > 0.2 + 1e-12
            for row in rows
        ),
        "correct_grounded_reward_lt_0_8": sum(
            _float(row.get("final_answer_correct")) > 0.5
            and _float(row.get("successful_relevant_readonly_sql")) > 0.5
            and _reward(row) < 0.8 - 1e-12
            for row in rows
        ),
        "guess_correct_nonzero_reward": sum(
            _float(row.get("guess_correct_blocked")) > 0.5 and _reward(row) > 1e-12
            for row in rows
        ),
        "unsafe_or_budget_nonzero_reward": sum(
            (
                _float(row.get("unsafe")) > 0.5
                or _float(row.get("budget_exceeded")) > 0.5
            )
            and _reward(row) > 1e-12
            for row in rows
        ),
        "unknown_nonzero_advantage": sum(
            _float(row.get("train_mask")) <= 0.5
            and abs(_float(row.get("trajectory_advantage_mean"))) > 1e-12
            for row in rows
        ),
        "uniform_group_nonzero_advantage": sum(
            group["uniform_or_unknown_nonzero_advantage_count"] for group in groups
        ),
        "wrong_above_correct_pairs": sum(group["wrong_above_correct_count"] for group in groups),
    }
    highest_cost_correct = sorted(
        (
            {
                "trajectory_identity_sha256": _row_digest(row),
                "reward": _reward(row),
                "q": int(_float(row.get("query_attempt_count"))),
                "t": int(_float(row.get("tool_response_tokens"))),
                "E": _float(row.get("E")),
            }
            for row in rows
            if _float(row.get("success")) > 0.5
        ),
        key=lambda value: (value["E"], value["q"], value["t"]),
        reverse=True,
    )[:5]
    highest_reward_wrong = sorted(
        (
            {
                "trajectory_identity_sha256": _row_digest(row),
                "reward": _reward(row),
                "q": int(_float(row.get("query_attempt_count"))),
                "t": int(_float(row.get("tool_response_tokens"))),
                "E": _float(row.get("E")),
            }
            for row in rows
            if _float(row.get("success")) <= 0.5
        ),
        key=lambda value: value["reward"],
        reverse=True,
    )[:5]
    return {
        "step": step,
        "rows": len(rows),
        "unique_trajectory_identities": len(
            {identity for row in rows if (identity := _row_digest(row))}
        ),
        "missing_task_identity": missing_task_identity,
        "judge_state_counts": dict(sorted(Counter(str(row.get("judge_state")) for row in rows).items())),
        "reward_layer_counts": dict(sorted(Counter(str(row.get("reward_layer")) for row in rows).items())),
        "judge_reason_counts": dict(sorted(Counter(str(row.get("judge_reason")) for row in rows).items())),
        "reward": _describe(rewards),
        "numeric_distributions": {
            key: _describe(row.get(key) for row in rows if row.get(key) is not None)
            for key in NUMERIC_FIELDS
        },
        "groups": groups,
        "strict_mixed_groups": sum(group["strict_mixed"] for group in groups),
        "strict_skipped_groups": sum(not group["strict_mixed"] for group in groups),
        "boundary_violations": violations,
        "highest_reward_wrong": highest_reward_wrong,
        "highest_cost_correct": highest_cost_correct,
    }


def _training_stage_summary(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    names = (
        "queue_wait_s",
        "deserialize_s",
        "assemble_s",
        "reward_s",
        "old_log_prob_s",
        "ref_log_prob_s",
        "adv_s",
        "update_actor_s",
        "step_s",
    )
    return [
        {"step": int(match.group(1)), **dict(zip(names, map(float, match.groups()[1:])))}
        for match in TRAIN_STAGE_RE.finditer(text)
    ]


def summarize(directory: Path, training_log: Path | None = None) -> dict[str, Any]:
    steps = [_step_summary(step, _read_rows(path)) for step, path in _iter_files(directory)]
    training_stages = _training_stage_summary(training_log)
    judge_state_counts: Counter[str] = Counter()
    reward_layer_counts: Counter[str] = Counter()
    judge_reason_counts: Counter[str] = Counter()
    for step in steps:
        judge_state_counts.update(step["judge_state_counts"])
        reward_layer_counts.update(step["reward_layer_counts"])
        judge_reason_counts.update(step["judge_reason_counts"])
    return {
        "schema_version": "qwen38-tiered-canary-safe-summary-v1",
        "sensitive_fields_emitted": False,
        "identity_policy": "sha256_only_fail_closed",
        "input_directory": directory.name,
        "files": len(steps),
        "steps": steps,
        "training_stages": training_stages,
        "training_stage_distributions": {
            key: _describe(stage[key] for stage in training_stages)
            for key in (
                "queue_wait_s",
                "ref_log_prob_s",
                "update_actor_s",
                "step_s",
            )
        },
        "totals": {
            "rows": sum(step["rows"] for step in steps),
            "nominal_batches": len(steps),
            "actual_optimizer_steps_implied": sum(step["strict_mixed_groups"] > 0 for step in steps),
            "strict_mixed_groups": sum(step["strict_mixed_groups"] for step in steps),
            "strict_skipped_groups": sum(step["strict_skipped_groups"] for step in steps),
            "judge_state_counts": dict(sorted(judge_state_counts.items())),
            "reward_layer_counts": dict(sorted(reward_layer_counts.items())),
            "judge_reason_counts": dict(sorted(judge_reason_counts.items())),
            "boundary_violations": {
                key: sum(step["boundary_violations"][key] for step in steps)
                for key in (
                    "wrong_reward_gt_0_2",
                    "correct_grounded_reward_lt_0_8",
                    "guess_correct_nonzero_reward",
                    "unsafe_or_budget_nonzero_reward",
                    "unknown_nonzero_advantage",
                    "uniform_group_nonzero_advantage",
                    "wrong_above_correct_pairs",
                )
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.rollout_dir, args.training_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["totals"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
