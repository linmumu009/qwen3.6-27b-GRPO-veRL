#!/usr/bin/env python3
"""Summarize correctness and reward separation in a bounded canary window."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _number(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    return float(value or 0)


def _average(values: list[float]) -> float | None:
    return mean(values) if values else None


def analyze(
    rollout_dir: Path,
    min_rollout_step: int,
    max_rollout_step: int,
    expected_group_size: int = 8,
) -> dict[str, Any]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    selected_files: list[str] = []
    for path in sorted(rollout_dir.glob("*.jsonl"), key=lambda item: int(item.stem)):
        step = int(path.stem)
        if step < min_rollout_step or step > max_rollout_step:
            continue
        selected_files.append(path.name)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_hash = hashlib.sha256(
                str(row.get("input") or "").encode("utf-8")
            ).hexdigest()
            groups[(step, prompt_hash)].append(row)

    valid_groups = [rows for rows in groups.values() if len(rows) == expected_group_size]
    mixed_groups: list[list[dict[str, Any]]] = []
    all_wrong_groups = 0
    all_correct_groups = 0
    variable_reward_groups = 0
    correct_scores: list[float] = []
    wrong_scores: list[float] = []
    mixed_margins: list[float] = []
    mixed_correct_counts: list[int] = []
    mixed_reward_ranges: list[float] = []
    all_wrong_reward_ranges: list[float] = []
    mixed_strict_ranked = 0

    for rows in valid_groups:
        flags = [_number(row, "final_answer_correct") > 0 for row in rows]
        scores = [_number(row, "score") for row in rows]
        if len(set(scores)) > 1:
            variable_reward_groups += 1
        if all(flags):
            all_correct_groups += 1
        elif not any(flags):
            all_wrong_groups += 1
            all_wrong_reward_ranges.append(max(scores) - min(scores))
        else:
            mixed_groups.append(rows)
            group_correct = [score for score, flag in zip(scores, flags) if flag]
            group_wrong = [score for score, flag in zip(scores, flags) if not flag]
            mixed_correct_counts.append(len(group_correct))
            mixed_reward_ranges.append(max(scores) - min(scores))
            margin = min(group_correct) - max(group_wrong)
            mixed_margins.append(margin)
            if margin > 0:
                mixed_strict_ranked += 1
        correct_scores.extend(score for score, flag in zip(scores, flags) if flag)
        wrong_scores.extend(score for score, flag in zip(scores, flags) if not flag)

    rows = [row for group in valid_groups for row in group]
    valid_count = len(valid_groups)
    mixed_count = len(mixed_groups)
    return {
        "contract": "banded-v1-canary-signal-summary-v1",
        "rollout_step_range": {
            "min": min_rollout_step,
            "max": max_rollout_step,
            "files": selected_files,
        },
        "expected_group_size": expected_group_size,
        "row_count": len(rows),
        "group_count": len(groups),
        "valid_group_count": valid_count,
        "valid_group_rate": valid_count / len(groups) if groups else 0.0,
        "mixed_correct_group_count": mixed_count,
        "mixed_correct_group_rate": mixed_count / valid_count if valid_count else 0.0,
        "all_wrong_group_count": all_wrong_groups,
        "all_wrong_group_rate": all_wrong_groups / valid_count if valid_count else 0.0,
        "all_correct_group_count": all_correct_groups,
        "reward_variable_group_count": variable_reward_groups,
        "reward_variable_group_rate": (
            variable_reward_groups / valid_count if valid_count else 0.0
        ),
        "correct_row_count": len(correct_scores),
        "correct_row_rate": len(correct_scores) / len(rows) if rows else 0.0,
        "correct_score_mean": _average(correct_scores),
        "wrong_score_mean": _average(wrong_scores),
        "mixed_correct_strict_rank_count": mixed_strict_ranked,
        "mixed_correct_strict_rank_rate": (
            mixed_strict_ranked / mixed_count if mixed_count else 0.0
        ),
        "mixed_correct_rows_mean": _average(
            [float(value) for value in mixed_correct_counts]
        ),
        "mixed_correct_min_margin_mean": _average(mixed_margins),
        "mixed_reward_range_mean": _average(mixed_reward_ranges),
        "all_wrong_reward_range_mean": _average(all_wrong_reward_ranges),
        "has_final_answer_rate": (
            sum(_number(row, "has_final_answer") > 0 for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "sql_evidence_correct_rate": (
            sum(_number(row, "sql_evidence_correct") > 0 for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "online_eligible_rate": (
            sum(_number(row, "online_eligible") > 0 for row in rows) / len(rows)
            if rows
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--min-rollout-step", type=int, required=True)
    parser.add_argument("--max-rollout-step", type=int, required=True)
    parser.add_argument("--expected-group-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.min_rollout_step > args.max_rollout_step:
        parser.error("--min-rollout-step cannot exceed --max-rollout-step")
    result = analyze(
        args.rollout_dir,
        args.min_rollout_step,
        args.max_rollout_step,
        args.expected_group_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
