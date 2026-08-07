#!/usr/bin/env python3
"""Summarize boss-aligned GRPO rollout signal drift without storing raw trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = (
    "score",
    "boss_reward",
    "boss_result_score",
    "boss_process_score",
    "boss_answer_correct",
    "boss_numbers_match",
    "boss_fields_used",
    "boss_task_fit",
    "has_final_answer",
    "required_table_used",
    "evidence_reward",
    "final_answer_correct",
    "sql_evidence_correct",
    "bash_command_count",
)


def _read_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        directory.glob("*.jsonl"), key=lambda item: int(item.stem) if item.stem.isdigit() else 0
    ):
        try:
            file_step = int(path.stem)
        except ValueError:
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_file_step"] = file_step
                rows.append(row)
    return rows


def _prompt_digest(row: dict[str, Any]) -> str:
    payload = json.dumps(row.get("input"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        result[f"{metric}_mean"] = mean(values) if values else None
        if metric in {
            "boss_answer_correct",
            "boss_numbers_match",
            "has_final_answer",
            "required_table_used",
            "final_answer_correct",
            "sql_evidence_correct",
        }:
            result[f"{metric}_count"] = sum(value > 0 for value in values)
    return result


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no rollout rows found")
    steps = sorted({int(row["_file_step"]) for row in rows})
    first_steps = set(steps[:25])
    last_steps = set(steps[-25:])
    first_rows = [row for row in rows if int(row["_file_step"]) in first_steps]
    last_rows = [row for row in rows if int(row["_file_step"]) in last_steps]
    first = _summary(first_rows)
    last = _summary(last_rows)

    prompt_counts = Counter(_prompt_digest(row) for row in rows)
    prompt_group_counts = Counter(count // 4 for count in prompt_counts.values())
    by_step = Counter(int(row["_file_step"]) for row in rows)
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["_file_step"]), _prompt_digest(row))].append(row)

    def mixed(group_rows: list[dict[str, Any]], field: str) -> bool:
        return len({float(row.get(field) or 0.0) for row in group_rows}) > 1

    def all_positive(group_rows: list[dict[str, Any]], field: str) -> bool:
        return all(float(row.get(field) or 0.0) > 0 for row in group_rows)

    def any_positive(group_rows: list[dict[str, Any]], field: str) -> bool:
        return any(float(row.get(field) or 0.0) > 0 for row in group_rows)

    group_rows = list(groups.values())
    deltas = {
        key: last[key] - first[key]
        for key in first
        if key.endswith("_mean") and first[key] is not None and last[key] is not None
    }
    return {
        "step_range": [steps[0], steps[-1]],
        "step_count": len(steps),
        "row_count": len(rows),
        "group_count": len(rows) // 4,
        "rows_per_step": dict(sorted(by_step.items())),
        "first_quartile": {
            "step_range": [min(first_steps), max(first_steps)],
            **first,
        },
        "last_quartile": {
            "step_range": [min(last_steps), max(last_steps)],
            **last,
        },
        "last_minus_first_mean_deltas": deltas,
        "prompt_exposure": {
            "unique_prompts": len(prompt_counts),
            "mean_groups_per_prompt": (len(rows) / 4) / len(prompt_counts),
            "minimum_groups_per_prompt": min(prompt_counts.values()) / 4,
            "maximum_groups_per_prompt": max(prompt_counts.values()) / 4,
            "prompt_count_by_group_exposure": {
                str(groups): count for groups, count in sorted(prompt_group_counts.items())
            },
        },
        "within_group_signal": {
            "total_groups": len(group_rows),
            "invalid_group_size_count": sum(len(group) != 4 for group in group_rows),
            "zero_score_variance_count": sum(not mixed(group, "score") for group in group_rows),
            "zero_score_variance_rate": sum(not mixed(group, "score") for group in group_rows)
            / len(group_rows),
            "numeric_correctness_mixed_count": sum(
                mixed(group, "boss_answer_correct") for group in group_rows
            ),
            "numeric_correctness_mixed_rate": sum(
                mixed(group, "boss_answer_correct") for group in group_rows
            )
            / len(group_rows),
            "numeric_correctness_all_wrong_count": sum(
                not any_positive(group, "boss_answer_correct") for group in group_rows
            ),
            "numeric_correctness_all_correct_count": sum(
                all_positive(group, "boss_answer_correct") for group in group_rows
            ),
            "completion_mixed_count": sum(mixed(group, "has_final_answer") for group in group_rows),
            "fields_used_mixed_count": sum(mixed(group, "boss_fields_used") for group in group_rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(_read_rows(args.rollout_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
