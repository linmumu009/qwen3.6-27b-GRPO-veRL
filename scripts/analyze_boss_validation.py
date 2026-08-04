#!/usr/bin/env python3
"""Summarize boss-primary validation JSONL without rerunning rollout."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_formal_grpo_50step import extract_bash_commands, unsafe_reasons
from llin_verl.pi_tool_contract import command_unsafe_reasons


METRICS = (
    "score",
    "boss_reward",
    "boss_result_score",
    "boss_process_score",
    "boss_efficiency_score",
    "boss_answer_correct",
    "boss_numbers_match",
    "boss_fields_used",
    "boss_task_fit",
    "evidence_reward",
    "acc",
    "final_answer_correct",
    "sql_evidence_correct",
    "required_table_used",
    "successful_bash",
    "safe",
    "valid_tool_protocol",
    "has_final_answer",
    "gold_sql_verified",
    "online_eligible",
)


def iter_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                yield row


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    output = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            output.append(float(value))
    return output


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = numeric_values(rows, key)
    return {
        "present": len(values),
        "missing_or_non_numeric": len(rows) - len(values),
        "mean": fmean(values) if values else None,
        "positive": sum(value > 0 for value in values),
        "positive_rate": sum(value > 0 for value in values) / len(values) if values else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("validation result is empty")
    tasks = [str((row.get("gts") or {}).get("task_id") or "") for row in rows]
    answer_types = [str((row.get("gts") or {}).get("answer_type") or "unknown") for row in rows]
    by_answer_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, answer_type in zip(rows, answer_types, strict=True):
        by_answer_type[answer_type].append(row)

    mismatches = []
    for row in rows:
        eligible = bool(row.get("safe") and row.get("valid_tool_protocol") and row.get("gold_sql_verified"))
        expected = (
            0.7 * float(row.get("boss_reward") or 0)
            + 0.3 * float(row.get("evidence_reward") or 0)
            if eligible
            else 0.0
        )
        recorded = float(row.get("score") or 0)
        if not math.isclose(recorded, expected, abs_tol=1e-6):
            mismatches.append({"task_id": str((row.get("gts") or {}).get("task_id") or ""), "recorded": recorded, "expected": expected})

    commands_by_row = [
        extract_bash_commands(str(row.get("output") or ""))
        for row in rows
    ]
    unsafe_by_row = [
        [reason for command in commands for reason in unsafe_reasons(command)]
        for commands in commands_by_row
    ]
    formal_unsafe_by_row = [
        [(command, command_unsafe_reasons(command)) for command in commands if command_unsafe_reasons(command)]
        for commands in commands_by_row
    ]

    return {
        "integrity": {
            "rows": len(rows),
            "unique_task_ids": len(set(tasks)),
            "duplicate_task_ids": sorted(task for task, count in Counter(tasks).items() if count > 1),
            "reward_formula_mismatches": len(mismatches),
            "verifier_errors": sum(bool(row.get("verifier_error")) for row in rows),
            "none_fields": {
                key: sum(row.get(key) is None for row in rows)
                for key in sorted({key for row in rows for key in row})
                if any(row.get(key) is None for row in rows)
            },
        },
        "answer_types": dict(Counter(answer_types)),
        "metrics": {key: metric_summary(rows, key) for key in METRICS},
        "by_answer_type": {
            answer_type: {
                "rows": len(group),
                "score_mean": metric_summary(group, "score")["mean"],
                "boss_reward_mean": metric_summary(group, "boss_reward")["mean"],
                "evidence_reward_mean": metric_summary(group, "evidence_reward")["mean"],
                "strict_correct": sum(float(row.get("acc") or 0) > 0 for row in group),
            }
            for answer_type, group in sorted(by_answer_type.items())
        },
        "safety_reconstruction": {
            "bash_commands": sum(len(commands) for commands in commands_by_row),
            "rows_with_unsafe_command": sum(bool(reasons) for reasons in unsafe_by_row),
            "unsafe_reasons": dict(Counter(reason for reasons in unsafe_by_row for reason in reasons)),
            "safe_field_vs_reconstruction_mismatches": sum(
                (float(row.get("safe") or 0) > 0) == bool(reasons)
                for row, reasons in zip(rows, unsafe_by_row, strict=True)
            ),
            "formal_rows_with_unsafe_command": sum(bool(commands) for commands in formal_unsafe_by_row),
            "formal_unsafe_commands": sum(len(commands) for commands in formal_unsafe_by_row),
            "formal_unsafe_reasons": dict(
                Counter(
                    reason
                    for commands in formal_unsafe_by_row
                    for _, reasons in commands
                    for reason in reasons
                )
            ),
            "safe_field_vs_formal_replay_mismatches": sum(
                (float(row.get("safe") or 0) > 0) == bool(commands)
                for row, commands in zip(rows, formal_unsafe_by_row, strict=True)
            ),
            "formal_unsafe_examples": [
                {
                    "task_id": str((row.get("gts") or {}).get("task_id") or ""),
                    "command": command[:240],
                    "reasons": reasons,
                }
                for row, commands in zip(rows, formal_unsafe_by_row, strict=True)
                for command, reasons in commands[:1]
            ][:10],
        },
        "reward_mismatch_examples": mismatches[:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = []
    for path in args.paths:
        paths.extend(sorted(path.glob("*.jsonl")) if path.is_dir() else [path])
    result = summarize(list(iter_rows(paths)))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
