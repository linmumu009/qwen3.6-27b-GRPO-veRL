#!/usr/bin/env python3
"""Apply the frozen native-anchored Pareto promotion gate to a candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.compare_boss_exact_evaluations import read_jsonl, summarize


ATTRIBUTION_CONTRACT = "native-vs-step120-full25-attribution-safe-summary-v1"
GATE_CONTRACT = "native-anchored-full25-pareto-promotion-gate-v1"


def evaluate(
    *,
    attribution: dict[str, Any],
    candidate_tools: dict[str, Any],
    candidate_reward_rows: list[dict[str, Any]],
    candidate_label: str,
) -> dict[str, Any]:
    if attribution.get("contract") != ATTRIBUTION_CONTRACT:
        raise ValueError("attribution contract mismatch")
    if candidate_tools.get("rows") != 64 or len(candidate_reward_rows) != 64:
        raise ValueError("Pareto gate requires the frozen 64-task protocol")
    thresholds = attribution["decision"]["future_pareto_gate"]
    boss = summarize(candidate_reward_rows)
    wrong_process_ok = int(
        boss["verdict_fine_counts"].get("result_wrong_process_ok") or 0
    )
    metrics = {
        "recognized_sqlite_tasks": int(
            candidate_tools["rows_with_recognized_readonly_sqlite"]
        ),
        "complete_count": int(boss["complete_count"]),
        "correct_numeric_count": int(boss["correct_numeric_count"]),
        "reward_total_mean": float(boss["reward_total_mean"]),
        "wrong_process_ok": wrong_process_ok,
    }
    checks = {
        "recognized_sqlite_task_floor": metrics["recognized_sqlite_tasks"]
        >= int(thresholds["recognized_sqlite_task_floor"]),
        "complete_count_floor": metrics["complete_count"]
        >= int(thresholds["complete_count_floor"]),
        "correct_numeric_count_floor": metrics["correct_numeric_count"]
        >= int(thresholds["correct_numeric_count_floor"]),
        "reward_total_mean_floor": metrics["reward_total_mean"]
        >= float(thresholds["reward_total_mean_floor"]),
        "wrong_process_ok_ceiling": metrics["wrong_process_ok"]
        <= int(thresholds["wrong_process_ok_ceiling"]),
    }
    passed = all(checks.values())
    return {
        "contract": GATE_CONTRACT,
        "candidate_label": candidate_label,
        "tasks": 64,
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": checks,
        "gate_passed": passed,
        "promotion_allowed": passed,
        "additional_training_allowed": False,
        "contains_raw_prompts_sql_answers_task_ids_or_tool_outputs": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-summary", type=Path, required=True)
    parser.add_argument("--candidate-tools", type=Path, required=True)
    parser.add_argument("--candidate-reward", type=Path, required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        attribution=json.loads(args.attribution_summary.read_text(encoding="utf-8")),
        candidate_tools=json.loads(args.candidate_tools.read_text(encoding="utf-8")),
        candidate_reward_rows=read_jsonl(args.candidate_reward),
        candidate_label=args.candidate_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
