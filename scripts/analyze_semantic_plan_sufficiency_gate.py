#!/usr/bin/env python3
"""Audit one generated recovery SQL across semantic-plan gate arms."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from scripts.analyze_repair_sft_first_query_semantics import (
    classify_first_query,
    ground_truth_by_task,
    summarize,
)
from scripts.prepare_boss_exact_evaluation import read_jsonl
from scripts.prepare_semantic_plan_sufficiency_gate import ARMS


def _generated_by_gate(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        gate_id = str(row.get("gate_id") or "")
        if not gate_id or gate_id in result:
            raise ValueError(f"missing or duplicate generated gate ID: {gate_id!r}")
        result[gate_id] = row
    return result


def decide(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    success = {
        (str(row["task_id"]), str(row["arm"])): bool(row["verified_or_equivalent"])
        for row in per_task
    }
    aggregation_tasks = sorted(
        {str(row["task_id"]) for row in per_task if row["aggregation_critical"]}
    )
    operator_aggregation = sum(success[(task_id, "operator_oracle")] for task_id in aggregation_tasks)
    control_successes = {task_id for task_id, arm in success if arm == "control" and success[(task_id, arm)]}
    operator_regressions = sorted(
        task_id
        for task_id in control_successes
        if not success[(task_id, "operator_oracle")]
    )
    full_success = sum(value for (task_id, arm), value in success.items() if arm == "full_plan_oracle")
    operator_pass = operator_aggregation >= 4 and not operator_regressions
    full_pass = full_success >= 8

    if operator_pass:
        target = "semantic_plan_selection_or_contrast_supervision"
        reason = "operator_oracle_recovers_at_least_4_of_9_aggregation_tasks_without_control_regression"
    elif full_pass:
        target = "schema_grounding_and_compositional_plan_training"
        reason = "only_full_semantic_plan_crosses_8_of_16_recovery_gate"
    else:
        target = "plan_to_sql_realization_and_recovery"
        reason = "even_full_semantic_plan_fails_8_of_16_recovery_gate"
    return {
        "operator_gate": {
            "aggregation_critical_recovered": operator_aggregation,
            "threshold": 4,
            "control_success_regressions": len(operator_regressions),
            "control_success_regression_task_ids": operator_regressions,
            "passed": operator_pass,
        },
        "full_plan_gate": {
            "verified_or_equivalent_recovered": full_success,
            "threshold": 8,
            "passed": full_pass,
        },
        "selected_training_target": target,
        "reason": reason,
        "training_should_start_only_after_margin_and_non_regression_audit": True,
    }


def analyze(
    *,
    replay_parquet: Path,
    generated_jsonl: Path,
    dataset_contract: Path,
    database: Path,
    max_rows: int = 10_000,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    order, truths = ground_truth_by_task(replay_parquet)
    generated = _generated_by_gate(generated_jsonl)
    contract = json.loads(dataset_contract.read_text(encoding="utf-8"))
    if contract.get("contract") != "semantic-plan-sufficiency-gate-dataset-v1":
        raise ValueError("semantic-plan analysis requires dataset contract v1")
    evidence = {str(row["gate_id"]): row for row in contract.get("evidence") or []}
    expected_ids = {f"{task_id}::{arm}" for task_id in order for arm in ARMS}
    if len(order) != 16 or set(generated) != expected_ids or set(evidence) != expected_ids:
        raise ValueError("semantic-plan generated rows, contract and replay tasks differ")

    per_task: list[dict[str, Any]] = []
    for arm in ARMS:
        for task_id in order:
            gate_id = f"{task_id}::{arm}"
            row = generated[gate_id]
            if row.get("task_id") != task_id or row.get("arm") != arm:
                raise ValueError(f"{gate_id}: generated identity differs")
            result = classify_first_query(
                database=database,
                messages=row.get("messages") or [],
                truth=truths[task_id],
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
            )
            per_task.append(
                {
                    "gate_id": gate_id,
                    "task_id": task_id,
                    "arm": arm,
                    "aggregation_critical": bool(evidence[gate_id]["aggregation_critical"]),
                    "semantic_difference_labels": evidence[gate_id]["semantic_difference_labels"],
                    **result,
                    "verified_or_equivalent": bool(
                        result["gold_supported"] or result["teacher_result_equivalent"]
                    ),
                }
            )

    arms: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in per_task if row["arm"] == arm]
        arm_summary = summarize(rows)
        arm_summary["verified_or_equivalent_count"] = sum(
            bool(row["verified_or_equivalent"]) for row in rows
        )
        arm_summary["aggregation_critical_verified_or_equivalent_count"] = sum(
            bool(row["verified_or_equivalent"] and row["aggregation_critical"])
            for row in rows
        )
        arm_summary["generated_bash_call_count"] = sum(
            row["category"] != "no_readonly_query" for row in rows
        )
        arm_summary["semantic_difference_recovery_counts"] = dict(
            sorted(
                Counter(
                    label
                    for row in rows
                    if row["verified_or_equivalent"]
                    for label in row["semantic_difference_labels"]
                ).items()
            )
        )
        arms[arm] = arm_summary
    return {
        "contract": "semantic-plan-sufficiency-gate-result-v1",
        "source_checkpoint": "step120",
        "task_count": len(order),
        "arms": arms,
        "decision": decide(per_task),
        "execution": {
            "generation_only_one_assistant_turn": True,
            "generated_tool_calls_not_executed_by_rollout": True,
            "analysis_read_only_sql_execution": True,
            "optimizer_initialized": False,
            "checkpoint_saved": False,
            "max_rows": max_rows,
            "timeout_seconds": timeout_seconds,
        },
        "per_task": per_task,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    result = analyze(
        replay_parquet=args.replay_parquet,
        generated_jsonl=args.generated_jsonl,
        dataset_contract=args.dataset_contract,
        database=args.database,
        max_rows=args.max_rows,
        timeout_seconds=args.timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
