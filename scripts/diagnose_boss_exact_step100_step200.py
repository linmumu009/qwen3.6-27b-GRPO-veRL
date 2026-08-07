#!/usr/bin/env python3
"""Explain a boss-original reward change with mutually exclusive task drivers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.compare_boss_exact_evaluations import _process_score, index_unique, read_jsonl
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from compare_boss_exact_evaluations import _process_score, index_unique, read_jsonl


EPSILON = 1e-9


def _number(value: Any) -> float:
    return float(value or 0.0)


def _task_metrics(row: dict[str, Any]) -> dict[str, Any]:
    reward = row.get("reward") or {}
    evidence = row.get("evidence") or {}
    return {
        "reward_total": _number(reward.get("reward_total")),
        "complete": int(reward.get("result_complete") or 0),
        "has_answer": int(reward.get("result_has_answer") or 0),
        "correct_numeric": int(reward.get("result_correct_numeric") or 0),
        "process_score": _process_score(reward),
        "tables_hit": _number(reward.get("process_tables_hit")),
        "fields_used": (
            None
            if reward.get("process_fields_used") is None
            else _number(reward.get("process_fields_used"))
        ),
        "task_fit": _number(reward.get("process_task_fit")),
        "n_turns": int(reward.get("efficiency_n_turns") or 0),
        "n_sql": int(reward.get("efficiency_n_sql") or 0),
        "n_cmds": int(reward.get("efficiency_n_cmds") or 0),
        "dup_cmd": int(reward.get("efficiency_dup_cmd") or 0),
        "answer_len": int(evidence.get("answer_len") or 0),
        "verdict": evidence.get("verdict"),
        "verdict_fine": evidence.get("verdict_fine"),
        "expected_tables": sorted(evidence.get("expected_tables") or []),
        "must_use_fields": sorted(evidence.get("must_use_fields") or []),
        "tables_queried": sorted(evidence.get("tables_queried") or []),
    }


def _delta(right: float | None, left: float | None) -> float | None:
    if right is None or left is None:
        return None
    return right - left


def diagnose(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    left_label: str = "step100",
    right_label: str = "step200",
) -> dict[str, Any]:
    left = index_unique(left_rows, left_label)
    right = index_unique(right_rows, right_label)
    if set(left) != set(right):
        raise ValueError("reward outputs do not contain identical task ids")

    bucket_totals = {
        "completion_churn": 0.0,
        "numeric_correctness": 0.0,
        "process_quality": 0.0,
        "unchanged_incomplete": 0.0,
    }
    component_totals = {
        "completion_churn": 0.0,
        "numeric_correctness": 0.0,
        "process_quality": 0.0,
    }
    tasks: list[dict[str, Any]] = []
    for task_id in sorted(left):
        left_metrics = _task_metrics(left[task_id])
        right_metrics = _task_metrics(right[task_id])
        reward_delta = right_metrics["reward_total"] - left_metrics["reward_total"]

        if left_metrics["has_answer"] != right_metrics["has_answer"]:
            bucket = "completion_churn"
            completion_contribution = reward_delta
            correctness_contribution = process_contribution = 0.0
        elif not left_metrics["has_answer"]:
            bucket = "unchanged_incomplete"
            completion_contribution = correctness_contribution = process_contribution = 0.0
        else:
            completion_contribution = 0.0
            correctness_contribution = 0.25 * (
                right_metrics["correct_numeric"] - left_metrics["correct_numeric"]
            )
            process_contribution = 0.5 * _delta(
                right_metrics["process_score"], left_metrics["process_score"]
            )
            if abs(correctness_contribution) > EPSILON:
                bucket = "numeric_correctness"
            else:
                bucket = "process_quality"

        contribution_total_before_rounding = (
            completion_contribution + correctness_contribution + process_contribution
        )
        scorer_rounding_residual = reward_delta - contribution_total_before_rounding
        contribution_total = contribution_total_before_rounding + scorer_rounding_residual
        if abs(contribution_total - reward_delta) > EPSILON:
            raise ValueError(
                f"{task_id}: decomposition does not reconcile: "
                f"{contribution_total} != {reward_delta}"
            )
        bucket_totals[bucket] += reward_delta
        component_totals["completion_churn"] += completion_contribution
        component_totals["numeric_correctness"] += correctness_contribution
        component_totals["process_quality"] += (
            process_contribution + scorer_rounding_residual
        )

        table_left = set(left_metrics["tables_queried"])
        table_right = set(right_metrics["tables_queried"])
        tasks.append(
            {
                "task_id": task_id,
                "driver_bucket": bucket,
                left_label: left_metrics,
                right_label: right_metrics,
                "delta": {
                    "reward_total": reward_delta,
                    "completion_contribution": completion_contribution,
                    "numeric_correctness_contribution": correctness_contribution,
                    "process_quality_contribution": process_contribution,
                    "scorer_rounding_residual": scorer_rounding_residual,
                    "process_score": _delta(
                        right_metrics["process_score"], left_metrics["process_score"]
                    ),
                    "fields_used": _delta(
                        right_metrics["fields_used"], left_metrics["fields_used"]
                    ),
                    "task_fit": right_metrics["task_fit"] - left_metrics["task_fit"],
                    "n_turns": right_metrics["n_turns"] - left_metrics["n_turns"],
                    "n_sql": right_metrics["n_sql"] - left_metrics["n_sql"],
                    "n_cmds": right_metrics["n_cmds"] - left_metrics["n_cmds"],
                    "dup_cmd": right_metrics["dup_cmd"] - left_metrics["dup_cmd"],
                    "answer_len": right_metrics["answer_len"] - left_metrics["answer_len"],
                },
                "tables_added": sorted(table_right - table_left),
                "tables_removed": sorted(table_left - table_right),
            }
        )

    total_delta = sum(task["delta"]["reward_total"] for task in tasks)
    explained_delta = sum(bucket_totals.values())
    if abs(total_delta - explained_delta) > EPSILON:
        raise ValueError("bucket totals do not reconcile to the paired reward delta")

    negative_total = -total_delta if total_delta < 0 else None
    driver_rows = []
    labels = {
        "completion_churn": "完成状态切换",
        "numeric_correctness": "数值正确性变化",
        "process_quality": "过程与字段质量变化",
    }
    for bucket in ("completion_churn", "numeric_correctness", "process_quality"):
        value = component_totals[bucket]
        driver_rows.append(
            {
                "driver": labels[bucket],
                "bucket": bucket,
                "reward_sum_delta": value,
                "reward_mean_delta": value / len(tasks),
                "share_of_net_decline": (
                    -value / negative_total if negative_total and value < 0 else 0.0
                ),
                "affected_task_count": sum(
                    1
                    for task in tasks
                    if abs(
                        task["delta"][
                            {
                                "completion_churn": "completion_contribution",
                                "numeric_correctness": "numeric_correctness_contribution",
                                "process_quality": "process_quality_contribution",
                            }[bucket]
                        ]
                        + (
                            task["delta"]["scorer_rounding_residual"]
                            if bucket == "process_quality"
                            else 0.0
                        )
                    )
                    > EPSILON
                ),
            }
        )

    return {
        "comparison": f"{left_label}_vs_{right_label}_boss_reward_driver_diagnosis",
        "task_count": len(tasks),
        "task_ids_identical": True,
        "reward_sum_delta": total_delta,
        "reward_mean_delta": total_delta / len(tasks),
        "decomposition_reconciles": abs(total_delta - explained_delta) <= EPSILON,
        "driver_rows": driver_rows,
        "component_totals": component_totals,
        "bucket_totals": bucket_totals,
        "loss_tasks": sorted(
            [task for task in tasks if task["delta"]["reward_total"] < -EPSILON],
            key=lambda task: task["delta"]["reward_total"],
        ),
        "win_tasks": sorted(
            [task for task in tasks if task["delta"]["reward_total"] > EPSILON],
            key=lambda task: task["delta"]["reward_total"],
            reverse=True,
        ),
        "all_tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", default="step100")
    parser.add_argument("--right-label", default="step200")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(
        read_jsonl(args.left),
        read_jsonl(args.right),
        args.left_label,
        args.right_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
