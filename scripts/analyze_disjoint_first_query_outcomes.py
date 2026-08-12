#!/usr/bin/env python3
"""Audit first-query outcomes without creating or authorizing training data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from scripts.analyze_repair_sft_first_query_semantics import classify_first_query
from scripts.analyze_repair_sft_free_run_divergence import (
    bash_calls,
    read_openai,
    tool_outputs,
)
from scripts.prepare_disjoint_first_error_pairs import source_contract
from scripts.prepare_repair_sft_dataset import load_parquet_rows, sha256_file, task_id


CONTRACT = "disjoint-first-query-outcome-audit-v1"


def classify_first_query_outcomes(
    *,
    replay_rows: list[dict[str, Any]],
    rollout_messages: dict[str, list[dict[str, Any]]],
    database: Path,
) -> dict[str, dict[str, str]]:
    truths: dict[str, dict[str, Any]] = {}
    for row in replay_rows:
        current_task_id = task_id(row)
        if not current_task_id or current_task_id in truths:
            raise ValueError(f"missing or duplicate replay task ID: {current_task_id!r}")
        truth = (row.get("reward_model") or {}).get("ground_truth") or {}
        truths[current_task_id] = {
            "answer_type": str(truth.get("answer_type") or ""),
            "expected": json.loads(str(truth.get("expected_value_json") or "null")),
            "verification_sql": str(truth.get("verification_sql") or ""),
        }
    if set(truths) != set(rollout_messages):
        raise ValueError("rollout and replay task IDs differ")

    classified: dict[str, dict[str, str]] = {}
    for current_task_id, truth in truths.items():
        messages = rollout_messages[current_task_id]
        first_sql_call = next((call for call in bash_calls(messages) if call["sql"]), None)
        if first_sql_call is None:
            classified[current_task_id] = {"outcome": "no_readonly_query"}
            continue
        if first_sql_call["call_id"] not in tool_outputs(messages):
            classified[current_task_id] = {
                "outcome": "first_readonly_query_without_observed_tool_result"
            }
            continue
        semantic = classify_first_query(database=database, messages=messages, truth=truth)
        if semantic["gold_supported"] or semantic["teacher_result_equivalent"]:
            classified[current_task_id] = {
                "outcome": "first_query_correct_or_equivalent"
            }
            continue
        classified[current_task_id] = {
            "outcome": "observed_first_query_error",
            "first_error_category": str(semantic["category"]),
        }
    return classified


def audit_first_query_outcomes(
    *,
    replay_rows: list[dict[str, Any]],
    rollout_messages: dict[str, list[dict[str, Any]]],
    database: Path,
    model_source: str,
) -> dict[str, Any]:
    classified = classify_first_query_outcomes(
        replay_rows=replay_rows,
        rollout_messages=rollout_messages,
        database=database,
    )
    outcomes: Counter[str] = Counter()
    error_categories: Counter[str] = Counter()
    for item in classified.values():
        outcomes[item["outcome"]] += 1
        if item.get("first_error_category"):
            error_categories[item["first_error_category"]] += 1

    return {
        "contract": CONTRACT,
        "model_source": model_source,
        "rows": len(classified),
        "outcome_counts": dict(sorted(outcomes.items())),
        "first_error_category_counts": dict(sorted(error_categories.items())),
        "all_rows_classified": sum(outcomes.values()) == len(classified),
        "contains_raw_prompts_sql_answers_task_ids_or_tool_outputs": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--rollout-openai", type=Path, required=True)
    parser.add_argument("--rollout-candidate-contract", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_contract = source_contract(
        args.rollout_candidate_contract, args.replay_parquet
    )
    result = audit_first_query_outcomes(
        replay_rows=load_parquet_rows(args.replay_parquet),
        rollout_messages=read_openai(args.rollout_openai),
        database=args.database,
        model_source=args.model_source,
    )
    result["source_candidate_contract_rows"] = int(candidate_contract["rows"])
    result["source_sha256"] = {
        "replay_parquet": sha256_file(args.replay_parquet),
        "rollout_openai": sha256_file(args.rollout_openai),
        "rollout_candidate_contract": sha256_file(args.rollout_candidate_contract),
        "database": sha256_file(args.database),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
