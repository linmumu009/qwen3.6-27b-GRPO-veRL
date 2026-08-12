#!/usr/bin/env python3
"""Decide the structured SQLite realization diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.prepare_structured_sqlite_realization_gate import CONTRACT


RESULT_CONTRACT = "structured-noninteractive-sqlite-realization-result-v1"


def decide(
    dataset_contract: dict[str, Any],
    outcome_audit: dict[str, Any],
    command_families: dict[str, Any],
) -> dict[str, Any]:
    if dataset_contract.get("contract") != CONTRACT:
        raise ValueError("structured SQLite dataset contract mismatch")
    rows = int(dataset_contract.get("rows") or 0)
    if rows <= 0 or int(outcome_audit.get("rows") or 0) != rows:
        raise ValueError("dataset and outcome row counts differ")
    if int(command_families.get("rows") or 0) != rows:
        raise ValueError("dataset and command-family row counts differ")
    if dataset_contract.get(
        "intervention_discloses_task_specific_answer_schema_query_or_literal"
    ) is not False:
        raise ValueError("task-specific intervention leakage guard failed")
    if dataset_contract.get("training_allowed") is not False:
        raise ValueError("dataset unexpectedly authorizes training")

    counts = outcome_audit.get("outcome_counts") or {}
    correct = int(counts.get("first_query_correct_or_equivalent") or 0)
    wrong = int(counts.get("observed_first_query_error") or 0)
    unobserved = int(
        counts.get("first_readonly_query_without_observed_tool_result") or 0
    )
    no_query = int(counts.get("no_readonly_query") or 0)
    if correct + wrong + unobserved + no_query != rows:
        raise ValueError("first-query outcomes do not cover all rows")
    observed = correct + wrong
    floor = int(dataset_contract.get("observed_readonly_query_recovery_floor") or 0)
    passed = observed >= floor

    if passed:
        diagnosis = "query_realization_is_recoverable_with_structured_runtime_workflow"
        next_action = "validate_structured_workflow_on_full64_and_frozen_val20"
    else:
        diagnosis = "task_specific_schema_grounding_or_action_supervision_required"
        next_action = "build_at_least_48_mechanically_verified_schema_grounded_action_pairs"

    return {
        "contract": RESULT_CONTRACT,
        "rows": rows,
        "first_query": {
            "observed_readonly_query_rows": observed,
            "correct_or_equivalent_rows": correct,
            "observed_wrong_rows": wrong,
            "unobserved_rows": unobserved,
            "no_readonly_query_rows": no_query,
        },
        "tools": {
            key: int(command_families.get(key) or 0)
            for key in (
                "rows_with_any_sqlite",
                "rows_with_schema_discovery_sqlite",
                "rows_with_recognized_readonly_sqlite",
                "duplicate_bash_calls",
            )
        },
        "gate": {
            "observed_readonly_query_recovery_floor": floor,
            "passed": passed,
        },
        "decision": {
            "diagnosis": diagnosis,
            "next_action": next_action,
            "training_allowed": False,
            "promotion_allowed": False,
        },
        "contains_raw_commands_prompts_sql_answers_task_ids_tool_outputs_or_server_paths": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--outcome-audit", type=Path, required=True)
    parser.add_argument("--command-families", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decide(
        json.loads(args.dataset_contract.read_text(encoding="utf-8")),
        json.loads(args.outcome_audit.read_text(encoding="utf-8")),
        json.loads(args.command_families.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
