#!/usr/bin/env python3
"""Decide runtime-schema sufficiency or same-state pair acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.prepare_schema_oracle_action_gate import CONTRACT


RESULT_CONTRACT = "task-specific-schema-oracle-action-result-v1"


def decide(
    dataset_contract: dict[str, Any],
    outcome_audit: dict[str, Any],
    command_families: dict[str, Any],
) -> dict[str, Any]:
    if dataset_contract.get("contract") != CONTRACT:
        raise ValueError("schema-oracle dataset contract mismatch")
    rows = int(dataset_contract.get("rows") or 0)
    if rows <= 0 or int(outcome_audit.get("rows") or 0) != rows:
        raise ValueError("dataset and outcome row counts differ")
    if int(command_families.get("rows") or 0) != rows:
        raise ValueError("dataset and command-family row counts differ")
    if dataset_contract.get(
        "schema_contains_database_rows_tool_results_answers_or_expected_values"
    ) is not False:
        raise ValueError("schema-oracle payload leakage guard failed")
    if dataset_contract.get("prompt_contains_gold_sql") is not False:
        raise ValueError("schema-oracle prompt leaked gold SQL")
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
    correct_floor = int(dataset_contract.get("correct_or_equivalent_runtime_floor") or 0)
    wrong_floor = int(dataset_contract.get("observed_wrong_pair_floor") or 0)
    runtime_passed = correct >= correct_floor
    pair_gate_passed = wrong >= wrong_floor

    if runtime_passed:
        diagnosis = "task_specific_schema_is_runtime_sufficient"
        next_action = (
            "validate_full_database_schema_injection_without_gold_table_selection_"
            "on_frozen_val20"
        )
    elif pair_gate_passed:
        diagnosis = "schema_grounding_enables_same_state_wrong_query_pair_acquisition"
        next_action = "build_and_cpu_audit_correct_vs_actual_wrong_pairs"
    else:
        diagnosis = "schema_oracle_still_insufficient_for_pair_acquisition"
        next_action = "design_chosen_only_schema_conditioned_action_supervision_gate"

    return {
        "contract": RESULT_CONTRACT,
        "rows": rows,
        "first_query": {
            "correct_or_equivalent_rows": correct,
            "observed_wrong_rows": wrong,
            "unobserved_rows": unobserved,
            "no_readonly_query_rows": no_query,
        },
        "tools": {
            key: int(command_families.get(key) or 0)
            for key in (
                "rows_with_any_sqlite",
                "rows_with_recognized_readonly_sqlite",
                "duplicate_bash_calls",
            )
        },
        "gates": {
            "runtime_correct_or_equivalent": {
                "observed": correct,
                "required": correct_floor,
                "passed": runtime_passed,
            },
            "observed_wrong_pair_acquisition": {
                "observed": wrong,
                "required": wrong_floor,
                "passed": pair_gate_passed,
            },
        },
        "decision": {
            "diagnosis": diagnosis,
            "next_action": next_action,
            "pair_construction_allowed": (not runtime_passed and pair_gate_passed),
            "training_allowed": False,
            "promotion_allowed": False,
        },
        "contains_raw_commands_prompts_schema_sql_answers_task_ids_tool_outputs_or_server_paths": False,
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
