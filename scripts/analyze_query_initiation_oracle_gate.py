#!/usr/bin/env python3
"""Choose the next target from a query-initiation oracle diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.prepare_query_initiation_oracle_candidates import CONTRACT


RESULT_CONTRACT = "query-initiation-oracle-diagnostic-result-v1"


def decide(dataset_contract: dict[str, Any], outcome_audit: dict[str, Any]) -> dict[str, Any]:
    if dataset_contract.get("contract") != CONTRACT:
        raise ValueError("query-initiation dataset contract mismatch")
    if outcome_audit.get("contract") != "disjoint-first-query-outcome-audit-v1":
        raise ValueError("first-query outcome audit contract mismatch")
    rows = int(dataset_contract.get("rows") or 0)
    if rows <= 0 or int(outcome_audit.get("rows") or 0) != rows:
        raise ValueError("dataset and outcome row counts differ")
    if dataset_contract.get("baseline_outcome_for_all_selected_rows") != "no_readonly_query":
        raise ValueError("diagnostic selection is not frozen to baseline no-query rows")
    if dataset_contract.get("intervention_discloses_answer_table_field_sql_or_literal") is not False:
        raise ValueError("intervention leakage guard failed")
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
        raise ValueError("first-query outcome counts do not cover all rows")
    observed_recovery = correct + wrong
    floor = int(dataset_contract.get("observed_readonly_query_recovery_floor") or 0)
    if not 1 <= floor <= rows:
        raise ValueError("invalid observed recovery floor")
    gate_passed = observed_recovery >= floor

    if gate_passed:
        diagnosis = "query_start_is_policy_routing_sensitive"
        next_target = "runtime_query_start_instruction_before_weight_updates"
        next_action = (
            "validate_the_same_non_answer_bearing_instruction_on_full64_and_frozen_val20"
        )
    elif observed_recovery * 2 >= rows:
        diagnosis = "query_start_is_partially_policy_routing_sensitive"
        next_target = "native_anchored_query_initiation_and_completion_contrasts"
        next_action = "collect_at_least_48_mechanically_observed_natural_contrast_pairs"
    else:
        diagnosis = "query_start_is_not_sufficiently_prompt_recoverable"
        next_target = "schema_discovery_and_tool_realization_repair"
        next_action = "build_a_mechanically_verified_schema_tool_realization_gate"

    return {
        "contract": RESULT_CONTRACT,
        "rows": rows,
        "baseline_observed_readonly_query_rows": 0,
        "intervention": {
            "observed_readonly_query_rows": observed_recovery,
            "correct_or_equivalent_first_query_rows": correct,
            "observed_wrong_first_query_rows": wrong,
            "unobserved_first_query_rows": unobserved,
            "no_readonly_query_rows": no_query,
        },
        "gate": {
            "observed_readonly_query_recovery_floor": floor,
            "passed": gate_passed,
        },
        "decision": {
            "diagnosis": diagnosis,
            "next_target": next_target,
            "next_action": next_action,
            "training_allowed": False,
            "promotion_allowed": False,
        },
        "contains_raw_prompts_sql_answers_task_ids_tool_outputs_or_server_paths": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--outcome-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decide(
        json.loads(args.dataset_contract.read_text(encoding="utf-8")),
        json.loads(args.outcome_audit.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
