from __future__ import annotations

import copy

import pytest

from scripts.analyze_structured_sqlite_realization_gate import decide
from scripts.prepare_query_initiation_oracle_candidates import (
    CONTRACT as QUERY_CONTRACT,
    INTERVENTION as QUERY_INTERVENTION,
)
from scripts.prepare_structured_sqlite_realization_gate import (
    CONTRACT,
    INTERVENTION,
    build_rows,
)


def _row(task: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": "boss system"},
            {"role": "user", "content": f"问题 {task}" + QUERY_INTERVENTION},
        ],
        "reward_model": {"ground_truth": {"task_id": task}},
        "extra_info": {"split": "diagnostic"},
    }


def _source_contract(rows: int = 2) -> dict:
    return {"contract": QUERY_CONTRACT, "rows": rows, "training_allowed": False}


def test_builder_replaces_generic_oracle_with_structured_workflow() -> None:
    source = [_row("a"), _row("b")]
    frozen = copy.deepcopy(source)
    rows, evidence = build_rows(
        query_oracle_rows=source,
        query_oracle_contract=_source_contract(),
        expected_rows=2,
    )

    assert source == frozen
    assert len(rows) == len(evidence) == 2
    assert rows[0]["prompt"][1]["content"] == "问题 a" + INTERVENTION
    assert QUERY_INTERVENTION not in rows[0]["prompt"][1]["content"]
    assert rows[0]["reward_model"] == source[0]["reward_model"]
    assert rows[0]["extra_info"]["structured_sqlite_realization_contract"] == CONTRACT


def test_builder_fails_closed_on_missing_source_intervention() -> None:
    source = _row("a")
    source["prompt"][1]["content"] = "问题 a"
    with pytest.raises(ValueError, match="intervention missing"):
        build_rows(
            query_oracle_rows=[source],
            query_oracle_contract=_source_contract(1),
            expected_rows=1,
        )


def _dataset(rows: int, floor: int) -> dict:
    return {
        "contract": CONTRACT,
        "rows": rows,
        "intervention_discloses_task_specific_answer_schema_query_or_literal": False,
        "observed_readonly_query_recovery_floor": floor,
        "training_allowed": False,
    }


def _outcome(rows: int, observed: int) -> dict:
    return {
        "rows": rows,
        "outcome_counts": {
            "observed_first_query_error": observed,
            "no_readonly_query": rows - observed,
        },
    }


def _tools(rows: int) -> dict:
    return {
        "rows": rows,
        "rows_with_any_sqlite": rows,
        "rows_with_schema_discovery_sqlite": rows - 2,
        "rows_with_recognized_readonly_sqlite": rows - 1,
        "duplicate_bash_calls": 3,
    }


def test_gate_routes_pass_to_runtime_validation_without_training() -> None:
    result = decide(_dataset(41, 31), _outcome(41, 31), _tools(41))

    assert result["gate"]["passed"] is True
    assert result["decision"]["diagnosis"] == (
        "query_realization_is_recoverable_with_structured_runtime_workflow"
    )
    assert result["decision"]["training_allowed"] is False


def test_gate_routes_failure_to_schema_grounded_data_gate() -> None:
    result = decide(_dataset(41, 31), _outcome(41, 20), _tools(41))

    assert result["gate"]["passed"] is False
    assert result["decision"]["next_action"] == (
        "build_at_least_48_mechanically_verified_schema_grounded_action_pairs"
    )
    assert result["decision"]["promotion_allowed"] is False
