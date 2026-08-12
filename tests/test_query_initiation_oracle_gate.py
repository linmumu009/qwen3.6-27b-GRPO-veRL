from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from scripts.analyze_query_initiation_oracle_gate import decide
from scripts.prepare_query_initiation_oracle_candidates import (
    CONTRACT,
    INTERVENTION,
    build_oracle_rows,
)


def _database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute("create table metric(amount integer)")
    connection.execute("insert into metric values (5)")
    connection.commit()
    connection.close()
    return path


def _row(task: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": "boss system"},
            {"role": "user", "content": f"问题 {task}"},
        ],
        "reward_model": {
            "ground_truth": {
                "task_id": task,
                "answer_type": "numeric",
                "expected_value_json": "5",
                "verification_sql": "SELECT SUM(amount) FROM metric",
            }
        },
        "extra_info": {"split": "train"},
    }


def _messages(task: str, sql: str | None) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": f"问题 {task}"}]
    if sql is None:
        messages.append({"role": "assistant", "content": "inspect files"})
        return messages
    call_id = f"call_{task}"
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "query",
                "tool_calls": [
                    {
                        "id": call_id,
                        "function": {
                            "name": "bash",
                            "arguments": {
                                "command": f"sqlite3 db.sqlite '{sql}'"
                            },
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": "observed"},
        ]
    )
    return messages


def test_builder_selects_only_full_budget_no_query_rows_without_mutating_source(
    tmp_path: Path,
) -> None:
    rows = [_row("no_query_a"), _row("queried"), _row("no_query_b")]
    output, evidence = build_oracle_rows(
        replay_rows=rows,
        baseline_messages={
            "no_query_a": _messages("no_query_a", None),
            "queried": _messages("queried", "SELECT SUM(amount) FROM metric"),
            "no_query_b": _messages("no_query_b", None),
        },
        database=_database(tmp_path / "db.sqlite"),
        expected_rows=2,
    )

    assert [item["task_id"] for item in evidence] == ["no_query_a", "no_query_b"]
    assert len(output) == 2
    assert rows[0]["prompt"][1]["content"] == "问题 no_query_a"
    assert output[0]["prompt"][1]["content"].endswith(INTERVENTION)
    assert output[0]["reward_model"] == rows[0]["reward_model"]
    assert output[0]["extra_info"]["query_initiation_baseline_outcome"] == (
        "no_readonly_query"
    )
    assert "SELECT" not in INTERVENTION.upper()


def test_builder_fails_closed_when_baseline_subset_size_drifts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 2 no-query"):
        build_oracle_rows(
            replay_rows=[_row("only")],
            baseline_messages={"only": _messages("only", None)},
            database=_database(tmp_path / "db.sqlite"),
            expected_rows=2,
        )


def _dataset(rows: int, floor: int) -> dict:
    return {
        "contract": CONTRACT,
        "rows": rows,
        "baseline_outcome_for_all_selected_rows": "no_readonly_query",
        "intervention_discloses_answer_table_field_sql_or_literal": False,
        "observed_readonly_query_recovery_floor": floor,
        "training_allowed": False,
    }


def _audit(rows: int, observed: int, correct: int = 0) -> dict:
    return {
        "contract": "disjoint-first-query-outcome-audit-v1",
        "rows": rows,
        "outcome_counts": {
            "first_query_correct_or_equivalent": correct,
            "observed_first_query_error": observed - correct,
            "no_readonly_query": rows - observed,
        },
    }


def test_gate_routes_prompt_recoverable_failures_without_authorizing_training() -> None:
    result = decide(_dataset(41, 31), _audit(41, 31, correct=3))

    assert result["gate"]["passed"] is True
    assert result["decision"]["diagnosis"] == (
        "query_start_is_policy_routing_sensitive"
    )
    assert result["decision"]["training_allowed"] is False
    assert result["decision"]["promotion_allowed"] is False


def test_gate_separates_partial_and_low_prompt_recovery() -> None:
    partial = decide(_dataset(41, 31), _audit(41, 21))
    low = decide(_dataset(41, 31), _audit(41, 8))

    assert partial["decision"]["diagnosis"] == (
        "query_start_is_partially_policy_routing_sensitive"
    )
    assert low["decision"]["next_target"] == (
        "schema_discovery_and_tool_realization_repair"
    )
