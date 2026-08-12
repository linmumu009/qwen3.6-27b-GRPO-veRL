from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from scripts.analyze_schema_oracle_action_gate import decide
from scripts.prepare_schema_oracle_action_gate import (
    CONTRACT,
    SOURCE_CONTRACT,
    build_rows,
    schema_payload,
)


def _database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE carriers(id INTEGER PRIMARY KEY, name TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE orders(id INTEGER PRIMARY KEY, carrier_id INTEGER, amount REAL, "
        "FOREIGN KEY(carrier_id) REFERENCES carriers(id))"
    )
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
                "expected_value_json": "10",
                "verification_sql": (
                    "SELECT SUM(o.amount) FROM orders o JOIN carriers c "
                    "ON o.carrier_id = c.id"
                ),
            }
        },
        "extra_info": {},
    }


def _candidate_contract(rows: int) -> dict:
    return {"contract": SOURCE_CONTRACT, "rows": rows, "training_allowed": False}


def test_schema_payload_contains_metadata_and_selected_foreign_keys_only(
    tmp_path: Path,
) -> None:
    schema = schema_payload(
        _database(tmp_path / "db.sqlite"), ["carriers", "orders"]
    )

    assert [table["name"] for table in schema["tables"]] == ["carriers", "orders"]
    orders = schema["tables"][1]
    assert [column["name"] for column in orders["columns"]] == [
        "id",
        "carrier_id",
        "amount",
    ]
    assert orders["foreign_keys"] == [
        {"from": "carrier_id", "to_table": "carriers", "to_column": "id"}
    ]


def test_builder_injects_schema_without_mutating_hidden_verifier(tmp_path: Path) -> None:
    database = _database(tmp_path / "db.sqlite")
    source = [_row("a"), _row("b")]
    rows, evidence = build_rows(
        candidate_rows=source,
        candidate_contract=_candidate_contract(2),
        database=database,
        expected_rows=2,
    )

    assert len(rows) == len(evidence) == 2
    assert source[0]["prompt"][1]["content"] == "问题 a"
    assert "SCHEMA_ORACLE_ACTION_GATE_V1" in rows[0]["prompt"][1]["content"]
    assert rows[0]["reward_model"] == source[0]["reward_model"]
    assert source[0]["reward_model"]["ground_truth"]["verification_sql"] not in (
        rows[0]["prompt"][1]["content"]
    )
    assert "10" not in rows[0]["prompt"][1]["content"]


def test_builder_fails_closed_on_unknown_gold_table(tmp_path: Path) -> None:
    database = _database(tmp_path / "db.sqlite")
    source = _row("a")
    source["reward_model"]["ground_truth"]["verification_sql"] = (
        "SELECT value FROM missing_table"
    )
    with pytest.raises(ValueError, match="unknown table"):
        build_rows(
            candidate_rows=[source],
            candidate_contract=_candidate_contract(1),
            database=database,
            expected_rows=1,
        )


def _dataset() -> dict:
    return {
        "contract": CONTRACT,
        "rows": 64,
        "schema_contains_database_rows_tool_results_answers_or_expected_values": False,
        "prompt_contains_gold_sql": False,
        "correct_or_equivalent_runtime_floor": 32,
        "observed_wrong_pair_floor": 48,
        "training_allowed": False,
    }


def _outcomes(correct: int, wrong: int) -> dict:
    return {
        "rows": 64,
        "outcome_counts": {
            "first_query_correct_or_equivalent": correct,
            "observed_first_query_error": wrong,
            "no_readonly_query": 64 - correct - wrong,
        },
    }


def _tools() -> dict:
    return {
        "rows": 64,
        "rows_with_any_sqlite": 64,
        "rows_with_recognized_readonly_sqlite": 60,
        "duplicate_bash_calls": 0,
    }


def test_gate_prefers_runtime_schema_when_correct_floor_passes() -> None:
    result = decide(_dataset(), _outcomes(32, 30), _tools())
    assert result["gates"]["runtime_correct_or_equivalent"]["passed"] is True
    assert result["decision"]["pair_construction_allowed"] is False
    assert result["decision"]["training_allowed"] is False


def test_gate_allows_pair_construction_but_not_training_on_48_wrong() -> None:
    result = decide(_dataset(), _outcomes(5, 48), _tools())
    assert result["gates"]["observed_wrong_pair_acquisition"]["passed"] is True
    assert result["decision"]["pair_construction_allowed"] is True
    assert result["decision"]["training_allowed"] is False


def test_gate_fails_closed_when_neither_threshold_passes() -> None:
    result = decide(_dataset(), _outcomes(4, 40), _tools())
    assert result["decision"]["pair_construction_allowed"] is False
    assert result["decision"]["next_action"] == (
        "design_chosen_only_schema_conditioned_action_supervision_gate"
    )
