from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from scripts.prepare_chosen_only_schema_action_sft import (
    SQLITE_COMMAND_PREFIX,
    build_action_row,
    split_rows,
)


def _database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE metric(category TEXT, amount INTEGER)")
    connection.executemany(
        "INSERT INTO metric VALUES (?, ?)", [("a", 3), ("b", 7)]
    )
    connection.commit()
    connection.close()
    return path


def _boss() -> dict:
    return {
        "system_prompt": "boss system",
        "tools": [
            {
                "type": "function",
                "function": {"name": "bash", "parameters": {"type": "object"}},
            }
        ],
    }


def _source(task: str, answer_type: str = "numeric") -> dict:
    return {
        "prompt": [
            {"role": "system", "content": "boss system"},
            {"role": "user", "content": f"问题 {task}"},
        ],
        "reward_model": {
            "ground_truth": {
                "task_id": task,
                "answer_type": answer_type,
                "expected_value_json": "10",
                "verification_sql": "SELECT SUM(amount) FROM metric",
                "task_family": "sum",
            }
        },
    }


def test_build_action_row_keeps_gold_sql_only_in_single_assistant_target(
    tmp_path: Path,
) -> None:
    row, evidence = build_action_row(
        source=_source("task_a"),
        boss_contract=_boss(),
        database=_database(tmp_path / "db.sqlite"),
    )

    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert "SELECT SUM(amount)" not in row["messages"][1]["content"]
    calls = row["messages"][2]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "bash"
    assert calls[0]["function"]["arguments"]["command"].startswith(
        SQLITE_COMMAND_PREFIX
    )
    assert row["supervised_assistant_turn_indices"] == [0]
    assert evidence["chosen_sql_rows"] == 1


def test_build_action_row_fails_when_expected_value_is_not_supported(
    tmp_path: Path,
) -> None:
    source = _source("task_a")
    source["reward_model"]["ground_truth"]["expected_value_json"] = "11"
    with pytest.raises(ValueError, match="does not support expected"):
        build_action_row(
            source=source,
            boss_contract=_boss(),
            database=_database(tmp_path / "db.sqlite"),
        )


def test_split_is_deterministic_disjoint_and_answer_type_stratified() -> None:
    rows = [{"task_id": f"task_{index}"} for index in range(8)]
    evidence = [
        {
            "task_id": f"task_{index}",
            "answer_type": "table" if index >= 6 else "numeric",
        }
        for index in range(8)
    ]
    first = split_rows(
        rows=rows,
        evidence=evidence,
        calibration_rows=2,
        seed="fixed",
    )
    second = split_rows(
        rows=rows,
        evidence=evidence,
        calibration_rows=2,
        seed="fixed",
    )

    train, calibration, summary = first
    assert first == second
    assert len(train) == 6
    assert len(calibration) == 2
    assert {row["task_id"] for row in train}.isdisjoint(
        row["task_id"] for row in calibration
    )
    assert sum(summary["calibration_answer_type_quotas"].values()) == 2
