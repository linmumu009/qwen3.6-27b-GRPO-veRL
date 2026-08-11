from __future__ import annotations

from pathlib import Path
import sqlite3

from scripts.prepare_repair_sft_dataset import (
    build_sft_row,
    eligible_candidates,
    select_diverse_candidates,
)


def _source_row(task_id: str, answer_type: str = "numeric", family: str = "count") -> dict:
    expected = "2" if answer_type == "numeric" else '[{"category":"A","value":2}]'
    sql = "SELECT COUNT(*) AS value FROM shipments" if answer_type == "numeric" else (
        "SELECT category, COUNT(*) AS value FROM shipments GROUP BY category"
    )
    return {
        "prompt": [
            {"role": "system", "content": "boss-system"},
            {"role": "user", "content": "question"},
        ],
        "reward_model": {
            "ground_truth": {
                "task_id": task_id,
                "answer_type": answer_type,
                "expected_value_json": expected,
                "verification_sql": sql,
                "task_family": family,
                "environment_id": "sft/test",
            }
        },
    }


def _review(task_id: str, instruction: str = "统计数量") -> dict:
    return {
        "task_id": task_id,
        "split": "train",
        "instruction": instruction,
        "gold": {"answer_type": "numeric", "verification_sql": "SELECT COUNT(*) FROM shipments"},
        "review_status": "approved",
        "approved_for_grpo": True,
        "source_instruction_in_current_task_definition": True,
    }


def test_candidate_gate_rejects_drift_and_disallowed_semantic_warnings():
    rows = [_source_row("task_clean"), _source_row("task_drift"), _source_row("task_broad")]
    reviews = [_review("task_clean"), _review("task_drift"), _review("task_broad", "请分析问题和建议")]
    reviews[1]["source_instruction_in_current_task_definition"] = False

    candidates = eligible_candidates(rows, reviews)

    assert [item["task_id"] for item in candidates] == ["task_clean"]


def test_selection_prioritizes_clean_rows_then_balances_answer_types():
    candidates = [
        {"task_id": "clean", "warnings": [], "answer_type": "numeric", "task_family": "a"},
        {
            "task_id": "numeric",
            "warnings": ["latest_instruction_without_temporal_sql"],
            "answer_type": "numeric",
            "task_family": "b",
        },
        {
            "task_id": "table",
            "warnings": ["latest_instruction_without_temporal_sql"],
            "answer_type": "table",
            "task_family": "c",
        },
    ]

    selected = select_diverse_candidates(candidates, 3, "seed")

    assert selected[0]["task_id"] == "clean"
    assert {item["answer_type"] for item in selected} == {"numeric", "table"}


def test_build_sft_row_uses_exact_boss_contract_and_one_readonly_query(tmp_path: Path):
    database = tmp_path / "logistics.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE shipments(category TEXT)")
    connection.executemany("INSERT INTO shipments VALUES (?)", [("A",), ("A",)])
    connection.commit()
    connection.close()
    candidate = {
        "row": _source_row("task_000001"),
        "task_id": "task_000001",
        "warnings": [],
        "answer_type": "numeric",
        "task_family": "count",
    }
    contract = {
        "system_prompt": "boss-system",
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }

    row, evidence = build_sft_row(candidate, contract, database)

    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    call = row["messages"][2]["tool_calls"][0]
    assert call["function"]["name"] == "bash"
    assert "sqlite3 -json /workspace/logistics.sqlite" in call["function"]["arguments"]["command"]
    assert row["messages"][3]["tool_call_id"] == call["id"]
    assert "**2**" in row["messages"][-1]["content"]
    assert evidence["sql_rows"] == 1
