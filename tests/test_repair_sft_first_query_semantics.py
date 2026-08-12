from __future__ import annotations

from pathlib import Path
import sqlite3

from scripts.analyze_repair_sft_first_query_semantics import (
    classify_first_query,
    results_equivalent,
    summarize,
    training_target_decision,
)
from scripts.analyze_repair_sft_all_query_semantics import (
    classify_query_sequence,
    summarize_query_sequences,
)


def _messages(command: str | None) -> list[dict]:
    if command is None:
        return [{"role": "assistant", "content": "I cannot query this."}]
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "bash", "arguments": {"command": command}},
                }
            ],
        }
    ]


def _database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE shipments(category TEXT, amount INTEGER)")
    connection.executemany(
        "INSERT INTO shipments VALUES (?, ?)",
        [("A", 1), ("A", 2), ("B", 10)],
    )
    connection.commit()
    connection.close()
    return path


def test_result_equivalence_ignores_row_order_but_tracks_column_names():
    result = results_equivalent(
        ["category", "value"],
        [("A", 3), ("B", 10)],
        ["bucket", "value"],
        [("B", 10.0), ("A", 3.0)],
    )

    assert result["row_value_multiset_equal"] is True
    assert result["column_and_row_multiset_equal"] is False


def test_first_query_semantic_gate_classifies_gold_support_and_wrong_evidence(tmp_path: Path):
    database = _database(tmp_path / "logistics.sqlite")
    truth = {
        "answer_type": "numeric",
        "expected": 3,
        "verification_sql": "SELECT SUM(amount) FROM shipments WHERE category = 'A'",
    }

    equivalent = classify_first_query(
        database=database,
        messages=_messages(
            "sqlite3 -json /workspace/logistics.sqlite \"SELECT 1 + 2 AS total\""
        ),
        truth=truth,
    )
    wrong = classify_first_query(
        database=database,
        messages=_messages(
            "sqlite3 -json /workspace/logistics.sqlite \"SELECT SUM(amount) FROM shipments\""
        ),
        truth=truth,
    )

    assert equivalent["category"] == "verified_gold_support"
    assert equivalent["teacher_result_equivalent"] is True
    assert wrong["category"] == "executable_wrong_or_insufficient_evidence"
    assert wrong["executable"] is True


def test_first_query_semantic_gate_fails_closed_on_missing_and_invalid_queries(tmp_path: Path):
    database = _database(tmp_path / "logistics.sqlite")
    truth = {
        "answer_type": "numeric",
        "expected": 3,
        "verification_sql": "SELECT SUM(amount) FROM shipments WHERE category = 'A'",
    }

    missing = classify_first_query(database=database, messages=_messages(None), truth=truth)
    invalid = classify_first_query(
        database=database,
        messages=_messages(
            "sqlite3 -json /workspace/logistics.sqlite \"SELECT nope FROM missing_table\""
        ),
        truth=truth,
    )

    assert missing["category"] == "no_readonly_query"
    assert invalid["category"] == "schema_syntax_or_execution_error"
    assert invalid["error_class"] == "OperationalError"


def test_training_target_requires_rank_gate_and_uses_half_support_boundary():
    rows = [
        {
            "category": "verified_gold_support" if index < 2 else "executable_wrong_or_insufficient_evidence",
            "gold_supported": index < 2,
            "teacher_result_equivalent": index < 2,
            "executable": True,
            "nonempty": True,
        }
        for index in range(4)
    ]
    decision = training_target_decision(summarize(rows))

    assert decision["selected_training_target"] == "post_evidence_synthesis_recovery_and_stopping"
    assert decision["token_rank_gate_still_required"] is True
    assert decision["training_must_not_start_from_this_cpu_gate_alone"] is True


def test_all_query_semantic_gate_locates_bounded_recovery(tmp_path: Path):
    database = _database(tmp_path / "logistics.sqlite")
    truth = {
        "answer_type": "numeric",
        "expected": 3,
        "verification_sql": "SELECT SUM(amount) FROM shipments WHERE category = 'A'",
    }
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "wrong",
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": "sqlite3 -json /workspace/logistics.sqlite \"SELECT SUM(amount) FROM shipments\""
                        },
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "recover",
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": "sqlite3 -json /workspace/logistics.sqlite \"SELECT SUM(amount) FROM shipments WHERE category = 'A'\""
                        },
                    },
                }
            ],
        },
    ]

    result = classify_query_sequence(database=database, messages=messages, truth=truth)
    summary = summarize_query_sequences([result])

    assert result["first_verified_or_equivalent_query_index"] == 2
    assert result["verified_or_equivalent_within_1"] is False
    assert result["verified_or_equivalent_within_2"] is True
    assert summary["tasks_within_1"] == 0
    assert summary["tasks_within_3"] == 1
