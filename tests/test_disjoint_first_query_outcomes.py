from __future__ import annotations

from pathlib import Path
import sqlite3

from scripts.analyze_disjoint_first_query_outcomes import audit_first_query_outcomes
from scripts.compare_disjoint_first_query_outcomes import compare_outcomes


def _database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute("create table metric(id integer, amount integer)")
    connection.executemany("insert into metric values (?, ?)", [(1, 5), (2, 15)])
    connection.commit()
    connection.close()
    return path


def _row(task: str) -> dict:
    return {
        "reward_model": {
            "ground_truth": {
                "task_id": task,
                "answer_type": "numeric",
                "expected_value_json": "20",
                "verification_sql": "SELECT SUM(amount) FROM metric",
            }
        }
    }


def _rollout(task: str, sql: str | None, *, observed: bool = True) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": "question"}]
    if sql is None:
        messages.append({"role": "assistant", "content": "no query"})
        return messages
    call_id = f"call_{task}"
    messages.append(
        {
            "role": "assistant",
            "content": "query",
            "tool_calls": [
                {
                    "id": call_id,
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": f"sqlite3 -json db.sqlite '{sql}'"
                        },
                    },
                }
            ],
        }
    )
    if observed:
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": "observed"}
        )
    return messages


def test_audit_classifies_all_rows_without_emitting_payloads(tmp_path: Path) -> None:
    tasks = ["correct", "wrong", "missing", "unobserved"]
    result = audit_first_query_outcomes(
        replay_rows=[_row(task) for task in tasks],
        rollout_messages={
            "correct": _rollout("correct", "SELECT SUM(amount) FROM metric"),
            "wrong": _rollout("wrong", "SELECT amount FROM metric LIMIT 1"),
            "missing": _rollout("missing", None),
            "unobserved": _rollout(
                "unobserved", "SELECT amount FROM metric LIMIT 1", observed=False
            ),
        },
        database=_database(tmp_path / "db.sqlite"),
        model_source="native",
    )

    assert result["rows"] == 4
    assert result["outcome_counts"] == {
        "first_query_correct_or_equivalent": 1,
        "first_readonly_query_without_observed_tool_result": 1,
        "no_readonly_query": 1,
        "observed_first_query_error": 1,
    }
    assert result["first_error_category_counts"] == {
        "executable_wrong_or_insufficient_evidence": 1
    }
    assert result["all_rows_classified"] is True
    assert result["contains_raw_prompts_sql_answers_task_ids_or_tool_outputs"] is False
    assert result["training_allowed"] is False


def test_paired_comparison_reports_only_aggregate_transitions() -> None:
    result = compare_outcomes(
        {
            "task_a": {"outcome": "no_readonly_query"},
            "task_b": {"outcome": "observed_first_query_error"},
            "task_c": {"outcome": "first_query_correct_or_equivalent"},
        },
        {
            "task_a": {"outcome": "observed_first_query_error"},
            "task_b": {"outcome": "no_readonly_query"},
            "task_c": {"outcome": "first_query_correct_or_equivalent"},
        },
    )

    assert result["outcome_transition_counts"] == {
        "first_query_correct_or_equivalent -> first_query_correct_or_equivalent": 1,
        "no_readonly_query -> observed_first_query_error": 1,
        "observed_first_query_error -> no_readonly_query": 1,
    }
    assert result["observed_first_query_presence"] == {
        "both_observed": 1,
        "native_only_observed": 1,
        "step120_only_observed": 1,
    }
    assert result["contains_raw_prompts_sql_answers_task_ids_or_tool_outputs"] is False
