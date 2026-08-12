from __future__ import annotations

from pathlib import Path
import sqlite3

from scripts.analyze_disjoint_pair_candidate_pool import audit_pool


def _source(task_id: str, instruction: str = "old instruction") -> dict:
    return {
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": instruction},
        ],
        "reward_model": {
            "ground_truth": {
                "task_id": task_id,
                "verification_sql": "SELECT COUNT(*) FROM shipments",
                "task_family": "count",
            }
        },
    }


def _review(task_id: str, *, current: bool) -> dict:
    return {
        "task_id": task_id,
        "split": "train",
        "review_status": "approved",
        "approved_for_grpo": True,
        "source_instruction_in_current_task_definition": current,
    }


def _manifest(task_id: str, instruction: str, sql: str = "SELECT COUNT(*) FROM shipments") -> dict:
    return {
        "task_id": task_id,
        "natural_language_instruction": instruction,
        "gold_answer": {
            "answer_type": "numeric",
            "value": 2,
            "verification_sql": sql,
        },
    }


def _database(tmp_path: Path) -> Path:
    database = tmp_path / "logistics.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE shipments(value INTEGER)")
    connection.executemany("INSERT INTO shipments VALUES (?)", [(1,), (2,)])
    connection.commit()
    connection.close()
    return database


def test_audit_rebuilds_drift_from_current_definition_and_enforces_identity_isolation(
    tmp_path: Path,
):
    rows = [_source("task_rebuilt"), _source("task_forbidden"), _source("task_broad")]
    reviews = [
        _review("task_rebuilt", current=False),
        _review("task_forbidden", current=True),
        _review("task_broad", current=True),
    ]
    manifest = {
        "task_rebuilt": _manifest("task_rebuilt", "统计 shipments 表的记录数。"),
        "task_forbidden": _manifest("task_forbidden", "统计 shipments 表的记录数。"),
        "task_broad": _manifest("task_broad", "请分析问题并给出建议。"),
    }
    forbidden = {
        "task_ids": {"task_forbidden"},
        "instruction_hashes": set(),
        "sql_hashes": set(),
    }

    result = audit_pool(
        train_rows=rows,
        review_rows=reviews,
        manifest_by_task=manifest,
        database=_database(tmp_path),
        forbidden=forbidden,
        minimum_available=1,
    )

    by_task = {row["task_id"]: row for row in result["records"]}
    assert by_task["task_rebuilt"]["tier"] == "strict_available"
    assert by_task["task_rebuilt"]["source_instruction_rebuilt"] is True
    assert by_task["task_forbidden"]["tier"] == "forbidden_overlap"
    assert by_task["task_broad"]["tier"] == "review_required"
    assert result["strict_available"] == 1
    assert result["data_gate_passed"] is True
    assert result["contains_prompts_sql_expected_values_or_tool_outputs"] is False


def test_audit_blocks_gold_mismatch_before_semantic_tiering(tmp_path: Path):
    rows = [_source("task_bad_gold")]
    reviews = [_review("task_bad_gold", current=True)]
    manifest = {
        "task_bad_gold": {
            "task_id": "task_bad_gold",
            "natural_language_instruction": "统计 shipments 表的记录数。",
            "gold_answer": {
                "answer_type": "numeric",
                "value": 99,
                "verification_sql": "SELECT COUNT(*) FROM shipments",
            },
        }
    }

    result = audit_pool(
        train_rows=rows,
        review_rows=reviews,
        manifest_by_task=manifest,
        database=_database(tmp_path),
        forbidden={"task_ids": set(), "instruction_hashes": set(), "sql_hashes": set()},
    )

    assert result["tier_counts"] == {"mechanically_blocked": 1}
    assert result["mechanical_failure_counts"] == {"gold_result_mismatch": 1}
    assert result["strict_available"] == 0
