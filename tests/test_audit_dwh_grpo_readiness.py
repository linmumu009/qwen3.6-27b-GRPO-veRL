import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.audit_dwh_grpo_readiness import (
    audit,
    canonical_hash,
    full_table_answer_match,
)


def test_full_table_match_requires_every_row_column_and_order():
    expected = [["a", 1.0, 2], ["b", 3.0, 4]]
    correct = """| name | value | count |
|---|---:|---:|
| a | 1.00001 | 2 |
| b | 3 | 4 |
"""
    missing_column = """| name | value |
|---|---:|
| a | 1 |
| b | 3 |
"""
    swapped = """| name | value | count |
|---|---:|---:|
| b | 3 | 4 |
| a | 1 | 2 |
"""

    assert full_table_answer_match(correct, expected, 1e-3, 1e-5)[0] is True
    assert full_table_answer_match(missing_column, expected, 1e-3, 1e-5)[0] is False
    assert full_table_answer_match(swapped, expected, 1e-3, 1e-5)[0] is False


def _task(index: int, *, answer_type: str, value, sql: str, instruction: str):
    table = "facts"
    table_task = answer_type == "table"
    return {
        "task_id": f"private-{index}",
        "task_type": "comparison_analysis" if table_task else "single_metric_query",
        "natural_language_instruction": instruction,
        "expected_tables": [table],
        "expected_fields": ["category", "value"] if table_task else ["value"],
        "expected_operations": ["filter", "group_by"] if table_task else ["filter", "aggregate"],
        "sample_sql": sql,
        "evidence_plan": {
            "aggregation": None if table_task else "SUM",
            "group_by": "category" if table_task else None,
            "order_by": "category" if table_task else None,
            "limit": None,
            "requires_percentage": False,
            "report_measures": [],
            "output_shape": "comparison" if table_task else "scalar",
        },
        "gold_answer": {
            "answer_type": answer_type,
            "value": value,
            "text": "private-gold",
            "verification_sql": sql,
        },
        "answerability_label": {"is_answerable": True, "reason": "validated"},
        "missing_requirements": {},
        "out_of_scope_reason": None,
        "validation": {
            "checked_against_database": True,
            "expected_result_exists": True,
            "validation_method": "sql_replay",
        },
        "_semantic_review": "strict_checker_pass",
    }


def _dataset_row(index: int, task: dict):
    instruction = task["natural_language_instruction"]
    gold = task["gold_answer"]
    return {
        "prompt": [
            {"role": "system", "content": "private-system"},
            {"role": "user", "content": "guidance\n" + instruction},
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answer_type": gold["answer_type"],
                "expected_value_json": json.dumps(gold["value"], ensure_ascii=False),
                "verification_sql": gold["verification_sql"],
                "abs_tol": 1e-3,
                "rel_tol": 1e-5,
            },
        },
        "extra_info": {
            "instruction_sha256": canonical_hash(instruction),
            "gold_sha256": canonical_hash(gold["value"]),
            "explicit_semantic_reviewed": True,
            "training_allowed": False,
            "promotion_allowed": False,
        },
    }


def test_audit_routes_mixed_and_all_wrong_without_leaking_sensitive_rows(tmp_path: Path):
    database = tmp_path / "logistics.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE facts(category TEXT, value REAL);
        INSERT INTO facts VALUES ('a', 1.0), ('b', 3.0);
        """
    )
    connection.close()

    # The dataset's broad ``filter`` operation tag does not imply a predicate;
    # no EvidencePlan filter means a valid query may omit WHERE entirely.
    numeric_sql = "SELECT SUM(value) FROM facts"
    table_sql = (
        "SELECT category, value FROM facts WHERE value >= 0 "
        "GROUP BY category ORDER BY category"
    )
    tasks = [
        _task(
            0,
            answer_type="numeric",
            value=4.0,
            sql=numeric_sql,
            instruction="private-mixed-secret",
        ),
        _task(
            1,
            answer_type="table",
            value=[["a", 1.0], ["b", 3.0]],
            sql=table_sql,
            instruction="private-table-secret",
        ),
        _task(
            2,
            answer_type="numeric",
            value=4.0,
            sql=numeric_sql,
            instruction="private-hard-secret",
        ),
    ]
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tasks),
        encoding="utf-8",
    )
    dataset = [_dataset_row(index, task) for index, task in enumerate(tasks)]
    dataset_path = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist(dataset), dataset_path)

    per_task = [
        {"source_task_index": 0, "bucket": "mixed", "correct_count": 4},
        {"source_task_index": 1, "bucket": "all_wrong", "correct_count": 0},
        {"source_task_index": 2, "bucket": "all_wrong", "correct_count": 0},
    ]
    per_task_path = tmp_path / "per_task.jsonl"
    per_task_path.write_text(
        "".join(json.dumps(row) + "\n" for row in per_task), encoding="utf-8"
    )

    correct_table = """| category | value |
|---|---:|
| a | 1 |
| b | 3 |
"""
    wrong_table = """| category | value |
|---|---:|
| a | 9 |
| b | 8 |
"""
    shards = tmp_path / "shards"
    shards.mkdir()
    observations = []
    for task_index in range(3):
        for sample_index in range(8):
            if task_index == 0:
                output = "final result: 4" if sample_index < 4 else "final result: 9"
            elif task_index == 1:
                output = correct_table if sample_index < 4 else wrong_table
            else:
                output = "final result: 9"
            observations.append(
                {
                    "source_task_index": task_index,
                    "sample_index": sample_index,
                    "output": output,
                    "response_tokens": 10,
                    "trajectory_timeout": False,
                    "runtime_error": False,
                }
            )
    (shards / "tasks_00000_00003.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in observations),
        encoding="utf-8",
    )

    output = tmp_path / "audit"
    summary = audit(
        dataset_path,
        tasks_path,
        database,
        shards,
        per_task_path,
        output,
        expected_tasks=3,
        samples_per_task=8,
    )

    assert summary["mixed_review"]["disposition_counts"] == {"可训练": 1}
    assert summary["all_wrong_review"]["primary_root_cause_counts"] == {
        "奖励器假阴性": 1,
        "真实高难": 1,
    }
    assert summary["all_wrong_review"]["audited_bucket_counts"] == {
        "all_wrong": 1,
        "mixed": 1,
    }
    assert summary["grpo_readiness"] == {
        "conditional_candidates_blocked_until_reward_route_fixed": 1,
        "directly_audited_mixed_candidates": 1,
        "reward_repair_conditional_mixed_candidates": 1,
        "total_nonzero_variance_candidates_after_reward_repair": 2,
    }
    assert summary["evidence_chain"]["legacy_to_audited_correct_count"] == {
        "0->0": 1,
        "0->4": 1,
        "4->4": 1,
    }
    assert summary["evidence_chain"]["plan_check_failure_counts"] == {}
    assert summary["evidence_chain"]["trajectory_signal_counts"] == {
        "environment_error_marker": 0,
        "format_valid": 24,
        "reviewed": 24,
        "runtime_error": 0,
        "sql_error_marker": 0,
        "timeout": 0,
    }
    assert summary["mixed_review"]["nonapproval_signal_counts"] == {}
    approved = pq.read_table(
        output / "private/mixed_approved_candidates.sensitive.parquet"
    ).to_pylist()
    assert len(approved) == 1
    assert approved[0]["extra_info"]["training_allowed"] is False
    repaired = pq.read_table(
        output / "private/reward_repaired_mixed_candidates.sensitive.parquet"
    ).to_pylist()
    assert len(repaired) == 1
    assert repaired[0]["extra_info"]["training_allowed"] is False

    safe_text = (output / "safe_summary.json").read_text(encoding="utf-8")
    assert "private-mixed-secret" not in safe_text
    assert "private-table-secret" not in safe_text
    assert "private-hard-secret" not in safe_text
    assert summary["contains_prompts_sql_gold_values_task_ids_final_answers_or_tool_outputs"] is False
