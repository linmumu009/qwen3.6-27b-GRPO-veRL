from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import build_grounded_verifier_casepack as module


def test_all_43_task_casepacks_run_and_block_training(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "sft" / "v1" / "logistics.sqlite"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE fact_metric(value REAL)")
        connection.executemany("INSERT INTO fact_metric VALUES (?)", [(10.0,), (20.0,)])

    approved = []
    manifest = []
    tasks = []
    for index in range(43):
        identity = f"instruction-{index:02d}"
        approved.append({
            "extra_info": {
                "instruction_sha256": identity,
                "global_index": index,
                "training_allowed": False,
            },
            "reward_model": {"ground_truth": {
                "environment_id": "sft/v1",
                "answer_type": "numeric",
                "expected_value": 30.0,
                "verification_sql": "SELECT SUM(value) FROM fact_metric",
                "required_tables": ["fact_metric"],
                "must_use_fields": ["value"],
                "abs_tol": 1e-3,
                "rel_tol": 1e-5,
            }},
        })
        manifest.append({"instruction_sha256": identity})
        tasks.append({
            "evidence_plan": {"metric": "sum"},
            "expected_tables": ["fact_metric"],
            "verification_criteria": {"must_use_fields": ["value"]},
        })
    approved_path = tmp_path / "approved.parquet"
    manifest_path = tmp_path / "manifest.jsonl"
    tasks_path = tmp_path / "tasks.jsonl"
    pq.write_table(pa.Table.from_pylist(approved), approved_path)
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
    tasks_path.write_text("".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8")
    monkeypatch.setattr(module, "APPROVED_SHA256", module.file_sha256(approved_path))
    monkeypatch.setattr(module, "MANIFEST_SHA256", module.file_sha256(manifest_path))

    output = tmp_path / "casepack"
    summary = module.build(argparse.Namespace(
        approved43=approved_path,
        manifest=manifest_path,
        tasks=tasks_path,
        database=database,
        output_dir=output,
    ))

    assert summary["status"] == "pass"
    assert summary["all_43_tasks_executed"] is True
    assert summary["tasks_all_pass"] == 43
    assert summary["failed_case_rows"] == 0
    assert summary["adversarial_wrong_sql_pass_count"] == 0
    assert summary["zero_wrong_semantic_mutations_pass"] is True
    assert summary["formal_training_allowed"] is False
    assert "multiquery_python_deterministic_composition" in summary["required_case_families"]
    assert "missing_tool_response" in summary["required_case_families"]
    assert "adversarial_delete_time_filter" in summary["required_case_families"]
    assert "adversarial_omit_join_condition" in summary["required_case_families"]
    assert "adversarial_wrong_unit_or_ratio" in summary["required_case_families"]
