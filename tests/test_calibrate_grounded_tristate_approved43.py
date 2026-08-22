from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import calibrate_grounded_tristate_approved43 as module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_calibration_is_exact_344_and_missing_human_labels_blocks(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "sft" / "v1" / "logistics.sqlite"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE fact_metric(value REAL)")
        connection.executemany("INSERT INTO fact_metric VALUES (?)", [(10.0,), (20.0,)])

    dataset: list[dict] = []
    tasks: list[dict] = []
    manifest: list[dict] = []
    observations: list[dict] = []
    for task_index in range(43):
        identity = f"instruction-{task_index:02d}"
        truth = {
            "environment_id": "sft/v1",
            "answer_type": "numeric",
            "expected_value": 30.0,
            "verification_sql": "SELECT SUM(value) FROM fact_metric",
            "required_tables": ["fact_metric"],
            "must_use_fields": ["value"],
            "abs_tol": 1e-3,
            "rel_tol": 1e-5,
        }
        dataset.append({
            "data_source": "dwh",
            "prompt": [{"role": "user", "content": f"task {task_index}"}],
            "reward_model": {"ground_truth": truth},
            "extra_info": {"instruction_sha256": identity, "training_allowed": False},
        })
        tasks.append({
            "evidence_plan": {"metric": "sum"},
            "expected_tables": ["fact_metric"],
            "verification_criteria": {"must_use_fields": ["value"]},
        })
        manifest.append({"instruction_sha256": identity})
        for sample in range(8):
            observations.append({
                "source_task_index": task_index,
                "sample_index": sample,
                "output": (
                    '<tool_call><function=bash><parameter=command>sqlite3 /workspace/logistics.sqlite '
                    '"SELECT SUM(value) FROM fact_metric"</parameter></tool_call>'
                    '<tool_response>30</tool_response>\n最终答案是 30。'
                ),
                "trajectory_timeout": False,
                "runtime_error": False,
            })

    dataset_path = tmp_path / "dataset.parquet"
    approved_path = tmp_path / "approved.parquet"
    tasks_path = tmp_path / "tasks.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    shards = tmp_path / "shards"
    shards.mkdir()
    pq.write_table(pa.Table.from_pylist(dataset), dataset_path)
    pq.write_table(pa.Table.from_pylist(dataset), approved_path)
    write_jsonl(tasks_path, tasks)
    write_jsonl(manifest_path, manifest)
    write_jsonl(shards / "tasks_00000_00043.jsonl", observations)
    monkeypatch.setattr(module, "APPROVED_SHA256", module.file_sha256(approved_path))
    monkeypatch.setattr(module, "MANIFEST_SHA256", module.file_sha256(manifest_path))

    output = tmp_path / "calibration"
    summary = module.calibrate(argparse.Namespace(
        dataset=dataset_path,
        tasks=tasks_path,
        shards=shards,
        database=database,
        approved43=approved_path,
        manifest=manifest_path,
        previous_shadow=None,
        human_labels=None,
        output_dir=output,
    ))

    assert summary["input_gate"]["trajectory_rows"] == 344
    assert summary["input_gate"]["unique_trajectory_identities"] == 344
    assert summary["automatic_tristate_shadow"]["PASS"] == 344
    assert summary["human_calibration"]["status"] == "pending"
    assert summary["human_calibration"]["completed_rows"] == 0
    assert "human_344_calibration_incomplete" in summary["blockers"]
    assert summary["formal_training_allowed"] is False
    assert len(module.read_jsonl(output / "private" / "human_labels_template.sensitive.jsonl")) == 344
    if os.name != "nt":
        assert (output / "private" / "human_calibration_packet.sensitive.jsonl").stat().st_mode & 0o777 == 0o600
