from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.outcome_gated_contract import evidence_binding_hash
from scripts.replay_qwen38_tiered_observability import replay


def test_offline_replay_repairs_database_path_but_never_invents_old_tool_tokens(
    tmp_path: Path,
) -> None:
    identity = "1" * 64
    truth = {
        "environment_id": "sft/v1",
        "answer_type": "numeric",
        "expected_value": 30.0,
        "verification_sql": "SELECT SUM(value) FROM fact_metric",
        "required_tables": ["fact_metric"],
        "must_use_fields": ["value"],
        "evidence_plan": {"aggregation": "sum"},
        "abs_tol": 1e-3,
        "rel_tol": 1e-5,
    }
    truth["process_evidence_binding_sha256"] = evidence_binding_hash(truth)
    dataset = tmp_path / "dataset.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "extra_info": {"instruction_sha256": identity},
                    "reward_model": {"ground_truth": truth},
                }
            ]
        ),
        dataset,
    )
    database = tmp_path / "sandbox/sft/v1/logistics.sqlite"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE fact_metric(value REAL)")
        connection.executemany("INSERT INTO fact_metric VALUES (?)", [(10.0,), (20.0,)])

    rollout = tmp_path / "rollouts"
    rollout.mkdir()
    rows = [
        {
            "task_identity_sha256": identity,
            "trajectory_identity_sha256": "2" * 64,
            "judge_state": "UNKNOWN",
            "judge_reason": "database_unavailable",
            "database_available": False,
            "output": "Final result: 30",
            "query_attempt_count": 1,
            "tool_event_count": 1,
            "unsafe": False,
            "budget_exceeded": False,
        },
        {
            "task_identity_sha256": identity,
            "trajectory_identity_sha256": "",
            "judge_state": "FAIL",
            "judge_reason": "no_relevant_readonly_attempt",
            "database_available": True,
            "output": "Final result: 30",
            "query_attempt_count": 0,
            "tool_event_count": 0,
            "unsafe": False,
            "budget_exceeded": False,
        },
    ]
    (rollout / "1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = replay(rollout, dataset, tmp_path / "sandbox")

    assert result["database_migration_counts"] == {
        "stored_False->replay_True": 1,
        "stored_True->replay_True": 1,
    }
    assert result["after"]["judge_state_counts"] == {"FAIL": 1, "UNKNOWN": 1}
    assert result["after"]["judge_reason_counts"][
        "historical_tool_response_cost_unobservable"
    ] == 1
    assert result["pass_promotion_from_missing_old_tool_evidence"] == 0
    assert result["identity"]["valid_task_sha256"] == 2
    assert result["identity"]["valid_trajectory_sha256"] == 1
    assert result["sensitive_fields_emitted"] is False
