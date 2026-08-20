import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.confirm_qwen38_strict_candidates import confirm


def _candidate(identity: str, difficulty: int) -> dict:
    return {
        "prompt": [{"role": "user", "content": "sensitive"}],
        "reward_model": {
            "ground_truth": {
                "answer_type": "table",
                "expected_value_json": "[]",
                "verification_sql": "select 1",
            }
        },
        "extra_info": {
            "instruction_sha256": identity,
            "source_version": "v23",
            "difficulty_level": difficulty,
            "adaptive_wave_contract": "llin-adaptive-dwh-2plus2plus2-v1",
            "adaptive_samples_observed": 2,
            "adaptive_correct_count": 1,
            "adaptive_completed_count": 2,
            "adaptive_timeout_count": 0,
            "adaptive_runtime_error_count": 0,
            "training_allowed": False,
            "promotion_allowed": False,
        },
    }


def _outcome(identity: str, *, correct: int, completed: int, runtime_errors: int = 0) -> dict:
    return {
        "instruction_sha256": identity,
        "outcome_contract": "banded-v2-strict-table-v1",
        "correct_count": correct,
        "completed_count": completed,
        "trajectory_timeout_count": 2 - completed,
        "runtime_error_count": runtime_errors,
    }


def test_confirmation_requires_two_correct_two_completed_wrong_and_zero_runtime_errors(
    tmp_path: Path,
):
    candidates = tmp_path / "candidates.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [_candidate("accept", 1), _candidate("no-second-correct", 2), _candidate("error", 3)]
        ),
        candidates,
    )
    outcomes = tmp_path / "per_task.jsonl"
    outcomes.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                _outcome("accept", correct=1, completed=2),
                _outcome("no-second-correct", correct=0, completed=2),
                _outcome("error", correct=1, completed=2, runtime_errors=1),
            )
        ),
        encoding="utf-8",
    )
    robust = tmp_path / "robust.parquet"
    rejected = tmp_path / "rejected.parquet"
    safe = tmp_path / "safe.json"

    summary = confirm(
        candidates,
        outcomes,
        robust,
        rejected,
        safe,
        expected_candidates=3,
        host_label="fixture",
    )

    assert summary["robust_candidates"] == 1
    assert summary["rejected_candidates"] == 2
    assert summary["rejection_reason_counts"] == {
        "fewer_than_two_strict_correct": 1,
        "runtime_error": 1,
    }
    accepted = pq.read_table(robust).to_pylist()[0]["extra_info"]
    assert accepted["robust_total_correct_count"] == 2
    assert accepted["robust_total_wrong_count"] == 2
    assert accepted["training_allowed"] is False
    assert "sensitive" not in safe.read_text(encoding="utf-8")
