from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.freeze_grpo_candidate_pool import file_sha256, instruction_identity
from scripts.split_grpo_candidate_pool import (
    allocate_train_counts,
    split_pool,
    validate_split,
)


def _row(index: int, difficulty: str, answer_type: str = "table") -> dict:
    instruction = f"question-{index}"
    environment_id = f"environment_{index % 3}"
    return {
        "data_source": "test",
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": instruction},
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answer_type": answer_type,
                "expected_value_json": "1",
                "verification_sql": "SELECT 1",
            },
        },
        "extra_info": {
            "instruction_sha256": __import__("hashlib")
            .sha256(instruction.encode())
            .hexdigest(),
            "difficulty_level": difficulty,
            "environment_id": environment_id,
            "candidate_pool_source": "source-a" if index % 2 else "source-b",
            "training_allowed": False,
            "promotion_allowed": False,
        },
    }


def _source(tmp_path: Path) -> Path:
    difficulties = ["1"] * 7 + ["2"] * 38 + ["3"] * 11 + ["4"] * 52 + ["5"] * 40 + ["unknown"] * 13
    rows = [
        _row(index, difficulty, "numeric" if index % 9 == 0 else "table")
        for index, difficulty in enumerate(difficulties)
    ]
    path = tmp_path / "pool.sensitive.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_hamilton_allocation_matches_requested_128_by_difficulty():
    allocation = allocate_train_counts(
        {"1": 7, "2": 38, "3": 11, "4": 52, "5": 40, "unknown": 13},
        128,
    )

    assert allocation == {"1": 6, "2": 30, "3": 9, "4": 41, "5": 32, "unknown": 10}


def test_split_is_deterministic_disjoint_and_owner_authorized(tmp_path: Path):
    source = _source(tmp_path)
    train = tmp_path / "train.sensitive.parquet"
    test = tmp_path / "test.sensitive.parquet"
    safe = tmp_path / "split.safe.json"
    environments = tmp_path / "environments.sensitive.json"

    summary = split_pool(
        source,
        train_path=train,
        test_path=test,
        safe_summary_path=safe,
        private_environment_manifest_path=environments,
        expected_source_sha256=file_sha256(source),
        expected_rows=161,
        train_rows=128,
        seed="fixed-seed",
        authorization_record="owner-request-test",
        owner_authorized_training=True,
    )

    train_rows = pq.read_table(train).to_pylist()
    test_rows = pq.read_table(test).to_pylist()
    assert len(train_rows) == 128
    assert len(test_rows) == 33
    assert summary["difficulty_counts"]["train"] == {
        "1": 6,
        "2": 30,
        "3": 9,
        "4": 41,
        "5": 32,
        "unknown": 10,
    }
    assert summary["difficulty_counts"]["test"] == {
        "1": 1,
        "2": 8,
        "3": 2,
        "4": 11,
        "5": 8,
        "unknown": 3,
    }
    assert {instruction_identity(row) for row in train_rows}.isdisjoint(
        {instruction_identity(row) for row in test_rows}
    )
    assert all(row["extra_info"]["training_allowed"] for row in train_rows)
    assert all(not row["extra_info"]["training_allowed"] for row in test_rows)
    assert all(row["extra_info"]["evaluation_only"] for row in test_rows)
    assert json.loads(environments.read_text(encoding="utf-8"))["environment_ids"] == [
        "environment_0",
        "environment_1",
        "environment_2",
    ]
    assert "environment_ids" not in json.loads(safe.read_text(encoding="utf-8"))

    second_train = tmp_path / "train2.parquet"
    second_test = tmp_path / "test2.parquet"
    split_pool(
        source,
        train_path=second_train,
        test_path=second_test,
        safe_summary_path=tmp_path / "safe2.json",
        private_environment_manifest_path=tmp_path / "env2.json",
        expected_source_sha256=file_sha256(source),
        expected_rows=161,
        train_rows=128,
        seed="fixed-seed",
        authorization_record="owner-request-test",
        owner_authorized_training=True,
    )
    assert [instruction_identity(row) for row in train_rows] == [
        instruction_identity(row) for row in pq.read_table(second_train).to_pylist()
    ]


def test_split_requires_owner_authorization_and_exact_source_hash(tmp_path: Path):
    source = _source(tmp_path)
    common = dict(
        train_path=tmp_path / "train.parquet",
        test_path=tmp_path / "test.parquet",
        safe_summary_path=tmp_path / "safe.json",
        private_environment_manifest_path=tmp_path / "env.json",
        expected_rows=161,
        train_rows=128,
        seed="fixed",
        authorization_record="owner-request-test",
    )
    with pytest.raises(ValueError, match="owner-authorized-training"):
        split_pool(
            source,
            expected_source_sha256=file_sha256(source),
            owner_authorized_training=False,
            **common,
        )
    with pytest.raises(ValueError, match="source sha256 mismatch"):
        split_pool(
            source,
            expected_source_sha256="0" * 64,
            owner_authorized_training=True,
            **common,
        )


def test_validation_checks_sandboxes_and_detects_tampering(tmp_path: Path):
    source = _source(tmp_path)
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    safe = tmp_path / "safe.json"
    split_pool(
        source,
        train_path=train,
        test_path=test,
        safe_summary_path=safe,
        private_environment_manifest_path=tmp_path / "env.json",
        expected_source_sha256=file_sha256(source),
        expected_rows=161,
        train_rows=128,
        seed="fixed",
        authorization_record="owner-request-test",
        owner_authorized_training=True,
    )
    sandbox = tmp_path / "sandbox"
    for index in range(3):
        (sandbox / f"environment_{index}").mkdir(parents=True)
    result = validate_split(
        train_path=train,
        test_path=test,
        safe_summary_path=safe,
        expected_rows=161,
        expected_train_rows=128,
        sandbox_root=sandbox,
    )
    assert result["status"] == "passed"
    (sandbox / "environment_2").rmdir()
    with pytest.raises(FileNotFoundError, match="split environments are missing"):
        validate_split(
            train_path=train,
            test_path=test,
            safe_summary_path=safe,
            expected_rows=161,
            expected_train_rows=128,
            sandbox_root=sandbox,
        )
