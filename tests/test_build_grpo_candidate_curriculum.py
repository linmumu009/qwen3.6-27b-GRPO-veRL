from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.build_grpo_candidate_curriculum import (
    build_curriculum,
    curriculum_bucket,
    validate_curriculum,
)
from scripts.freeze_grpo_candidate_pool import file_sha256, instruction_identity


def _row(index: int, bucket: int) -> dict:
    instruction = f"curriculum-question-{index}"
    extra = {
        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "difficulty_level": str(index % 3 + 1),
        "environment_id": f"environment_{index % 2}",
        "training_allowed": True,
        "promotion_allowed": False,
    }
    if bucket == 2 and index % 2 == 0:
        extra.update(
            {
                "candidate_pool_source": "v20_adaptive_h05",
                "adaptive_screen_samples": 2,
                "adaptive_topup_samples": 0,
                "adaptive_screen_correct_count": 1,
            }
        )
    elif bucket in (2, 4, 6):
        extra.update(
            {
                "candidate_pool_source": "v21_h06",
                "adaptive_mixed_after_samples": bucket,
            }
        )
    else:
        extra["candidate_pool_source"] = "planfirst300"
    return {
        "data_source": "test",
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": instruction},
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answer_type": "numeric",
                "expected_value_json": "1",
                "verification_sql": "SELECT 1",
            },
        },
        "extra_info": extra,
    }


def _canonical(tmp_path: Path) -> Path:
    rows = [
        _row(0, 2),
        _row(1, 2),
        _row(2, 4),
        _row(3, 4),
        _row(4, 6),
        _row(5, 6),
        _row(6, 8),
        _row(7, 8),
    ]
    path = tmp_path / "train.sensitive.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_curriculum_bucket_keeps_legacy_eight_distinct():
    assert curriculum_bucket(_row(0, 2)) == 2
    assert curriculum_bucket(_row(1, 2)) == 2
    assert curriculum_bucket(_row(2, 4)) == 4
    assert curriculum_bucket(_row(4, 6)) == 6
    assert curriculum_bucket(_row(6, 8)) == 8


def test_build_curriculum_orders_phases_and_repeats_every_task(tmp_path: Path):
    canonical = _canonical(tmp_path)
    curriculum = tmp_path / "train8x3.curriculum.sensitive.parquet"
    safe = tmp_path / "curriculum.safe.json"
    summary = build_curriculum(
        canonical,
        output_path=curriculum,
        safe_summary_path=safe,
        expected_canonical_sha256=file_sha256(canonical),
        expected_tasks=8,
        exposures=3,
        groups_per_step=2,
        seed="test-seed",
    )

    rows = pq.read_table(curriculum).to_pylist()
    assert len(rows) == 24
    assert summary["task_bucket_counts"] == {2: 2, 4: 2, 6: 2, 8: 2}
    assert summary["scheduled_bucket_counts"] == {2: 6, 4: 6, 6: 6, 8: 6}
    assert [row["extra_info"]["candidate_curriculum_bucket"] for row in rows] == (
        [2] * 6 + [4] * 6 + [6] * 6 + [8] * 6
    )
    identities = [instruction_identity(row) for row in rows]
    assert set(
        {identity: identities.count(identity) for identity in set(identities)}.values()
    ) == {3}
    assert validate_curriculum(
        canonical,
        curriculum,
        safe,
        expected_tasks=8,
        exposures=3,
    )["status"] == "passed"

    second = tmp_path / "second.parquet"
    build_curriculum(
        canonical,
        output_path=second,
        safe_summary_path=tmp_path / "second.safe.json",
        expected_canonical_sha256=file_sha256(canonical),
        expected_tasks=8,
        exposures=3,
        groups_per_step=2,
        seed="test-seed",
    )
    assert file_sha256(curriculum) == file_sha256(second)


def test_curriculum_rejects_wrong_hash_and_tampered_order(tmp_path: Path):
    canonical = _canonical(tmp_path)
    with pytest.raises(ValueError, match="canonical train sha256 mismatch"):
        build_curriculum(
            canonical,
            output_path=tmp_path / "bad.parquet",
            safe_summary_path=tmp_path / "bad.json",
            expected_canonical_sha256="0" * 64,
            expected_tasks=8,
        )

    curriculum = tmp_path / "curriculum.parquet"
    safe = tmp_path / "safe.json"
    build_curriculum(
        canonical,
        output_path=curriculum,
        safe_summary_path=safe,
        expected_canonical_sha256=file_sha256(canonical),
        expected_tasks=8,
        exposures=1,
    )
    rows = pq.read_table(curriculum).to_pylist()
    rows[0], rows[-1] = rows[-1], rows[0]
    pq.write_table(pa.Table.from_pylist(rows), curriculum)
    with pytest.raises(ValueError, match="not ordered"):
        validate_curriculum(
            canonical,
            curriculum,
            safe,
            expected_tasks=8,
            exposures=1,
        )
