from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.assemble_qwen38_step70_mixed27 import assemble, identity, validate


ROOT = Path(__file__).resolve().parents[1]


def make_rows(version_counts: dict[str, int], *, offset: int, original: bool) -> list[dict]:
    rows: list[dict] = []
    index = offset
    for version, count in version_counts.items():
        for _ in range(count):
            instruction = hashlib.sha256(f"instruction-{index}".encode()).hexdigest()
            gold = hashlib.sha256(f"gold-{index}".encode()).hexdigest()
            extra = {
                "instruction_sha256": instruction,
                "gold_sha256": gold,
                "source_version": f"20260628_{version}",
                "difficulty_level": index % 5 + 1,
                "training_allowed": False,
                "promotion_allowed": False,
            }
            if original:
                extra["strict_reward_contract"] = "banded-v2-strict-table-v1"
            rows.append(
                {
                    "data_source": "test",
                    "prompt": [{"role": "user", "content": f"question-{index}"}],
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": {
                            "answer_type": "table",
                            "expected_value_json": json.dumps([{"category": "A", "value": index}]),
                            "verification_sql": f"SELECT 'A', {index}",
                        },
                    },
                    "extra_info": extra,
                }
            )
            index += 1
    return rows


def write_sources(tmp_path: Path, rows: list[dict], prefix: str) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    chunks = [rows[0::3], rows[1::3], rows[2::3]]
    for host, chunk in zip(("m00", "m05", "m06"), chunks, strict=True):
        path = tmp_path / f"{prefix}-{host}.parquet"
        pq.write_table(pa.Table.from_pylist(chunk), path)
        result.append((host, path))
    return result


def build(tmp_path: Path):
    original = make_rows({"v15": 4, "v20": 9, "v21": 2}, offset=0, original=True)
    heldout = make_rows({"v15": 5, "v20": 11, "v21": 2}, offset=100, original=False)
    canonical = tmp_path / "train27.parquet"
    schedule = tmp_path / "train27x4.parquet"
    sealed = tmp_path / "sealed6.parquet"
    safe = tmp_path / "safe.json"
    summary = assemble(
        write_sources(tmp_path, original, "original"),
        write_sources(tmp_path, heldout, "heldout"),
        canonical_path=canonical,
        schedule_path=schedule,
        sealed_path=sealed,
        safe_summary_path=safe,
        seed="test-seed",
    )
    return summary, canonical, schedule, sealed, safe


def test_assemble_freezes_27x4_and_disjoint_sealed6(tmp_path: Path) -> None:
    summary, canonical, schedule, sealed, safe = build(tmp_path)
    train_rows = pq.read_table(canonical).to_pylist()
    schedule_rows = pq.read_table(schedule).to_pylist()
    sealed_rows = pq.read_table(sealed).to_pylist()
    train_ids = {identity(row) for row in train_rows}
    sealed_ids = {identity(row) for row in sealed_rows}

    assert summary["canonical_tasks"] == 27
    assert summary["sealed_tasks"] == 6
    assert summary["schedule_groups"] == 108
    assert summary["new_rollout_trajectories"] == 864
    assert summary["optimizer_steps"] == 54
    assert summary["train_source_version_counts"] == {"v15": 7, "v20": 17, "v21": 3}
    assert summary["sealed_source_version_counts"] == {"v15": 2, "v20": 3, "v21": 1}
    assert not train_ids & sealed_ids
    assert len(schedule_rows) == 108
    assert {sum(identity(row) == task for row in schedule_rows) for task in train_ids} == {4}
    assert all(row["extra_info"]["training_allowed"] is True for row in schedule_rows)
    assert all(row["extra_info"]["training_allowed"] is False for row in sealed_rows)
    assert validate(canonical, schedule, sealed, safe)["status"] == "passed"


def test_assemble_rejects_overlap_between_original_and_heldout(tmp_path: Path) -> None:
    original = make_rows({"v15": 4, "v20": 9, "v21": 2}, offset=0, original=True)
    heldout = make_rows({"v15": 5, "v20": 11, "v21": 2}, offset=100, original=False)
    heldout[0]["extra_info"]["instruction_sha256"] = original[0]["extra_info"]["instruction_sha256"]
    heldout[0]["extra_info"]["gold_sha256"] = original[0]["extra_info"]["gold_sha256"]
    with pytest.raises(ValueError, match="overlap"):
        assemble(
            write_sources(tmp_path, original, "original"),
            write_sources(tmp_path, heldout, "heldout"),
            canonical_path=tmp_path / "train.parquet",
            schedule_path=tmp_path / "schedule.parquet",
            sealed_path=tmp_path / "sealed.parquet",
            safe_summary_path=tmp_path / "safe.json",
        )


def test_runner_uses_step70_27x4_n8_and_final_only_checkpoint() -> None:
    script = (ROOT / "scripts" / "run_pi_qwen38_step70_mixed27_4x_banded_v2.sh").read_text(encoding="utf-8")
    assert "TRAIN_TASKS=27" in script
    assert "SEALED_TASKS=6" in script
    assert "EXPOSURES_PER_TASK=4" in script
    assert "GROUPS_PER_STEP=2" in script
    assert "RESPONSES_PER_GROUP=8" in script
    assert "TOTAL_ROLLOUT_GROUPS != 108" in script
    assert "TOTAL_TRAINING_STEPS != 54" in script
    assert "global_step_70" in script
    assert 'SAVE_FREQ="${TOTAL_TRAINING_STEPS}"' in script
    assert "trainer.max_actor_ckpt_to_keep=1" in script
    assert "compute_score_banded_v2" in script
    assert "reasoning_effort=medium" in script


def test_host_wrapper_requires_step70_export_and_step54_gate() -> None:
    script = (ROOT / "scripts" / "launch_qwen38_step70_mixed27_host.sh").read_text(encoding="utf-8")
    assert "llin-qwen38-grpo-step70-hf-20260819-02" in script
    assert "MODEL_EXPORT_POLICY_STEP=70" in script
    assert "EXPECTED_CHECKPOINT_STEP=54" in script
    assert "assemble_qwen38_step70_mixed27.py" in script
    assert "run_pi_qwen38_step70_mixed27_4x_banded_v2.sh" in script
