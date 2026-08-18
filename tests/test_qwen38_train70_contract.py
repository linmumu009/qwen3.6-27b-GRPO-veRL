from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.assemble_qwen38_train70 import assemble, identity, validate


ROOT = Path(__file__).resolve().parents[1]


def _rows() -> list[dict]:
    versions = ["v15"] * 12 + ["v20"] * 39 + ["v21"] * 19
    difficulties = ["1"] * 2 + ["2"] * 25 + ["3"] * 17 + ["4"] * 16 + ["5"] * 10
    rows = []
    for index, (version, difficulty) in enumerate(zip(versions, difficulties, strict=True)):
        instruction_hash = hashlib.sha256(f"instruction-{index}".encode()).hexdigest()
        gold_hash = hashlib.sha256(f"gold-{index}".encode()).hexdigest()
        rows.append(
            {
                "data_source": "test",
                "prompt": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": f"question-{index}"},
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "answer_type": "table",
                        "expected_value_json": json.dumps([{"category": "A", "value": index}]),
                        "verification_sql": f"SELECT 'A', {index}",
                    },
                },
                "extra_info": {
                    "instruction_sha256": instruction_hash,
                    "gold_sha256": gold_hash,
                    "source_version": f"20260628_{version}",
                    "difficulty_level": difficulty,
                    "explicit_semantic_reviewed": True,
                    "semantic_review_decision": "approved_candidate",
                    "training_allowed": False,
                    "promotion_allowed": False,
                },
            }
        )
    return rows


def _sources(tmp_path: Path) -> list[tuple[str, Path]]:
    rows = _rows()
    result = []
    start = 0
    for host, count in (("m00", 21), ("m05", 20), ("m06", 29)):
        path = tmp_path / f"{host}.parquet"
        pq.write_table(pa.Table.from_pylist(rows[start : start + count]), path)
        result.append((host, path))
        start += count
    return result


def test_assemble_authorizes_exactly_two_exposures_and_keeps_promotion_disabled(tmp_path: Path) -> None:
    canonical = tmp_path / "train70.parquet"
    schedule = tmp_path / "train70x2.parquet"
    safe = tmp_path / "safe.json"
    summary = assemble(
        _sources(tmp_path),
        canonical_path=canonical,
        schedule_path=schedule,
        safe_summary_path=safe,
        seed="test-seed",
    )

    assert summary["canonical_tasks"] == 70
    assert summary["schedule_groups"] == 140
    assert summary["optimizer_steps"] == 70
    assert summary["strict_baseline_variance_tasks"] == 20
    rows = pq.read_table(schedule).to_pylist()
    counts = {key: sum(identity(row) == key for row in rows) for key in {identity(row) for row in rows}}
    assert set(counts.values()) == {2}
    assert [row["extra_info"]["qwen38_exposure"] for row in rows] == [1] * 70 + [2] * 70
    assert all(row["extra_info"]["training_allowed"] is True for row in rows)
    assert all(row["extra_info"]["promotion_allowed"] is False for row in rows)
    assert validate(canonical, schedule, safe)["status"] == "passed"


def test_assemble_rejects_a_source_row_that_was_already_training_enabled(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    table = pq.read_table(sources[0][1]).to_pylist()
    table[0]["extra_info"]["training_allowed"] = True
    pq.write_table(pa.Table.from_pylist(table), sources[0][1])
    with pytest.raises(ValueError, match="approved, disabled table contract"):
        assemble(
            sources,
            canonical_path=tmp_path / "canonical.parquet",
            schedule_path=tmp_path / "schedule.parquet",
            safe_summary_path=tmp_path / "safe.json",
        )


def test_assemble_accepts_legacy_review_schema_without_decision(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    for _, source in sources:
        rows = pq.read_table(source).to_pylist()
        for row in rows:
            row["extra_info"].pop("semantic_review_decision")
        pq.write_table(pa.Table.from_pylist(rows), source)
    summary = assemble(
        sources,
        canonical_path=tmp_path / "canonical.parquet",
        schedule_path=tmp_path / "schedule.parquet",
        safe_summary_path=tmp_path / "safe.json",
    )
    assert summary["canonical_tasks"] == 70
    assert summary["legacy_review_schema_missing_decision_is_accepted"] is True


def test_formal_qwen38_train70_script_freezes_shape_context_reward_and_final_only_save() -> None:
    script = (ROOT / "scripts" / "run_pi_qwen38_train70_2x_banded_v2.sh").read_text(encoding="utf-8")
    assert "TRAIN_TASKS=70" in script
    assert "EXPOSURES_PER_TASK=2" in script
    assert "GROUPS_PER_STEP=2" in script
    assert "RESPONSES_PER_GROUP=8" in script
    assert "MAX_PROMPT_TOKENS=4096" in script
    assert "MAX_RESPONSE_TOKENS=49152" in script
    assert "MAX_CONTEXT_TOKENS=53248" in script
    assert "AGENT_TIMEOUT_SECONDS=1800" in script
    assert "ROLLOUT_TP=4 ROLLOUT_NPUS=16" in script
    assert "OPTIMIZER_CPU_OFFLOAD=false" in script
    assert "ENGINE_OPTIMIZER_OFFLOAD=false" in script
    assert "PI_REWARD_MODE=banded_v2" in script
    assert "compute_score_banded_v2" in script
    assert "+data.apply_chat_template_kwargs.reasoning_effort=medium" in script
    assert "trainer.test_freq=-1" in script
    assert 'SAVE_FREQ="${TOTAL_TRAINING_STEPS}"' in script
    assert "trainer.max_actor_ckpt_to_keep=1" in script
    assert "temperature=1.0" in script
    assert "top_p=0.95" in script
    assert "top_k=20" in script


def test_fully_async_runner_exposes_qwen38_device_optimizer_switches() -> None:
    script = (ROOT / "scripts" / "run_pi_grpo_fully_async_tp4_pp2_cp2.sh").read_text(encoding="utf-8")
    assert 'OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"' in script
    assert 'ENGINE_OPTIMIZER_OFFLOAD="${ENGINE_OPTIMIZER_OFFLOAD:-true}"' in script
    assert '"${OPTIMIZER_CPU_OFFLOAD_ARG}"' in script
    assert '"${ENGINE_OPTIMIZER_OFFLOAD_ARG}"' in script


def test_host_launcher_uses_only_m05_trainer_and_m06_rollout() -> None:
    script = (ROOT / "scripts" / "launch_qwen38_train70_host.sh").read_text(encoding="utf-8")
    assert "llin-verl-qwen38-smoke-m05-20260817" in script
    assert "192.168.202.4" in script
    assert "llin-verl-qwen38-smoke-m06-20260817" in script
    assert "ROLLOUT_RANKS=16" in script
    assert "start_ray_qwen38_smoke_m05.sh" in script
    assert "start_ray_qwen38_smoke_m06.sh" in script
    assert "pi_runtime_preflight.py" in script
