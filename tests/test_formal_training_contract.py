from pathlib import Path

import pytest

from scripts.check_formal_data_on_ray import check_manifests, file_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_formal_50step_uses_isolated_data_and_full_pi_contract():
    script = (ROOT / "scripts" / "run_pi_formal_50step.sh").read_text(encoding="utf-8")

    assert "boss_pi_train.parquet" in script
    assert "boss_pi_val.parquet" in script
    assert "boss_pi_test.parquet" not in script
    assert "formal_pi_v2_20260803" not in script
    assert "check_boss_alignment_contract.py" in script
    assert "pi_workspace_tools.yaml" in script
    assert "pi_agent_loops.yaml" in script
    assert 'MAX_ASSISTANT_TURNS=26' in script
    assert 'MAX_USER_TURNS=25' in script
    assert 'MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS}"' in script


def test_formal_50step_is_exact_4of4_with_periodic_greedy_validation():
    script = (ROOT / "scripts" / "run_pi_formal_50step.sh").read_text(encoding="utf-8")

    expected = (
        'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"',
        'GROUPS_PER_STEP="${GROUPS_PER_STEP:-4}"',
        'EVAL_FREQ="${EVAL_FREQ:-10}"',
        'SAVE_FREQ="${SAVE_FREQ:-10}"',
        'LEARNING_RATE="${LEARNING_RATE:-1e-7}"',
        'PREWARM_GROUPS="${PREWARM_GROUPS:-8}"',
        'STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-1.0}"',
        'FASTEST_K=4',
        'OVERSAMPLE_CANDIDATES=4',
        'actor_rollout_ref.actor.megatron.optimizer_offload=False',
        'actor_rollout_ref.rollout.val_kwargs.n=1',
        'actor_rollout_ref.rollout.val_kwargs.temperature=0',
        'actor_rollout_ref.rollout.val_kwargs.do_sample=False',
        'trainer.test_freq="${EVAL_FREQ}"',
        'trainer.validation_data_dir="${OUTPUT_DIR}/validation"',
        'trainer.max_actor_ckpt_to_keep=1',
        'async_training.use_trainer_do_validate=False',
    )
    for item in expected:
        assert item in script
    assert "check_formal_data_on_ray.py" in script


def test_formal_launcher_records_lifecycle_and_exit_code():
    script = (ROOT / "scripts" / "launch_pi_formal_50step.sh").read_text(encoding="utf-8")

    assert "run_pi_formal_50step.sh" in script
    assert 'driver.pid' in script
    assert 'started_at' in script
    assert 'exit_code' in script
    assert 'finished_at' in script


def test_five_step_gate_waits_for_successful_frozen_baseline():
    script = (ROOT / "scripts" / "launch_v15_dwh_gate_after_baseline.sh").read_text(
        encoding="utf-8"
    )

    assert 'while [[ ! -f "${BASELINE_DIR}/exit_code" ]]' in script
    assert 'if [[ "${baseline_exit}" != "0" ]]' in script
    assert 'TOTAL_TRAINING_STEPS=5' in script
    assert 'EVAL_FREQ=5' in script
    assert 'SAVE_FREQ=5' in script
    assert 'launch_pi_formal_50step.sh' in script
    assert 'baseline_failed' in script
    assert 'target_failed' in script


def test_formal_data_manifest_requires_identical_files_on_both_roles(tmp_path: Path):
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    train.write_bytes(b"train")
    val.write_bytes(b"val")
    manifest = file_manifest([str(train), str(val)])

    check_manifests({"trainer": manifest, "rollout": dict(manifest)})

    mismatch = {path: dict(item) for path, item in manifest.items()}
    mismatch[str(train)]["sha256"] = "different"
    with pytest.raises(ValueError, match="data mismatch"):
        check_manifests({"trainer": manifest, "rollout": mismatch})


def test_formal_data_manifest_rejects_missing_remote_file(tmp_path: Path):
    missing = str(tmp_path / "missing.parquet")
    manifest = file_manifest([missing])

    with pytest.raises(FileNotFoundError, match="cannot read formal PI data"):
        check_manifests({"trainer": manifest, "rollout": manifest})
