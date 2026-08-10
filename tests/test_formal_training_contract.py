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


def test_formal_50step_uses_four_groups_and_only_saves_the_final_model():
    script = (ROOT / "scripts" / "run_pi_formal_50step.sh").read_text(encoding="utf-8")

    expected = (
        'export PYTHONPATH="${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"',
        'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"',
        'GROUPS_PER_STEP="${GROUPS_PER_STEP:-4}"',
        'EVAL_FREQ="${EVAL_FREQ:-10}"',
        'SAVE_FREQ="${TOTAL_TRAINING_STEPS}"',
        'LEARNING_RATE="${LEARNING_RATE:-1e-7}"',
        'PREWARM_GROUPS="${PREWARM_GROUPS:-$((GROUPS_PER_STEP * 2))}"',
        'MAX_QUEUE_GROUPS="${MAX_QUEUE_GROUPS:-$((GROUPS_PER_STEP * 2))}"',
        'STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-1.0}"',
        'FASTEST_K=4',
        'OVERSAMPLE_CANDIDATES=4',
        'actor_rollout_ref.actor.megatron.optimizer_offload=False',
        'actor_rollout_ref.actor.megatron.use_dist_checkpointing=True',
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
    assert 'GROUPS_PER_STEP != 4' in script
    assert 'SAVE_FREQ="${SAVE_FREQ:-' not in script


def test_formal_launcher_records_lifecycle_and_exit_code():
    script = (ROOT / "scripts" / "launch_pi_formal_50step.sh").read_text(encoding="utf-8")

    assert "run_pi_formal_50step.sh" in script
    assert 'driver.pid' in script
    assert 'started_at' in script
    assert 'exit_code' in script
    assert 'finished_at' in script
    assert 'verify_checkpoint_integrity.py' in script
    assert 'checkpoint_integrity.json' in script
    assert 'CHECKPOINT_INVALID' in script
    assert 'exit_code=8' in script


def test_fully_async_runner_supports_dynamic_group_shape_without_selection():
    script = (
        ROOT / "scripts" / "run_pi_grpo_fully_async_tp4_pp2_cp2.sh"
    ).read_text(encoding="utf-8")

    assert 'RESPONSES_PER_GROUP="${RESPONSES_PER_GROUP:-4}"' in script
    assert 'actor_rollout_ref.rollout.n="${RESPONSES_PER_GROUP}"' in script
    assert 'rollout.n="${RESPONSES_PER_GROUP}"' in script
    assert 'FASTEST_K != RESPONSES_PER_GROUP' in script
    assert 'CONCURRENT_SAMPLES_PER_REPLICA="${CONCURRENT_SAMPLES_PER_REPLICA:-16}"' in script


def test_banded_2x8_resume_contract_and_final_only_checkpoint():
    run_script = (ROOT / "scripts" / "run_pi_banded_2x8_resume.sh").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts" / "launch_pi_banded_2x8_resume.sh").read_text(
        encoding="utf-8"
    )

    expected = (
        "GROUPS_PER_STEP=2",
        "RESPONSES_PER_GROUP=8",
        'FASTEST_K="${RESPONSES_PER_GROUP}"',
        'OVERSAMPLE_CANDIDATES="${RESPONSES_PER_GROUP}"',
        "CONCURRENT_SAMPLES_PER_REPLICA=6",
        "reward.custom_reward_function.name=compute_score_banded_v1",
        'trainer.test_freq="${FINAL_POLICY_STEP}"',
        'trainer.save_freq="${FINAL_POLICY_STEP}"',
        "[model,optimizer,extra]",
    )
    for item in expected:
        assert item in run_script
    assert "expected_global_step_%s_got_%s" in launcher
    assert "verify_checkpoint_integrity.py" in launcher


def test_unattended_pipeline_is_fail_closed_between_stages():
    script = (
        ROOT / "scripts" / "run_unattended_accuracy_pipeline_host.sh"
    ).read_text(encoding="utf-8")

    assert "set -Eeuo pipefail" in script
    assert "flock -n" in script
    assert "replay_banded_reward_gate.py" in script
    assert "analyze_accuracy_gate.py" in script
    assert "NEW_TRAINING_STEPS=5" in script
    assert "NEW_TRAINING_STEPS=20" in script
    assert "CHECKPOINT_INVALID" not in script
    assert "ray stop --force" in script


def test_formal_100step_uses_twelve_inflight_groups_and_final_only_artifacts():
    script = (
        ROOT / "scripts" / "run_pi_formal_100step_12groups.sh"
    ).read_text(encoding="utf-8")

    expected = (
        'TOTAL_TRAINING_STEPS=100',
        'GROUPS_PER_STEP=4',
        'FINAL_EVAL_STEP="${TOTAL_TRAINING_STEPS}"',
        'FINAL_SAVE_STEP="${TOTAL_TRAINING_STEPS}"',
        'PREWARM_GROUPS=8',
        'MAX_QUEUE_GROUPS=8',
        'STALENESS_THRESHOLD=2.0',
        'TARGET_CONCURRENT_GROUPS=12',
        'ROLLOUT_GPU_MEMORY_UTILIZATION=0.80',
        'ROLLOUT_MAX_BATCHED_TOKENS=16384',
        'ROLLOUT_MAX_SEQS=24',
        'AGENT_WORKERS=12',
        'WEIGHT_BUCKET_MB=2560',
        'TOTAL_ROLLOUT_GROUPS="$((TOTAL_TRAINING_STEPS * GROUPS_PER_STEP))"',
        'FASTEST_K=4',
        'OVERSAMPLE_CANDIDATES=4',
        'WEIGHT_BUCKET_MB="${WEIGHT_BUCKET_MB}"',
        "'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]'",
        'trainer.val_before_train=False',
        'trainer.test_freq="${FINAL_EVAL_STEP}"',
        'trainer.save_freq="${FINAL_SAVE_STEP}"',
        'trainer.max_actor_ckpt_to_keep=1',
        'async_training.use_trainer_do_validate=False',
    )
    for item in expected:
        assert item in script

    assert 'EVAL_FREQ=' not in script
    assert 'SAVE_FREQ="${FINAL_SAVE_STEP}"' in script
    assert 'OVERSAMPLE_CANDIDATES=6' not in script


def test_formal_100step_launcher_requires_exact_final_checkpoint():
    script = (
        ROOT / "scripts" / "launch_pi_formal_100step_12groups.sh"
    ).read_text(encoding="utf-8")

    assert "run_pi_formal_100step_12groups.sh" in script
    assert 'latest_iteration' in script
    assert 'if [[ "${latest_iteration}" != "100" ]]' in script
    assert 'expected_global_step_100_got_%s' in script
    assert 'verify_checkpoint_integrity.py' in script
    assert 'CHECKPOINT_INVALID' in script
    assert 'exit_code=8' in script


def test_step100_to_step200_continuation_runs_exactly_100_new_updates():
    script = (
        ROOT / "scripts" / "run_pi_formal_step100_to_step200_12groups.sh"
    ).read_text(encoding="utf-8")

    expected = (
        'START_POLICY_STEP=100',
        'FINAL_POLICY_STEP="${FINAL_POLICY_STEP:-200}"',
        'TOTAL_TRAINING_STEPS="${FINAL_POLICY_STEP}"',
        'GROUPS_PER_STEP=4',
        'TOTAL_ROLLOUT_GROUPS="$((FINAL_POLICY_STEP * GROUPS_PER_STEP))"',
        'TARGET_CONCURRENT_GROUPS=12',
        'ROLLOUT_GPU_MEMORY_UTILIZATION=0.80',
        'ROLLOUT_MAX_BATCHED_TOKENS=16384',
        'ROLLOUT_MAX_SEQS=24',
        'AGENT_WORKERS=12',
        'FASTEST_K=4',
        'OVERSAMPLE_CANDIDATES=4',
        "'actor_rollout_ref.actor.checkpoint.load_contents=[model,extra]'",
        "'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]'",
        'trainer.resume_mode=resume_path',
        'trainer.resume_from_path="${RESUME_CHECKPOINT}"',
        'trainer.del_local_ckpt_after_load=False',
        'trainer.test_freq="${FINAL_EVAL_STEP}"',
        'trainer.save_freq="${FINAL_SAVE_STEP}"',
        'trainer.max_actor_ckpt_to_keep=1',
    )
    for item in expected:
        assert item in script

    assert 'optimizer_state=reset_missing_from_source' in script
    assert 'dataloader_state=reset_for_corrected_train236' in script


def test_dense_correctness_trial_is_twenty_updates_from_step100_with_same_topology():
    run_script = (
        ROOT / "scripts" / "run_pi_dense_correctness_step100_to_step120.sh"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "scripts" / "launch_pi_dense_correctness_step100_to_step120.sh"
    ).read_text(encoding="utf-8")

    assert "FINAL_POLICY_STEP=120" in run_script
    assert "PI_DENSE_CORRECTNESS_WEIGHT=0.30" in run_script
    assert "reward.custom_reward_function.name=compute_score_dense30" in run_script
    assert "run_pi_formal_step100_to_step200_12groups.sh" in run_script
    assert 'if [[ "${latest_iteration}" != "120" ]]' in launcher
    assert "expected_global_step_120_got_%s" in launcher
    assert "verify_checkpoint_integrity.py" in launcher


def test_step100_resume_view_resets_only_the_incompatible_data_cursor():
    script = (ROOT / "scripts" / "prepare_pi_step100_resume_view.sh").read_text(
        encoding="utf-8"
    )

    assert 'ln -s "${SOURCE_CHECKPOINT}/actor" "${RESUME_CHECKPOINT}/actor"' in script
    assert 'Refusing stale dataloader state' in script
    assert 'dataloader_state=reset_for_train236' in script
    assert 'rm ' not in script


def test_step100_to_step200_launcher_requires_exact_final_checkpoint():
    script = (
        ROOT / "scripts" / "launch_pi_formal_step100_to_step200_12groups.sh"
    ).read_text(encoding="utf-8")

    assert "run_pi_formal_step100_to_step200_12groups.sh" in script
    assert 'if [[ "${latest_iteration}" != "200" ]]' in script
    assert 'expected_global_step_200_got_%s' in script
    assert 'verify_checkpoint_integrity.py' in script
    assert 'CHECKPOINT_INVALID' in script
    assert 'exit_code=8' in script


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
