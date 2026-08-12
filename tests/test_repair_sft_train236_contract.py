from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repair_sft_overfit_is_train_only_and_saves_final_model_without_optimizer():
    script = (ROOT / "scripts" / "run_repair_sft_train236_overfit.sh").read_text(encoding="utf-8")

    assert "train_split=train236_only" in script
    assert "heldout_overlap=0" in script
    assert "intermediate_validation=false" in script
    assert "trainer.save_freq=-1" in script
    assert "trainer.test_freq=-1" in script
    assert "'checkpoint.save_contents=[model,extra]'" in script
    assert "optimizer_checkpoint_saved=false" in script
    assert "checkpoint.save_contents=[model,optimizer" not in script


def test_repair_sft_overfit_uses_step120_model_only_and_verified_qwen_dataset():
    script = (ROOT / "scripts" / "run_repair_sft_train236_overfit.sh").read_text(encoding="utf-8")

    assert "global_step_120" in script
    assert "checkpoint_initialization=model_only_dist_ckpt" in script
    assert 'DATASET_NAME="${DATASET_NAME:-Qwen36AssistantMaskSFTDataset}"' in script
    assert '"data.custom_cls.name=${DATASET_NAME}"' in script
    assert "engine.tensor_model_parallel_size=${TP}" in script
    assert "engine.pipeline_model_parallel_size=${PP}" in script
    assert "engine.context_parallel_size=${CP}" in script
    assert "replay_gate=at_least_14_of_16_exact_boss_reward_success" in script


def test_sql_weighted_canary_is_one_variable_one_step_from_step120():
    script = (ROOT / "scripts" / "run_repair_sft_sql_weighted_canary.sh").read_text(
        encoding="utf-8"
    )

    assert "semantic_gate_verified_first_query_support=0_of_16" in script
    assert "intervention=sql_payload_weight_only" in script
    assert "model_state_correction_examples=0" in script
    assert "TOTAL_STEPS=1" in script
    assert "TOTAL_EPOCHS=1" in script
    assert "MAX_LENGTH=8192" in script
    assert "--max-length 8192" in script
    assert 'SQL_PAYLOAD_WEIGHT="${SQL_PAYLOAD_WEIGHT:-8.0}"' in script
    assert 'TOOL_STRUCTURE_WEIGHT="${TOOL_STRUCTURE_WEIGHT:-0.25}"' in script
    assert "Qwen36SQLWeightedSFTDataset" in script
    assert "python3 -m scripts.check_sql_weighted_sft_dataset" in script


def test_state_conditioned_canary_keeps_wrong_assistant_turn_as_zero_loss_context():
    script = (ROOT / "scripts" / "run_repair_sft_state_conditioned_canary.sh").read_text(
        encoding="utf-8"
    )
    core = (ROOT / "scripts" / "run_repair_sft_train236_overfit.sh").read_text(
        encoding="utf-8"
    )

    assert "only_causal_change=step120_first_wrong_sql_and_observed_tool_result_as_zero_loss_context" in script
    assert "error_context_assistant_loss_weight=0" in script
    assert "supervised_assistant_turn_indices=1,2" in script
    assert "TOTAL_STEPS=1" in script
    assert "TOTAL_EPOCHS=1" in script
    assert "Qwen36SQLWeightedSFTDataset" in script
    assert "python3 -m scripts.check_state_conditioned_sft_dataset" in script
    assert "train236-state-conditioned-repair-sft-dataset-v1" in core
    assert "all_error_context_loss_mass_zero" in core
