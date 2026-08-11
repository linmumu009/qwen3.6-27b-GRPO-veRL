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
    assert "data.custom_cls.name=Qwen36AssistantMaskSFTDataset" in script
    assert "engine.tensor_model_parallel_size=${TP}" in script
    assert "engine.pipeline_model_parallel_size=${PP}" in script
    assert "engine.context_parallel_size=${CP}" in script
    assert "replay_gate=at_least_14_of_16_exact_boss_reward_success" in script
