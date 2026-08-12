from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pairwise_trainer_accepts_one_full_even_dataset_batch():
    source = (ROOT / "scripts" / "run_semantic_delta_pairwise_training.py").read_text(
        encoding="utf-8"
    )
    assert "positive even global batch" in source
    assert "exactly one full-dataset batch" in source
    assert "global_batch_size != 32" not in source


def test_disjoint_pairwise_canary_is_one_step_and_frozen16_gated():
    source = (ROOT / "scripts" / "run_disjoint_pairwise_canary.sh").read_text(
        encoding="utf-8"
    )
    assert "48 <= pairs <= 64" in source
    assert "data.train_batch_size=${ROWS}" in source
    assert "data.micro_batch_size_per_gpu=2" in source
    assert "trainer.total_training_steps=1" in source
    assert "pair_order=chosen_then_rejected_no_shuffle" in source
    assert "full_replay_before_frozen16_probability_gate=false" in source
    assert "original_frozen16_chosen_preferred_12_margin_improved_12_no_earlier_regressions" in source
    assert "checkpoint.save_contents=[model,extra]" in source
    assert "optimizer_checkpoint_saved=false" in source
