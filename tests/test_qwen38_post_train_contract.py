from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_success_requires_exact_step70_checkpoint() -> None:
    source = (ROOT / "scripts" / "launch_qwen38_train70_host.sh").read_text(encoding="utf-8")
    for fragment in (
        "verifying_step70_checkpoint",
        'root = run / "checkpoints"',
        'expected = root / "global_step_70"',
        'model_format == "megatron_dist_checkpoint"',
        "model_shards > 0",
        "training_exit=86",
    ):
        assert fragment in source


def test_post_train_evaluation_is_step70_leak_free_and_two_host() -> None:
    source = (ROOT / "scripts" / "run_qwen38_post_train_heldout_host.py").read_text(
        encoding="utf-8"
    )
    for fragment in (
        'checkpoint_root = args.training_run / "checkpoints"',
        "if steps != [expected]",
        'manifest.get("training_overlap_tasks") != 0',
        '"m05": 715, "m06": 715',
        '"--tensor-parallel-size", "4"',
        '"--data-parallel-size", "4"',
        '"--max-num-seqs", "16"',
        '"copying_verified_model_to_m06"',
        '"adaptive_sampling": "strict_2_plus_2_plus_2_max_6"',
    ):
        assert fragment in source


def test_qwen38_training_and_evaluation_ray_starts_install_none_logprob_patch() -> None:
    for name in (
        "start_ray_qwen38_smoke_m05.sh",
        "start_ray_qwen38_smoke_m06.sh",
        "start_ray_qwen38_topology_benchmark.sh",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "patch_verl_none_rollout_logprobs.py" in source


def test_qwen38_formal_training_saves_dist_checkpoint_for_exact_export() -> None:
    source = (ROOT / "scripts" / "run_pi_qwen38_train70_2x_banded_v2.sh").read_text(
        encoding="utf-8"
    )
    assert "actor_rollout_ref.actor.megatron.use_dist_checkpointing=True" in source
