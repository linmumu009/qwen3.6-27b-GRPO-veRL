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


def test_post_train_evaluation_is_step70_leak_free_and_three_host() -> None:
    source = (ROOT / "scripts" / "run_qwen38_post_train_heldout_host.py").read_text(
        encoding="utf-8"
    )
    for fragment in (
        'checkpoint_root = args.training_run / "checkpoints"',
        "if steps != [expected]",
        'manifest.get("training_overlap_tasks") != 0',
        '"m00": {"v15": 162, "v20": 153, "v21": 161}',
        '"m00": {"ssh": args.m00_host',
        '"--tensor-parallel-size", "4"',
        '"--data-parallel-size", "4"',
        '"--max-num-seqs", "16"',
        '"copying_verified_model_to_m06_and_m00"',
        '"starting_three_independent_tp4dp4_clusters"',
        '"env", f"PYTHONPATH={args.container_project}"',
        'f"{args.container_project}/reference/Megatron-Bridge-de93536e/src"',
        'f"PYTHONPATH={export_pythonpath}"',
        'remote_verifier = f"{remote_supervisor}/verify_model_transfer.py"',
        'f"root@{ssh_host}:{remote_verifier}"',
        '["python3", remote_verifier, "verify"',
        '"adaptive_sampling": "strict_2_plus_2_plus_2_max_6"',
    ):
        assert fragment in source

    preflight = source.index('status(args.supervisor_dir, "preflighting_frozen_heldout_data")')
    wait = source.index("wait_for_training(args)", preflight)
    reverify = source.index('status(args.supervisor_dir, "reverifying_frozen_heldout_data")')
    assert preflight < wait < reverify


def test_qwen38_training_and_evaluation_ray_starts_install_none_logprob_patch() -> None:
    for name in (
        "start_ray_qwen38_smoke_m05.sh",
        "start_ray_qwen38_smoke_m06.sh",
        "start_ray_qwen38_topology_benchmark.sh",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "patch_verl_none_rollout_logprobs.py" in source
        assert "--detach-utils /verl/verl/experimental/fully_async_policy/detach_utils.py" in source


def test_qwen38_formal_training_saves_dist_checkpoint_for_exact_export() -> None:
    source = (ROOT / "scripts" / "run_pi_qwen38_train70_2x_banded_v2.sh").read_text(
        encoding="utf-8"
    )
    assert "actor_rollout_ref.actor.megatron.use_dist_checkpointing=True" in source


def test_qwen38_host_queue_clears_only_its_stale_exit_code() -> None:
    source = (ROOT / "scripts" / "run_qwen38_host_rerun_queue.py").read_text(encoding="utf-8")
    assert 'stale_exit = args.queue_dir / "exit_code"' in source
    assert "stale_exit.unlink()" in source
