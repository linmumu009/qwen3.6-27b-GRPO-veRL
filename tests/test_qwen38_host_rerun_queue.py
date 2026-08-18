from argparse import Namespace
from pathlib import Path

from scripts.run_qwen38_host_rerun_queue import arm_command, parse_arm


def test_host_queue_builds_sequential_strict_medium_arm_command(tmp_path: Path) -> None:
    args = Namespace(
        project_root=tmp_path,
        model=tmp_path / "model",
        model_label="qwen38-27b-grpo-step70",
        policy_step=70,
        rollout_resource="q38_m00",
        tensor_parallel_size=4,
        data_parallel_size=3,
        rollout_npus=12,
        max_num_seqs=16,
        task_batch_size=24,
        rolling_window_trajectories=60,
        monitor_first_card=2,
        monitor_num_cards=6,
        runs_dir=tmp_path / "runs",
        run_prefix="q38-rerun",
        state_root=tmp_path / "state",
        final_root=tmp_path / "final",
        queue_dir=tmp_path / "queue",
        ray_address="10.10.2.2:46379",
        poll_seconds=30,
        stage_timeout_seconds=1209600,
    )
    command = arm_command(args, "v21", tmp_path / "v21.parquet", 300)
    rendered = " ".join(command)
    assert "--reasoning-effort medium" in rendered
    assert "--model-label qwen38-27b-grpo-step70" in rendered
    assert "--policy-step 70" in rendered
    assert "--tensor-parallel-size 4" in rendered
    assert "--data-parallel-size 3" in rendered
    assert "--rollout-npus 12" in rendered
    assert "--max-num-seqs 16" in rendered
    assert "--rolling-window-trajectories 60" in rendered
    assert "--monitor-first-card 2" in rendered
    assert "--monitor-num-cards 6" in rendered
    assert "--expected-tasks 300" in rendered


def test_parse_arm_preserves_version_and_private_dataset_path() -> None:
    version, path = parse_arm("v20=/private/v20.parquet")
    assert version == "v20"
    assert path == Path("/private/v20.parquet")
