from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_open_dwh_admission_recovery_queue import (
    launch_v20_finalizer,
    rollout_environment,
)


def args(tmp_path: Path, *, max_num_seqs: int) -> Namespace:
    return Namespace(
        project_root=tmp_path,
        model=tmp_path / "model",
        max_num_seqs=max_num_seqs,
        ray_address="ray",
        rollout_resource="resource",
    )


def test_retry_environment_fills_exact_h05_capacity(tmp_path: Path):
    environment = rollout_environment(
        args(tmp_path, max_num_seqs=24),
        run_name="retry",
        dataset=tmp_path / "retry.parquet",
        expected_tasks=1936,
        samples_per_task=1,
        task_batch_size=48,
        analyze_on_success=False,
    )

    assert environment["EXPECTED_TASKS"] == "1936"
    assert environment["SAMPLES_PER_TASK"] == "1"
    assert environment["TASK_BATCH_SIZE"] == "48"
    assert environment["ANALYZE_ON_SUCCESS"] == "0"
    assert environment["ROLLING_ADMISSION"] == "1"
    assert environment["ROLLING_WINDOW_TRAJECTORIES"] == "48"


def test_full_environment_fills_exact_h06_capacity(tmp_path: Path):
    environment = rollout_environment(
        args(tmp_path, max_num_seqs=32),
        run_name="full",
        dataset=tmp_path / "full.parquet",
        expected_tasks=250,
        samples_per_task=8,
        task_batch_size=8,
        analyze_on_success=True,
    )

    assert environment["SAMPLES_PER_TASK"] == "8"
    assert environment["TASK_BATCH_SIZE"] == "8"
    assert environment["ANALYZE_ON_SUCCESS"] == "1"
    assert environment["ROLLING_ADMISSION"] == "1"
    assert environment["ROLLING_WINDOW_TRAJECTORIES"] == "64"


def test_queue_rejects_oversubscribed_full_batch(tmp_path: Path):
    with pytest.raises(ValueError, match="requested=384, capacity=48"):
        rollout_environment(
            args(tmp_path, max_num_seqs=24),
            run_name="bad",
            dataset=tmp_path / "bad.parquet",
            expected_tasks=250,
            samples_per_task=8,
            task_batch_size=48,
            analyze_on_success=True,
        )


def test_v20_finalizer_freezes_single_arm_shape(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_run(command, *, env, check):
        captured.update({"command": command, "env": env, "check": check})

    full_args = args(tmp_path, max_num_seqs=32)
    full_args.v20_reconciled_dir = tmp_path / "reconciled"
    full_args.v20_original_dataset = tmp_path / "original.parquet"
    full_args.v20_original_run_dir = tmp_path / "original-run"
    full_args.v20_retry_dataset = tmp_path / "retry.parquet"
    full_args.v20_retry_run_dir = tmp_path / "retry-run"
    monkeypatch.setattr(
        "scripts.run_open_dwh_admission_recovery_queue.subprocess.run", fake_run
    )

    launch_v20_finalizer(full_args)

    assert captured["env"]["EXPECTED_TASKS"] == "250"
    assert captured["env"]["SAMPLES_PER_TASK"] == "8"
    assert captured["check"] is True
