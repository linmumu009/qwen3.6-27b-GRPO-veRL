#!/usr/bin/env python3
"""Run one host's v15 -> v20 -> v21 admission-safe recovery queue.

The v15 timeout retry is already active when this controller starts.  The
controller waits for its reconciled arm, launches and reconciles the v20
timeout-only retry, then launches a fresh complete v21 arm.  Every transition
is fail-closed and recorded in an aggregate-only status file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.standalone_rollout_shards import trajectory_admission_contract


CONTRACT = "llin-open-dwh-admission-recovery-queue-v1"


def write_status(queue_dir: Path, *, stage: str, arm_label: str, **fields: object) -> None:
    queue_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": CONTRACT,
        "stage": stage,
        "arm_label": arm_label,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "training_allowed": False,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
        **fields,
    }
    path = queue_dir / "queue.safe.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def wait_for_exit(run_dir: Path, *, poll_seconds: float, timeout_seconds: float) -> int:
    deadline = time.monotonic() + timeout_seconds
    exit_path = run_dir / "exit_code"
    while time.monotonic() < deadline:
        if exit_path.is_file():
            return int(exit_path.read_text(encoding="utf-8").strip())
        time.sleep(poll_seconds)
    return 124


def rollout_environment(
    args: argparse.Namespace,
    *,
    run_name: str,
    dataset: Path,
    expected_tasks: int,
    samples_per_task: int,
    task_batch_size: int,
    analyze_on_success: bool,
) -> dict[str, str]:
    trajectory_admission_contract(
        task_batch_size=task_batch_size,
        samples_per_task=samples_per_task,
        max_num_seqs_per_dp_engine=args.max_num_seqs,
        data_parallel_size=2,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PROJECT_ROOT": str(args.project_root),
            "RUN_NAME": run_name,
            "OUTPUT_DIR": str(args.project_root / "runs" / run_name),
            "MODEL": str(args.model),
            "MODEL_LABEL": "step120",
            "POLICY_STEP": "120",
            "DATASET": str(dataset),
            "EXPECTED_TASKS": str(expected_tasks),
            "SAMPLES_PER_TASK": str(samples_per_task),
            "TASK_BATCH_SIZE": str(task_batch_size),
            "MAX_NUM_SEQS": str(args.max_num_seqs),
            "AGENT_WORKERS": "16",
            "MAX_NUM_BATCHED_TOKENS": "16384",
            "GPU_MEMORY_UTILIZATION": "0.80",
            "MAX_PROMPT_TOKENS": "4096",
            "MAX_RESPONSE_TOKENS": "90112",
            "MAX_CONTEXT_TOKENS": "94208",
            "TRAJECTORY_TIMEOUT_SECONDS": "1800",
            "ROLLING_ADMISSION": "1",
            "ROLLING_WINDOW_TRAJECTORIES": str(args.max_num_seqs * 2),
            "RAY_ADDRESS": args.ray_address,
            "ROLLOUT_RESOURCE": args.rollout_resource,
            "ANALYZE_ON_SUCCESS": "1" if analyze_on_success else "0",
            "MONITOR_NPU": "1",
            "MAX_ATTEMPTS": "3",
            "RETRY_DELAY_SECONDS": "60",
        }
    )
    return environment


def launch_rollout(
    args: argparse.Namespace,
    *,
    run_name: str,
    dataset: Path,
    expected_tasks: int,
    samples_per_task: int,
    task_batch_size: int,
    analyze_on_success: bool,
) -> Path:
    run_dir = args.project_root / "runs" / run_name
    if (run_dir / "exit_code").is_file():
        code = int((run_dir / "exit_code").read_text(encoding="utf-8").strip())
        if code != 0:
            raise RuntimeError(f"existing rollout failed with exit code {code}")
        return run_dir
    subprocess.run(
        ["bash", str(args.project_root / "scripts" / "launch_multisandbox_dwh_standalone.sh")],
        env=rollout_environment(
            args,
            run_name=run_name,
            dataset=dataset,
            expected_tasks=expected_tasks,
            samples_per_task=samples_per_task,
            task_batch_size=task_batch_size,
            analyze_on_success=analyze_on_success,
        ),
        check=True,
    )
    return run_dir


def launch_v20_finalizer(args: argparse.Namespace) -> None:
    if (args.v20_reconciled_dir / "exit_code").is_file():
        return
    environment = dict(os.environ)
    environment.update(
        {
            "PROJECT_ROOT": str(args.project_root),
            "ORIGINAL_DATASET": str(args.v20_original_dataset),
            "ORIGINAL_SHARDS_DIR": str(args.v20_original_run_dir / "shards"),
            "RETRY_DATASET": str(args.v20_retry_dataset),
            "RETRY_RUN_DIR": str(args.v20_retry_run_dir),
            "OUTPUT_DIR": str(args.v20_reconciled_dir),
            "EXPECTED_TASKS": "250",
            "SAMPLES_PER_TASK": "8",
        }
    )
    subprocess.run(
        ["bash", str(args.project_root / "scripts" / "launch_plan_first_dwh_timeout_retry_arm_finalizer.sh")],
        env=environment,
        check=True,
    )


def require_success(run_dir: Path, *, args: argparse.Namespace, label: str) -> None:
    code = wait_for_exit(
        run_dir,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.stage_timeout_seconds,
    )
    if code != 0:
        raise RuntimeError(f"{label} exited with {code}")


def run(args: argparse.Namespace) -> None:
    write_status(args.queue_dir, stage="waiting_v15_reconciliation", arm_label=args.arm_label)
    require_success(args.v15_reconciled_dir, args=args, label="v15 reconciliation")

    write_status(
        args.queue_dir,
        stage="launching_v20_timeout_retry",
        arm_label=args.arm_label,
        v20_retry_slots=args.v20_retry_tasks,
    )
    launch_rollout(
        args,
        run_name=args.v20_retry_run_dir.name,
        dataset=args.v20_retry_dataset,
        expected_tasks=args.v20_retry_tasks,
        samples_per_task=1,
        task_batch_size=args.retry_task_batch_size,
        analyze_on_success=False,
    )
    require_success(args.v20_retry_run_dir, args=args, label="v20 timeout retry")

    write_status(args.queue_dir, stage="reconciling_v20", arm_label=args.arm_label)
    launch_v20_finalizer(args)
    require_success(args.v20_reconciled_dir, args=args, label="v20 reconciliation")

    write_status(args.queue_dir, stage="launching_v21_full", arm_label=args.arm_label)
    launch_rollout(
        args,
        run_name=args.v21_run_dir.name,
        dataset=args.v21_dataset,
        expected_tasks=250,
        samples_per_task=8,
        task_batch_size=args.full_task_batch_size,
        analyze_on_success=True,
    )
    require_success(args.v21_run_dir, args=args, label="v21 full rerun")
    write_status(args.queue_dir, stage="complete", arm_label=args.arm_label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/llin-verl-grpo"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--arm-label", required=True)
    parser.add_argument("--rollout-resource", required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--retry-task-batch-size", type=int, required=True)
    parser.add_argument("--full-task-batch-size", type=int, required=True)
    parser.add_argument("--v15-reconciled-dir", type=Path, required=True)
    parser.add_argument("--v20-original-dataset", type=Path, required=True)
    parser.add_argument("--v20-original-run-dir", type=Path, required=True)
    parser.add_argument("--v20-retry-dataset", type=Path, required=True)
    parser.add_argument("--v20-retry-tasks", type=int, required=True)
    parser.add_argument("--v20-retry-run-dir", type=Path, required=True)
    parser.add_argument("--v20-reconciled-dir", type=Path, required=True)
    parser.add_argument("--v21-dataset", type=Path, required=True)
    parser.add_argument("--v21-run-dir", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--ray-address", default="192.168.202.5:26379")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--stage-timeout-seconds", type=float, default=604800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except BaseException as exc:
        write_status(
            args.queue_dir,
            stage="failed",
            arm_label=args.arm_label,
            error_type=type(exc).__name__,
        )
        (args.queue_dir / "exit_code").write_text("1\n", encoding="utf-8")
        raise
    else:
        (args.queue_dir / "exit_code").write_text("0\n", encoding="utf-8")


if __name__ == "__main__":
    main()
