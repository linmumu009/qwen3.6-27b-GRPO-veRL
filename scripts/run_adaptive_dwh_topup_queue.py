#!/usr/bin/env python3
"""Unattended variance screen with early-stop and uncertain-task top-up."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.adaptive_dwh_topup import (
    SELECTION_CONTRACT,
    TOPUP_SAMPLES,
    finalize,
    prepare_topup,
)
from scripts.standalone_rollout_shards import trajectory_admission_contract


CONTRACT = "llin-adaptive-dwh-topup-queue-v1"


def write_status(queue_dir: Path, *, stage: str, arm_label: str, **fields: object) -> None:
    payload = {
        "contract": CONTRACT,
        "stage": stage,
        "arm_label": arm_label,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "training_allowed": False,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
        **fields,
    }
    queue_dir.mkdir(parents=True, exist_ok=True)
    temporary = queue_dir / "queue.safe.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(queue_dir / "queue.safe.json")


def wait_for_exit(run_dir: Path, *, poll_seconds: float, timeout_seconds: float) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        exit_path = run_dir / "exit_code"
        if exit_path.is_file():
            return int(exit_path.read_text(encoding="utf-8").strip())
        time.sleep(poll_seconds)
    return 124


def launch_topup(args: argparse.Namespace, selected_tasks: int) -> None:
    if (args.topup_run_dir / "exit_code").is_file():
        code = int((args.topup_run_dir / "exit_code").read_text(encoding="utf-8").strip())
        if code != 0:
            raise RuntimeError(f"existing top-up rollout failed with exit code {code}")
        return
    admission = trajectory_admission_contract(
        task_batch_size=args.topup_task_batch_size,
        samples_per_task=TOPUP_SAMPLES,
        max_num_seqs_per_dp_engine=args.max_num_seqs,
        data_parallel_size=2,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PROJECT_ROOT": str(args.project_root),
            "RUN_NAME": args.topup_run_dir.name,
            "OUTPUT_DIR": str(args.topup_run_dir),
            "MODEL": str(args.model),
            "MODEL_LABEL": "step120",
            "POLICY_STEP": "120",
            "DATASET": str(args.topup_dataset),
            "EXPECTED_TASKS": str(selected_tasks),
            "SAMPLES_PER_TASK": str(TOPUP_SAMPLES),
            "TASK_BATCH_SIZE": str(args.topup_task_batch_size),
            "MAX_NUM_SEQS": str(args.max_num_seqs),
            "AGENT_WORKERS": "16",
            "MAX_NUM_BATCHED_TOKENS": "16384",
            "GPU_MEMORY_UTILIZATION": "0.80",
            "MAX_PROMPT_TOKENS": "4096",
            "MAX_RESPONSE_TOKENS": "90112",
            "MAX_CONTEXT_TOKENS": "94208",
            "TRAJECTORY_TIMEOUT_SECONDS": "1800",
            "ROLLING_ADMISSION": "1",
            "ROLLING_WINDOW_TRAJECTORIES": str(admission["aggregate_sequence_capacity"]),
            "RAY_ADDRESS": args.ray_address,
            "ROLLOUT_RESOURCE": args.rollout_resource,
            "ANALYZE_ON_SUCCESS": "0",
            "MONITOR_NPU": "1",
            "MAX_ATTEMPTS": "3",
            "RETRY_DELAY_SECONDS": "60",
        }
    )
    subprocess.run(
        ["bash", str(args.project_root / "scripts" / "launch_multisandbox_dwh_standalone.sh")],
        env=environment,
        check=True,
    )


def run(args: argparse.Namespace) -> None:
    write_status(
        args.queue_dir,
        stage="waiting_screen",
        arm_label=args.arm_label,
        expected_screen_tasks=args.expected_screen_tasks,
    )
    screen_code = wait_for_exit(
        args.screen_run_dir,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.stage_timeout_seconds,
    )
    if screen_code != 0:
        raise RuntimeError(f"two-sample screen exited with {screen_code}")

    write_status(args.queue_dir, stage="preparing_topup", arm_label=args.arm_label)
    if args.selection_manifest.is_file() and args.topup_dataset.is_file():
        manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        if str(manifest.get("contract")) != SELECTION_CONTRACT:
            raise ValueError("existing adaptive selection manifest contract mismatch")
    else:
        manifest = prepare_topup(
            args.screen_dataset,
            args.screen_run_dir / "outcomes" / "per_task.sensitive.jsonl",
            args.source_tasks,
            args.reference_profile,
            args.topup_dataset,
            args.selection_manifest,
            reference_per_signature=args.reference_per_signature,
            exploration_per_level=args.exploration_per_level,
            seed=args.seed,
        )
    selected_tasks = int(manifest["selected_tasks"])
    confirmed_mixed_tasks = int(manifest["confirmed_mixed_tasks"])
    write_status(
        args.queue_dir,
        stage="launching_topup",
        arm_label=args.arm_label,
        selected_tasks=selected_tasks,
        confirmed_mixed_tasks=confirmed_mixed_tasks,
        confirmed_mixed_avoided_topup_trajectories=(
            confirmed_mixed_tasks * TOPUP_SAMPLES
        ),
        topup_trajectories=selected_tasks * TOPUP_SAMPLES,
    )
    launch_topup(args, selected_tasks)
    topup_code = wait_for_exit(
        args.topup_run_dir,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.stage_timeout_seconds,
    )
    if topup_code != 0:
        raise RuntimeError(f"six-sample top-up exited with {topup_code}")

    write_status(
        args.queue_dir,
        stage="finalizing_eight_trajectory_groups",
        arm_label=args.arm_label,
        selected_tasks=selected_tasks,
        confirmed_mixed_tasks=confirmed_mixed_tasks,
    )
    final_summary_path = args.final_output_dir / "adaptive_final_safe_summary.json"
    if final_summary_path.is_file():
        summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    else:
        summary = finalize(
            args.screen_dataset,
            args.screen_run_dir / "shards",
            args.topup_dataset,
            args.topup_run_dir / "shards",
            args.final_output_dir,
            expected_screen_tasks=args.expected_screen_tasks,
        )
    write_status(
        args.queue_dir,
        stage="complete",
        arm_label=args.arm_label,
        selected_tasks=selected_tasks,
        confirmed_mixed_tasks=confirmed_mixed_tasks,
        strict_mixed_tasks=int(summary["strict_mixed_tasks"]),
        relaxed_explicit_mixed_tasks=int(summary["relaxed_explicit_mixed_tasks"]),
        grpo_variance_candidate_tasks=int(summary["grpo_variance_candidate_tasks"]),
        avoided_trajectories=int(summary["avoided_trajectories"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/llin-verl-grpo"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--arm-label", required=True)
    parser.add_argument("--rollout-resource", required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--topup-task-batch-size", type=int, required=True)
    parser.add_argument("--screen-dataset", type=Path, required=True)
    parser.add_argument("--screen-run-dir", type=Path, required=True)
    parser.add_argument("--expected-screen-tasks", type=int, default=250)
    parser.add_argument("--source-tasks", type=Path, required=True)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--topup-dataset", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--topup-run-dir", type=Path, required=True)
    parser.add_argument("--final-output-dir", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--reference-per-signature", type=int, default=2)
    parser.add_argument("--exploration-per-level", type=int, default=2)
    parser.add_argument("--seed", default="adaptive-dwh-topup-v1")
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
