#!/usr/bin/env python3
"""Run one server arm of a resumable 2+2+2 DWH variance screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import pyarrow.parquet as pq

from scripts.adaptive_dwh_wave_earlystop import (
    CONTRACT,
    WAVE_SAMPLES,
    file_sha256,
    finalize,
    prepare_remaining_pool,
    select_after_wave,
    write_private_parquet,
)
from scripts.standalone_rollout_shards import trajectory_admission_contract


QUEUE_CONTRACT = "llin-adaptive-dwh-2plus2plus2-queue-v1"
DEFAULT_STEP120_MANIFEST_SHA256 = (
    "bf363a04d460a389021cf5e6e0c7013552ea88df6c5bd7bdb5ffd43300057208"
)


def write_status(queue_dir: Path, *, stage: str, arm_label: str, **fields: object) -> None:
    payload = {
        "contract": QUEUE_CONTRACT,
        "wave_contract": CONTRACT,
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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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


def launch_wave(args: argparse.Namespace, dataset: Path, run_dir: Path, tasks: int) -> None:
    if tasks <= 0:
        return
    if (run_dir / "exit_code").is_file():
        code = int((run_dir / "exit_code").read_text(encoding="utf-8").strip())
        if code != 0:
            raise RuntimeError(f"existing wave rollout failed with exit code {code}")
        return
    admission = trajectory_admission_contract(
        task_batch_size=args.task_batch_size,
        samples_per_task=WAVE_SAMPLES,
        max_num_seqs_per_dp_engine=args.max_num_seqs,
        data_parallel_size=2,
    )
    if args.rolling_window_trajectories <= int(admission["aggregate_sequence_capacity"]):
        raise ValueError("2+2+2 queue requires explicit logical oversubscription")
    environment = dict(os.environ)
    environment.update(
        {
            "PROJECT_ROOT": str(args.project_root),
            "RUN_NAME": run_dir.name,
            "OUTPUT_DIR": str(run_dir),
            "MODEL": str(args.model),
            "MODEL_LABEL": "step120",
            "POLICY_STEP": "120",
            "DATASET": str(dataset),
            "EXPECTED_TASKS": str(tasks),
            "SAMPLES_PER_TASK": str(WAVE_SAMPLES),
            "TASK_BATCH_SIZE": str(args.task_batch_size),
            "MAX_NUM_SEQS": str(args.max_num_seqs),
            "AGENT_WORKERS": "16",
            "MAX_NUM_BATCHED_TOKENS": "16384",
            "GPU_MEMORY_UTILIZATION": "0.80",
            "MAX_PROMPT_TOKENS": "4096",
            "MAX_RESPONSE_TOKENS": "90112",
            "MAX_CONTEXT_TOKENS": "94208",
            "TRAJECTORY_TIMEOUT_SECONDS": "1800",
            "ROLLING_ADMISSION": "1",
            "ROLLING_WINDOW_TRAJECTORIES": str(args.rolling_window_trajectories),
            "ROLLING_WINDOW_MAX_MULTIPLIER": str(args.rolling_window_max_multiplier),
            "RAY_ADDRESS": args.ray_address,
            "ROLLOUT_RESOURCE": args.rollout_resource,
            "ANALYZE_ON_SUCCESS": "1",
            "MONITOR_NPU": "1",
            "MONITOR_ROLE": "rollout",
            "MONITOR_INTERVAL": "5",
            "MAX_ATTEMPTS": "3",
            "RETRY_DELAY_SECONDS": "60",
        }
    )
    subprocess.run(
        ["bash", str(args.project_root / "scripts" / "launch_multisandbox_dwh_standalone.sh")],
        env=environment,
        check=True,
    )


def run(args: argparse.Namespace) -> dict:
    export_manifest = args.model / "llin_export_manifest.json"
    if not export_manifest.is_file():
        raise FileNotFoundError("Step120 export manifest is missing")
    observed_manifest_sha256 = file_sha256(export_manifest)
    if observed_manifest_sha256 != args.expected_model_manifest_sha256:
        raise ValueError("Step120 export manifest SHA256 mismatch")
    write_status(args.queue_dir, stage="preparing_remaining_pool", arm_label=args.arm_label)
    if args.pool_dataset.is_file() and args.pool_manifest.is_file():
        pool_manifest = json.loads(args.pool_manifest.read_text(encoding="utf-8"))
        if str(pool_manifest.get("contract")) != CONTRACT:
            raise ValueError("existing pool manifest contract mismatch")
        if int(pool_manifest.get("remaining_tasks", -1)) != args.expected_remaining_tasks:
            raise ValueError("existing pool manifest task count mismatch")
        if str(pool_manifest.get("remaining_dataset_sha256")) != file_sha256(
            args.pool_dataset
        ):
            raise ValueError("existing pool dataset SHA256 mismatch")
    else:
        pool_manifest = prepare_remaining_pool(
            args.screen_dataset,
            args.screen_per_task,
            args.excluded_probe_dataset,
            args.excluded_direct_dataset,
            args.pool_dataset,
            args.pool_manifest,
            expected_remaining_tasks=args.expected_remaining_tasks,
        )
    pool_tasks = int(pool_manifest["remaining_tasks"])
    write_status(
        args.queue_dir,
        stage="running_wave_to_four",
        arm_label=args.arm_label,
        initial_tasks=pool_tasks,
        current_wave_tasks=pool_tasks,
        current_wave_samples_per_task=WAVE_SAMPLES,
    )
    launch_wave(args, args.pool_dataset, args.wave4_run_dir, pool_tasks)
    code = wait_for_exit(
        args.wave4_run_dir,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.stage_timeout_seconds,
    )
    if code != 0:
        raise RuntimeError(f"wave-to-four exited with {code}")
    wave4_manifest = select_after_wave(
        args.pool_dataset,
        args.wave4_run_dir / "outcomes" / "per_task.sensitive.jsonl",
        args.wave6_dataset,
        args.mixed4_dataset,
        args.wave4_manifest,
        expected_prior_samples=2,
    )
    unresolved = int(wave4_manifest["unresolved_tasks"])
    write_status(
        args.queue_dir,
        stage="running_wave_to_six" if unresolved else "finalizing",
        arm_label=args.arm_label,
        initial_tasks=pool_tasks,
        mixed_after_four_tasks=int(wave4_manifest["new_mixed_tasks"]),
        current_wave_tasks=unresolved,
        current_wave_samples_per_task=WAVE_SAMPLES if unresolved else 0,
    )
    if unresolved:
        launch_wave(args, args.wave6_dataset, args.wave6_run_dir, unresolved)
        code = wait_for_exit(
            args.wave6_run_dir,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.stage_timeout_seconds,
        )
        if code != 0:
            raise RuntimeError(f"wave-to-six exited with {code}")
        wave6_manifest = select_after_wave(
            args.wave6_dataset,
            args.wave6_run_dir / "outcomes" / "per_task.sensitive.jsonl",
            args.unresolved6_dataset,
            args.mixed6_dataset,
            args.wave6_manifest,
            expected_prior_samples=4,
        )
    else:
        empty = pq.read_table(args.pool_dataset).slice(0, 0)
        for path in (args.mixed6_dataset, args.unresolved6_dataset):
            write_private_parquet(path, empty.to_pylist(), empty_from=args.pool_dataset)
        wave6_manifest = {"new_mixed_tasks": 0, "unresolved_tasks": 0}
    summary = finalize(
        args.pool_dataset,
        args.mixed4_dataset,
        args.mixed6_dataset,
        args.unresolved6_dataset,
        args.final_output_dir,
    )
    write_status(
        args.queue_dir,
        stage="complete",
        arm_label=args.arm_label,
        initial_tasks=pool_tasks,
        mixed_after_four_tasks=int(wave4_manifest["new_mixed_tasks"]),
        mixed_after_six_tasks=int(wave6_manifest["new_mixed_tasks"]),
        variance_candidate_tasks=int(summary["variance_candidate_tasks"]),
        unresolved_after_six_tasks=int(summary["unresolved_after_six_tasks"]),
        actual_trajectories_including_existing_two=int(
            summary["actual_trajectories_including_existing_two"]
        ),
        avoided_trajectories_vs_full_six=int(summary["avoided_trajectories_vs_full_six"]),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/llin-verl-grpo"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--expected-model-manifest-sha256",
        default=DEFAULT_STEP120_MANIFEST_SHA256,
    )
    parser.add_argument("--arm-label", required=True)
    parser.add_argument("--rollout-resource", required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--task-batch-size", type=int, required=True)
    parser.add_argument("--rolling-window-trajectories", type=int, required=True)
    parser.add_argument("--rolling-window-max-multiplier", type=float, default=1.25)
    parser.add_argument("--screen-dataset", type=Path, required=True)
    parser.add_argument("--screen-per-task", type=Path, required=True)
    parser.add_argument("--excluded-probe-dataset", type=Path, required=True)
    parser.add_argument("--excluded-direct-dataset", type=Path, required=True)
    parser.add_argument("--expected-remaining-tasks", type=int, required=True)
    parser.add_argument("--pool-dataset", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--wave4-run-dir", type=Path, required=True)
    parser.add_argument("--wave4-manifest", type=Path, required=True)
    parser.add_argument("--mixed4-dataset", type=Path, required=True)
    parser.add_argument("--wave6-dataset", type=Path, required=True)
    parser.add_argument("--wave6-run-dir", type=Path, required=True)
    parser.add_argument("--wave6-manifest", type=Path, required=True)
    parser.add_argument("--mixed6-dataset", type=Path, required=True)
    parser.add_argument("--unresolved6-dataset", type=Path, required=True)
    parser.add_argument("--final-output-dir", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--ray-address", default="192.168.202.5:26379")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--stage-timeout-seconds", type=float, default=604800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
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
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
