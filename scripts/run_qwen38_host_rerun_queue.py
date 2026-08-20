#!/usr/bin/env python3
"""Run multiple private Qwen3.8 2+2+2 arms sequentially on one host."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq


CONTRACT = "llin-qwen38-host-rerun-queue-v1"


def parse_arm(value: str) -> tuple[str, Path]:
    version, separator, raw_path = value.partition("=")
    if not separator or not version or not raw_path:
        raise argparse.ArgumentTypeError("expected VERSION=DATASET")
    return version, Path(raw_path)


def write_status(queue_dir: Path, *, stage: str, host_label: str, **fields: object) -> None:
    queue_dir.mkdir(parents=True, exist_ok=True)
    temporary = queue_dir / "host_queue.safe.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "contract": CONTRACT,
                "stage": stage,
                "host_label": host_label,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "model_label": fields.pop("model_label", "qwen38-27b-native-hf"),
                "policy_step": fields.pop("policy_step", 0),
                "reasoning_effort": "medium",
                "training_allowed": False,
                "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
                **fields,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(queue_dir / "host_queue.safe.json")


def arm_command(args: argparse.Namespace, version: str, dataset: Path, tasks: int) -> list[str]:
    command = [
        sys.executable,
        str(args.project_root / "scripts" / "run_qwen38_adaptive_dwh_three_wave_queue.py"),
        "--project-root",
        str(args.project_root),
        "--model",
        str(args.model),
        "--model-label",
        args.model_label,
        "--policy-step",
        str(args.policy_step),
        "--reasoning-effort",
        "medium",
        "--arm-label",
        version,
        "--host-label",
        args.host_label,
        "--rollout-resource",
        args.rollout_resource,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--rollout-npus",
        str(args.rollout_npus),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--task-batch-size",
        str(args.task_batch_size),
        "--rolling-window-trajectories",
        str(args.rolling_window_trajectories),
        "--rolling-window-max-multiplier",
        "1.25",
        "--monitor-first-card",
        str(args.monitor_first_card),
        "--monitor-num-cards",
        str(args.monitor_num_cards),
        "--source-dataset",
        str(dataset),
        "--expected-tasks",
        str(tasks),
        "--runs-dir",
        str(args.runs_dir),
        "--run-prefix",
        f"{args.run_prefix}-{version}",
        "--work-dir",
        str(args.state_root / version),
        "--final-output-dir",
        str(args.final_root / version),
        "--queue-dir",
        str(args.queue_dir / version),
        "--ray-address",
        args.ray_address,
        "--poll-seconds",
        str(args.poll_seconds),
        "--stage-timeout-seconds",
        str(args.stage_timeout_seconds),
    ]
    if args.confirm_candidates:
        command.append("--confirm-candidates")
    return command


def run(args: argparse.Namespace) -> dict:
    arms = list(args.arm)
    if len({version for version, _ in arms}) != len(arms):
        raise ValueError("host queue versions must be unique")
    task_counts = {}
    for version, dataset in arms:
        if not dataset.is_file():
            raise FileNotFoundError(dataset)
        task_counts[version] = pq.read_metadata(dataset).num_rows
        if task_counts[version] <= 0:
            raise ValueError("host queue arm cannot be empty")
    completed = []
    write_status(
        args.queue_dir,
        stage="running",
        host_label=args.host_label,
        arm_order=[version for version, _ in arms],
        arm_task_counts=task_counts,
        completed_arms=completed,
        model_label=args.model_label,
        policy_step=args.policy_step,
    )
    for version, dataset in arms:
        arm_exit = args.queue_dir / version / "exit_code"
        if arm_exit.is_file() and arm_exit.read_text(encoding="utf-8").strip() == "0":
            completed.append(version)
        else:
            subprocess.run(
                arm_command(args, version, dataset, task_counts[version]),
                check=True,
            )
            completed.append(version)
        write_status(
            args.queue_dir,
            stage="running",
            host_label=args.host_label,
            arm_order=[item[0] for item in arms],
            arm_task_counts=task_counts,
            completed_arms=list(completed),
            model_label=args.model_label,
            policy_step=args.policy_step,
        )
    result = {
        "arm_order": [version for version, _ in arms],
        "arm_task_counts": task_counts,
        "completed_arms": completed,
        "total_tasks": sum(task_counts.values()),
    }
    write_status(
        args.queue_dir,
        stage="complete",
        host_label=args.host_label,
        model_label=args.model_label,
        policy_step=args.policy_step,
        **result,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/llin-verl-grpo"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", default="qwen38-27b-native-hf")
    parser.add_argument("--policy-step", type=int, default=0)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--confirm-candidates", action="store_true")
    parser.add_argument("--arm", type=parse_arm, action="append", required=True)
    parser.add_argument("--rollout-resource", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--data-parallel-size", type=int, required=True)
    parser.add_argument("--rollout-npus", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--task-batch-size", type=int, required=True)
    parser.add_argument("--rolling-window-trajectories", type=int, required=True)
    parser.add_argument("--monitor-first-card", type=int, default=0)
    parser.add_argument("--monitor-num-cards", type=int, default=8)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--stage-timeout-seconds", type=float, default=1209600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stale_exit = args.queue_dir / "exit_code"
    if stale_exit.is_file():
        stale_exit.unlink()
    try:
        result = run(args)
    except BaseException as exc:
        write_status(
            args.queue_dir,
            stage="failed",
            host_label=args.host_label,
            error_type=type(exc).__name__,
            model_label=args.model_label,
            policy_step=args.policy_step,
        )
        (args.queue_dir / "exit_code").write_text("1\n", encoding="utf-8")
        raise
    else:
        (args.queue_dir / "exit_code").write_text("0\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
