#!/usr/bin/env python3
"""Run one Qwen3.8 native-HF DWH arm with strict 2+2+2 early stopping."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from scripts.adaptive_dwh_wave_earlystop import (
    CONTRACT,
    WAVE_SAMPLES,
    file_sha256,
    finalize_three_wave,
    prepare_initial_pool,
    select_after_wave,
    write_json,
    write_private_parquet,
)
from scripts.run_adaptive_dwh_wave_earlystop_queue import wait_for_exit
from scripts.standalone_rollout_shards import trajectory_admission_contract


QUEUE_CONTRACT = "llin-qwen38-adaptive-dwh-2plus2plus2-queue-v1"
WAVE_TARGETS = (2, 4, 6)


def native_model_identity(model: Path) -> dict[str, object]:
    config = model / "config.json"
    index = model / "model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        raise FileNotFoundError("native HF model config or safetensor index is missing")
    if (model / "llin_export_manifest.json").exists():
        raise ValueError("Qwen3.8 rerun must not use a converted training checkpoint")
    return {
        "valid": True,
        "kind": "native_hf_checkpoint",
        "policy_step": 0,
        "config_sha256": file_sha256(config),
        "safetensor_index_sha256": file_sha256(index),
        "export_manifest_sha256": None,
    }


def write_status(queue_dir: Path, *, stage: str, arm_label: str, **fields: object) -> None:
    write_json(
        queue_dir / "queue.safe.json",
        {
            "contract": QUEUE_CONTRACT,
            "wave_contract": CONTRACT,
            "stage": stage,
            "arm_label": arm_label,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "model_label": "qwen38-27b-native-hf",
            "policy_step": 0,
            "reasoning_effort": "medium",
            "training_allowed": False,
            "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
            **fields,
        },
    )


def validate_topology(args: argparse.Namespace) -> dict[str, int]:
    if args.reasoning_effort != "medium":
        raise ValueError("Qwen3.8 rerun contract requires reasoning_effort=medium")
    if args.tensor_parallel_size * args.data_parallel_size != args.rollout_npus:
        raise ValueError("TP × DP must equal rollout NPUs")
    admission = trajectory_admission_contract(
        task_batch_size=args.task_batch_size,
        samples_per_task=WAVE_SAMPLES,
        max_num_seqs_per_dp_engine=args.max_num_seqs,
        data_parallel_size=args.data_parallel_size,
    )
    capacity = int(admission["aggregate_sequence_capacity"])
    if args.rolling_window_trajectories <= capacity:
        raise ValueError("Qwen3.8 queue requires explicit 1.25× logical window")
    if args.rolling_window_trajectories > int(capacity * args.rolling_window_max_multiplier):
        raise ValueError("logical window exceeds configured multiplier")
    return {
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "rollout_npus": args.rollout_npus,
        "max_num_seqs_per_dp_engine": args.max_num_seqs,
        "physical_sequence_capacity": capacity,
        "logical_window_trajectories": args.rolling_window_trajectories,
    }


def launch_wave(args: argparse.Namespace, dataset: Path, run_dir: Path, tasks: int) -> None:
    if tasks <= 0:
        return
    exit_path = run_dir / "exit_code"
    if exit_path.is_file():
        code = int(exit_path.read_text(encoding="utf-8").strip())
        if code != 0:
            raise RuntimeError(f"existing Qwen3.8 wave failed with exit code {code}")
        return
    environment = dict(os.environ)
    environment.update(
        {
            "PROJECT_ROOT": str(args.project_root),
            "RUN_NAME": run_dir.name,
            "OUTPUT_DIR": str(run_dir),
            "MODEL": str(args.model),
            "MODEL_LABEL": "qwen38-27b-native-hf",
            "POLICY_STEP": "0",
            "REASONING_EFFORT": args.reasoning_effort,
            "DATASET": str(dataset),
            "EXPECTED_TASKS": str(tasks),
            "SAMPLES_PER_TASK": str(WAVE_SAMPLES),
            "TASK_BATCH_SIZE": str(args.task_batch_size),
            "MAX_NUM_SEQS": str(args.max_num_seqs),
            "TENSOR_PARALLEL_SIZE": str(args.tensor_parallel_size),
            "DATA_PARALLEL_SIZE": str(args.data_parallel_size),
            "ROLLOUT_NPUS": str(args.rollout_npus),
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
            "MONITOR_ROLE": "qwen38_rollout",
            "MONITOR_INTERVAL": "5",
            "MONITOR_FIRST_CARD": str(args.monitor_first_card),
            "MONITOR_NUM_CARDS": str(args.monitor_num_cards),
            "MAX_ATTEMPTS": "3",
            "RETRY_DELAY_SECONDS": "60",
        }
    )
    subprocess.run(
        ["bash", str(args.project_root / "scripts" / "launch_multisandbox_dwh_standalone.sh")],
        env=environment,
        check=True,
    )


def prepare_or_validate_pool(args: argparse.Namespace) -> dict:
    dataset = args.work_dir / "initial.sensitive.parquet"
    manifest_path = args.work_dir / "pool.safe.json"
    if dataset.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("contract")) != CONTRACT:
            raise ValueError("existing Qwen3.8 pool contract mismatch")
        if int(manifest.get("remaining_tasks", -1)) != args.expected_tasks:
            raise ValueError("existing Qwen3.8 pool task count mismatch")
        if str(manifest.get("remaining_dataset_sha256")) != file_sha256(dataset):
            raise ValueError("existing Qwen3.8 pool SHA256 mismatch")
        return manifest
    return prepare_initial_pool(
        args.source_dataset,
        dataset,
        manifest_path,
        expected_tasks=args.expected_tasks,
    )


def run(args: argparse.Namespace) -> dict:
    topology = validate_topology(args)
    identity = native_model_identity(args.model)
    write_status(
        args.queue_dir,
        stage="preparing_initial_pool",
        arm_label=args.arm_label,
        expected_tasks=args.expected_tasks,
        topology=topology,
        model_identity=identity,
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    pool = prepare_or_validate_pool(args)
    initial = args.work_dir / "initial.sensitive.parquet"
    current_dataset = initial
    current_tasks = int(pool["remaining_tasks"])
    mixed_paths: list[Path] = []
    prior_samples = 0
    for target_samples in WAVE_TARGETS:
        run_dir = args.runs_dir / f"{args.run_prefix}-wave{target_samples}"
        mixed_path = args.work_dir / f"mixed{target_samples}.sensitive.parquet"
        unresolved_path = args.work_dir / f"unresolved{target_samples}.sensitive.parquet"
        manifest_path = args.work_dir / f"wave{target_samples}.safe.json"
        write_status(
            args.queue_dir,
            stage=f"running_wave_to_{target_samples}",
            arm_label=args.arm_label,
            expected_tasks=args.expected_tasks,
            current_wave_tasks=current_tasks,
            current_wave_samples_per_task=WAVE_SAMPLES,
            samples_observed_after_wave=target_samples,
            topology=topology,
        )
        if current_tasks:
            launch_wave(args, current_dataset, run_dir, current_tasks)
            code = wait_for_exit(
                run_dir,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.stage_timeout_seconds,
            )
            if code != 0:
                raise RuntimeError(f"Qwen3.8 wave-to-{target_samples} exited with {code}")
            wave = select_after_wave(
                current_dataset,
                run_dir / "outcomes" / "per_task.sensitive.jsonl",
                unresolved_path,
                mixed_path,
                manifest_path,
                expected_prior_samples=prior_samples,
                max_samples=6,
            )
            current_tasks = int(wave["unresolved_tasks"])
        else:
            for path in (mixed_path, unresolved_path):
                write_private_parquet(path, [], empty_from=current_dataset)
            write_json(
                manifest_path,
                {
                    "contract": CONTRACT,
                    "stage": f"decision_after_{target_samples}_samples",
                    "input_tasks": 0,
                    "new_mixed_tasks": 0,
                    "unresolved_tasks": 0,
                    "training_allowed": False,
                    "promotion_allowed": False,
                },
            )
        mixed_paths.append(mixed_path)
        current_dataset = unresolved_path
        prior_samples = target_samples
    summary = finalize_three_wave(
        initial,
        mixed_paths[0],
        mixed_paths[1],
        mixed_paths[2],
        current_dataset,
        args.final_output_dir,
    )
    write_status(
        args.queue_dir,
        stage="complete",
        arm_label=args.arm_label,
        topology=topology,
        initial_tasks=int(summary["initial_tasks"]),
        mixed_after_two_tasks=int(summary["mixed_after_two_tasks"]),
        mixed_after_four_tasks=int(summary["mixed_after_four_tasks"]),
        mixed_after_six_tasks=int(summary["mixed_after_six_tasks"]),
        variance_candidate_tasks=int(summary["variance_candidate_tasks"]),
        unresolved_after_six_tasks=int(summary["unresolved_after_six_tasks"]),
        actual_sampling_trajectories=int(summary["actual_sampling_trajectories"]),
        avoided_trajectories_vs_full_six=int(summary["avoided_trajectories_vs_full_six"]),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/llin-verl-grpo"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reasoning-effort", choices=("medium",), default="medium")
    parser.add_argument("--arm-label", required=True)
    parser.add_argument("--rollout-resource", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--data-parallel-size", type=int, required=True)
    parser.add_argument("--rollout-npus", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--task-batch-size", type=int, required=True)
    parser.add_argument("--rolling-window-trajectories", type=int, required=True)
    parser.add_argument("--rolling-window-max-multiplier", type=float, default=1.25)
    parser.add_argument("--monitor-first-card", type=int, default=0)
    parser.add_argument("--monitor-num-cards", type=int, default=8)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--final-output-dir", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--stage-timeout-seconds", type=float, default=1209600)
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
