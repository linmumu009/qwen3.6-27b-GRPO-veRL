#!/usr/bin/env python3
"""Run an unattended Step120 2+2+2+2 variance screen after another arm finishes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.adaptive_dwh_wave_earlystop import (
    CONTRACT,
    file_sha256,
    finalize_four_wave,
    finalize_three_wave,
    prepare_initial_pool,
    select_after_wave,
    write_json,
    write_private_parquet,
)
from scripts.run_adaptive_dwh_wave_earlystop_queue import (
    DEFAULT_STEP120_MANIFEST_SHA256,
    launch_wave,
    wait_for_exit,
)


QUEUE_CONTRACT = "llin-adaptive-dwh-2plus2plus2plus2-queue-v1"
WAVE_TARGETS = (2, 4, 6, 8)


def write_status(queue_dir: Path, *, stage: str, arm_label: str, **fields: object) -> None:
    write_json(
        queue_dir / "queue.safe.json",
        {
            "contract": QUEUE_CONTRACT,
            "wave_contract": CONTRACT,
            "stage": stage,
            "arm_label": arm_label,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "training_allowed": False,
            "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
            **fields,
        },
    )


def prepare_or_validate_pool(args: argparse.Namespace) -> dict:
    dataset = args.work_dir / "initial.sensitive.parquet"
    manifest_path = args.work_dir / "pool.safe.json"
    if dataset.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("contract")) != CONTRACT:
            raise ValueError("existing initial pool contract mismatch")
        if int(manifest.get("remaining_tasks", -1)) != args.expected_tasks:
            raise ValueError("existing initial pool task count mismatch")
        if str(manifest.get("remaining_dataset_sha256")) != file_sha256(dataset):
            raise ValueError("existing initial pool SHA256 mismatch")
        return manifest
    return prepare_initial_pool(
        args.source_dataset,
        dataset,
        manifest_path,
        expected_tasks=args.expected_tasks,
    )


def run(args: argparse.Namespace) -> dict:
    export_manifest = args.model / "llin_export_manifest.json"
    if not export_manifest.is_file():
        raise FileNotFoundError("Step120 export manifest is missing")
    if file_sha256(export_manifest) != args.expected_model_manifest_sha256:
        raise ValueError("Step120 export manifest SHA256 mismatch")
    write_status(
        args.queue_dir,
        stage="waiting_for_previous_arm",
        arm_label=args.arm_label,
        expected_tasks=args.expected_tasks,
    )
    previous_code = wait_for_exit(
        args.previous_queue_dir,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.stage_timeout_seconds,
    )
    if previous_code != 0:
        raise RuntimeError(f"previous arm exited with {previous_code}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    pool = prepare_or_validate_pool(args)
    current_dataset = args.work_dir / "initial.sensitive.parquet"
    current_tasks = int(pool["remaining_tasks"])
    mixed_paths: list[Path] = []
    prior_samples = 0
    wave_targets = tuple(
        target for target in WAVE_TARGETS if target <= args.max_target_samples
    )
    for target_samples in wave_targets:
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
            current_wave_samples_per_task=2,
            samples_observed_after_wave=target_samples,
        )
        if current_tasks:
            launch_wave(args, current_dataset, run_dir, current_tasks)
            code = wait_for_exit(
                run_dir,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.stage_timeout_seconds,
            )
            if code != 0:
                raise RuntimeError(f"wave-to-{target_samples} exited with {code}")
            wave = select_after_wave(
                current_dataset,
                run_dir / "outcomes" / "per_task.sensitive.jsonl",
                unresolved_path,
                mixed_path,
                manifest_path,
                expected_prior_samples=prior_samples,
                max_samples=args.max_target_samples,
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
    if args.max_target_samples == 6:
        summary = finalize_three_wave(
            args.work_dir / "initial.sensitive.parquet",
            mixed_paths[0],
            mixed_paths[1],
            mixed_paths[2],
            current_dataset,
            args.final_output_dir,
        )
    else:
        summary = finalize_four_wave(
            args.work_dir / "initial.sensitive.parquet",
            mixed_paths[0],
            mixed_paths[1],
            mixed_paths[2],
            mixed_paths[3],
            current_dataset,
            args.final_output_dir,
        )
    completion_fields = {
        "initial_tasks": int(summary["initial_tasks"]),
        "mixed_after_two_tasks": int(summary["mixed_after_two_tasks"]),
        "mixed_after_four_tasks": int(summary["mixed_after_four_tasks"]),
        "mixed_after_six_tasks": int(summary["mixed_after_six_tasks"]),
        "variance_candidate_tasks": int(summary["variance_candidate_tasks"]),
        "maximum_samples_per_task": args.max_target_samples,
    }
    if args.max_target_samples == 6:
        completion_fields.update(
            {
                "unresolved_after_six_tasks": int(
                    summary["unresolved_after_six_tasks"]
                ),
                "actual_sampling_trajectories": int(
                    summary["actual_sampling_trajectories"]
                ),
                "avoided_trajectories_vs_full_six": int(
                    summary["avoided_trajectories_vs_full_six"]
                ),
            }
        )
    else:
        completion_fields.update(
            {
                "mixed_after_eight_tasks": int(summary["mixed_after_eight_tasks"]),
                "unresolved_after_eight_tasks": int(
                    summary["unresolved_after_eight_tasks"]
                ),
                "actual_sampling_trajectories": int(
                    summary["actual_sampling_trajectories"]
                ),
                "avoided_trajectories_vs_full_eight": int(
                    summary["avoided_trajectories_vs_full_eight"]
                ),
            }
        )
    write_status(
        args.queue_dir,
        stage="complete",
        arm_label=args.arm_label,
        **completion_fields,
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
    parser.add_argument("--previous-queue-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--max-target-samples", type=int, choices=(6, 8), default=8)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--final-output-dir", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--ray-address", default="192.168.202.5:26379")
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
