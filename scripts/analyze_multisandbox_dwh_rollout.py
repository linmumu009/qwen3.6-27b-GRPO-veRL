#!/usr/bin/env python3
"""Score standalone shards and select mixed groups without exposing content."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import statistics

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.outcome_shadow import score_final_outcome
from llin_verl.pi_reward import extract_final_assistant_answer
from scripts.standalone_rollout_shards import completed_shard_rows


_SHARD_RE = re.compile(r"^tasks_(\d+)_(\d+)\.jsonl$")


def write_private_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def write_private_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def bucket(
    correct: int,
    samples: int,
    timeout_count: int = 0,
    runtime_error_count: int = 0,
) -> str:
    if runtime_error_count:
        return "runtime_error"
    if timeout_count:
        return "timed_out"
    if correct == 0:
        return "all_wrong"
    if correct == samples:
        return "all_correct"
    return "mixed"


def analyze(
    dataset_path: Path,
    shards_dir: Path,
    output_dir: Path,
    *,
    expected_tasks: int,
    samples_per_task: int,
    allow_partial: bool = False,
) -> dict:
    dataset = pq.read_table(dataset_path).to_pylist()
    if len(dataset) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} dataset rows, got {len(dataset)}")
    observations: dict[tuple[int, int], dict] = {}
    complete_shard_ranges: list[tuple[int, int]] = []
    for path in sorted(shards_dir.glob("tasks_*.jsonl")):
        match = _SHARD_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid shard filename: {path.name}")
        start, stop = (int(value) for value in match.groups())
        if not 0 <= start < stop <= expected_tasks:
            raise ValueError(f"shard range outside dataset: {path.name}")
        expected_rows = (stop - start) * samples_per_task
        actual_rows = completed_shard_rows(
            path,
            start=start,
            stop=stop,
            samples_per_task=samples_per_task,
        )
        if actual_rows != expected_rows:
            raise ValueError(f"incomplete or invalid shard: {path.name}")
        complete_shard_ranges.append((start, stop))
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (int(row["source_task_index"]), int(row["sample_index"]))
                if key in observations:
                    raise ValueError(f"duplicate task/sample observation: {key}")
                observations[key] = row
    observed_tasks = sorted({task for task, _ in observations})
    expected_keys = {
        (task, sample) for task in observed_tasks for sample in range(samples_per_task)
    }
    if set(observations) != expected_keys:
        missing = len(expected_keys - set(observations))
        extra = len(set(observations) - expected_keys)
        raise ValueError(f"rollout shape mismatch: missing={missing}, extra={extra}")
    if not allow_partial and len(observed_tasks) != expected_tasks:
        missing = expected_tasks - len(observed_tasks)
        raise ValueError(f"rollout shape mismatch: missing_tasks={missing}")
    if not observed_tasks:
        raise ValueError("no complete rollout shards found")

    per_task = []
    selected_rows = []
    histogram: Counter[int] = Counter()
    bucket_counts: Counter[str] = Counter()
    answer_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    version_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    total_completed = total_runtime_errors = total_timeouts = total_correct = 0
    total_abort_acks = total_abort_physical = total_abort_errors = 0
    review_rows = []
    for task_index in observed_tasks:
        dataset_row = dataset[task_index]
        truth = dataset_row["reward_model"]["ground_truth"]
        task_observations = [observations[(task_index, sample)] for sample in range(samples_per_task)]
        scores = [
            score_final_outcome(str(row.get("output") or ""), truth)
            for row in task_observations
        ]
        correct = sum(int(score["final_answer_correct"]) for score in scores)
        completed = sum(int(score["has_final_answer"]) for score in scores)
        runtime_errors = sum(bool(row.get("runtime_error")) for row in task_observations)
        timeouts = sum(bool(row.get("trajectory_timeout")) for row in task_observations)
        abort_acks = sum(
            int(row.get("trajectory_abort_acknowledged_count", 0) or 0)
            for row in task_observations
        )
        abort_physical = sum(
            int(row.get("trajectory_abort_physical_request_count", 0) or 0)
            for row in task_observations
        )
        abort_errors = sum(
            int(row.get("trajectory_abort_error_count", 0) or 0)
            for row in task_observations
        )
        classification = bucket(
            correct,
            samples_per_task,
            timeout_count=timeouts,
            runtime_error_count=runtime_errors,
        )
        extra = dataset_row["extra_info"]
        answer_type = str(truth["answer_type"])
        version = str(extra["source_version"])
        response_lengths = [int(row.get("response_tokens", 0)) for row in task_observations]
        per_task.append(
            {
                "source_task_index": task_index,
                "verifier_id": str(extra["verifier_id"]),
                "instruction_sha256": str(extra["instruction_sha256"]),
                "source_version": version,
                "answer_type": answer_type,
                "correct_count": correct,
                "completed_count": completed,
                "runtime_error_count": runtime_errors,
                "trajectory_timeout_count": timeouts,
                "trajectory_abort_acknowledged_count": abort_acks,
                "trajectory_abort_physical_request_count": abort_physical,
                "trajectory_abort_error_count": abort_errors,
                "dense_mean": statistics.fmean(
                    float(score["dense_final_answer_correctness"]) for score in scores
                ),
                "response_tokens_mean": statistics.fmean(response_lengths),
                "response_tokens_max": max(response_lengths),
                "bucket": classification,
                "training_allowed": False,
            }
        )
        if classification == "mixed":
            selected_rows.append(dataset_row)
            review_rows.append(
                {
                    "source_task_index": task_index,
                    "prompt": dataset_row["prompt"],
                    "reward_model": dataset_row["reward_model"],
                    "extra_info": dataset_row["extra_info"],
                    "trajectory_final_answers": [
                        {
                            "sample_index": sample,
                            "final_answer": extract_final_assistant_answer(
                                str(task_observations[sample].get("output") or "")
                            ),
                            "final_answer_correct": bool(scores[sample]["final_answer_correct"]),
                            "has_final_answer": bool(scores[sample]["has_final_answer"]),
                            "response_tokens": int(
                                task_observations[sample].get("response_tokens", 0)
                            ),
                            "runtime_error": bool(
                                task_observations[sample].get("runtime_error")
                            ),
                            "trajectory_timeout": bool(
                                task_observations[sample].get("trajectory_timeout")
                            ),
                        }
                        for sample in range(samples_per_task)
                    ],
                    "review_decision": {
                        "instruction_unambiguously_entails_gold": None,
                        "verification_sql_fully_answers_instruction": None,
                        "expected_value_supported_by_query_result": None,
                        "final_outcome_routing_trustworthy": None,
                        "verdict": "needs_review",
                        "reason_codes": [],
                        "notes": "",
                    },
                    "training_allowed": False,
                    "promotion_allowed": False,
                }
            )
        histogram[correct] += 1
        bucket_counts[classification] += 1
        answer_buckets[answer_type][classification] += 1
        version_buckets[version][classification] += 1
        total_correct += correct
        total_completed += completed
        total_runtime_errors += runtime_errors
        total_timeouts += timeouts
        total_abort_acks += abort_acks
        total_abort_physical += abort_physical
        total_abort_errors += abort_errors

    output_dir.mkdir(parents=True, exist_ok=True)
    write_private_jsonl(output_dir / "per_task.sensitive.jsonl", per_task)
    write_private_parquet(output_dir / "mixed_groups.sensitive.parquet", selected_rows)
    write_private_jsonl(output_dir / "mixed_review_queue.sensitive.jsonl", review_rows)
    analyzed_tasks = len(observed_tasks)
    analyzed_trajectories = analyzed_tasks * samples_per_task
    summary = {
        "contract": "boss-multisandbox-dwh-rollout-outcomes-v2",
        "expected_tasks": expected_tasks,
        "tasks": analyzed_tasks,
        "partial": analyzed_tasks != expected_tasks,
        "complete_shards": len(complete_shard_ranges),
        "complete_shard_ranges": [
            {"start": start, "stop": stop} for start, stop in complete_shard_ranges
        ],
        "samples_per_task": samples_per_task,
        "trajectories": analyzed_trajectories,
        "correct_trajectories": total_correct,
        "completed_trajectories": total_completed,
        "runtime_error_trajectories": total_runtime_errors,
        "timeout_trajectories": total_timeouts,
        "timeout_abort_acknowledged_count": total_abort_acks,
        "timeout_abort_physical_request_count": total_abort_physical,
        "timeout_abort_error_count": total_abort_errors,
        "evaluable_trajectories": (
            analyzed_trajectories - total_runtime_errors - total_timeouts
        ),
        "correct_rate": total_correct / analyzed_trajectories,
        "completion_rate": total_completed / analyzed_trajectories,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "correct_count_histogram": {
            str(value): histogram.get(value, 0) for value in range(samples_per_task + 1)
        },
        "answer_type_bucket_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(answer_buckets.items())
        },
        "version_bucket_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(version_buckets.items())
        },
        "mixed_screening_rows": len(selected_rows),
        "mixed_review_queue_rows": len(review_rows),
        "selection_rule": (
            "exactly_8_complete_usable_trajectories_and_1_to_7_final_outcome_correct"
        ),
        "runtime_error_or_timeout_fail_closed": True,
        "explicit_semantic_review_completed": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
    }
    (output_dir / "safe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="analyze every fully written shard without waiting for the full dataset",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                args.dataset,
                args.shards_dir,
                args.output_dir,
                expected_tasks=args.expected_tasks,
                samples_per_task=args.samples_per_task,
                allow_partial=args.allow_partial,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
