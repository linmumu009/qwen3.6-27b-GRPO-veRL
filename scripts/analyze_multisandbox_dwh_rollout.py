#!/usr/bin/env python3
"""Score standalone shards and select mixed groups without exposing content."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import statistics

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.outcome_shadow import score_final_outcome


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


def bucket(correct: int, samples: int) -> str:
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
) -> dict:
    dataset = pq.read_table(dataset_path).to_pylist()
    if len(dataset) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} dataset rows, got {len(dataset)}")
    observations: dict[tuple[int, int], dict] = {}
    for path in sorted(shards_dir.glob("tasks_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (int(row["source_task_index"]), int(row["sample_index"]))
                if key in observations:
                    raise ValueError(f"duplicate task/sample observation: {key}")
                observations[key] = row
    expected_keys = {
        (task, sample)
        for task in range(expected_tasks)
        for sample in range(samples_per_task)
    }
    if set(observations) != expected_keys:
        missing = len(expected_keys - set(observations))
        extra = len(set(observations) - expected_keys)
        raise ValueError(f"rollout shape mismatch: missing={missing}, extra={extra}")

    per_task = []
    selected_rows = []
    histogram: Counter[int] = Counter()
    bucket_counts: Counter[str] = Counter()
    answer_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    version_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    total_completed = total_runtime_errors = total_correct = 0
    for task_index, dataset_row in enumerate(dataset):
        truth = dataset_row["reward_model"]["ground_truth"]
        task_observations = [observations[(task_index, sample)] for sample in range(samples_per_task)]
        scores = [
            score_final_outcome(str(row.get("output") or ""), truth)
            for row in task_observations
        ]
        correct = sum(int(score["final_answer_correct"]) for score in scores)
        completed = sum(int(score["has_final_answer"]) for score in scores)
        runtime_errors = sum(bool(row.get("runtime_error")) for row in task_observations)
        classification = bucket(correct, samples_per_task)
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
        histogram[correct] += 1
        bucket_counts[classification] += 1
        answer_buckets[answer_type][classification] += 1
        version_buckets[version][classification] += 1
        total_correct += correct
        total_completed += completed
        total_runtime_errors += runtime_errors

    output_dir.mkdir(parents=True, exist_ok=True)
    write_private_jsonl(output_dir / "per_task.sensitive.jsonl", per_task)
    write_private_parquet(output_dir / "mixed_groups.sensitive.parquet", selected_rows)
    summary = {
        "contract": "boss-multisandbox-dwh-rollout-outcomes-v1",
        "tasks": expected_tasks,
        "samples_per_task": samples_per_task,
        "trajectories": expected_tasks * samples_per_task,
        "correct_trajectories": total_correct,
        "completed_trajectories": total_completed,
        "runtime_error_trajectories": total_runtime_errors,
        "correct_rate": total_correct / (expected_tasks * samples_per_task),
        "completion_rate": total_completed / (expected_tasks * samples_per_task),
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
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                args.dataset,
                args.shards_dir,
                args.output_dir,
                expected_tasks=args.expected_tasks,
                samples_per_task=args.samples_per_task,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
