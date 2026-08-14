#!/usr/bin/env python3
"""Merge two safe rollout summaries and their server-only mixed datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def merge_counts(items: list[dict]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for item in items:
        total.update({str(key): int(value) for key, value in item.items()})
    return dict(sorted(total.items()))


def merge_nested(items: list[dict]) -> dict[str, dict[str, int]]:
    total: dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        for outer, counts in item.items():
            total[str(outer)].update({str(key): int(value) for key, value in counts.items()})
    return {key: dict(sorted(value.items())) for key, value in sorted(total.items())}


def token_distribution_from_histogram(histogram: dict[str, int]) -> dict[str, float | int]:
    counts = sorted((int(token_count), int(count)) for token_count, count in histogram.items())
    total = sum(count for _, count in counts)
    if total == 0:
        return {"count": 0, "mean": 0.0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}

    def nearest_rank(percentile: float) -> int:
        target = max(1, math.ceil(percentile * total))
        cumulative = 0
        for token_count, count in counts:
            cumulative += count
            if cumulative >= target:
                return token_count
        raise AssertionError("histogram rank exceeds total")

    return {
        "count": total,
        "mean": sum(token_count * count for token_count, count in counts) / total,
        "p50": nearest_rank(0.50),
        "p90": nearest_rank(0.90),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "max": counts[-1][0],
    }


def write_private_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def merge(summary_paths: list[Path], mixed_paths: list[Path], output_dir: Path) -> dict:
    if len(summary_paths) != 2 or len(mixed_paths) != 2:
        raise ValueError("exactly two arm summaries and mixed datasets are required")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    contracts = {str(item.get("contract")) for item in summaries}
    supported_contracts = {
        "boss-multisandbox-dwh-rollout-outcomes-v1",
        "boss-multisandbox-dwh-rollout-outcomes-v2",
    }
    if len(contracts) != 1 or not contracts <= supported_contracts:
        raise ValueError("unexpected arm summary contract")
    samples = {int(item["samples_per_task"]) for item in summaries}
    if len(samples) != 1:
        raise ValueError("arm samples_per_task mismatch")

    mixed_rows = []
    identities = set()
    for path in mixed_paths:
        for row in pq.read_table(path).to_pylist():
            identity = str(row["extra_info"]["verifier_id"])
            if identity in identities:
                raise ValueError("mixed arm datasets overlap")
            identities.add(identity)
            mixed_rows.append(row)
    expected_mixed = sum(int(item["mixed_screening_rows"]) for item in summaries)
    if len(mixed_rows) != expected_mixed:
        raise ValueError(f"expected {expected_mixed} mixed rows, got {len(mixed_rows)}")

    samples_per_task = samples.pop()
    tasks = sum(int(item["tasks"]) for item in summaries)
    trajectories = sum(int(item["trajectories"]) for item in summaries)
    correct = sum(int(item["correct_trajectories"]) for item in summaries)
    completed = sum(int(item["completed_trajectories"]) for item in summaries)
    runtime_errors = sum(int(item["runtime_error_trajectories"]) for item in summaries)
    timeouts = sum(int(item.get("timeout_trajectories", 0)) for item in summaries)
    abort_acks = sum(int(item.get("timeout_abort_acknowledged_count", 0)) for item in summaries)
    abort_physical = sum(
        int(item.get("timeout_abort_physical_request_count", 0)) for item in summaries
    )
    abort_errors = sum(int(item.get("timeout_abort_error_count", 0)) for item in summaries)
    summary = {
        "contract": "boss-multisandbox-dwh-dual-server-outcomes-v1",
        "arms": 2,
        "tasks": tasks,
        "samples_per_task": samples_per_task,
        "trajectories": trajectories,
        "correct_trajectories": correct,
        "completed_trajectories": completed,
        "runtime_error_trajectories": runtime_errors,
        "timeout_trajectories": timeouts,
        "timeout_abort_acknowledged_count": abort_acks,
        "timeout_abort_physical_request_count": abort_physical,
        "timeout_abort_error_count": abort_errors,
        "evaluable_trajectories": trajectories - runtime_errors - timeouts,
        "correct_rate": correct / trajectories,
        "completion_rate": completed / trajectories,
        "bucket_counts": merge_counts([item["bucket_counts"] for item in summaries]),
        "correct_count_histogram": merge_counts(
            [item["correct_count_histogram"] for item in summaries]
        ),
        "answer_type_bucket_counts": merge_nested(
            [item["answer_type_bucket_counts"] for item in summaries]
        ),
        "version_bucket_counts": merge_nested(
            [item["version_bucket_counts"] for item in summaries]
        ),
        "mixed_screening_rows": len(mixed_rows),
        "explicit_semantic_review_completed": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
    }
    if all("difficulty_bucket_counts" in item for item in summaries):
        summary["difficulty_bucket_counts"] = merge_nested(
            [item["difficulty_bucket_counts"] for item in summaries]
        )
    if all("difficulty_correct_count_histogram" in item for item in summaries):
        summary["difficulty_correct_count_histogram"] = merge_nested(
            [item["difficulty_correct_count_histogram"] for item in summaries]
        )
    if all("response_token_histogram" in item for item in summaries):
        token_histogram = merge_counts(
            [item["response_token_histogram"] for item in summaries]
        )
        summary["response_token_histogram"] = token_histogram
        summary["response_token_distribution"] = token_distribution_from_histogram(
            token_histogram
        )
    if all("response_token_histogram_by_difficulty" in item for item in summaries):
        by_difficulty = merge_nested(
            [item["response_token_histogram_by_difficulty"] for item in summaries]
        )
        summary["response_token_histogram_by_difficulty"] = by_difficulty
        summary["response_token_distribution_by_difficulty"] = {
            level: token_distribution_from_histogram(histogram)
            for level, histogram in by_difficulty.items()
        }
    if tasks * samples_per_task != trajectories:
        raise ValueError("merged trajectory shape mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_private_parquet(output_dir / "mixed_groups.sensitive.parquet", mixed_rows)
    (output_dir / "safe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-summary", type=Path, action="append", required=True)
    parser.add_argument("--mixed-dataset", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            merge(args.arm_summary, args.mixed_dataset, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
