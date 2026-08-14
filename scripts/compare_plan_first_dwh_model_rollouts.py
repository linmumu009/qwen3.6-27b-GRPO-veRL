#!/usr/bin/env python3
"""Compare native and Step120 plan-first DWH outcomes without exposing rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import statistics
from typing import Any

import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row["instruction_sha256"])
        if identity in result:
            raise ValueError(f"duplicate {label} instruction hash")
        result[identity] = row
    return result


def arm_summary(rows: list[dict[str, Any]], samples: int) -> dict[str, Any]:
    correct = sum(int(row["correct_count"]) for row in rows)
    trajectories = len(rows) * samples
    return {
        "tasks": len(rows),
        "trajectories": trajectories,
        "correct_trajectories": correct,
        "correct_rate": correct / trajectories,
        "runtime_error_trajectories": sum(int(row["runtime_error_count"]) for row in rows),
        "timeout_trajectories": sum(int(row["trajectory_timeout_count"]) for row in rows),
        "bucket_counts": dict(sorted(Counter(str(row["bucket"]) for row in rows).items())),
        "correct_count_histogram": {
            str(value): sum(int(row["correct_count"]) == value for row in rows)
            for value in range(samples + 1)
        },
    }


def compare(
    dataset_path: Path,
    native_path: Path,
    step120_path: Path,
    output_dir: Path,
    *,
    samples_per_task: int = 8,
) -> dict[str, Any]:
    dataset = pq.read_table(dataset_path).to_pylist()
    native = index_unique(read_jsonl(native_path), "native")
    step120 = index_unique(read_jsonl(step120_path), "step120")
    metadata = {
        str(row["extra_info"]["instruction_sha256"]): {
            "difficulty_band": int(row["extra_info"]["difficulty_band"]),
            "comparison_split": str(row["extra_info"]["comparison_split"]),
            "answer_type": str(row["reward_model"]["ground_truth"]["answer_type"]),
        }
        for row in dataset
    }
    if set(native) != set(step120) or set(native) != set(metadata):
        raise ValueError("native, Step120, and dataset task identities differ")

    transitions: Counter[str] = Counter()
    delta_histogram: Counter[int] = Counter()
    wins = Counter()
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    candidates: list[dict[str, Any]] = []
    deltas: list[float] = []
    for identity in sorted(metadata):
        left, right = native[identity], step120[identity]
        left_correct, right_correct = int(left["correct_count"]), int(right["correct_count"])
        delta = right_correct - left_correct
        deltas.append(delta / samples_per_task)
        delta_histogram[delta] += 1
        transitions[f"{left['bucket']}->{right['bucket']}"] += 1
        wins["step120_win" if delta > 0 else "native_win" if delta < 0 else "tie"] += 1
        meta = metadata[identity]
        grouped[("band", str(meta["difficulty_band"]))].append((left_correct, right_correct))
        grouped[("answer_type", str(meta["answer_type"]))].append((left_correct, right_correct))
        grouped[("split", str(meta["comparison_split"]))].append((left_correct, right_correct))
        if meta["comparison_split"] == "training_candidate" and right["bucket"] == "mixed":
            candidates.append(
                {
                    "instruction_sha256": identity,
                    "native_correct_count": left_correct,
                    "step120_correct_count": right_correct,
                    "difficulty_band": meta["difficulty_band"],
                    "answer_type": meta["answer_type"],
                    "training_allowed": False,
                }
            )

    def aggregate(pairs: list[tuple[int, int]]) -> dict[str, Any]:
        denominator = len(pairs) * samples_per_task
        native_correct = sum(left for left, _ in pairs)
        step_correct = sum(right for _, right in pairs)
        return {
            "tasks": len(pairs),
            "native_correct_rate": native_correct / denominator,
            "step120_correct_rate": step_correct / denominator,
            "step120_minus_native": (step_correct - native_correct) / denominator,
        }

    strata = {
        family: {
            value: aggregate(pairs)
            for (current_family, value), pairs in sorted(grouped.items())
            if current_family == family
        }
        for family in ("band", "answer_type", "split")
    }
    summary = {
        "contract": "llin-plan-first-dwh-base-step120-comparison-summary-v1",
        "tasks": len(dataset),
        "samples_per_task": samples_per_task,
        "arms": {
            "native": arm_summary(list(native.values()), samples_per_task),
            "step120": arm_summary(list(step120.values()), samples_per_task),
        },
        "paired_task_outcomes": dict(sorted(wins.items())),
        "paired_mean_success_rate_delta": statistics.fmean(deltas),
        "correct_count_delta_histogram": {
            str(value): delta_histogram.get(value, 0)
            for value in range(-samples_per_task, samples_per_task + 1)
        },
        "bucket_transition_counts": dict(sorted(transitions.items())),
        "strata": strata,
        "step120_mixed_training_candidates": len(candidates),
        "candidate_rule": "training_candidate_split_and_step120_1_to_7_of_8_without_runtime_error_or_timeout",
        "request_rng_pairing": "unpaired_stochastic_requests; comparison paired at task level",
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "candidate_hashes.sensitive.jsonl.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(output_dir / "candidate_hashes.sensitive.jsonl")
    (output_dir / "safe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--native-per-task", type=Path, required=True)
    parser.add_argument("--step120-per-task", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=8)
    args = parser.parse_args()
    result = compare(
        args.dataset,
        args.native_per_task,
        args.step120_per_task,
        args.output_dir,
        samples_per_task=args.samples_per_task,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
