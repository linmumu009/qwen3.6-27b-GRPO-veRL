#!/usr/bin/env python3
"""Compare normalized 10x8 PI-Agent and veRL runtime outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bucket(correct: int, n: int) -> str:
    if correct == 0:
        return "all_wrong"
    if correct == n:
        return "all_correct"
    return "mixed"


def bernoulli_js(left: float, right: float) -> float:
    midpoint = (left + right) / 2

    def kl(p: float, q: float) -> float:
        total = 0.0
        for a, b in ((p, q), (1 - p, 1 - q)):
            if a > 0:
                total += a * math.log2(a / b)
        return total

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


def arm_summary(rows: list[dict[str, Any]], expected_n: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["task_key"])].append(row)
    group_buckets = Counter()
    per_task = {}
    for key, group in sorted(groups.items()):
        correct = sum(float(row.get("final_answer_correct") or 0) > 0 for row in group)
        name = bucket(correct, expected_n) if len(group) == expected_n else "invalid_size"
        group_buckets[name] += 1
        per_task[key] = {
            "rows": len(group),
            "correct": correct,
            "accuracy": correct / len(group),
            "bucket": name,
            "complete": sum(bool(row.get("completed", True)) for row in group),
            "timeouts": sum(bool(row.get("timeout", False)) for row in group),
        }
    return {
        "rows": len(rows),
        "tasks": len(groups),
        "accuracy": mean([float(row.get("final_answer_correct") or 0) for row in rows]),
        "has_final_answer_rate": mean([float(row.get("has_final_answer") or 0) for row in rows]),
        "completion_rate": mean([float(bool(row.get("completed", True))) for row in rows]),
        "timeout_rate": mean([float(bool(row.get("timeout", False))) for row in rows]),
        "bucket_counts": dict(sorted(group_buckets.items())),
        "per_task": per_task,
    }


def paired_bootstrap(deltas: list[float], samples: int = 20_000) -> list[float]:
    rng = random.Random(20260813)
    values = sorted(
        mean(rng.choice(deltas) for _ in deltas)
        for _ in range(samples)
    )
    return [values[int(0.025 * samples)], values[int(0.975 * samples)]]


def analyze(
    pi_rows: list[dict[str, Any]],
    verl_rows: list[dict[str, Any]],
    expected_tasks: int = 10,
    expected_n: int = 8,
) -> dict[str, Any]:
    pi = arm_summary(pi_rows, expected_n)
    verl = arm_summary(verl_rows, expected_n)
    task_keys_identical = set(pi["per_task"]) == set(verl["per_task"])
    common = sorted(set(pi["per_task"]) & set(verl["per_task"]))
    comparisons = []
    for key in common:
        left, right = pi["per_task"][key], verl["per_task"][key]
        comparisons.append(
            {
                "task_key": key,
                "pi_correct": left["correct"],
                "verl_correct": right["correct"],
                "accuracy_delta_verl_minus_pi": right["accuracy"] - left["accuracy"],
                "absolute_accuracy_delta": abs(right["accuracy"] - left["accuracy"]),
                "pi_bucket": left["bucket"],
                "verl_bucket": right["bucket"],
            }
        )
    deltas = [row["accuracy_delta_verl_minus_pi"] for row in comparisons]
    bucket_agreement = (
        mean(float(row["pi_bucket"] == row["verl_bucket"]) for row in comparisons)
        if comparisons
        else 0.0
    )
    structural = {
        "task_keys_identical": task_keys_identical,
        "both_have_10_tasks": pi["tasks"] == expected_tasks and verl["tasks"] == expected_tasks,
        "both_have_80_rows": pi["rows"] == expected_tasks * expected_n and verl["rows"] == expected_tasks * expected_n,
        "every_group_has_8": all(
            item["rows"] == expected_n
            for arm in (pi, verl)
            for item in arm["per_task"].values()
        ),
        "sample_indexes_unique": all(
            len({(row["task_key"], int(row["sample_index"])) for row in rows}) == len(rows)
            for rows in (pi_rows, verl_rows)
        ),
    }
    metrics = {
        "global_accuracy_delta_verl_minus_pi": verl["accuracy"] - pi["accuracy"],
        "global_accuracy_absolute_delta": abs(verl["accuracy"] - pi["accuracy"]),
        "mean_per_task_absolute_accuracy_delta": mean(
            row["absolute_accuracy_delta"] for row in comparisons
        ) if comparisons else 1.0,
        "paired_task_mean_delta_bootstrap95": paired_bootstrap(deltas) if deltas else [None, None],
        "global_bernoulli_jensen_shannon_divergence_bits": bernoulli_js(pi["accuracy"], verl["accuracy"]),
        "completion_rate_absolute_delta": abs(pi["completion_rate"] - verl["completion_rate"]),
        "bucket_agreement_rate": bucket_agreement,
    }
    parity_checks = {
        **structural,
        "no_timeouts": pi["timeout_rate"] == 0 and verl["timeout_rate"] == 0,
        "global_accuracy_delta_at_most_10pp": metrics["global_accuracy_absolute_delta"] <= 0.10,
        "mean_per_task_delta_at_most_20pp": metrics["mean_per_task_absolute_accuracy_delta"] <= 0.20,
        "completion_delta_at_most_10pp": metrics["completion_rate_absolute_delta"] <= 0.10,
        "js_divergence_at_most_0_02_bits": metrics["global_bernoulli_jensen_shannon_divergence_bits"] <= 0.02,
    }
    parity_passed = all(parity_checks.values())
    verl_buckets = verl["bucket_counts"]
    routing = {
        "decision_enabled": parity_passed,
        "fresh_grpo_eligible_mixed_tasks": verl_buckets.get("mixed", 0),
        "exclude_from_that_optimizer_update_all_correct": verl_buckets.get("all_correct", 0),
        "exclude_from_that_optimizer_update_all_wrong": verl_buckets.get("all_wrong", 0),
        "all_correct_destination": "retain_for_evaluation_and_anti_regression",
        "all_wrong_destination": "retain_for_correction_sft_curriculum_or_future_rescreen",
        "permanent_deletion_allowed": False,
        "individual_trajectory_cherry_picking_allowed": False,
        "training_requires_fresh_rollout_after_screening": True,
    }
    return {
        "contract": "pi-vs-verl-runtime-parity-10x8-v1",
        "interpretation": "engineering_parity_smoke_not_formal_statistical_equivalence",
        "pi_agent": pi,
        "verl_rollout": verl,
        "per_task_comparison": comparisons,
        "parity_metrics": metrics,
        "parity_checks": parity_checks,
        "parity_smoke_passed": parity_passed,
        "group_routing": routing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi", type=Path, required=True)
    parser.add_argument("--verl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--expected-n", type=int, default=8)
    args = parser.parse_args()
    result = analyze(
        read_jsonl(args.pi),
        read_jsonl(args.verl),
        args.expected_tasks,
        args.expected_n,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["parity_smoke_passed"] else 3)


if __name__ == "__main__":
    main()

