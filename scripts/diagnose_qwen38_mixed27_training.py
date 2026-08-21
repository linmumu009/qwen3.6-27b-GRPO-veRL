#!/usr/bin/env python3
"""Produce a privacy-safe diagnostic summary for a PI GRPO training run."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean, pstdev
from typing import Any, Iterable


METRIC_RE = re.compile(
    r"(?:^| - )([A-Za-z0-9_./@-]+):(?:np\.(?:float64|int32|int64)\()?([-+0-9.eE]+)\)?"
)
STEP_RE = re.compile(r"\bstep:(\d+)\b")


def as_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def mean(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return fmean(clean) if clean else None


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean, y_mean = fmean(xs), fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def digest_input(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def iter_rollouts(directory: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    files: list[tuple[int, Path]] = []
    for path in directory.glob("*.jsonl"):
        try:
            files.append((int(path.stem), path))
        except ValueError:
            continue
    for step, path in sorted(files):
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    yield step, json.loads(line)


def describe_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "score",
        "base_score",
        "dense_final_answer_correctness",
        "evidence_reward",
        "boss_reward",
        "boss_result_score",
        "boss_process_score",
        "boss_efficiency_score",
        "required_table_used",
        "sql_evidence_correct",
        "has_final_answer",
        "online_eligible",
        "safe",
        "valid_tool_protocol",
        "successful_bash",
        "bash_command_count",
        "sql_evidence_queries_checked",
        "sql_evidence_queries_truncated",
        "gold_sql_verified",
    )
    return {
        "rows": len(rows),
        "means": {key: mean(as_float(row, key) for row in rows) for key in keys},
        "output_chars_mean": mean(len(str(row.get("output") or "")) for row in rows),
        "verifier_error_rows": sum(bool(row.get("verifier_error")) for row in rows),
    }


def summarize_rollouts(directory: Path, expected_steps: int, rows_per_step: int) -> tuple[dict[str, Any], set[int]]:
    rows: list[dict[str, Any]] = []
    step_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    step_groups: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for step, row in iter_rollouts(directory):
        row = dict(row)
        row["_step"] = step
        prompt = digest_input(row.get("input"))
        rows.append(row)
        step_rows[step].append(row)
        step_groups[step][prompt].append(row)

    groups: list[dict[str, Any]] = []
    exposure_counts: dict[str, int] = defaultdict(int)
    for step in sorted(step_groups):
        for prompt, group_rows in step_groups[step].items():
            exposure_counts[prompt] += 1
            scores = [as_float(row, "score") for row in group_rows]
            correct = sum(as_float(row, "final_answer_correct") > 0 for row in group_rows)
            groups.append(
                {
                    "step": step,
                    "exposure": exposure_counts[prompt],
                    "size": len(group_rows),
                    "correct": correct,
                    "score_mean": fmean(scores),
                    "score_std": pstdev(scores),
                    "output_chars_mean": fmean(len(str(row.get("output") or "")) for row in group_rows),
                    "has_final_mean": fmean(as_float(row, "has_final_answer") for row in group_rows),
                }
            )

    steps_any_correct = {
        step
        for step, current in step_rows.items()
        if any(as_float(row, "final_answer_correct") > 0 for row in current)
    }
    all_wrong = [group for group in groups if group["correct"] == 0]
    mixed = [group for group in groups if 0 < group["correct"] < group["size"]]
    all_correct = [group for group in groups if group["correct"] == group["size"]]

    exposure_summary = []
    for exposure in sorted({group["exposure"] for group in groups}):
        selected = [group for group in groups if group["exposure"] == exposure]
        exposure_summary.append(
            {
                "exposure": exposure,
                "groups": len(selected),
                "correct_trajectories": sum(group["correct"] for group in selected),
                "mixed_groups": sum(group["correct"] > 0 for group in selected),
                "all_wrong_groups": sum(group["correct"] == 0 for group in selected),
                "score_mean": mean(group["score_mean"] for group in selected),
                "has_final_answer_mean": mean(group["has_final_mean"] for group in selected),
                "output_chars_mean": mean(group["output_chars_mean"] for group in selected),
            }
        )

    wrong = [row for row in rows if as_float(row, "final_answer_correct") <= 0]
    correct = [row for row in rows if as_float(row, "final_answer_correct") > 0]
    correlation_keys = (
        "base_score",
        "boss_reward",
        "boss_result_score",
        "boss_process_score",
        "boss_efficiency_score",
        "dense_final_answer_correctness",
        "evidence_reward",
        "required_table_used",
        "bash_command_count",
    )
    wrong_scores = [as_float(row, "score") for row in wrong]

    step_bins = []
    bin_width = max(1, math.ceil(expected_steps / 6))
    for start in range(1, expected_steps + 1, bin_width):
        end = min(expected_steps, start + bin_width - 1)
        selected = [group for group in groups if start <= group["step"] <= end]
        step_bins.append(
            {
                "steps": f"{start}-{end}",
                "groups": len(selected),
                "correct_trajectories": sum(group["correct"] for group in selected),
                "mixed_groups": sum(group["correct"] > 0 for group in selected),
                "all_wrong_groups": sum(group["correct"] == 0 for group in selected),
            }
        )

    actual_steps = set(step_rows)
    integrity = {
        "files": len(actual_steps),
        "rows": len(rows),
        "groups": len(groups),
        "missing_steps": sorted(set(range(1, expected_steps + 1)) - actual_steps),
        "wrong_rows_per_step": {
            str(step): len(current)
            for step, current in step_rows.items()
            if len(current) != rows_per_step
        },
        "group_sizes": sorted({group["size"] for group in groups}),
        "unique_prompts": len(exposure_counts),
        "prompt_exposure_counts": sorted(set(exposure_counts.values())),
    }
    return (
        {
            "integrity": integrity,
            "overall": describe_rows(rows),
            "correctness": {
                "correct_trajectories": len(correct),
                "correct_trajectory_rate": len(correct) / len(rows) if rows else None,
                "mixed_groups": len(mixed),
                "all_wrong_groups": len(all_wrong),
                "all_correct_groups": len(all_correct),
                "all_wrong_groups_with_nonzero_reward_variance": sum(
                    group["score_std"] > 0 for group in all_wrong
                ),
                "optimizer_steps_with_any_correct": len(steps_any_correct),
                "optimizer_steps_with_no_correct": len(actual_steps - steps_any_correct),
            },
            "group_score_std_mean": mean(group["score_std"] for group in groups),
            "by_exposure": exposure_summary,
            "by_step_bin": step_bins,
            "correct_rows": describe_rows(correct),
            "wrong_rows": describe_rows(wrong),
            "wrong_row_score_correlations": {
                key: corr(wrong_scores, [as_float(row, key) for row in wrong])
                for key in correlation_keys
            },
        },
        steps_any_correct,
    )


def summarize_log(path: Path, expected_steps: int, steps_any_correct: set[int]) -> dict[str, Any]:
    records: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "training/global_step:" not in line:
                continue
            match = STEP_RE.search(line)
            if not match:
                continue
            records[int(match.group(1))] = {
                key: float(value) for key, value in METRIC_RE.findall(line)
            }

    metric_keys = (
        "actor/ppo_kl",
        "actor/pg_clipfrac",
        "actor/grad_norm",
        "actor/pg_loss",
        "actor/loss",
        "critic/score/mean",
        "response_length/mean",
        "response_length/max",
        "response/aborted_ratio",
        "num_turns/mean",
        "timing_s/update_actor",
        "timing_s/step",
        "fully_async/total_wait_time",
        "fully_async/count/stale_trajectory_processed",
        "fully_async/count/staleness_samples",
    )

    def describe(selected_steps: Iterable[int]) -> dict[str, float | None]:
        selected = [records[step] for step in selected_steps if step in records]
        return {
            key: mean(record[key] for record in selected if key in record)
            for key in metric_keys
        }

    ordered = sorted(records)
    no_correct = set(ordered) - steps_any_correct
    return {
        "records": len(records),
        "missing_steps": sorted(set(range(1, expected_steps + 1)) - set(records)),
        "overall_means": describe(ordered),
        "first_9_means": describe(ordered[:9]),
        "last_9_means": describe(ordered[-9:]),
        "steps_with_any_correct": {"count": len(set(ordered) & steps_any_correct), "means": describe(steps_any_correct)},
        "steps_with_no_correct": {"count": len(no_correct), "means": describe(no_correct)},
        "ppo_kl_sum_diagnostic_not_reference_divergence": sum(
            records[step].get("actor/ppo_kl", 0.0) for step in ordered
        ),
        "final_cumulative_counters": {
            key: records[ordered[-1]].get(key) if ordered else None
            for key in (
                "fully_async/count/stale_trajectory_processed",
                "fully_async/count/staleness_samples",
                "fully_async/count/dropped_stale_samples",
                "fully_async/count/total_generated_samples",
            )
        },
        "nonzero_grad_norm_steps": sum(records[step].get("actor/grad_norm", 0.0) > 0 for step in ordered),
        "nonzero_grad_norm_no_correct_steps": sum(
            records[step].get("actor/grad_norm", 0.0) > 0 for step in no_correct
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=54)
    parser.add_argument("--rows-per-step", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_summary, steps_any_correct = summarize_rollouts(
        args.rollout_dir, args.expected_steps, args.rows_per_step
    )
    result = {
        "schema_version": "qwen38-mixed27-collapse-diagnostic-safe-v1",
        "rollouts": rollout_summary,
        "driver": summarize_log(args.driver_log, args.expected_steps, steps_any_correct),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
