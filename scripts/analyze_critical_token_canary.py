#!/usr/bin/env python3
"""Audit whether a critical-token SFT canary repaired its frozen SQL divergence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from scripts.analyze_repair_sft_free_run_divergence import normalize_container


DIAGNOSTIC_CONTRACT = "repair-sft-teacher-forced-component-diagnostic-v3"
COMPARISON_CONTRACT = "repair-sft-teacher-forced-prepost-comparison-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_by_task(diagnostic: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["task_id"]): row["sql_token_rank"]
        for row in diagnostic["per_task"]
    }


def analyze_critical_token_recovery(
    critical_rows: Iterable[dict[str, Any]],
    baseline: dict[str, Any],
    post: dict[str, Any],
    comparison: dict[str, Any],
    *,
    required_full_sql_probability_tasks: int = 12,
) -> dict[str, Any]:
    """Return a safe aggregate and bounded per-task critical-token audit."""

    if baseline.get("contract") != DIAGNOSTIC_CONTRACT:
        raise ValueError("baseline must use semantic diagnostic v3")
    if post.get("contract") != DIAGNOSTIC_CONTRACT:
        raise ValueError("post model must use semantic diagnostic v3")
    if comparison.get("contract") != COMPARISON_CONTRACT:
        raise ValueError("unexpected pre/post comparison contract")
    if baseline.get("task_ids") != post.get("task_ids"):
        raise ValueError("baseline and post task IDs differ")
    if baseline.get("data_sha256") != post.get("data_sha256"):
        raise ValueError("baseline and post data hashes differ")
    if comparison.get("task_ids_identical") is not True:
        raise ValueError("comparison did not validate identical task IDs")
    if comparison.get("data_sha256_identical") is not True:
        raise ValueError("comparison did not validate identical data hashes")
    if comparison.get("forward_only_both") is not True:
        raise ValueError("critical-token attribution requires forward-only diagnostics")
    if comparison.get("optimizer_initialized_either") is not False:
        raise ValueError("diagnostic unexpectedly initialized an optimizer")

    baseline_by_task = _rank_by_task(baseline)
    post_by_task = _rank_by_task(post)
    statuses: Counter[str] = Counter()
    family_stats: dict[str, Counter[str]] = defaultdict(Counter)
    rank_direction: Counter[str] = Counter()
    evidence: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()

    for raw_row in critical_rows:
        row = normalize_container(raw_row)
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen_task_ids:
            raise ValueError(f"invalid or duplicate critical task ID: {task_id!r}")
        seen_task_ids.add(task_id)
        if task_id not in baseline_by_task or task_id not in post_by_task:
            raise ValueError(f"missing diagnostic evidence for {task_id!r}")

        critical_offset = int(row["critical_sql_token_offset"])
        critical_target_id = int(row["critical_sql_target_id"])
        family = str(row["critical_token_family"])
        before = baseline_by_task[task_id]
        after = post_by_task[task_id]
        if int(before["first_nongreedy_offset"]) != critical_offset:
            raise ValueError(f"{task_id}: frozen offset differs from baseline")
        if int(before["first_nongreedy_target_id"]) != critical_target_id:
            raise ValueError(f"{task_id}: frozen target ID differs from baseline")

        after_offset = after.get("first_nongreedy_offset")
        if after_offset is None or int(after_offset) > critical_offset:
            status = "original_critical_became_greedy"
        elif int(after_offset) < critical_offset:
            status = "new_earlier_nongreedy_blocks_direct_attribution"
        else:
            if int(after["first_nongreedy_target_id"]) != critical_target_id:
                raise ValueError(f"{task_id}: target ID changed at the frozen offset")
            status = "original_critical_still_first_nongreedy"
            before_rank = int(before["first_nongreedy_rank"])
            after_rank = int(after["first_nongreedy_rank"])
            if after_rank < before_rank:
                rank_direction["improved"] += 1
            elif after_rank > before_rank:
                rank_direction["worsened"] += 1
            else:
                rank_direction["tied"] += 1

        statuses[status] += 1
        family_stats[family]["tasks"] += 1
        family_stats[family][status] += 1
        evidence.append(
            {
                "task_id": task_id,
                "critical_token_family": family,
                "critical_sql_token_offset": critical_offset,
                "baseline_rank": int(before["first_nongreedy_rank"]),
                "baseline_probability": before["first_nongreedy_target_probability"],
                "post_first_nongreedy_offset": after_offset,
                "post_rank_at_critical_offset": (
                    int(after["first_nongreedy_rank"])
                    if status == "original_critical_still_first_nongreedy"
                    else None
                ),
                "post_probability_at_critical_offset": (
                    after["first_nongreedy_target_probability"]
                    if status == "original_critical_still_first_nongreedy"
                    else None
                ),
                "status": status,
            }
        )

    expected_ids = {str(task_id) for task_id in baseline["task_ids"]}
    if seen_task_ids != expected_ids:
        raise ValueError("critical dataset task IDs differ from diagnostic task IDs")

    sql = comparison["components"]["sql_shell"]
    probability = sql["per_task_probability"]
    post_above_half = int(probability["post_sft_above_0_5"])
    full_sql_gate_passed = post_above_half >= required_full_sql_probability_tasks
    return {
        "contract": "repair-sft-critical-token-canary-analysis-v1",
        "task_count": len(seen_task_ids),
        "data_sha256_identical": True,
        "forward_only_both": True,
        "optimizer_initialized_either": False,
        "critical_token_recovery": {
            "status_counts": dict(sorted(statuses.items())),
            "still_nongreedy_rank_direction": dict(sorted(rank_direction.items())),
            "by_family": {
                family: dict(sorted(counts.items()))
                for family, counts in sorted(family_stats.items())
            },
        },
        "full_correction_sql_gate": {
            "required_tasks_above_0_5": required_full_sql_probability_tasks,
            "step120_tasks_above_0_5": int(probability["step120_above_0_5"]),
            "post_sft_tasks_above_0_5": post_above_half,
            "passed": full_sql_gate_passed,
            "step120_mean_nll": sql["step120_mean_nll"],
            "post_sft_mean_nll": sql["post_sft_mean_nll"],
            "per_task_nll_improved": int(sql["per_task_nll"]["improved"]),
        },
        "decision": (
            "eligible_for_short_replay"
            if full_sql_gate_passed
            else "stop_no_replay_probability_gate_failed"
        ),
        "per_task": evidence,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critical-data", type=Path, required=True)
    parser.add_argument("--step120-diagnostic", type=Path, required=True)
    parser.add_argument("--post-diagnostic", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-full-sql-probability-tasks", type=int, default=12)
    args = parser.parse_args()

    baseline = json.loads(args.step120_diagnostic.read_text(encoding="utf-8"))
    post = json.loads(args.post_diagnostic.read_text(encoding="utf-8"))
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    if sha256_file(args.critical_data) != baseline.get("data_sha256"):
        raise ValueError("critical dataset hash differs from diagnostic data hash")
    rows = [normalize_container(row) for row in pd.read_parquet(args.critical_data).to_dict("records")]
    result = analyze_critical_token_recovery(
        rows,
        baseline,
        post,
        comparison,
        required_full_sql_probability_tasks=args.required_full_sql_probability_tasks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
