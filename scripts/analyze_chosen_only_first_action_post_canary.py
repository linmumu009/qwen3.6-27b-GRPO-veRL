#!/usr/bin/env python3
"""Apply sealed calibration16 gates to the one-step chosen-only canary."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from scripts.analyze_state_recovery_semantics import critical_token_family
from scripts.prepare_repair_sft_dataset import sha256_file


CONTRACT = "chosen-only-first-action-post-canary-decision-v1"


def _task_sql_nll(row: dict[str, Any]) -> float:
    return float(((row.get("components") or {}).get("sql_shell") or {})["mean_nll"])


def _first_barrier_family(rank: dict[str, Any]) -> str:
    if rank.get("first_nongreedy_offset") is None:
        return "all_sql_tokens_greedy"
    token = rank.get("first_nongreedy_target_token")
    if not isinstance(token, str):
        raise ValueError("non-greedy SQL barrier is missing its decoded target token")
    return critical_token_family(token)


def decide(
    baseline: dict[str, Any],
    post: dict[str, Any],
    authorization: dict[str, Any],
) -> dict[str, Any]:
    if authorization.get("contract") != "chosen-only-first-action-baseline-decision-v1":
        raise ValueError("chosen-only post gate authorization contract mismatch")
    if (authorization.get("one_step_canary") or {}).get("allowed") is not True:
        raise ValueError("chosen-only one-step canary was not authorized")
    for label, result in (("baseline", baseline), ("post", post)):
        if result.get("contract") != "repair-sft-teacher-forced-component-diagnostic-v3":
            raise ValueError(f"{label} teacher-forced contract mismatch")
        if result.get("forward_only") is not True or result.get("optimizer_initialized") is not False:
            raise ValueError(f"{label} evaluation was not forward-only")
        if int(result.get("task_count") or 0) != 16:
            raise ValueError(f"{label} evaluation is not calibration16")
    if baseline.get("model_label") != "step120" or post.get("model_label") != "chosen_only_post_step1":
        raise ValueError("chosen-only pre/post model labels drifted")
    if baseline.get("data_sha256") != post.get("data_sha256"):
        raise ValueError("chosen-only pre/post calibration data hashes differ")
    baseline_tasks = baseline.get("task_ids") or []
    post_tasks = post.get("task_ids") or []
    if baseline_tasks != post_tasks or len(set(baseline_tasks)) != 16:
        raise ValueError("chosen-only pre/post task ordering differs")

    base_components = baseline["components"]
    post_components = post["components"]
    base_rank = baseline["sql_token_rank"]
    post_rank = post["sql_token_rank"]
    base_sql_nll = float(base_components["sql_shell"]["mean_nll"])
    post_sql_nll = float(post_components["sql_shell"]["mean_nll"])
    relative_improvement = (base_sql_nll - post_sql_nll) / base_sql_nll
    by_task_base = {row["task_id"]: row for row in baseline.get("per_task") or []}
    by_task_post = {row["task_id"]: row for row in post.get("per_task") or []}
    if set(by_task_base) != set(baseline_tasks) or set(by_task_post) != set(baseline_tasks):
        raise ValueError("chosen-only pre/post per-task evidence is incomplete")
    improved_tasks = sum(
        _task_sql_nll(by_task_post[current]) < _task_sql_nll(by_task_base[current])
        for current in baseline_tasks
    )
    transition_counts: Counter[str] = Counter()
    baseline_family_counts: Counter[str] = Counter()
    post_family_counts: Counter[str] = Counter()
    by_baseline_family: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for current in baseline_tasks:
        before = by_task_base[current]["sql_token_rank"]
        after = by_task_post[current]["sql_token_rank"]
        before_offset = before.get("first_nongreedy_offset")
        after_offset = after.get("first_nongreedy_offset")
        before_family = _first_barrier_family(before)
        after_family = _first_barrier_family(after)
        baseline_family_counts[before_family] += 1
        post_family_counts[after_family] += 1
        if before_offset is None:
            transition = (
                "stable_all_sql_tokens_greedy"
                if after_offset is None
                else "earlier_regression"
            )
        elif after_offset is None or int(after_offset) > int(before_offset):
            transition = "cleared_to_later_or_all_greedy"
        elif int(after_offset) == int(before_offset):
            transition = "same_first_offset"
        else:
            transition = "earlier_regression"
        transition_counts[transition] += 1
        by_baseline_family[before_family][transition] += 1

    earlier_regressions = transition_counts["earlier_regression"]

    thresholds = authorization.get("post_canary_gates") or {}
    checks = {
        "sql_nll_relative_improvement": {
            "observed": relative_improvement,
            "required_min": float(
                thresholds["calibration_sql_nll_relative_improvement_min"]
            ),
        },
        "tasks_sql_nll_improved": {
            "observed": improved_tasks,
            "required_min": int(thresholds["calibration_tasks_sql_nll_improved_min"]),
        },
        "sql_greedy_token_gain": {
            "observed": int(post_rank["greedy_token_count"])
            - int(base_rank["greedy_token_count"]),
            "required_min": int(thresholds["calibration_sql_greedy_token_gain_min"]),
        },
        "sql_top5_token_count": {
            "observed": int(post_rank["top5_token_count"]),
            "required_min": int(thresholds["calibration_sql_top5_token_count_min"]),
        },
        "sql_mean_rank": {
            "before": float(base_rank["mean_rank"]),
            "after": float(post_rank["mean_rank"]),
        },
        "tool_structure_nll_relative_regression": {
            "observed": (
                float(post_components["tool_structure"]["mean_nll"])
                - float(base_components["tool_structure"]["mean_nll"])
            )
            / float(base_components["tool_structure"]["mean_nll"]),
            "allowed_max": float(
                thresholds["calibration_tool_structure_nll_relative_regression_max"]
            ),
        },
        "earlier_sql_boundary_regressions": {
            "observed": earlier_regressions,
            "allowed_max": int(
                thresholds["earlier_template_or_sql_boundary_regressions_max"]
            ),
        },
    }
    checks["sql_nll_relative_improvement"]["passed"] = (
        checks["sql_nll_relative_improvement"]["observed"]
        >= checks["sql_nll_relative_improvement"]["required_min"]
    )
    checks["tasks_sql_nll_improved"]["passed"] = (
        checks["tasks_sql_nll_improved"]["observed"]
        >= checks["tasks_sql_nll_improved"]["required_min"]
    )
    checks["sql_greedy_token_gain"]["passed"] = (
        checks["sql_greedy_token_gain"]["observed"]
        >= checks["sql_greedy_token_gain"]["required_min"]
    )
    checks["sql_top5_token_count"]["passed"] = (
        checks["sql_top5_token_count"]["observed"]
        >= checks["sql_top5_token_count"]["required_min"]
    )
    checks["sql_mean_rank"]["passed"] = (
        checks["sql_mean_rank"]["after"] < checks["sql_mean_rank"]["before"]
    )
    checks["tool_structure_nll_relative_regression"]["passed"] = (
        checks["tool_structure_nll_relative_regression"]["observed"]
        <= checks["tool_structure_nll_relative_regression"]["allowed_max"]
    )
    checks["earlier_sql_boundary_regressions"]["passed"] = (
        checks["earlier_sql_boundary_regressions"]["observed"]
        <= checks["earlier_sql_boundary_regressions"]["allowed_max"]
    )
    passed = all(item["passed"] for item in checks.values())
    return {
        "contract": CONTRACT,
        "rows": 16,
        "before": {
            "sql_mean_nll": base_sql_nll,
            "sql_greedy_tokens": int(base_rank["greedy_token_count"]),
            "sql_top5_tokens": int(base_rank["top5_token_count"]),
            "sql_mean_rank": float(base_rank["mean_rank"]),
            "tool_structure_mean_nll": float(
                base_components["tool_structure"]["mean_nll"]
            ),
        },
        "after": {
            "sql_mean_nll": post_sql_nll,
            "sql_greedy_tokens": int(post_rank["greedy_token_count"]),
            "sql_top5_tokens": int(post_rank["top5_token_count"]),
            "sql_mean_rank": float(post_rank["mean_rank"]),
            "tool_structure_mean_nll": float(
                post_components["tool_structure"]["mean_nll"]
            ),
        },
        "checks": checks,
        "first_nongreedy_boundary_diagnostic": {
            "baseline_family_counts": dict(sorted(baseline_family_counts.items())),
            "post_family_counts": dict(sorted(post_family_counts.items())),
            "transition_counts": dict(sorted(transition_counts.items())),
            "by_baseline_family": {
                family: {
                    "total": sum(counts.values()),
                    **dict(sorted(counts.items())),
                }
                for family, counts in sorted(by_baseline_family.items())
            },
        },
        "gate_passed": passed,
        "decision": {
            "next_action": (
                "run_calibration16_one_action_free_rollout"
                if passed
                else "stop_chosen_only_canary_and_do_not_free_rollout"
            ),
            "additional_training_allowed": False,
            "free_rollout_allowed": passed,
            "promotion_allowed": False,
        },
        "contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decide(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.post.read_text(encoding="utf-8")),
        json.loads(args.authorization.read_text(encoding="utf-8")),
    )
    result["source_sha256"] = {
        "baseline": sha256_file(args.baseline),
        "post": sha256_file(args.post),
        "authorization": sha256_file(args.authorization),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
