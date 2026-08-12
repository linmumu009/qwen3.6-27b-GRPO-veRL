#!/usr/bin/env python3
"""Build a payload-free native-vs-Step120 full-protocol attribution summary."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path
from typing import Any

from scripts.compare_boss_exact_evaluations import index_unique, read_jsonl


CONTRACT = "native-vs-step120-full25-attribution-safe-summary-v1"


def exact_sign_pvalue(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    lower = min(left_only, right_only)
    one_sided = sum(comb(n, k) for k in range(lower + 1)) / (2**n)
    return min(1.0, 2.0 * one_sided)


def paired_boolean(
    native: dict[str, dict[str, Any]],
    step120: dict[str, dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for current_task_id in native:
        left = bool((native[current_task_id].get("reward") or {}).get(key))
        right = bool((step120[current_task_id].get("reward") or {}).get(key))
        if left and right:
            counts["both"] += 1
        elif left:
            counts["native_only"] += 1
        elif right:
            counts["step120_only"] += 1
        else:
            counts["neither"] += 1
    return {
        **dict(sorted(counts.items())),
        "exact_two_sided_p": exact_sign_pvalue(
            counts["native_only"], counts["step120_only"]
        ),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_tools(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "rows",
            "tool_calls",
            "tool_call_name_counts",
            "bash_calls",
            "bash_command_family_counts",
            "rows_with_recognized_readonly_sqlite",
            "rows_with_unobserved_tool_calls",
            "observed_tool_calls",
            "unobserved_tool_calls",
            "duplicate_bash_calls",
        )
    }


def build_summary(
    *,
    native_reward_rows: list[dict[str, Any]],
    step120_reward_rows: list[dict[str, Any]],
    native_tools: dict[str, Any],
    step120_tools: dict[str, Any],
    native_conversion: dict[str, Any],
    step120_conversion: dict[str, Any],
    native_first_query: dict[str, Any],
    step120_first_query: dict[str, Any],
    first_query_comparison: dict[str, Any],
    boss_comparison: dict[str, Any],
) -> dict[str, Any]:
    native = index_unique(native_reward_rows, "native")
    step120 = index_unique(step120_reward_rows, "step120")
    if set(native) != set(step120) or len(native) != 64:
        raise ValueError("expected identical 64-task reward outputs")
    if boss_comparison.get("prompt_identity", {}).get("identical_prompt_count") != 64:
        raise ValueError("native and Step 120 prompts are not identical")
    if first_query_comparison.get("rows") != 64:
        raise ValueError("first-query paired comparison is not 64 rows")

    reward_wins = boss_comparison["paired_reward"]
    query_presence = first_query_comparison["observed_first_query_presence"]
    native_summary = boss_comparison["native"]
    step_summary = boss_comparison["step120"]
    return {
        "contract": CONTRACT,
        "date": "2026-08-13",
        "scope": {
            "tasks": 64,
            "same_tasks_and_prompts": True,
            "greedy_n1": True,
            "context_tokens": 49152,
            "max_assistant_turns": 26,
            "max_tool_result_turns": 25,
            "native_weight_identity": "original_hf_base_weights_with_initial_sync_skipped",
            "step120_weight_identity": "dist_checkpoint_forced_actor_to_rollout_sync",
            "boss_original_reward_judge": True,
            "contains_original_prompts_sql_answers_task_ids_tool_outputs_or_server_paths": False,
        },
        "native": {
            "conversion": {
                key: native_conversion[key]
                for key in (
                    "validation_rows",
                    "tool_calls",
                    "missing_tool_responses",
                    "terminal_assistant_answers",
                    "truncated_terminal_tool_calls",
                )
            },
            "tools": compact_tools(native_tools),
            "first_query": {
                "outcome_counts": native_first_query["outcome_counts"],
                "first_error_category_counts": native_first_query[
                    "first_error_category_counts"
                ],
            },
            "boss": native_summary,
        },
        "step120": {
            "conversion": {
                key: step120_conversion[key]
                for key in (
                    "validation_rows",
                    "tool_calls",
                    "missing_tool_responses",
                    "terminal_assistant_answers",
                    "truncated_terminal_tool_calls",
                )
            },
            "tools": compact_tools(step120_tools),
            "first_query": {
                "outcome_counts": step120_first_query["outcome_counts"],
                "first_error_category_counts": step120_first_query[
                    "first_error_category_counts"
                ],
            },
            "boss": step_summary,
        },
        "paired": {
            "first_query_transitions": first_query_comparison[
                "outcome_transition_counts"
            ],
            "observed_first_query_presence": {
                **query_presence,
                "exact_two_sided_p": exact_sign_pvalue(
                    int(query_presence.get("native_only_observed") or 0),
                    int(query_presence.get("step120_only_observed") or 0),
                ),
            },
            "complete": paired_boolean(native, step120, "result_complete"),
            "has_answer": paired_boolean(native, step120, "result_has_answer"),
            "correct_numeric": paired_boolean(
                native, step120, "result_correct_numeric"
            ),
            "reward": {
                **reward_wins,
                "exact_two_sided_p": exact_sign_pvalue(
                    int(reward_wins["losses"]), int(reward_wins["wins"])
                ),
            },
        },
        "attribution": {
            "proxy_aligned_wrong_process_ok_preexists_in_native": True,
            "created_by_step120_training": False,
            "amplified_by_step120_training": False,
            "native_wrong_process_ok": int(
                native_summary["verdict_fine_counts"]["result_wrong_process_ok"]
            ),
            "step120_wrong_process_ok": int(
                step_summary["verdict_fine_counts"]["result_wrong_process_ok"]
            ),
            "directional_step120_query_start_coverage_regression": True,
            "query_start_regression_statistically_conclusive_at_0_05": False,
            "tradeoff": "more_correct_answers_but_less_query_coverage_completion_and_total_reward",
        },
        "decision": {
            "training_allowed": False,
            "promotion_allowed": False,
            "pair_gate_observed": 22,
            "pair_gate_required": 48,
            "next_training_target": "native_anchored_query_initiation_and_completion_repair_with_correctness_preservation",
            "future_pareto_gate": {
                "recognized_sqlite_task_floor": int(
                    native_tools["rows_with_recognized_readonly_sqlite"]
                ),
                "complete_count_floor": int(native_summary["complete_count"]),
                "correct_numeric_count_floor": int(
                    step_summary["correct_numeric_count"]
                ),
                "reward_total_mean_floor": float(native_summary["reward_total_mean"]),
                "wrong_process_ok_ceiling": int(
                    step_summary["verdict_fine_counts"]["result_wrong_process_ok"]
                ),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-reward", type=Path, required=True)
    parser.add_argument("--step120-reward", type=Path, required=True)
    parser.add_argument("--native-tools", type=Path, required=True)
    parser.add_argument("--step120-tools", type=Path, required=True)
    parser.add_argument("--native-conversion", type=Path, required=True)
    parser.add_argument("--step120-conversion", type=Path, required=True)
    parser.add_argument("--native-first-query", type=Path, required=True)
    parser.add_argument("--step120-first-query", type=Path, required=True)
    parser.add_argument("--first-query-comparison", type=Path, required=True)
    parser.add_argument("--boss-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_summary(
        native_reward_rows=read_jsonl(args.native_reward),
        step120_reward_rows=read_jsonl(args.step120_reward),
        native_tools=load_json(args.native_tools),
        step120_tools=load_json(args.step120_tools),
        native_conversion=load_json(args.native_conversion),
        step120_conversion=load_json(args.step120_conversion),
        native_first_query=load_json(args.native_first_query),
        step120_first_query=load_json(args.step120_first_query),
        first_query_comparison=load_json(args.first_query_comparison),
        boss_comparison=load_json(args.boss_comparison),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
