#!/usr/bin/env python3
"""Compare Step 120 and post-SFT component diagnostics without exposing row content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COMPONENTS = ("assistant", "tool_turn", "tool_structure", "sql_shell", "final_answer")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step120", type=Path, required=True)
    parser.add_argument("--post-sft", type=Path, required=True)
    parser.add_argument("--rollout-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.step120)
    post = load(args.post_sft)
    rollout = load(args.rollout_comparison)
    expected_contract = "repair-sft-teacher-forced-component-diagnostic-v1"
    if baseline.get("contract") != expected_contract or post.get("contract") != expected_contract:
        raise ValueError("unexpected teacher-forced diagnostic contract")
    if baseline["task_ids"] != post["task_ids"]:
        raise ValueError("teacher-forced task IDs differ")
    if baseline["data_sha256"] != post["data_sha256"]:
        raise ValueError("teacher-forced data hashes differ")
    if rollout.get("task_ids_identical") is not True or rollout.get("prompt_identity", {}).get("identical_prompt_count") != 16:
        raise ValueError("free-rollout comparison does not use identical 16 prompts")

    component_comparison: dict[str, Any] = {}
    per_task_by_model = {
        "step120": {row["task_id"]: row for row in baseline["per_task"]},
        "post_sft": {row["task_id"]: row for row in post["per_task"]},
    }
    for component in COMPONENTS:
        before = baseline["components"][component]
        after = post["components"][component]
        nll_delta = after["mean_nll"] - before["mean_nll"]
        probability_ratio = (
            after["geometric_mean_target_probability"]
            / before["geometric_mean_target_probability"]
        )
        wins = losses = ties = 0
        task_deltas: list[float] = []
        for task_id in baseline["task_ids"]:
            task_before = per_task_by_model["step120"][task_id]["components"][component]["mean_nll"]
            task_after = per_task_by_model["post_sft"][task_id]["components"][component]["mean_nll"]
            delta = task_after - task_before
            task_deltas.append(delta)
            if delta < -1e-9:
                wins += 1
            elif delta > 1e-9:
                losses += 1
            else:
                ties += 1
        component_comparison[component] = {
            "token_count": before["token_count"],
            "step120_mean_nll": before["mean_nll"],
            "post_sft_mean_nll": after["mean_nll"],
            "mean_nll_delta": nll_delta,
            "relative_nll_change": nll_delta / before["mean_nll"],
            "step120_geometric_mean_target_probability": before["geometric_mean_target_probability"],
            "post_sft_geometric_mean_target_probability": after["geometric_mean_target_probability"],
            "target_probability_ratio": probability_ratio,
            "per_task_nll": {"improved": wins, "worsened": losses, "tied": ties},
            "per_task_nll_deltas": task_deltas,
        }

    core_components = ("tool_structure", "sql_shell", "final_answer")
    all_teacher_targets_improved = all(
        component_comparison[name]["mean_nll_delta"] < 0 for name in core_components
    )
    rollout_reward_delta = rollout["numeric_deltas"]["reward_total_mean"]["delta"]
    rollout_complete_delta = rollout["numeric_deltas"]["complete_count"]["delta"]
    rollout_worsened = rollout_reward_delta < 0 and rollout_complete_delta < 0
    if all_teacher_targets_improved and rollout_worsened:
        diagnosis = "teacher_forced_targets_improved_but_free_running_degraded"
    elif not all_teacher_targets_improved:
        diagnosis = "one_or_more_teacher_forced_target_components_did_not_improve"
    else:
        diagnosis = "teacher_forced_targets_improved_without_joint_free_running_regression"

    result = {
        "contract": "repair-sft-teacher-forced-prepost-comparison-v1",
        "task_count": 16,
        "task_ids_identical": True,
        "data_sha256_identical": True,
        "forward_only_both": baseline["forward_only"] is True and post["forward_only"] is True,
        "optimizer_initialized_either": baseline["optimizer_initialized"] or post["optimizer_initialized"],
        "official_assistant_loss": {
            "step120": baseline["official_assistant_loss"],
            "post_sft": post["official_assistant_loss"],
            "delta": post["official_assistant_loss"] - baseline["official_assistant_loss"],
        },
        "components": component_comparison,
        "free_rollout_reference": {
            "reward_mean_delta": rollout_reward_delta,
            "complete_count_delta": rollout_complete_delta,
            "correct_count_delta": rollout["numeric_deltas"]["correct_numeric_count"]["delta"],
            "sql_mean_delta": rollout["numeric_deltas"]["n_sql_mean"]["delta"],
            "duplicate_command_mean_delta": rollout["numeric_deltas"]["dup_cmd_mean"]["delta"],
            "paired_reward": rollout["paired_reward"],
        },
        "diagnostic_classification": diagnosis,
        "all_teacher_targets_improved": all_teacher_targets_improved,
        "free_rollout_worsened": rollout_worsened,
        "runtime_seconds": {
            "step120": baseline["runtime_seconds"],
            "post_sft": post["runtime_seconds"],
            "total": baseline["runtime_seconds"] + post["runtime_seconds"],
        },
        "promotion_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "components"}, indent=2))


if __name__ == "__main__":
    main()
