#!/usr/bin/env python3
"""Compare chosen-vs-actual-wrong SQL likelihood on semantic edit tokens."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any


def _index_diagnostic(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in result.get("per_task") or []:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in indexed:
            raise ValueError(f"missing or duplicate diagnostic task ID: {task_id!r}")
        indexed[task_id] = row
    return indexed


def analyze(
    diagnostic: dict[str, Any],
    contract: dict[str, Any],
    *,
    preference_threshold: int = 12,
    source_checkpoint: str = "step120",
    attribution_only: bool = False,
) -> dict[str, Any]:
    if diagnostic.get("contract") != "repair-sft-teacher-forced-component-diagnostic-v3":
        raise ValueError("semantic-delta margin requires teacher-forced diagnostic v3")
    if diagnostic.get("forward_only") is not True or diagnostic.get("optimizer_initialized") is not False:
        raise ValueError("semantic-delta margin requires forward-only execution without optimizer")
    if contract.get("contract") != "semantic-delta-margin-gate-dataset-v1":
        raise ValueError("unexpected semantic-delta dataset contract")
    if diagnostic.get("data_sha256") != contract.get("output_sha256"):
        raise ValueError("semantic-delta diagnostic and dataset hashes differ")
    if "semantic_delta" not in diagnostic.get("components", {}):
        raise ValueError("diagnostic is missing semantic-delta metrics")
    rows = _index_diagnostic(diagnostic)
    evidence = {str(row["task_id"]): row for row in contract.get("evidence") or []}
    expected = {
        f"{task_id}::{label}"
        for task_id in evidence
        for label in ("chosen", "rejected")
    }
    if len(evidence) != 16 or set(rows) != expected:
        raise ValueError("semantic-delta diagnostic does not contain 16 complete pairs")

    per_task: list[dict[str, Any]] = []
    family_margins: dict[str, list[float]] = defaultdict(list)
    family_preferred: Counter[str] = Counter()
    baseline_consistent = 0
    for task_id, frozen in sorted(evidence.items()):
        chosen = rows[f"{task_id}::chosen"]
        rejected = rows[f"{task_id}::rejected"]
        chosen_delta_nll = float(chosen["components"]["semantic_delta"]["mean_nll"])
        rejected_delta_nll = float(rejected["components"]["semantic_delta"]["mean_nll"])
        chosen_sql_nll = float(chosen["components"]["sql_shell"]["mean_nll"])
        rejected_sql_nll = float(rejected["components"]["sql_shell"]["mean_nll"])
        delta_margin = rejected_delta_nll - chosen_delta_nll
        sql_margin = rejected_sql_nll - chosen_sql_nll
        rank = chosen["sql_token_rank"]
        exact_reconstruction = (
            rank.get("first_nongreedy_offset") == int(frozen["critical_sql_token_offset"])
            and rank.get("first_nongreedy_target_id") == int(frozen["critical_sql_target_id"])
        )
        actual_offset = rank.get("first_nongreedy_offset")
        earlier_regression = actual_offset is not None and int(actual_offset) < int(
            frozen["critical_sql_token_offset"]
        )
        invalid_target_at_frozen_offset = (
            actual_offset == int(frozen["critical_sql_token_offset"])
            and rank.get("first_nongreedy_target_id")
            != int(frozen["critical_sql_target_id"])
        )
        baseline_consistent += int(exact_reconstruction)
        family = str(frozen["critical_token_family"])
        family_margins[family].append(delta_margin)
        family_preferred[family] += int(delta_margin > 0)
        per_task.append(
            {
                "task_id": task_id,
                "critical_token_family": family,
                "semantic_delta_log_probability_margin_per_token": delta_margin,
                "full_sql_log_probability_margin_per_token": sql_margin,
                "chosen_preferred_on_semantic_delta": delta_margin > 0,
                "frozen_first_nongreedy_token_reconstructed": exact_reconstruction,
                "new_earlier_first_nongreedy_regression": earlier_regression,
                "invalid_target_at_frozen_offset": invalid_target_at_frozen_offset,
            }
        )

    margins = [row["semantic_delta_log_probability_margin_per_token"] for row in per_task]
    chosen_preferred = sum(row["chosen_preferred_on_semantic_delta"] for row in per_task)
    earlier_regressions = sum(
        row["new_earlier_first_nongreedy_regression"] for row in per_task
    )
    invalid_targets = sum(row["invalid_target_at_frozen_offset"] for row in per_task)
    non_regression_passed = earlier_regressions == 0 and invalid_targets == 0
    if attribution_only:
        target = "attribution_probe_only_no_training"
        training_allowed = False
        reason = "cross_model_attribution_does_not_authorize_training"
    elif not non_regression_passed:
        target = "blocked_inconsistent_teacher_forced_reconstruction"
        training_allowed = False
        reason = "frozen_step120_first_nongreedy_token_did_not_reconstruct_for_all_pairs"
    elif chosen_preferred >= preference_threshold:
        target = "constrained_sql_planner_and_bash_only_tool_policy"
        training_allowed = False
        reason = "correct_semantic_delta_is_already_preferred_but_greedy_realization_still_fails"
    else:
        target = "one_step_pairwise_chosen_vs_rejected_plan_to_sql_canary"
        training_allowed = True
        reason = "correct_semantic_delta_is_not_preferred_on_at_least_five_of_sixteen_pairs"

    by_family = {
        family: {
            "tasks": len(values),
            "chosen_preferred": family_preferred[family],
            "mean_margin": fmean(values),
            "median_margin": median(values),
        }
        for family, values in sorted(family_margins.items())
    }
    return {
        "contract": "semantic-delta-margin-gate-result-v2",
        "source_checkpoint": source_checkpoint,
        "model_label": diagnostic.get("model_label"),
        "task_count": 16,
        "semantic_delta_margin": {
            "definition": "rejected_mean_nll_minus_chosen_mean_nll_positive_prefers_chosen",
            "chosen_preferred": chosen_preferred,
            "threshold": preference_threshold,
            "mean_margin": fmean(margins),
            "median_margin": median(margins),
            "by_critical_token_family": by_family,
        },
        "frozen_critical_token_audit": {
            "frozen_first_nongreedy_token_reconstructed": baseline_consistent,
            "new_earlier_first_nongreedy_regressions": earlier_regressions,
            "invalid_targets_at_frozen_offset": invalid_targets,
            "passed": non_regression_passed,
        },
        "decision": {
            "analysis_mode": "attribution_only" if attribution_only else "training_gate",
            "selected_next_action": target,
            "one_step_training_allowed": training_allowed,
            "reason": reason,
        },
        "frozen_post_training_gate": {
            "semantic_delta_chosen_preferred_min": preference_threshold,
            "per_task_margin_improved_min": preference_threshold,
            "new_earlier_first_nongreedy_regressions_max": 0,
            "optimizer_steps": 1,
            "full_replay_before_probability_gate": False,
        },
        "execution": {
            "forward_only": True,
            "optimizer_initialized": False,
            "checkpoint_saved": False,
        },
        "per_task": per_task,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preference-threshold", type=int, default=12)
    parser.add_argument("--source-checkpoint", default="step120")
    parser.add_argument("--attribution-only", action="store_true")
    args = parser.parse_args()
    result = analyze(
        json.loads(args.diagnostic.read_text(encoding="utf-8")),
        json.loads(args.dataset_contract.read_text(encoding="utf-8")),
        preference_threshold=args.preference_threshold,
        source_checkpoint=args.source_checkpoint,
        attribution_only=args.attribution_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
