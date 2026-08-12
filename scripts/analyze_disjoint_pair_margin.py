#!/usr/bin/env python3
"""Analyze Step 120 correct-vs-actual-wrong margins on disjoint pairs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any

from scripts.analyze_state_recovery_semantics import critical_token_family


def _index_diagnostic(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in result.get("per_task") or []:
        current_task_id = str(row.get("task_id") or "")
        if not current_task_id or current_task_id in indexed:
            raise ValueError(f"missing or duplicate diagnostic task ID: {current_task_id!r}")
        indexed[current_task_id] = row
    return indexed


def analyze(diagnostic: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    if diagnostic.get("contract") != "repair-sft-teacher-forced-component-diagnostic-v3":
        raise ValueError("disjoint margin requires teacher-forced diagnostic v3")
    if diagnostic.get("forward_only") is not True or diagnostic.get("optimizer_initialized") is not False:
        raise ValueError("disjoint margin requires forward-only execution without optimizer")
    if contract.get("contract") != "current-definition-disjoint-first-error-pairs-v1":
        raise ValueError("unexpected disjoint first-error pair contract")
    pairs = int(contract.get("pairs") or 0)
    minimum_pairs = int(contract.get("minimum_pairs") or 0)
    if contract.get("pair_count_gate_passed") is not True or pairs < minimum_pairs:
        raise ValueError("disjoint pair count gate did not pass")
    if diagnostic.get("data_sha256") != contract.get("output_sha256"):
        raise ValueError("disjoint margin diagnostic and dataset hashes differ")
    if "semantic_delta" not in diagnostic.get("components", {}):
        raise ValueError("diagnostic is missing semantic-delta metrics")
    evidence = {str(row["task_id"]): row for row in contract.get("evidence") or []}
    if len(evidence) != pairs:
        raise ValueError("disjoint pair evidence count differs from contract")
    expected = {
        f"{task_id}::{label}" for task_id in evidence for label in ("chosen", "rejected")
    }
    rows = _index_diagnostic(diagnostic)
    if set(rows) != expected:
        raise ValueError("diagnostic does not contain the complete disjoint pair set")

    per_task: list[dict[str, Any]] = []
    family_margins: dict[str, list[float]] = defaultdict(list)
    family_preferred: Counter[str] = Counter()
    for task_id in sorted(evidence):
        chosen = rows[f"{task_id}::chosen"]
        rejected = rows[f"{task_id}::rejected"]
        chosen_delta_nll = float(chosen["components"]["semantic_delta"]["mean_nll"])
        rejected_delta_nll = float(rejected["components"]["semantic_delta"]["mean_nll"])
        chosen_sql_nll = float(chosen["components"]["sql_shell"]["mean_nll"])
        rejected_sql_nll = float(rejected["components"]["sql_shell"]["mean_nll"])
        margin = rejected_delta_nll - chosen_delta_nll
        rank = chosen["sql_token_rank"]
        if rank.get("first_nongreedy_offset") is None:
            family = "all_chosen_sql_tokens_greedy"
        else:
            family = critical_token_family(rank.get("first_nongreedy_target_token"))
        family_margins[family].append(margin)
        family_preferred[family] += int(margin > 0)
        per_task.append(
            {
                "task_id": task_id,
                "critical_token_family": family,
                "first_nongreedy_offset": rank.get("first_nongreedy_offset"),
                "first_nongreedy_target_id": rank.get("first_nongreedy_target_id"),
                "first_nongreedy_rank": rank.get("first_nongreedy_rank"),
                "semantic_delta_log_probability_margin_per_token": margin,
                "full_sql_log_probability_margin_per_token": rejected_sql_nll - chosen_sql_nll,
                "chosen_preferred_on_semantic_delta": margin > 0,
            }
        )

    margins = [float(row["semantic_delta_log_probability_margin_per_token"]) for row in per_task]
    chosen_preferred = sum(bool(row["chosen_preferred_on_semantic_delta"]) for row in per_task)
    threshold = math.ceil(0.75 * pairs)
    training_allowed = chosen_preferred < threshold
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
        "contract": "current-definition-disjoint-pair-margin-result-v1",
        "source_checkpoint": "step120",
        "task_count": pairs,
        "semantic_delta_margin": {
            "definition": "rejected_mean_nll_minus_chosen_mean_nll_positive_prefers_chosen",
            "chosen_preferred": chosen_preferred,
            "preference_threshold_75pct": threshold,
            "mean_margin": fmean(margins),
            "median_margin": median(margins),
            "by_critical_token_family": by_family,
        },
        "decision": {
            "one_step_training_allowed": training_allowed,
            "selected_next_action": (
                "one_step_disjoint_pairwise_canary_then_frozen16_gate"
                if training_allowed
                else "do_not_pairwise_train_correct_delta_already_preferred"
            ),
            "reason": (
                "correct_semantic_delta_is_not_preferred_on_at_least_one_quarter_of_pairs"
                if training_allowed
                else "correct_semantic_delta_is_already_preferred_on_at_least_three_quarters_of_pairs"
            ),
        },
        "frozen_post_training_gate": {
            "evaluation_set": "original_frozen16_never_used_for_this_training_batch",
            "semantic_delta_chosen_preferred_min": 12,
            "per_task_margin_improved_min": 12,
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
        "training_allowed": training_allowed,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        json.loads(args.diagnostic.read_text(encoding="utf-8")),
        json.loads(args.dataset_contract.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
