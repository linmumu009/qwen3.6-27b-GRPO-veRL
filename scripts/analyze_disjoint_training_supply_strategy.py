#!/usr/bin/env python3
"""Rebase disjoint training-pair capacity after freezing eval22."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.prepare_repair_sft_dataset import sha256_file


CONTRACT = "disjoint-training-supply-strategy-v3"
NATIVE_SUPPLY_CONTRACT = "native-disjoint-real-state-supply-audit-v1"
NATIVE_MARGIN_CONTRACT = "native-disjoint-real-state-step120-margin-safe-summary-v1"
EVAL_SUMMARY_CONTRACT = "disjoint-real-state-eval22-chosen-candidate-safe-summary-v2"


def beta_binomial_tail(
    trials: int,
    minimum_successes: int,
    *,
    alpha: float,
    beta: float,
) -> float:
    if not 0 <= minimum_successes <= trials:
        return 0.0 if minimum_successes > trials else 1.0
    log_beta_prior = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    total = 0.0
    for successes in range(minimum_successes, trials + 1):
        log_probability = (
            math.lgamma(trials + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(trials - successes + 1)
            + math.lgamma(successes + alpha)
            + math.lgamma(trials - successes + beta)
            - math.lgamma(trials + alpha + beta)
            - log_beta_prior
        )
        total += math.exp(log_probability)
    return min(1.0, max(0.0, total))


def minimum_trials_for_probability(
    *,
    minimum_successes: int,
    alpha: float,
    beta: float,
    target_probability: float,
    maximum_trials: int = 1000,
) -> int:
    for trials in range(minimum_successes, maximum_trials + 1):
        if (
            beta_binomial_tail(
                trials, minimum_successes, alpha=alpha, beta=beta
            )
            >= target_probability
        ):
            return trials
    raise ValueError("probability target not reached within maximum_trials")


def build_strategy(
    *,
    native_supply: dict[str, Any],
    native_margin: dict[str, Any],
    eval_summary: dict[str, Any],
    review_queue_tasks: int = 138,
    low_risk_review_tasks: int = 42,
    observed_pair_successes: int = 22,
    observed_candidate_tasks: int = 64,
    required_training_pairs: int = 48,
) -> dict[str, Any]:
    if native_supply.get("contract") != NATIVE_SUPPLY_CONTRACT:
        raise ValueError("native supply contract mismatch")
    if native_supply.get("training_allowed") is not False:
        raise ValueError("native supply audit is not fail-closed")
    if native_margin.get("contract") != NATIVE_MARGIN_CONTRACT:
        raise ValueError("native margin contract mismatch")
    if native_margin.get("training_allowed") is not False:
        raise ValueError("native margin screen is not fail-closed")
    if eval_summary.get("contract") != EVAL_SUMMARY_CONTRACT:
        raise ValueError("eval22 summary contract mismatch")
    if eval_summary.get("training_allowed") is not False:
        raise ValueError("eval22 training prohibition is missing")

    eval_pairs = int(eval_summary["data_gate"]["unique_source_tasks"])
    native_states = int(native_supply["native_error_states_outside_all_frozen_sets"])
    if int(native_margin["scope"]["pairs"]) != native_states:
        raise ValueError("native supply and margin pair counts differ")
    if native_margin["decision"]["retain_as_candidate_training_source_stratum"] is not True:
        raise ValueError("native margin screen did not retain the source stratum")
    remaining_if_all_native_pass = max(0, required_training_pairs - native_states)
    alpha = observed_pair_successes + 0.5
    beta = observed_candidate_tasks - observed_pair_successes + 0.5
    queue_probability_without_native = beta_binomial_tail(
        review_queue_tasks, required_training_pairs, alpha=alpha, beta=beta
    )
    queue_probability_with_native = beta_binomial_tail(
        review_queue_tasks, remaining_if_all_native_pass, alpha=alpha, beta=beta
    )
    approved_for_90 = minimum_trials_for_probability(
        minimum_successes=remaining_if_all_native_pass,
        alpha=alpha,
        beta=beta,
        target_probability=0.90,
    )
    approved_for_95 = minimum_trials_for_probability(
        minimum_successes=remaining_if_all_native_pass,
        alpha=alpha,
        beta=beta,
        target_probability=0.95,
    )
    low_risk_probability = beta_binomial_tail(
        low_risk_review_tasks,
        remaining_if_all_native_pass,
        alpha=alpha,
        beta=beta,
    )

    return {
        "contract": CONTRACT,
        "date": "2026-08-13",
        "objective": "maximize_expected_accuracy_learning_per_unit_of_cpu_and_npu_time",
        "corrected_capacity_logic": {
            "required_training_pairs": required_training_pairs,
            "eval22_pairs": eval_pairs,
            "eval22_pairs_reusable_for_training": 0,
            "previous_reuse_assumption_invalid": True,
            "observed_pair_yield": observed_pair_successes / observed_candidate_tasks,
            "yield_observations": {
                "pairs": observed_pair_successes,
                "candidate_tasks": observed_candidate_tasks,
            },
            "uncertainty_model": "Jeffreys-posterior beta-binomial predictive",
            "review_queue_tasks": review_queue_tasks,
            "probability_queue_alone_reaches_48_if_every_task_is_approved": queue_probability_without_native,
        },
        "existing_zero_npu_opportunity": {
            "native_observed_error_states": int(
                native_supply["native_observed_first_query_errors"]
            ),
            "overlap_with_eval22_excluded": int(
                native_supply["native_error_overlap_with_eval22"]
            ),
            "native_error_states_outside_eval22": int(
                native_supply["native_error_states_outside_eval22"]
            ),
            "native_error_states_outside_all_frozen_sets": native_states,
            "additional_frozen_overlap_outside_eval22": int(
                native_supply["additional_frozen_overlap_outside_eval22"]
            ),
            "error_categories": native_supply[
                "outside_all_frozen_first_error_category_counts"
            ],
            "currently_training_ready_pairs": 0,
            "retained_pair_candidates_after_step120_margin": native_states,
            "remaining_pairs_if_all_native_states_pass_pair_audit": remaining_if_all_native_pass,
        },
        "rebased_capacity": {
            "probability_review_queue_supplies_remaining_pairs_if_all_review_tasks_are_approved": queue_probability_with_native,
            "approved_candidate_tasks_for_90pct_predictive_probability": approved_for_90,
            "approved_candidate_tasks_for_95pct_predictive_probability": approved_for_95,
            "shortfall_vs_review138_at_90pct_even_if_all_are_approved": max(
                0, approved_for_90 - review_queue_tasks
            ),
            "shortfall_vs_review138_at_95pct_even_if_all_are_approved": max(
                0, approved_for_95 - review_queue_tasks
            ),
            "low_risk_review_subset_tasks": low_risk_review_tasks,
            "low_risk_expected_pairs_at_observed_yield": (
                low_risk_review_tasks * observed_pair_successes / observed_candidate_tasks
            ),
            "probability_low_risk_subset_supplies_remaining_pairs": low_risk_probability,
            "low_risk_subset_can_reliably_fill_remaining_pairs_by_itself": False,
        },
        "decision": {
            "highest_value_next_action": "adjudicate_42_lowest_risk_review_required_tasks_as_supply_pilot",
            "native_pair_build_and_step120_margin_completed": True,
            "retained_native_pairs": native_states,
            "remaining_pair_gap": remaining_if_all_native_pass,
            "training_now": False,
            "full25_now": False,
            "promotion_allowed": False,
            "why": "native_pairs_passed_all_mechanical_token_and_margin_screens_but_the_remaining_gap_requires_measuring_semantic_approval_before_more_rollout_spend",
        },
        "next_action_contract": [
            {
                "order": 1,
                "action": "mechanically build native-sourced pairs outside all frozen sets",
                "resource": "CPU only",
                "status": "completed",
                "gate": "actual observed first error, verified chosen, identical prefix through real tool result, zero eval22/chosen-calibration16/frozen16/val20/test20 overlap",
                "stop": "any identity, semantic, tool-result, or pair-integrity failure",
            },
            {
                "order": 2,
                "action": "run Step 120 forward-only margin on retained native pairs",
                "resource": "NPU forward only",
                "status": "completed",
                "gate": "nonempty delta masks and systematic correct-vs-actual-error misranking",
                "stop": "no optimizer, no checkpoint, no training authorization",
            },
            {
                "order": 3,
                "action": "adjudicate the 42 lowest-risk review-required tasks as a supply pilot",
                "resource": "CPU only",
                "status": "next",
                "gate": "measure semantic approval rate before committing to all 138",
                "stop": "do not mistake 42 tasks for capacity to supply the remaining pair target",
            },
            {
                "order": 4,
                "action": "expand approved candidate capacity and collect Step 120 states in 32-task batches",
                "resource": "CPU then NPU rollout",
                "status": "blocked_on_review_pilot",
                "gate": "capacity target rebased from retained native pairs and measured approval/yield",
                "stop": "stop acquisition at 48 distinct training pairs; keep eval22 and chosen-calibration16 frozen",
            },
        ],
        "rejected_immediate_actions": [
            "train_on_eval22",
            "continue_the_rejected_chosen_only_checkpoint",
            "review_all_138_before_measuring_the_42_task_approval_rate",
            "run_full64_or_additional_optimizer_steps_now",
            "lower_the_48_pair_gate",
        ],
        "caveats": [
            "the retained native states passed pair, token, and Step120 margin screens but remain a separately labeled off-policy source stratum",
            "the beta-binomial calculation assumes future approved tasks have exchangeable pair yield with the observed strict64; review-required tasks may yield less",
            "semantic approval is below 100 percent by construction, so queue-only probabilities are optimistic upper bounds",
            "native-sourced states are off-policy relative to Step 120 and must be labeled and evaluated as a separate source stratum",
        ],
        "contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-supply", type=Path, required=True)
    parser.add_argument("--native-margin", type=Path, required=True)
    parser.add_argument("--eval22-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native_supply = json.loads(args.native_supply.read_text(encoding="utf-8"))
    native_margin = json.loads(args.native_margin.read_text(encoding="utf-8"))
    eval_summary = json.loads(args.eval22_summary.read_text(encoding="utf-8"))
    result = build_strategy(
        native_supply=native_supply,
        native_margin=native_margin,
        eval_summary=eval_summary,
    )
    result["source_sha256"] = {
        "native_supply": sha256_file(args.native_supply),
        "native_margin": sha256_file(args.native_margin),
        "eval22_summary": sha256_file(args.eval22_summary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
