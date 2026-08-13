#!/usr/bin/env python3
"""Rebase disjoint training supply after the 42-task semantic review pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from scripts.analyze_disjoint_real_state_evaluation import wilson_interval
from scripts.prepare_repair_sft_dataset import sha256_file


CONTRACT = "disjoint-pair-semantic-review-supply-rebase-v1"
DECISION_CONTRACT = "disjoint-pair-semantic-review-decisions-v1"
PILOT_CONTRACT = "disjoint-pair-review-pilot42-safe-summary-v1"
STRATEGY_CONTRACT = "disjoint-training-supply-strategy-v3"


def beta_binomial_pmf(
    trials: int,
    successes: int,
    *,
    alpha: float,
    beta: float,
) -> float:
    if trials < 0 or not 0 <= successes <= trials or alpha <= 0 or beta <= 0:
        return 0.0
    log_probability = (
        math.lgamma(trials + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(trials - successes + 1)
        + math.lgamma(successes + alpha)
        + math.lgamma(trials - successes + beta)
        - math.lgamma(trials + alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + math.lgamma(alpha + beta)
    )
    return math.exp(log_probability)


def beta_binomial_distribution(
    trials: int,
    *,
    alpha: float,
    beta: float,
) -> list[float]:
    values = [
        beta_binomial_pmf(trials, successes, alpha=alpha, beta=beta)
        for successes in range(trials + 1)
    ]
    total = sum(values)
    if total <= 0:
        raise ValueError("invalid beta-binomial distribution")
    return [value / total for value in values]


def discrete_quantile(probabilities: list[float], probability: float) -> int:
    if not 0 <= probability <= 1:
        raise ValueError("quantile probability must be in [0, 1]")
    cumulative = 0.0
    for index, value in enumerate(probabilities):
        cumulative += value
        if cumulative >= probability:
            return index
    return len(probabilities) - 1


def compound_pair_distribution(
    tasks: int,
    *,
    approval_alpha: float,
    approval_beta: float,
    pair_alpha: float,
    pair_beta: float,
) -> list[float]:
    approvals = beta_binomial_distribution(
        tasks, alpha=approval_alpha, beta=approval_beta
    )
    pairs = [0.0] * (tasks + 1)
    for approved, approval_probability in enumerate(approvals):
        conditional = beta_binomial_distribution(
            approved, alpha=pair_alpha, beta=pair_beta
        )
        for pair_count, pair_probability in enumerate(conditional):
            pairs[pair_count] += approval_probability * pair_probability
    total = sum(pairs)
    return [value / total for value in pairs]


def _validate_decisions(
    decisions: dict[str, Any], expected_tasks: int
) -> tuple[Counter[str], Counter[str], Counter[str], Counter[str], dict[str, int]]:
    if decisions.get("contract") != DECISION_CONTRACT:
        raise ValueError("semantic decision contract mismatch")
    if decisions.get("training_allowed") is not False:
        raise ValueError("semantic decisions are not fail closed")
    rows = list(decisions.get("decisions") or [])
    if len(rows) != expected_tasks:
        raise ValueError("semantic decision count differs from pilot")
    indices = [int(row.get("review_index", -1)) for row in rows]
    if sorted(indices) != list(range(expected_tasks)) or len(set(indices)) != expected_tasks:
        raise ValueError("semantic review indices are incomplete or duplicated")

    verdicts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    severity: Counter[str] = Counter()
    evidence = {
        "instruction_unambiguously_entails_gold": 0,
        "verification_sql_fully_answers_instruction": 0,
        "expected_value_supported_by_query_result": 0,
    }
    for row in rows:
        verdict = str(row.get("decision") or "")
        entails = row.get("instruction_unambiguously_entails_gold") is True
        answers = row.get("verification_sql_fully_answers_instruction") is True
        supported = row.get("expected_value_supported_by_query_result") is True
        if verdict == "approved" and not (entails and answers and supported):
            raise ValueError("approved decision lacks all semantic evidence gates")
        if verdict == "rejected" and entails and answers and supported:
            raise ValueError("rejected decision has no failed semantic evidence gate")
        if verdict not in {"approved", "rejected"}:
            raise ValueError(f"unsupported semantic decision: {verdict!r}")
        reason = str(row.get("reason_code") or "")
        confidence_value = str(row.get("confidence") or "")
        severity_value = str(row.get("severity") or "")
        if not reason or confidence_value not in {"medium", "high"}:
            raise ValueError("semantic review metadata is incomplete")
        if severity_value != "high":
            raise ValueError("semantic mismatch is not marked high severity")
        verdicts[verdict] += 1
        reasons[reason] += 1
        confidence[confidence_value] += 1
        severity[severity_value] += 1
        evidence["instruction_unambiguously_entails_gold"] += entails
        evidence["verification_sql_fully_answers_instruction"] += answers
        evidence["expected_value_supported_by_query_result"] += supported
    return verdicts, reasons, confidence, severity, evidence


def build_rebase(
    *,
    decisions: dict[str, Any],
    pilot: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    if pilot.get("contract") != PILOT_CONTRACT:
        raise ValueError("pilot safe-summary contract mismatch")
    if strategy.get("contract") != STRATEGY_CONTRACT:
        raise ValueError("training supply strategy contract mismatch")
    for value, label in ((pilot, "pilot"), (strategy, "strategy")):
        if value.get("training_allowed") is not False:
            raise ValueError(f"{label} is not fail closed")

    pool_tasks = int(pilot["source_pool"]["review_required_tasks"])
    reviewed_tasks = int(pilot["source_pool"]["selected_tasks"])
    stable_tasks = int(pilot["query_stability_gate"]["stable"])
    if stable_tasks != reviewed_tasks:
        raise ValueError("not every reviewed task passed the query-stability gate")
    if pilot["source_pool"]["selection_role"] != (
        "lowest_mechanical_risk_semantic_approval_rate_pilot"
    ):
        raise ValueError("pilot is not the expected lowest-risk stratum")

    verdicts, reasons, confidence, severity, evidence = _validate_decisions(
        decisions, reviewed_tasks
    )
    approved = verdicts["approved"]
    rejected = verdicts["rejected"]
    remaining_tasks = pool_tasks - reviewed_tasks
    if remaining_tasks < 0:
        raise ValueError("reviewed tasks exceed source pool")

    # Jeffreys posterior after the semantic pilot.
    approval_alpha = approved + 0.5
    approval_beta = rejected + 0.5
    approval_predictive = beta_binomial_distribution(
        remaining_tasks, alpha=approval_alpha, beta=approval_beta
    )

    yield_observations = strategy["corrected_capacity_logic"]["yield_observations"]
    observed_pairs = int(yield_observations["pairs"])
    observed_candidates = int(yield_observations["candidate_tasks"])
    pair_alpha = observed_pairs + 0.5
    pair_beta = observed_candidates - observed_pairs + 0.5
    pair_predictive = compound_pair_distribution(
        remaining_tasks,
        approval_alpha=approval_alpha,
        approval_beta=approval_beta,
        pair_alpha=pair_alpha,
        pair_beta=pair_beta,
    )

    retained_native_pairs = int(strategy["decision"]["retained_native_pairs"])
    remaining_pair_gap = int(strategy["decision"]["remaining_pair_gap"])
    if retained_native_pairs + remaining_pair_gap != int(
        strategy["corrected_capacity_logic"]["required_training_pairs"]
    ):
        raise ValueError("retained supply and remaining gap do not match the training gate")

    expected_approvals = remaining_tasks * approval_alpha / (
        approval_alpha + approval_beta
    )
    expected_pairs = expected_approvals * pair_alpha / (pair_alpha + pair_beta)
    return {
        "contract": CONTRACT,
        "date": "2026-08-13",
        "question": "should_the_remaining_review96_be_adjudicated_or_should_supply_be_rebuilt",
        "semantic_review": {
            "reviewed_tasks": reviewed_tasks,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approved / reviewed_tasks,
            "approval_rate_wilson95": wilson_interval(approved, reviewed_tasks),
            "decision_counts": dict(sorted(verdicts.items())),
            "reason_counts": dict(sorted(reasons.items())),
            "confidence_counts": dict(sorted(confidence.items())),
            "severity_counts": dict(sorted(severity.items())),
            **evidence,
            "all_reviewed_tasks_mechanically_valid_and_query_stable": True,
            "finding": "mechanically_supported_but_semantically_misaligned_labels",
        },
        "remaining_queue_exchangeability_stress_test": {
            "remaining_tasks": remaining_tasks,
            "approval_model": {
                "name": "Jeffreys-posterior beta-binomial predictive",
                "posterior_alpha": approval_alpha,
                "posterior_beta": approval_beta,
                "expected_approved_tasks": expected_approvals,
                "median_approved_tasks": discrete_quantile(approval_predictive, 0.5),
                "p90_upper_approved_tasks": discrete_quantile(approval_predictive, 0.9),
                "p95_upper_approved_tasks": discrete_quantile(approval_predictive, 0.95),
                "probability_at_least_one_approved_task": 1.0 - approval_predictive[0],
            },
            "conditional_pair_yield_model": {
                "observed_pairs": observed_pairs,
                "observed_candidate_tasks": observed_candidates,
                "posterior_alpha": pair_alpha,
                "posterior_beta": pair_beta,
            },
            "expected_additional_pairs": expected_pairs,
            "median_additional_pairs": discrete_quantile(pair_predictive, 0.5),
            "p90_upper_additional_pairs": discrete_quantile(pair_predictive, 0.9),
            "p95_upper_additional_pairs": discrete_quantile(pair_predictive, 0.95),
            "probability_at_least_one_additional_pair": 1.0 - pair_predictive[0],
            "probability_remaining_queue_fills_pair_gap": sum(
                pair_predictive[remaining_pair_gap:]
            ),
            "interpretation": "optimistic_exchangeability_stress_test_not_a_strict_upper_bound",
        },
        "training_capacity": {
            "required_pairs": int(
                strategy["corrected_capacity_logic"]["required_training_pairs"]
            ),
            "retained_native_pair_candidates": retained_native_pairs,
            "remaining_pair_gap": remaining_pair_gap,
            "approved_candidate_tasks_for_90pct_pair_supply_probability": int(
                strategy["rebased_capacity"][
                    "approved_candidate_tasks_for_90pct_predictive_probability"
                ]
            ),
            "approved_candidate_tasks_for_95pct_pair_supply_probability": int(
                strategy["rebased_capacity"][
                    "approved_candidate_tasks_for_95pct_predictive_probability"
                ]
            ),
        },
        "decision": {
            "stop_reviewing_remaining96_from_same_queue_now": approved == 0,
            "do_not_spend_npu_on_remaining_queue": approved == 0,
            "training_now": False,
            "promotion_allowed": False,
            "highest_value_next_action": "author_fresh_unambiguous_current_definition_tasks_with_explicit_metric_grouping_and_time_scope",
            "fresh_authoring_pilot_tasks": 32,
            "fresh_authoring_pilot_gate": "every_task_must_pass_instruction_gold_sql_semantic_review_before_rollout",
            "scale_target_after_pilot": "172_semantically_approved_candidate_tasks_for_95pct_pair_supply_probability",
            "acquisition_batch_size": 32,
            "acquisition_stop": "stop_when_41_additional_distinct_pairs_pass_all_gates",
        },
        "limitations": [
            "the 42 tasks were deliberately selected as the lowest mechanical-risk stratum, so treating the remaining 96 as exchangeable is optimistic and not a population estimate",
            "the pair-yield posterior comes from strict64 rather than the rejected review-required queue",
            "the seven retained native pairs are a separately labeled off-policy candidate stratum",
            "no rejected task may be used for rollout, training, evaluation, or model promotion",
        ],
        "contains_task_ids_prompts_sql_gold_values_tool_outputs_or_server_paths": False,
        "may_be_used_as_training_or_rollout_data": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--strategy-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_rebase(
        decisions=json.loads(args.decisions.read_text(encoding="utf-8")),
        pilot=json.loads(args.pilot_summary.read_text(encoding="utf-8")),
        strategy=json.loads(args.strategy_summary.read_text(encoding="utf-8")),
    )
    result["source_sha256"] = {
        "decisions": sha256_file(args.decisions),
        "pilot_summary": sha256_file(args.pilot_summary),
        "strategy_summary": sha256_file(args.strategy_summary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
