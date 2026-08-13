#!/usr/bin/env python3
"""Analyze the frozen eval22 Step 120 real first-error margin baseline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any

from scripts.analyze_state_recovery_semantics import critical_token_family


DATA_CONTRACT = "current-definition-disjoint-first-error-evaluation-v1"
TOKEN_CONTRACT = "current-definition-disjoint-pair-evaluation-token-gate-v1"


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _index_diagnostic(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in result.get("per_task") or []:
        task = str(row.get("task_id") or "")
        if not task or task in indexed:
            raise ValueError(f"missing or duplicate diagnostic task ID: {task!r}")
        indexed[task] = row
    return indexed


def analyze(
    diagnostic: dict[str, Any], contract: dict[str, Any], token_gate: dict[str, Any]
) -> dict[str, Any]:
    if diagnostic.get("contract") != "repair-sft-teacher-forced-component-diagnostic-v3":
        raise ValueError("eval22 margin requires teacher-forced diagnostic v3")
    if diagnostic.get("forward_only") is not True or diagnostic.get("optimizer_initialized") is not False:
        raise ValueError("eval22 margin requires forward-only execution without optimizer")
    if contract.get("contract") != DATA_CONTRACT:
        raise ValueError("unexpected eval22 data contract")
    pairs = int(contract.get("pairs") or 0)
    expected_pairs = int(contract.get("expected_pairs") or 0)
    if (
        contract.get("pair_evaluation_gate_passed") is not True
        or contract.get("evaluation_only") is not True
        or contract.get("may_be_used_as_training_data") is not False
        or contract.get("training_allowed") is not False
        or contract.get("promotion_allowed") is not False
        or pairs != expected_pairs
        or pairs <= 0
    ):
        raise ValueError("eval22 data contract is not a frozen fail-closed evaluation")
    if token_gate.get("contract") != TOKEN_CONTRACT:
        raise ValueError("unexpected eval22 token gate contract")
    if token_gate.get("evaluation_only") is not True or int(token_gate.get("pairs") or 0) != pairs:
        raise ValueError("eval22 token gate identity differs from the data contract")
    if diagnostic.get("data_sha256") != contract.get("output_sha256"):
        raise ValueError("eval22 diagnostic and dataset hashes differ")
    if "semantic_delta" not in diagnostic.get("components", {}):
        raise ValueError("eval22 diagnostic is missing semantic-delta metrics")

    evidence = {str(row["task_id"]): row for row in contract.get("evidence") or []}
    if len(evidence) != pairs:
        raise ValueError("eval22 evidence count differs from contract")
    expected_rows = {
        f"{task}::{label}" for task in evidence for label in ("chosen", "rejected")
    }
    rows = _index_diagnostic(diagnostic)
    if set(rows) != expected_rows:
        raise ValueError("diagnostic does not contain the complete eval22 pair set")

    per_task: list[dict[str, Any]] = []
    family_margins: dict[str, list[float]] = defaultdict(list)
    family_preferred: Counter[str] = Counter()
    full_sql_margins: list[float] = []
    for task in sorted(evidence):
        chosen = rows[f"{task}::chosen"]
        rejected = rows[f"{task}::rejected"]
        chosen_delta_nll = float(chosen["components"]["semantic_delta"]["mean_nll"])
        rejected_delta_nll = float(rejected["components"]["semantic_delta"]["mean_nll"])
        chosen_sql_nll = float(chosen["components"]["sql_shell"]["mean_nll"])
        rejected_sql_nll = float(rejected["components"]["sql_shell"]["mean_nll"])
        margin = rejected_delta_nll - chosen_delta_nll
        full_sql_margin = rejected_sql_nll - chosen_sql_nll
        full_sql_margins.append(full_sql_margin)
        rank = chosen["sql_token_rank"]
        if rank.get("first_nongreedy_offset") is None:
            family = "all_chosen_sql_tokens_greedy"
        else:
            family = critical_token_family(rank.get("first_nongreedy_target_token"))
        family_margins[family].append(margin)
        family_preferred[family] += int(margin > 0)
        per_task.append(
            {
                "task_id": task,
                "critical_token_family": family,
                "first_nongreedy_offset": rank.get("first_nongreedy_offset"),
                "first_nongreedy_target_id": rank.get("first_nongreedy_target_id"),
                "first_nongreedy_rank": rank.get("first_nongreedy_rank"),
                "semantic_delta_log_probability_margin_per_token": margin,
                "full_sql_log_probability_margin_per_token": full_sql_margin,
                "chosen_preferred_on_semantic_delta": margin > 0,
                "chosen_preferred_on_full_sql": full_sql_margin > 0,
            }
        )

    margins = [float(row["semantic_delta_log_probability_margin_per_token"]) for row in per_task]
    chosen_preferred = sum(bool(row["chosen_preferred_on_semantic_delta"]) for row in per_task)
    preferred_threshold = int(contract["future_candidate_gate"]["chosen_preferred_min"])
    systematic_misranking = chosen_preferred < preferred_threshold
    return {
        "contract": "disjoint-real-state-eval22-margin-baseline-v1",
        "state_source_checkpoint": "step120",
        "evaluated_model_label": diagnostic.get("model_label"),
        "evaluation_role": contract["evaluation_role"],
        "selection_bias": contract["selection_bias"],
        "task_count": pairs,
        "semantic_delta_margin": {
            "definition": "rejected_mean_nll_minus_chosen_mean_nll_positive_prefers_chosen",
            "chosen_preferred": chosen_preferred,
            "chosen_preferred_rate": chosen_preferred / pairs,
            "chosen_preferred_wilson95": wilson_interval(chosen_preferred, pairs),
            "future_candidate_chosen_preferred_min": preferred_threshold,
            "mean_margin": fmean(margins),
            "median_margin": median(margins),
            "by_critical_token_family": {
                family: {
                    "tasks": len(values),
                    "chosen_preferred": family_preferred[family],
                    "mean_margin": fmean(values),
                    "median_margin": median(values),
                }
                for family, values in sorted(family_margins.items())
            },
        },
        "full_sql_margin": {
            "definition": "rejected_mean_nll_minus_chosen_mean_nll_positive_prefers_chosen",
            "chosen_preferred": sum(
                bool(row["chosen_preferred_on_full_sql"]) for row in per_task
            ),
            "chosen_preferred_rate": sum(
                bool(row["chosen_preferred_on_full_sql"]) for row in per_task
            )
            / pairs,
            "mean_margin": fmean(full_sql_margins),
            "median_margin": median(full_sql_margins),
        },
        "baseline_diagnosis": {
            "systematic_correct_vs_actual_wrong_misranking_observed": systematic_misranking,
            "training_data_expansion_justified": systematic_misranking,
            "selected_next_action": (
                "expand_to_at_least_48_disjoint_training_pairs_then_one_step_canary"
                if systematic_misranking
                else "do_not_expand_pairwise_data_correct_delta_already_preferred"
            ),
        },
        "future_candidate_gate": contract["future_candidate_gate"],
        "execution": {
            "forward_only": True,
            "optimizer_initialized": False,
            "checkpoint_saved": False,
        },
        "per_task": per_task,
        "evaluation_only": True,
        "may_be_used_as_training_data": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--token-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        json.loads(args.diagnostic.read_text(encoding="utf-8")),
        json.loads(args.dataset_contract.read_text(encoding="utf-8")),
        json.loads(args.token_gate.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
