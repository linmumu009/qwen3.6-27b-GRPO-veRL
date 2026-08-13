#!/usr/bin/env python3
"""Analyze Step 120 margins on native-model real first-error candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any

from scripts.analyze_disjoint_real_state_evaluation import wilson_interval
from scripts.analyze_state_recovery_semantics import critical_token_family


DATA_CONTRACT = "current-definition-native-first-error-training-candidates-v1"
TOKEN_CONTRACT = "current-definition-disjoint-pair-candidate-token-gate-v1"


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get("task_id") or "")
        if not identity or identity in output:
            raise ValueError(f"missing or duplicate diagnostic task ID: {identity!r}")
        output[identity] = row
    return output


def analyze(
    diagnostic: dict[str, Any], contract: dict[str, Any], token_gate: dict[str, Any]
) -> dict[str, Any]:
    if diagnostic.get("contract") != "repair-sft-teacher-forced-component-diagnostic-v3":
        raise ValueError("native candidate margin requires teacher-forced diagnostic v3")
    if diagnostic.get("forward_only") is not True or diagnostic.get("optimizer_initialized") is not False:
        raise ValueError("native candidate margin requires forward-only execution without optimizer")
    if contract.get("contract") != DATA_CONTRACT:
        raise ValueError("unexpected native candidate data contract")
    pairs = int(contract.get("pairs") or 0)
    expected_pairs = int(contract.get("expected_pairs") or 0)
    if (
        contract.get("candidate_pair_gate_passed") is not True
        or contract.get("candidate_only") is not True
        or contract.get("may_be_used_as_training_data") is not False
        or contract.get("training_allowed") is not False
        or contract.get("promotion_allowed") is not False
        or pairs != expected_pairs
        or pairs <= 0
    ):
        raise ValueError("native candidate data contract is not fail closed")
    if (
        token_gate.get("contract") != TOKEN_CONTRACT
        or token_gate.get("candidate_only") is not True
        or token_gate.get("may_be_used_as_training_data") is not False
        or int(token_gate.get("pairs") or 0) != pairs
    ):
        raise ValueError("native candidate token gate identity differs")
    if diagnostic.get("data_sha256") != contract.get("output_sha256"):
        raise ValueError("native candidate diagnostic and dataset hashes differ")
    if "semantic_delta" not in diagnostic.get("components", {}):
        raise ValueError("native candidate diagnostic is missing semantic-delta metrics")

    evidence = {str(row["task_id"]): row for row in contract.get("evidence") or []}
    if len(evidence) != pairs:
        raise ValueError("native candidate evidence count differs")
    rows = _index(list(diagnostic.get("per_task") or []))
    expected_rows = {f"{task}::{label}" for task in evidence for label in ("chosen", "rejected")}
    if set(rows) != expected_rows:
        raise ValueError("diagnostic does not contain the complete native candidate set")

    per_task: list[dict[str, Any]] = []
    family_margins: dict[str, list[float]] = defaultdict(list)
    family_preferred: Counter[str] = Counter()
    full_sql_margins: list[float] = []
    for task in sorted(evidence):
        chosen = rows[f"{task}::chosen"]
        rejected = rows[f"{task}::rejected"]
        margin = float(rejected["components"]["semantic_delta"]["mean_nll"]) - float(
            chosen["components"]["semantic_delta"]["mean_nll"]
        )
        full_sql_margin = float(rejected["components"]["sql_shell"]["mean_nll"]) - float(
            chosen["components"]["sql_shell"]["mean_nll"]
        )
        if not math.isfinite(margin) or not math.isfinite(full_sql_margin):
            raise ValueError("native candidate margin contains a non-finite value")
        rank = chosen["sql_token_rank"]
        family = (
            "all_chosen_sql_tokens_greedy"
            if rank.get("first_nongreedy_offset") is None
            else critical_token_family(rank.get("first_nongreedy_target_token"))
        )
        family_margins[family].append(margin)
        family_preferred[family] += int(margin > 0)
        full_sql_margins.append(full_sql_margin)
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

    margins = [row["semantic_delta_log_probability_margin_per_token"] for row in per_task]
    chosen_preferred = sum(row["chosen_preferred_on_semantic_delta"] for row in per_task)
    preferred_threshold = math.ceil(0.75 * pairs)
    systematic_misranking = chosen_preferred < preferred_threshold
    full_chosen = sum(row["chosen_preferred_on_full_sql"] for row in per_task)
    return {
        "contract": "native-disjoint-real-state-step120-margin-screen-v1",
        "state_source_checkpoint": "native_base",
        "evaluated_model_label": diagnostic.get("model_label"),
        "candidate_role": contract["candidate_role"],
        "task_count": pairs,
        "semantic_delta_margin": {
            "definition": "rejected_mean_nll_minus_chosen_mean_nll_positive_prefers_chosen",
            "chosen_preferred": chosen_preferred,
            "chosen_preferred_rate": chosen_preferred / pairs,
            "chosen_preferred_wilson95": wilson_interval(chosen_preferred, pairs),
            "systematic_misranking_screen_threshold": preferred_threshold,
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
            "chosen_preferred": full_chosen,
            "chosen_preferred_rate": full_chosen / pairs,
            "mean_margin": fmean(full_sql_margins),
            "median_margin": median(full_sql_margins),
        },
        "screening_decision": {
            "systematic_correct_vs_native_wrong_misranking_observed": systematic_misranking,
            "retain_as_candidate_training_source_stratum": systematic_misranking,
            "selected_next_action": (
                "retain_native_pairs_then_measure_42task_review_approval_rate"
                if systematic_misranking
                else "discard_native_pairwise_stratum_correct_delta_already_preferred"
            ),
            "remaining_gap_if_retained": 48 - pairs if systematic_misranking else 48,
        },
        "execution": {"forward_only": True, "optimizer_initialized": False, "checkpoint_saved": False},
        "per_task": per_task,
        "candidate_only": True,
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
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
