from pathlib import Path

import pytest

from scripts.analyze_disjoint_real_state_evaluation import analyze, wilson_interval
from scripts.compare_disjoint_real_state_evaluation import compare


def fixtures(chosen_preferred: int):
    pairs = 22
    evidence = [{"task_id": f"task_{index:06d}"} for index in range(pairs)]
    rows = []
    for index in range(pairs):
        task = f"task_{index:06d}"
        preferred = index < chosen_preferred
        for label in ("chosen", "rejected"):
            chosen = label == "chosen"
            nll = 1.0 if chosen == preferred else 2.0
            rows.append(
                {
                    "task_id": f"{task}::{label}",
                    "components": {
                        "semantic_delta": {"mean_nll": nll},
                        "sql_shell": {"mean_nll": nll + 0.1},
                    },
                    "sql_token_rank": {
                        "first_nongreedy_offset": 0,
                        "first_nongreedy_target_id": 1,
                        "first_nongreedy_rank": 2,
                        "first_nongreedy_target_token": "SUM" if index % 2 else "SELECT",
                    },
                }
            )
    diagnostic = {
        "contract": "repair-sft-teacher-forced-component-diagnostic-v3",
        "forward_only": True,
        "optimizer_initialized": False,
        "data_sha256": "hash",
        "components": {"semantic_delta": {}},
        "model_label": "step120_eval22",
        "per_task": rows,
    }
    contract = {
        "contract": "current-definition-disjoint-first-error-evaluation-v1",
        "evaluation_role": "postselected_real_failure_state_diagnostic_only",
        "selection_bias": "conditioned_on_step120_failure_not_population_representative",
        "pairs": pairs,
        "rows": 2 * pairs,
        "expected_pairs": pairs,
        "pair_evaluation_gate_passed": True,
        "output_sha256": "hash",
        "evidence": evidence,
        "evaluation_only": True,
        "may_be_used_as_training_data": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "future_candidate_gate": {
            "chosen_preferred_min": 17,
            "per_task_margin_improved_min": 18,
            "new_earlier_first_nongreedy_regressions_max": 0,
            "full64_pareto_required_after_pass": True,
        },
    }
    token_gate = {
        "contract": "current-definition-disjoint-pair-evaluation-token-gate-v1",
        "evaluation_only": True,
        "pairs": pairs,
    }
    return diagnostic, contract, token_gate


def test_eval22_baseline_authorizes_only_data_expansion_when_misranking_is_systematic():
    diagnostic, contract, token_gate = fixtures(chosen_preferred=3)
    result = analyze(diagnostic, contract, token_gate)

    assert result["task_count"] == 22
    assert result["semantic_delta_margin"]["chosen_preferred"] == 3
    assert result["baseline_diagnosis"] == {
        "systematic_correct_vs_actual_wrong_misranking_observed": True,
        "training_data_expansion_justified": True,
        "selected_next_action": "expand_to_at_least_48_disjoint_training_pairs_then_one_step_canary",
    }
    assert result["training_allowed"] is False
    assert result["may_be_used_as_training_data"] is False
    assert result["evaluated_model_label"] == "step120_eval22"
    assert result["full_sql_margin"]["chosen_preferred"] == 3


def test_eval22_baseline_stops_pairwise_path_when_17_are_already_preferred():
    diagnostic, contract, token_gate = fixtures(chosen_preferred=17)
    result = analyze(diagnostic, contract, token_gate)
    assert result["baseline_diagnosis"]["systematic_correct_vs_actual_wrong_misranking_observed"] is False
    assert result["baseline_diagnosis"]["training_data_expansion_justified"] is False


def test_eval22_contract_cannot_enable_training():
    diagnostic, contract, token_gate = fixtures(chosen_preferred=3)
    contract["training_allowed"] = True
    with pytest.raises(ValueError, match="fail-closed evaluation"):
        analyze(diagnostic, contract, token_gate)


def test_wilson_interval_is_bounded_and_contains_observed_rate():
    low, high = wilson_interval(17, 22)
    assert 0 <= low < 17 / 22 < high <= 1


def test_eval22_runner_is_forward_only_and_never_saves_or_trains():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_disjoint_real_state_eval22_margin.sh").read_text(
        encoding="utf-8"
    )
    assert "PAIRS != EXPECTED_PAIRS" in source
    assert "evaluation_only=true" in source
    assert "may_be_used_as_training_data=false" in source
    assert "engine.forward_only=true" in source
    assert "checkpoint.load_contents=[]" in source
    assert "checkpoint.save_contents=[]" in source
    assert "optimizer_initialized=false" in source
    assert "training_allowed=false" in source


def test_future_candidate_gate_requires_17_preferred_18_improved_and_no_regression():
    diagnostic, contract, token_gate = fixtures(chosen_preferred=3)
    baseline = analyze(diagnostic, contract, token_gate)
    candidate_diagnostic, _, _ = fixtures(chosen_preferred=22)
    candidate = analyze(candidate_diagnostic, contract, token_gate)
    for index, row in enumerate(candidate["per_task"]):
        before = baseline["per_task"][index]["semantic_delta_log_probability_margin_per_token"]
        row["semantic_delta_log_probability_margin_per_token"] = (
            before + 0.5 if index < 18 else before
        )
    result = compare(baseline, candidate)
    assert result["checks"]["chosen_preferred"]["observed"] == 22
    assert result["checks"]["per_task_margin_improved"]["observed"] == 18
    assert result["checks"]["new_earlier_first_nongreedy_regressions"]["observed"] == 0
    assert result["gate_passed"] is True
    assert result["decision"]["full64_pareto_replay_allowed"] is True
    assert result["decision"]["additional_training_allowed"] is False
