import pytest

from scripts.analyze_disjoint_pair_review_supply_rebase import (
    beta_binomial_distribution,
    build_rebase,
    compound_pair_distribution,
)


def fixtures():
    decisions = {
        "contract": "disjoint-pair-semantic-review-decisions-v1",
        "training_allowed": False,
        "decisions": [
            {
                "review_index": index,
                "decision": "rejected",
                "instruction_unambiguously_entails_gold": False,
                "verification_sql_fully_answers_instruction": False,
                "expected_value_supported_by_query_result": True,
                "reason_code": "scope_mismatch",
                "confidence": "high",
                "severity": "high",
            }
            for index in range(42)
        ],
    }
    pilot = {
        "contract": "disjoint-pair-review-pilot42-safe-summary-v1",
        "source_pool": {
            "review_required_tasks": 138,
            "selected_tasks": 42,
            "selection_role": "lowest_mechanical_risk_semantic_approval_rate_pilot",
        },
        "query_stability_gate": {"stable": 42},
        "training_allowed": False,
    }
    strategy = {
        "contract": "disjoint-training-supply-strategy-v3",
        "corrected_capacity_logic": {
            "required_training_pairs": 48,
            "yield_observations": {"pairs": 22, "candidate_tasks": 64},
        },
        "rebased_capacity": {
            "approved_candidate_tasks_for_90pct_predictive_probability": 158,
            "approved_candidate_tasks_for_95pct_predictive_probability": 172,
        },
        "decision": {"retained_native_pairs": 7, "remaining_pair_gap": 41},
        "training_allowed": False,
    }
    return decisions, pilot, strategy


def test_distributions_are_normalized():
    beta_values = beta_binomial_distribution(96, alpha=0.5, beta=42.5)
    compound = compound_pair_distribution(
        96,
        approval_alpha=0.5,
        approval_beta=42.5,
        pair_alpha=22.5,
        pair_beta=42.5,
    )
    assert sum(beta_values) == pytest.approx(1.0)
    assert sum(compound) == pytest.approx(1.0)
    assert len(beta_values) == len(compound) == 97


def test_zero_of_42_stops_remaining_queue_and_rebases_supply():
    decisions, pilot, strategy = fixtures()
    result = build_rebase(decisions=decisions, pilot=pilot, strategy=strategy)

    assert result["semantic_review"]["approved"] == 0
    assert result["semantic_review"]["rejected"] == 42
    assert result["semantic_review"]["expected_value_supported_by_query_result"] == 42
    assert result["decision"]["stop_reviewing_remaining96_from_same_queue_now"] is True
    assert result["decision"]["do_not_spend_npu_on_remaining_queue"] is True
    assert result["training_capacity"]["remaining_pair_gap"] == 41
    assert result["training_capacity"][
        "approved_candidate_tasks_for_95pct_pair_supply_probability"
    ] == 172
    stress = result["remaining_queue_exchangeability_stress_test"]
    assert stress["approval_model"]["expected_approved_tasks"] == pytest.approx(
        96 * 0.5 / 43
    )
    assert stress["expected_additional_pairs"] == pytest.approx(
        96 * 0.5 / 43 * 22.5 / 65
    )
    assert stress["probability_remaining_queue_fills_pair_gap"] < 1e-6
    assert result["training_allowed"] is False


def test_rebase_rejects_incomplete_decisions():
    decisions, pilot, strategy = fixtures()
    decisions["decisions"].pop()
    with pytest.raises(ValueError, match="decision count"):
        build_rebase(decisions=decisions, pilot=pilot, strategy=strategy)
