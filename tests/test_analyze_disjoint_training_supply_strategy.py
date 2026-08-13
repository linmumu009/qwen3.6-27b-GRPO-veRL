from scripts.analyze_disjoint_training_supply_strategy import (
    EVAL_SUMMARY_CONTRACT,
    NATIVE_SUPPLY_CONTRACT,
    beta_binomial_tail,
    build_strategy,
)


def test_beta_binomial_tail_and_rebased_capacity():
    native = {
        "contract": NATIVE_SUPPLY_CONTRACT,
        "training_allowed": False,
        "native_observed_first_query_errors": 27,
        "native_error_overlap_with_eval22": 16,
        "native_error_states_outside_eval22": 11,
        "additional_frozen_overlap_outside_eval22": 0,
        "native_error_states_outside_all_frozen_sets": 11,
        "outside_all_frozen_first_error_category_counts": {
            "executable_wrong_or_insufficient_evidence": 10,
            "schema_syntax_or_execution_error": 1,
        },
    }
    evaluation = {
        "contract": EVAL_SUMMARY_CONTRACT,
        "training_allowed": False,
        "data_gate": {"unique_source_tasks": 22},
    }
    result = build_strategy(native_supply=native, eval_summary=evaluation)
    assert result["corrected_capacity_logic"]["eval22_pairs_reusable_for_training"] == 0
    assert result["existing_zero_npu_opportunity"]["remaining_pairs_if_all_native_states_pass_pair_audit"] == 37
    assert 0.49 < result["corrected_capacity_logic"]["probability_queue_alone_reaches_48_if_every_task_is_approved"] < 0.51
    assert 0.87 < result["rebased_capacity"]["probability_review138_supplies_remaining_pairs_if_all_11_native_states_pass_and_all_138_are_approved"] < 0.88
    assert result["rebased_capacity"]["approved_candidate_tasks_for_90pct_predictive_probability"] == 143
    assert result["rebased_capacity"]["approved_candidate_tasks_for_95pct_predictive_probability"] == 156
    assert result["rebased_capacity"]["low_risk_expected_pairs_at_observed_yield"] == 14.4375
    assert result["rebased_capacity"]["probability_low_risk_subset_supplies_remaining_pairs"] < 0.000001
    assert result["rebased_capacity"]["low_risk_subset_can_reliably_fill_remaining_pairs_by_itself"] is False
    assert result["decision"]["training_now"] is False


def test_predictive_probability_increases_with_more_candidates():
    alpha, beta = 22.5, 42.5
    assert beta_binomial_tail(138, 37, alpha=alpha, beta=beta) > beta_binomial_tail(
        100, 37, alpha=alpha, beta=beta
    )
