from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_validator_recomputes_comparison_and_stays_fail_closed():
    source = (
        ROOT / "scripts" / "validate_disjoint_real_state_eval22_candidate.py"
    ).read_text(encoding="utf-8")
    assert "recomputed_comparison = compare(baseline, candidate)" in source
    assert '"chosen_preferred": (3, 17, False)' in source
    assert '"per_task_margin_improved": (18, 18, True)' in source
    assert '"new_earlier_first_nongreedy_regressions": (0, 0, True)' in source
    assert '"eval22_was_sufficient_to_reject_existing_candidate": True' in source
    assert '"use_eval22_as_training_data": False' in source
    assert '"run_full64_now": False' in source
    assert '"additional_chosen_only_steps": False' in source
    assert '"training_allowed": False' in source
    assert '"promotion_allowed": False' in source
