import pytest

from llin_verl.force_final_policy import build_force_final_instruction, decide_force_final
from scripts.prepare_force_final_sentinel import DEFAULT_TASK_IDS, select_rows


def test_turn_budget_triggers_before_hard_cap():
    decision = decide_force_final(
        assistant_turns=22,
        response_tokens=20_000,
        response_length=45_056,
        after_assistant_turns=22,
        reserve_response_tokens=4_096,
    )
    assert decision.triggered
    assert decision.reason == "assistant_turn_budget"
    assert decision.remaining_response_tokens == 25_056


def test_token_budget_triggers_independently():
    decision = decide_force_final(
        assistant_turns=12,
        response_tokens=41_000,
        response_length=45_056,
        after_assistant_turns=22,
        reserve_response_tokens=4_096,
    )
    assert decision.triggered
    assert decision.reason == "response_token_budget"
    assert "Do not call any more tools" in build_force_final_instruction(decision, assistant_turns=12)


def test_disabled_policy_does_not_trigger():
    decision = decide_force_final(
        assistant_turns=99,
        response_tokens=45_000,
        response_length=45_056,
    )
    assert not decision.triggered
    with pytest.raises(ValueError):
        build_force_final_instruction(decision, assistant_turns=99)


def test_sentinel_selection_is_ordered_and_fail_closed():
    rows = [
        {"reward_model": {"ground_truth": {"task_id": value}}}
        for value in reversed(DEFAULT_TASK_IDS)
    ]
    selected = select_rows(rows, DEFAULT_TASK_IDS)
    assert [row["reward_model"]["ground_truth"]["task_id"] for row in selected] == list(DEFAULT_TASK_IDS)
    with pytest.raises(ValueError, match="missing"):
        select_rows(rows[:-1], DEFAULT_TASK_IDS)
