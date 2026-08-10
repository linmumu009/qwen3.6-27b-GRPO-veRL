from scripts.replay_banded_reward_gate import replay


def trajectory(prompt: str, correct: bool, *, sql: bool = False) -> dict:
    return {
        "input": prompt,
        "has_final_answer": 1.0,
        "final_answer_correct": float(correct),
        "sql_evidence_correct": float(sql),
        "safe": 1.0,
        "valid_tool_protocol": 1.0,
        "gold_sql_verified": 1.0,
        "base_score": 1.0,
        "_source": "ignored",
    }


def test_banded_replay_ranks_correct_above_wrong_inside_group():
    rows = [
        trajectory("same", False),
        trajectory("same", False, sql=True),
        trajectory("same", True),
        trajectory("same", True, sql=True),
    ]
    result = replay(rows, expected_group_size=4)

    assert result["valid_group_count"] == 1
    assert result["mixed_correct_group_count"] == 1
    assert result["mixed_correct_rank_rate"] == 1.0
    assert result["gate_checks"]["eligible_wrong_capped_at_0_5"] is True
    assert result["gate_checks"]["eligible_correct_floor_0_65"] is True


def test_banded_replay_hard_zeros_ineligible_and_no_answer():
    rows = [trajectory("same", False) for _ in range(4)]
    rows[0]["safe"] = 0.0
    rows[1]["has_final_answer"] = 0.0
    result = replay(rows, expected_group_size=4)

    assert result["gate_checks"]["ineligible_always_zero"] is True
    assert result["gate_checks"]["no_final_answer_always_zero"] is True
