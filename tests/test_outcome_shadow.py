import pytest

from llin_verl.outcome_shadow import expected_value_from_ground_truth, score_final_outcome


def test_outcome_shadow_uses_only_final_visible_answer():
    solution = """assistant
<think>the tool returned 10</think><tool_call>query</tool_call>
user
<tool_response>10</tool_response>
assistant
最终答案是 9。"""
    result = score_final_outcome(
        solution,
        {
            "answer_type": "numeric",
            "expected_value_json": "10",
            "verification_sql": "SELECT 10",
            "abs_tol": 1e-3,
            "rel_tol": 1e-5,
        },
    )
    assert result["outcome_only_score"] == 0
    assert 0 < result["dense_final_answer_correctness"] < 1
    assert result["has_final_answer"] == 1


def test_outcome_shadow_accepts_table_labels_and_values():
    result = score_final_outcome(
        "assistant\n最终：\n| 类别 | 数值 |\n|---|---:|\n| A | 10 |\n| B | 20 |",
        {
            "answer_type": "table",
            "expected_value_json": '[{"category":"A","value":10},{"category":"B","value":20}]',
        },
    )
    assert result["outcome_only_score"] == 1


def test_outcome_shadow_rejects_swapped_table_bindings():
    result = score_final_outcome(
        "assistant\n最终：\n| 类别 | 数值 |\n|---|---:|\n| A | 20 |\n| B | 10 |",
        {
            "answer_type": "table",
            "expected_value_json": '[{"category":"A","value":10},{"category":"B","value":20}]',
        },
    )
    assert result["outcome_only_score"] == 0


def test_expected_value_is_required():
    with pytest.raises(ValueError, match="expected final value"):
        expected_value_from_ground_truth({"answer_type": "numeric"})
