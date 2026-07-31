from llin_verl.pi_reward import compute_score, contains_expected_number, extract_final_assistant_answer


def test_contains_expected_number_handles_commas_and_tolerance():
    assert contains_expected_number("合计为 1,581.5300", 1581.53)
    assert not contains_expected_number("合计为 158.153", 1581.53)


def test_reward_requires_tool_table_and_answer():
    truth = {"expected_value": 621.62, "required_tables": ["fact_quality_incident"]}
    info = {
        "llin_sql_queries": [
            {
                "ok": True,
                "tables": ["fact_quality_incident"],
                "sql": "select sum(cargo_declared_value) from fact_quality_incident",
            }
        ]
    }
    result = compute_score("llin_pi_dwh", "最终结果为 621.62。", truth, info)
    assert result["score"] == 1.0
    assert result["acc"] == 1.0


def test_reward_only_gives_progress_for_wrong_answer():
    truth = {"expected_value": 621.62, "required_tables": ["fact_quality_incident"]}
    info = {"llin_sql_queries": [{"ok": True, "tables": ["fact_quality_incident"]}]}
    assert compute_score("llin_pi_dwh", "结果为 0。", truth, info)["score"] == 0.2


def test_strict_final_answer_excludes_tool_results_and_thought():
    response = """assistant
<think>正确值可能是 621.62，但还要查工具。</think>
<tool_call>SELECT value FROM fact_quality_incident</tool_call>
user
<tool_response>{"value": 621.62}</tool_response>
assistant
<think>工具已经返回结果。</think>
最终结论是 600.00。"""

    assert extract_final_assistant_answer(response) == "最终结论是 600.00。"

    truth = {"expected_value": 621.62, "required_tables": ["fact_quality_incident"]}
    info = {"llin_sql_queries": [{"ok": True, "tables": ["fact_quality_incident"]}]}
    result = compute_score("llin_pi_dwh", response, truth, info)

    # Tool evidence is preserved for diagnosis, but the incorrect visible
    # final answer no longer receives full reward.
    assert result["score"] == 0.2
    assert result["answer_correct"] == 0.0
    assert result["evidence_contains_expected"] == 1.0
    assert result["final_answer_correct"] == 0.0


def test_strict_final_answer_is_empty_when_trajectory_ends_with_tool_call():
    response = """assistant
<think>继续查询。</think>
<tool_call>SELECT 621.62</tool_call>
user
<tool_response>{"value": 621.62}</tool_response>"""

    assert extract_final_assistant_answer(response) == ""
