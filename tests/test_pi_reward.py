from llin_verl.pi_reward import compute_score, contains_expected_number


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
