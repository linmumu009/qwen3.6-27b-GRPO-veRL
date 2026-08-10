import sqlite3
from pathlib import Path

from llin_verl.pi_reward import (
    boss_numbers_match,
    boss_reward_components,
    compute_score,
    compute_score_dense30,
    contains_expected_number,
    dense_final_answer_correctness,
    extract_final_assistant_answer,
    extract_selects,
    final_answer_correct,
    rows_contain_unique_projection,
    safe_projection_sql,
)
from llin_verl.pi_tool_contract import command_is_safe


def make_database(root: Path) -> None:
    database = root / "sft" / "version" / "logistics.sqlite"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database)
    connection.execute("create table fact_quality_incident(value real)")
    connection.executemany("insert into fact_quality_incident values (?)", [(600.0,), (21.62,)])
    connection.commit()
    connection.close()


def truth() -> dict:
    return {
        "environment_id": "sft/version",
        "answer_type": "numeric",
        "expected_value": 621.62,
        "verification_sql": "SELECT SUM(value) FROM fact_quality_incident",
        "required_tables": ["fact_quality_incident"],
    }


def evidence(command: str | None = None) -> dict:
    command = command or 'sqlite3 /workspace/logistics.sqlite "SELECT SUM(value) FROM fact_quality_incident"'
    return {
        "pi_tool_events": [
            {
                "name": "bash",
                "arguments": {"command": command},
                "ok": True,
            }
        ]
    }


def test_contains_expected_number_handles_commas_and_tolerance():
    assert contains_expected_number("合计为 1,581.5300", 1581.53)
    assert not contains_expected_number("合计为 158.153", 1581.53)


def test_extract_selects_supports_sqlite_and_python_execute():
    commands = [
        'sqlite3 /workspace/logistics.sqlite "SELECT SUM(value) FROM fact_quality_incident"',
        "python3 -c 'c.execute(\"SELECT value FROM fact_quality_incident\")'",
    ]
    assert extract_selects(commands) == [
        "SELECT SUM(value) FROM fact_quality_incident",
        "SELECT value FROM fact_quality_incident",
    ]


def test_reward_requires_final_answer_and_exact_executable_sql(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))

    result = compute_score(
        "llin_pi_dwh_v2",
        "查询与复核已经完成，最终确认的合计结果为 621.62。",
        truth(),
        evidence(),
    )

    assert result["score"] == 1.0
    assert result["acc"] == 1.0
    assert result["sql_evidence_correct"] == 1.0
    assert result["sql_evidence_mode"] == "exact"
    assert result["boss_reward"] == 1.0
    assert result["boss_fields_used"] == 1.0
    assert result["evidence_reward"] == 1.0


def test_reward_accepts_unique_safe_projection_with_extra_aggregates(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))
    superset_sql = (
        "SELECT COUNT(*), SUM(value), AVG(value), MIN(value), MAX(value) "
        "FROM fact_quality_incident"
    )

    result = compute_score(
        "llin_pi_dwh_v2",
        "查询与复核已经完成，最终确认的合计结果为 621.62。",
        truth(),
        evidence(f'sqlite3 /workspace/logistics.sqlite "{superset_sql}"'),
    )

    assert result["score"] == 1.0
    assert result["acc"] == 1.0
    assert result["sql_evidence_mode"] == "safe_projection"


def test_projection_rejects_ambiguous_duplicate_columns_and_wrong_aggregate(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))
    duplicate = "SELECT SUM(value), SUM(value) FROM fact_quality_incident"
    wrong_aggregate = "SELECT MAX(621.62) FROM fact_quality_incident"

    duplicate_result = compute_score(
        "llin_pi_dwh_v2",
        "查询与复核已经完成，最终确认的合计结果为 621.62。",
        truth(),
        evidence(f'sqlite3 /workspace/logistics.sqlite "{duplicate}"'),
    )
    wrong_result = compute_score(
        "llin_pi_dwh_v2",
        "查询与复核已经完成，最终确认的合计结果为 621.62。",
        truth(),
        evidence(f'sqlite3 /workspace/logistics.sqlite "{wrong_aggregate}"'),
    )

    assert duplicate_result["sql_evidence_mode"] == "none"
    assert wrong_result["sql_evidence_mode"] == "none"
    assert rows_contain_unique_projection([(621.62, 621.62)], [(621.62,)], 1e-3, 1e-5) is False


def test_safe_projection_preserves_tables_aggregates_and_date_filter():
    gold = (
        "SELECT SUM(dwell_minutes) FROM fact_waybill_event "
        "WHERE DATE(occurred_at) = '2026-06-23'"
    )
    candidate = (
        "SELECT COUNT(*), SUM(dwell_minutes), AVG(dwell_minutes) FROM fact_waybill_event "
        "WHERE occurred_at >= '2026-06-23 00:00:00' "
        "AND occurred_at < '2026-06-24 00:00:00' AND dwell_minutes IS NOT NULL"
    )

    assert safe_projection_sql(candidate, gold, {"fact_waybill_event"}) is True
    assert safe_projection_sql(candidate.replace("SUM(dwell_minutes)", "SUM(other_value)"), gold, {"fact_waybill_event"}) is False


def test_wrong_answer_gets_evidence_progress_but_not_accuracy(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))

    result = compute_score(
        "llin_pi_dwh_v2",
        "查询与复核已经完成，但最终错误地报告合计结果为 0。",
        truth(),
        evidence(),
    )

    assert result["score"] == 0.68
    assert result["acc"] == 0.0
    assert result["final_answer_correct"] == 0.0
    assert result["boss_reward"] == 0.8


def test_answer_without_matching_sql_cannot_receive_full_reward(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))
    wrong_sql = 'sqlite3 /workspace/logistics.sqlite "SELECT value FROM fact_quality_incident LIMIT 1"'

    result = compute_score(
        "llin_pi_dwh_v2",
        "查询与复核已经完成，最终确认的合计结果为 621.62。",
        truth(),
        evidence(wrong_sql),
    )

    assert result["score"] == 0.925
    assert result["acc"] == 0.0
    assert result["sql_evidence_correct"] == 0.0
    assert result["boss_reward"] == 1.0


def test_unsafe_tool_attempt_is_hard_zero(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))

    result = compute_score(
        "llin_pi_dwh_v2",
        "查询与复核已经完成，最终确认的合计结果为 621.62。",
        truth(),
        evidence("curl https://example.com"),
    )

    assert result["score"] == 0.0
    assert result["safe"] == 0.0
    assert result["bash_command_count"] == 1.0
    assert result["unsafe_command_count"] == 1.0
    assert result["unsafe_network_count"] == 1.0
    assert result["unsafe_destructive_count"] == 0.0
    assert result["unsafe_host_path_escape_count"] == 0.0


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


def test_strict_final_answer_is_empty_when_trajectory_ends_with_tool_call():
    response = """assistant
<think>继续查询。</think>
<tool_call>SELECT 621.62</tool_call>
user
<tool_response>{"value": 621.62}</tool_response>"""
    assert extract_final_assistant_answer(response) == ""


def test_workspace_guard_blocks_root_enumeration_but_allows_workspace_queries():
    assert not command_is_safe("find / -name '*.sqlite'")
    assert not command_is_safe("ls -la /")
    assert command_is_safe("ls -la /workspace/")
    assert command_is_safe("find /workspace -name '*.md'")
    assert command_is_safe(
        'sqlite3 /workspace/logistics.sqlite "SELECT COUNT(*) FROM fact_quality_incident"'
    )


def test_boss_number_match_allows_extra_values_but_strict_table_answer_does_not():
    expected = [
        {"category": "A", "value": 10},
        {"category": "B", "value": 20},
    ]

    assert boss_numbers_match("复核结果包括 10、20，另外观察到 30。", expected) is True
    assert final_answer_correct("复核结果包括 10、20，另外观察到 30。", "table", expected, 1e-3, 1e-5) is False


def test_boss_reward_components_port_upstream_result_process_efficiency_formula():
    components = boss_reward_components(
        "已经完成查询和复核，最终合计结果为 621.62。",
        621.62,
        ['sqlite3 /workspace/logistics.sqlite "SELECT SUM(value) FROM fact_quality_incident"'],
        ["SELECT SUM(value) FROM fact_quality_incident"],
        {"fact_quality_incident"},
        {"fact_quality_incident"},
        ["value"],
        True,
        [],
    )

    assert components["result_score"] == 1.0
    assert components["process_score"] == 1.0
    assert components["efficiency_score"] == 1.0
    assert components["reward"] == 1.0


def test_dense_correctness_orders_near_wrong_far_and_exact_numeric_answers():
    near = dense_final_answer_correctness("最终结果为 590。", "numeric", 600.0)
    far = dense_final_answer_correctness("最终结果为 60。", "numeric", 600.0)

    assert dense_final_answer_correctness("最终结果为 600。", "numeric", 600.0) == 1.0
    assert 1.0 > near > far > 0.0


def test_dense_correctness_uses_table_labels_and_penalizes_number_dumping():
    expected = [{"category": "A", "value": 10}, {"category": "B", "value": 20}]
    labeled = dense_final_answer_correctness("A 为 9，B 为 19。", "table", expected)
    unlabeled = dense_final_answer_correctness("结果为 9 和 19。", "table", expected)
    dumped = dense_final_answer_correctness(
        "结果候选为 1、2、3、4、5、6、7、8、9、19。", "table", expected
    )

    assert labeled > unlabeled > dumped


def test_dense_correctness_ignores_dates_and_requires_visible_final_answer():
    assert dense_final_answer_correctness("", "numeric", 2026.0) == 0.0
    assert dense_final_answer_correctness("统计周期是 2026-06-28，结果为 100。", "numeric", 2026.0) < 0.5


def test_dense_reward_weight_is_opt_in_and_preserves_safety_gate(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))
    baseline = compute_score(
        "llin_pi_dwh_v2",
        "查询与复核已经完成，但最终错误地报告合计结果为 600。",
        truth(),
        evidence(),
    )
    monkeypatch.setenv("PI_DENSE_CORRECTNESS_WEIGHT", "0.3")
    dense = compute_score(
        "llin_pi_dwh_v2",
        "查询与复核已经完成，但最终错误地报告合计结果为 600。",
        truth(),
        evidence(),
    )
    unsafe = compute_score(
        "llin_pi_dwh_v2",
        "查询与复核已经完成，但最终错误地报告合计结果为 600。",
        truth(),
        evidence("curl https://example.com"),
    )

    assert baseline["score"] == baseline["base_score"]
    assert dense["dense_correctness_weight"] == 0.3
    assert dense["score"] != dense["base_score"]
    assert unsafe["score"] == 0.0


def test_dense30_entry_point_does_not_depend_on_worker_environment(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))
    monkeypatch.setenv("PI_DENSE_CORRECTNESS_WEIGHT", "0")

    result = compute_score_dense30(
        "llin_pi_dwh_v2",
        "查询与复核已经完成，但最终错误地报告合计结果为 600。",
        truth(),
        evidence(),
    )

    assert result["dense_correctness_weight"] == 0.3
