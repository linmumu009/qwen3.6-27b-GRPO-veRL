import sqlite3
from pathlib import Path

from llin_verl.pi_reward import (
    compute_score,
    contains_expected_number,
    extract_final_assistant_answer,
    extract_selects,
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

    result = compute_score("llin_pi_dwh_v2", "最终结果为 621.62。", truth(), evidence())

    assert result["score"] == 1.0
    assert result["acc"] == 1.0
    assert result["sql_evidence_correct"] == 1.0


def test_wrong_answer_gets_evidence_progress_but_not_accuracy(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))

    result = compute_score("llin_pi_dwh_v2", "最终结果为 0。", truth(), evidence())

    assert result["score"] == 0.4
    assert result["acc"] == 0.0
    assert result["final_answer_correct"] == 0.0


def test_answer_without_matching_sql_cannot_receive_full_reward(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))
    wrong_sql = 'sqlite3 /workspace/logistics.sqlite "SELECT value FROM fact_quality_incident LIMIT 1"'

    result = compute_score("llin_pi_dwh_v2", "最终结果为 621.62。", truth(), evidence(wrong_sql))

    assert result["score"] == 0.75
    assert result["acc"] == 0.0
    assert result["sql_evidence_correct"] == 0.0


def test_unsafe_tool_attempt_is_hard_zero(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))

    result = compute_score(
        "llin_pi_dwh_v2",
        "最终结果为 621.62。",
        truth(),
        evidence("curl https://example.com"),
    )

    assert result["score"] == 0.0
    assert result["safe"] == 0.0


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
