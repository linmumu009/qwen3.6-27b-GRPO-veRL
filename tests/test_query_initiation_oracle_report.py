import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_query_initiation_oracle_safe_summary_locks_actual_gate_result() -> None:
    summary = json.loads(
        (
            ROOT / "docs" / "query_initiation_oracle_gate_20260813_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["scope"]["rows"] == 41
    assert summary["scope"]["exit_code"] == 0
    assert summary["scope"]["optimizer_initialized"] is False
    assert summary["scope"]["checkpoint_saved"] is False
    assert summary["first_query"] == {
        "observed_correct_or_equivalent": 0,
        "observed_wrong": 0,
        "unobserved_recognized_readonly": 2,
        "no_recognized_readonly": 39,
    }
    assert summary["tools"]["rows_with_any_sqlite"] == 41
    assert summary["tools"]["rows_with_schema_discovery_sqlite"] == 18
    assert summary["tools"]["duplicate_bash_calls"] == 112
    assert summary["gate"] == {
        "observed_readonly_query_rows": 0,
        "required": 31,
        "passed": False,
    }
    assert summary["decision"]["training_allowed"] is False
    assert summary["decision"]["promotion_allowed"] is False
    assert summary[
        "contains_raw_commands_prompts_sql_answers_task_ids_tool_outputs_or_server_paths"
    ] is False

