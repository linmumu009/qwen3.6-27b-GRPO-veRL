import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_schema_oracle_action_safe_summary_locks_actual_failure() -> None:
    summary = json.loads(
        (
            ROOT / "docs" / "schema_oracle_action_gate_20260813_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["scope"]["rows"] == 64
    assert summary["scope"]["exit_code"] == 0
    assert summary["scope"]["forced_checkpoint_to_rollout_sync"] is True
    assert summary["scope"]["optimizer_initialized"] is False
    assert summary["scope"]["checkpoint_saved"] is False
    assert summary["conversion"] == {
        "tool_calls": 127,
        "observed_tool_calls": 64,
        "unobserved_terminal_tool_calls": 63,
        "truncated_nonterminal_tool_calls_with_real_error_response": 1,
    }
    assert summary["first_query"] == {
        "correct_or_equivalent": 4,
        "observed_wrong": 35,
        "no_readonly_query": 25,
        "wrong_categories": {
            "executable_empty_evidence": 10,
            "executable_wrong_or_insufficient_evidence": 23,
            "schema_syntax_or_execution_error": 2,
        },
    }
    assert summary["gates"]["runtime_correct_or_equivalent"] == {
        "observed": 4,
        "required": 32,
        "passed": False,
    }
    assert summary["gates"]["observed_wrong_pair_acquisition"] == {
        "observed": 35,
        "required": 48,
        "passed": False,
    }
    assert summary["decision"]["pair_construction_allowed"] is False
    assert summary["decision"]["training_allowed"] is False
    assert summary["invalid_predecessor_run"]["used_for_gate"] is False
    assert summary[
        "contains_raw_commands_prompts_schema_sql_answers_task_ids_tool_outputs_or_server_paths"
    ] is False
