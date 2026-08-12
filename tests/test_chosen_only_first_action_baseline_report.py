import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_chosen_only_baseline_summary_locks_pre_registered_canary() -> None:
    summary = json.loads(
        (
            ROOT
            / "docs"
            / "chosen_only_first_action_baseline_20260813_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["scope"]["calibration_rows"] == 16
    assert summary["scope"]["optimizer_initialized"] is False
    assert summary["cpu_gate"]["rows"] == 64
    assert summary["cpu_gate"]["passed"] is True
    assert summary["baseline"]["sql_token_count"] == 381
    assert summary["baseline"]["sql_greedy_token_count"] == 277
    assert summary["baseline"]["tasks_all_sql_tokens_greedy"] == 0
    assert summary["one_step_canary"]["training_steps"] == 1
    assert summary["one_step_canary"]["calibration_rows_excluded_from_training"] == 16
    assert summary["post_canary_gates"]["calibration_tasks_sql_nll_improved_min"] == 12
    assert summary["decision"]["training_scope"] == "one_step_train48_only"
    assert summary["decision"]["free_rollout_allowed"] is False
    assert summary["decision"]["promotion_allowed"] is False
    assert summary[
        "contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths"
    ] is False
