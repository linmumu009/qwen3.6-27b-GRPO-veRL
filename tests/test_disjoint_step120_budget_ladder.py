from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_safe_budget_ladder_keeps_training_closed() -> None:
    path = ROOT / "docs" / "disjoint_step120_budget_ladder_20260813_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    payload = path.read_text(encoding="utf-8")

    assert summary["contract"] == "disjoint-step120-budget-ladder-safe-summary-v1"
    assert summary["scope"]["forced_step120_actor_to_rollout_sync"] is True
    assert summary["scope"]["disjoint_from_frozen16_val20_test20"] is True
    assert summary["runs"]["short3"]["pairs"] == 1
    assert summary["runs"]["long12"]["pairs"] == 23
    assert summary["runs"]["full25"]["pairs"] == 22
    assert summary["runs"]["full25"]["no_readonly_query"] == 41
    assert summary["decision"]["minimum_pairs"] == 48
    assert summary["decision"]["maximum_observed_pairs"] < 48
    assert summary["decision"]["training_allowed"] is False
    assert summary["decision"]["optimizer_initialized"] is False
    assert summary["decision"]["checkpoint_saved"] is False
    assert summary["decision"]["do_not_lower_pair_threshold"] is True
    assert summary["decision"]["do_not_use_no_sql_or_unobserved_calls_as_rejected"] is True
    assert summary["scope"][
        "contains_original_prompts_sql_answers_task_ids_tool_outputs_or_server_paths"
    ] is False

    for forbidden in ("SELECT ", "task_000", "/workspace/", "tool_call_id"):
        assert forbidden not in payload
