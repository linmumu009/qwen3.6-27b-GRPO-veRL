from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full25_summary_freezes_attribution_and_pareto_gate() -> None:
    path = ROOT / "docs" / "native_vs_step120_full25_attribution_20260813_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    payload = path.read_text(encoding="utf-8")

    assert summary["contract"] == "native-vs-step120-full25-attribution-safe-summary-v1"
    assert summary["scope"]["same_tasks_and_prompts"] is True
    assert summary["native"]["recognized_sqlite_tasks"] == 30
    assert summary["step120"]["recognized_sqlite_tasks"] == 23
    assert summary["native"]["wrong_process_ok_count"] == 13
    assert summary["step120"]["wrong_process_ok_count"] == 8
    assert summary["attribution"]["created_by_step120_training"] is False
    assert summary["attribution"]["amplified_by_step120_training"] is False
    assert summary["decision"]["training_allowed"] is False
    assert summary["decision"]["pair_gate_observed"] < summary["decision"]["pair_gate_required"]
    assert summary["decision"]["future_pareto_gate"] == {
        "recognized_sqlite_task_floor": 30,
        "complete_count_floor": 40,
        "correct_numeric_count_floor": 7,
        "reward_total_mean_floor": 0.285840625,
        "wrong_process_ok_ceiling": 8,
    }
    for forbidden in ("SELECT ", "task_000", "/workspace/", "tool_call_id"):
        assert forbidden not in payload
