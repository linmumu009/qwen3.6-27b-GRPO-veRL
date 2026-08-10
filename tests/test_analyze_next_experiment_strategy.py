from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_next_experiment_strategy import capacity_scenarios, runtime_and_experiments
from scripts.build_next_experiment_strategy_report import build as build_report


ROOT = Path(__file__).resolve().parents[1]


def test_96k_training_fits_but_24_sequence_rollout_does_not() -> None:
    capacity = capacity_scenarios(262_144)
    rows = capacity["scenario_rows"]
    target = next(
        row
        for row in rows
        if row["context_tokens"] == 98_304 and row["max_sequences_per_replica"] == 24
    )
    assert target["training_planning_peak_gib"] < capacity["usable_hbm_gib"]
    assert target["rollout_planning_fit"] is False
    assert capacity["cache_increment_48k_to_96k_per_full_sequence_gib"] == pytest.approx(0.75)


def test_runtime_projection_reproduces_observed_cost() -> None:
    step120 = json.loads((ROOT / "docs" / "step120_dense_trial_20260810_summary.json").read_text(encoding="utf-8"))
    result = runtime_and_experiments(step120)
    hundred = next(row for row in result["training_cost_projection"] if row["steps"] == 100)
    assert hundred["with_full_val_and_save_hours"] == pytest.approx(18.1416955229)
    assert result["ranked_experiments"][0]["training_steps"] == 0


def test_strategy_report_preserves_recommended_order() -> None:
    summary = json.loads((ROOT / "docs" / "next_experiment_strategy_20260810_summary.json").read_text(encoding="utf-8"))
    report = build_report(summary)
    assert report["manifest"]["title"] == "先修复收尾，再决定是否上96K"
    assert report["snapshot"]["datasets"]["experiment_table"][0]["实验"] == "48K 强制收尾哨兵集"
