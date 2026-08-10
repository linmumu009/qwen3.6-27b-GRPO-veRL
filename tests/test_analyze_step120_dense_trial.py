from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_step120_dense_trial_report import build as build_report
from scripts.analyze_step120_dense_trial import (
    bootstrap_mean_ci,
    exact_sign_test_p,
    normalize_validation_row,
    paired_metric,
)


def test_exact_sign_test_uses_only_non_ties() -> None:
    assert exact_sign_test_p(7, 3) == 0.34375
    assert exact_sign_test_p(0, 0) is None


def test_bootstrap_interval_is_deterministic_and_ordered() -> None:
    first = bootstrap_mean_ci([-1.0, 0.0, 1.0], draws=2000)
    second = bootstrap_mean_ci([-1.0, 0.0, 1.0], draws=2000)
    assert first == second
    assert first[0] <= 0 <= first[1]


def test_normalize_validation_row_replays_dense30_formula(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.analyze_step120_dense_trial.extract_final_assistant_answer",
        lambda _: "answer",
    )
    monkeypatch.setattr(
        "scripts.analyze_step120_dense_trial.dense_final_answer_correctness",
        lambda *args: 0.5,
    )
    row = {
        "output": "unused",
        "gts": {
            "answer_type": "numeric",
            "expected_value_json": "1",
            "abs_tol": 0,
            "rel_tol": 0,
            "task_id": "task_1",
        },
        "boss_reward": 0.6,
        "evidence_reward": 0.2,
        "safe": 1,
        "valid_tool_protocol": 1,
        "gold_sql_verified": 1,
    }
    result = normalize_validation_row(row)
    assert result["base_score"] == pytest.approx(0.48)
    assert result["dense30_score"] == pytest.approx(0.486)


def test_paired_metric_reports_wins_losses_and_ties() -> None:
    left = {
        "a": {"metric": 0.0},
        "b": {"metric": 1.0},
        "c": {"metric": 1.0},
    }
    right = {
        "a": {"metric": 1.0},
        "b": {"metric": 0.0},
        "c": {"metric": 1.0},
    }
    result = paired_metric(left, right, "metric")
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["ties"] == 1
    assert result["mean_delta"] == 0.0


def test_report_payload_preserves_decision_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "docs" / "step120_dense_trial_20260810_summary.json").read_text(encoding="utf-8"))
    report = build_report(summary)
    assert report["surface"] == "report"
    assert report["snapshot"]["datasets"]["boss_total_scores"] == [
        {"checkpoint": "Step 100", "value": 0.44375, "task_count": 20},
        {"checkpoint": "Step 120", "value": 0.563745, "task_count": 20},
        {"checkpoint": "Step 200", "value": 0.399685, "task_count": 20},
    ]
    assert report["snapshot"]["datasets"]["runtime_table"][-1]["seconds"] == 89.0
