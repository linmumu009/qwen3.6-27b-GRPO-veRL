import json
from pathlib import Path

from scripts.build_boss_exact_step100_step200_report import build_artifact


ROOT = Path(__file__).parents[1]


def _load(name):
    return json.loads((ROOT / "docs" / name).read_text(encoding="utf-8"))


def test_report_preserves_comparison_and_adds_driver_diagnosis():
    artifact = build_artifact(
        _load("boss_exact_step100_step200_20260807_summary.json"),
        _load("boss_exact_step200_20260807_adapter_summary.json"),
        _load("boss_exact_step100_step200_20260807_audit.json"),
        _load("boss_exact_step100_step200_20260807_diagnosis.json"),
        _load("boss_exact_step100_step200_20260807_training_signal.json"),
        _load("boss_exact_step100_step200_20260807_failure_review.json"),
        _load("boss_exact_step100_step200_20260807_runtime_audit.json"),
    )

    block_ids = {block["id"] for block in artifact["manifest"]["blocks"]}
    assert "score_chart_block" in block_ids
    assert "driver_contribution_block" in block_ids
    assert "failure_task_table_block" in block_ids
    assert "training_signal_chart_block" in block_ids
    assert sum(
        row["reward_sum_delta"]
        for row in artifact["snapshot"]["datasets"]["driver_contributions"]
    ) == -0.8813
    assert len(artifact["snapshot"]["datasets"]["failure_rows"]) == 6
