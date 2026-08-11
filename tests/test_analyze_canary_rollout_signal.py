import json
from pathlib import Path

import pytest

from scripts.analyze_canary_rollout_signal import analyze


def _write_group(path: Path, flags: list[int], scores: list[float]) -> None:
    rows = [
        {
            "input": f"prompt-{path.stem}",
            "final_answer_correct": flag,
            "score": score,
            "has_final_answer": 1,
            "sql_evidence_correct": flag,
            "online_eligible": 1,
        }
        for flag, score in zip(flags, scores)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_analyze_reports_correctness_density_and_strict_reward_order(tmp_path: Path):
    _write_group(tmp_path / "122.jsonl", [0] * 8, [0.2] * 8)
    _write_group(
        tmp_path / "123.jsonl",
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0.8, 0.2, 0.3, 0.1, 0.2, 0.2, 0.1, 0.3],
    )
    _write_group(tmp_path / "124.jsonl", [1] * 8, [0.9] * 8)

    result = analyze(tmp_path, 122, 123, expected_group_size=8)

    assert result["row_count"] == 16
    assert result["valid_group_count"] == 2
    assert result["mixed_correct_group_count"] == 1
    assert result["all_wrong_group_count"] == 1
    assert result["correct_row_count"] == 1
    assert result["correct_row_rate"] == pytest.approx(1 / 16)
    assert result["mixed_correct_strict_rank_rate"] == 1.0
    assert result["mixed_correct_rows_mean"] == 1.0
    assert result["mixed_correct_min_margin_mean"] == pytest.approx(0.5)
    assert result["mixed_reward_range_mean"] == pytest.approx(0.7)
    assert result["all_wrong_reward_range_mean"] == 0.0
