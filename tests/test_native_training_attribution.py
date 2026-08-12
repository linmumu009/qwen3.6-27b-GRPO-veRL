from scripts.analyze_native_training_attribution import analyze
from pathlib import Path
import json
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _margin(mean, preferred, values):
    return {
        "execution": {"forward_only": True},
        "semantic_delta_margin": {
            "mean_margin": mean,
            "chosen_preferred": preferred,
        },
        "per_task": [
            {
                "task_id": f"task_{index:06d}",
                "semantic_delta_log_probability_margin_per_token": value,
            }
            for index, value in enumerate(values)
        ],
    }


def _boss_summary(*, wrong_ok, duplicate):
    return {
        "n": 16,
        "reward_total_mean": 0.5,
        "process_score_mean": 0.8,
        "correct_numeric_count": 2,
        "complete_count": 15,
        "n_sql_mean": 8.0,
        "dup_cmd_mean": duplicate,
        "verdict_fine_counts": {"result_wrong_process_ok": wrong_ok},
    }


def test_native_attribution_distinguishes_preexisting_failure_from_amplification():
    native_values = [-1.0] * 16
    trained_values = [-0.9] * 10 + [-1.1] * 6
    boss = {
        "task_ids_identical": True,
        "prompt_identity": {
            "task_ids_identical": True,
            "identical_prompt_count": 16,
        },
        "native": _boss_summary(wrong_ok=8, duplicate=12.0),
        "step120": _boss_summary(wrong_ok=7, duplicate=10.0),
    }

    result = analyze(
        _margin(-1.0, 0, native_values),
        _margin(-0.975, 0, trained_values),
        boss,
    )

    assert result["conditional_margin"]["preexisting_in_native"] is True
    assert result["conditional_margin"]["amplified_by_step120"] is False
    changes = result["conditional_margin"]["paired_task_margin_change"]
    assert changes["toward_correct"] == 10
    assert changes["toward_wrong"] == 6
    assert changes["ties"] == 0
    assert changes["mean_change"] == pytest.approx(0.025)
    assert changes["mean_absolute_change"] == pytest.approx(0.1)
    rollout = result["natural_rollout_boss_original"]
    assert rollout["preexisting_wrong_result_process_ok"] is True
    assert rollout["wrong_result_process_ok_amplified_by_step120"] is False
    assert rollout["duplicate_commands_amplified_by_step120"] is False
    assert result["interpretation"]["core_failure_created_by_training"] is False


def test_safe_native_attribution_summary_has_no_raw_assets():
    path = ROOT / "docs" / "native_vs_step120_reward_behavior_attribution_20260812_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    payload = path.read_text(encoding="utf-8")

    assert summary["conditional_margin"]["preexisting_in_native"] is True
    assert summary["conditional_margin"]["amplified_by_step120"] is False
    rollout = summary["natural_rollout_boss_original"]
    assert rollout["native"]["wrong_result_process_ok_count"] == 13
    assert rollout["step120"]["wrong_result_process_ok_count"] == 12
    assert rollout["wrong_result_process_ok_amplified_by_step120"] is False
    assert summary["interpretation"]["core_failure_created_by_training"] is False
    assert "/workspace/" not in payload
    assert "SELECT " not in payload
    assert "task_000" not in payload
