from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from scripts.analyze_chosen_only_first_action_post_canary import decide


ROOT = Path(__file__).resolve().parents[1]


def _result(label: str, sql_nll: float, greedy: int, top5: int, rank: float) -> dict:
    per_task = []
    for index in range(16):
        per_task.append(
            {
                "task_id": f"task_{index}",
                "components": {"sql_shell": {"mean_nll": sql_nll + index / 1000}},
                "sql_token_rank": {"first_nongreedy_offset": 5},
            }
        )
    return {
        "contract": "repair-sft-teacher-forced-component-diagnostic-v3",
        "model_label": label,
        "forward_only": True,
        "optimizer_initialized": False,
        "task_count": 16,
        "data_sha256": "cal",
        "task_ids": [f"task_{index}" for index in range(16)],
        "components": {
            "tool_structure": {"mean_nll": 1.4},
            "sql_shell": {"mean_nll": sql_nll},
        },
        "sql_token_rank": {
            "greedy_token_count": greedy,
            "top5_token_count": top5,
            "mean_rank": rank,
        },
        "per_task": per_task,
    }


def _authorization() -> dict:
    return {
        "contract": "chosen-only-first-action-baseline-decision-v1",
        "one_step_canary": {"allowed": True},
        "post_canary_gates": {
            "calibration_sql_nll_relative_improvement_min": 0.05,
            "calibration_tasks_sql_nll_improved_min": 12,
            "calibration_sql_greedy_token_gain_min": 12,
            "calibration_sql_top5_token_count_min": 344,
            "calibration_sql_mean_rank_must_improve": True,
            "calibration_tool_structure_nll_relative_regression_max": 0.05,
            "earlier_template_or_sql_boundary_regressions_max": 0,
        },
    }


def test_post_canary_opens_only_free_rollout_when_all_gates_pass() -> None:
    baseline = _result("step120", 1.30, 277, 344, 18.8)
    post = _result("chosen_only_post_step1", 1.20, 290, 350, 15.0)

    result = decide(baseline, post, _authorization())

    assert result["gate_passed"] is True
    assert result["decision"]["free_rollout_allowed"] is True
    assert result["decision"]["additional_training_allowed"] is False
    assert result["decision"]["promotion_allowed"] is False


def test_post_canary_stops_on_earlier_sql_boundary_regression() -> None:
    baseline = _result("step120", 1.30, 277, 344, 18.8)
    post = _result("chosen_only_post_step1", 1.20, 290, 350, 15.0)
    post = deepcopy(post)
    post["per_task"][0]["sql_token_rank"]["first_nongreedy_offset"] = 4

    result = decide(baseline, post, _authorization())

    assert result["checks"]["earlier_sql_boundary_regressions"]["passed"] is False
    assert result["gate_passed"] is False
    assert result["decision"]["free_rollout_allowed"] is False


def test_one_step_script_is_train48_only_and_model_only_checkpoint() -> None:
    script = (
        ROOT / "scripts" / "run_chosen_only_first_action_one_step.sh"
    ).read_text(encoding="utf-8")

    assert "chosen_only_schema_action_train48.parquet" in script
    assert 'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"' in script
    assert "total_training_steps=1" in script
    assert "calibration_rows_excluded=16" in script
    assert "optimizer_state=fresh_cpu_offloaded_adam" in script
    assert "+optim.override_optimizer_config.optimizer_cpu_offload=true" in script
    assert "'checkpoint.load_contents=[]'" in script
    assert "'checkpoint.save_contents=[model,extra]'" in script
    assert "promotion_allowed=false" in script
