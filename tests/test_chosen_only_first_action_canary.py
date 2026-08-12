from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from scripts.analyze_chosen_only_first_action_post_canary import decide


ROOT = Path(__file__).resolve().parents[1]


def _result(label: str, sql_nll: float, greedy: int, top5: int, rank: float) -> dict:
    per_task = []
    for index in range(16):
        token = "SUM" if index < 9 else "SELECT"
        per_task.append(
            {
                "task_id": f"task_{index}",
                "components": {"sql_shell": {"mean_nll": sql_nll + index / 1000}},
                "sql_token_rank": {
                    "first_nongreedy_offset": 5,
                    "first_nongreedy_target_token": token,
                },
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
    assert result["first_nongreedy_boundary_diagnostic"]["baseline_family_counts"] == {
        "aggregation_function": 9,
        "query_start": 7,
    }
    assert result["first_nongreedy_boundary_diagnostic"]["transition_counts"] == {
        "same_first_offset": 16
    }


def test_post_canary_stops_on_earlier_sql_boundary_regression() -> None:
    baseline = _result("step120", 1.30, 277, 344, 18.8)
    post = _result("chosen_only_post_step1", 1.20, 290, 350, 15.0)
    post = deepcopy(post)
    post["per_task"][0]["sql_token_rank"]["first_nongreedy_offset"] = 4

    result = decide(baseline, post, _authorization())

    assert result["checks"]["earlier_sql_boundary_regressions"]["passed"] is False
    assert result["gate_passed"] is False
    assert result["decision"]["free_rollout_allowed"] is False


def test_post_canary_treats_loss_of_an_all_greedy_sql_as_regression() -> None:
    baseline = _result("step120", 1.30, 277, 344, 18.8)
    post = _result("chosen_only_post_step1", 1.20, 290, 350, 15.0)
    baseline["per_task"][0]["sql_token_rank"] = {
        "first_nongreedy_offset": None,
        "first_nongreedy_target_token": None,
    }

    result = decide(baseline, post, _authorization())

    assert result["checks"]["earlier_sql_boundary_regressions"]["observed"] == 1
    assert result["gate_passed"] is False


def test_one_step_script_is_train48_only_and_model_only_checkpoint() -> None:
    script = (
        ROOT / "scripts" / "run_chosen_only_first_action_one_step.sh"
    ).read_text(encoding="utf-8")

    assert "chosen_only_schema_action_train48.parquet" in script
    assert 'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"' in script
    assert "total_training_steps=1" in script
    assert "calibration_rows_excluded=16" in script
    assert "optimizer_state=fresh_cpu_offloaded_adam" in script
    assert (
        'export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}:${PROJECT_ROOT}/runtime:'
        '${PROJECT_ROOT}:/verl:${PYTHONPATH:-}"' in script
    )
    assert "+optim.override_optimizer_config.optimizer_cpu_offload=true" in script
    assert "'checkpoint.load_contents=[]'" in script
    assert "'checkpoint.save_contents=[model,extra]'" in script
    assert "promotion_allowed=false" in script


def test_actual_canary_summary_is_fail_closed_and_safe() -> None:
    path = ROOT / "docs" / "chosen_only_first_action_canary_20260813_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))

    assert summary["contract"] == "chosen-only-first-action-canary-summary-v1"
    assert summary["training"]["exit_code"] == 0
    assert summary["training"]["checkpoint"]["model_distcp_shards"] == 32
    assert summary["training"]["checkpoint"]["optimizer_files"] == 0
    assert summary["checks"]["sql_nll_relative_improvement"]["passed"] is True
    assert summary["checks"]["tasks_sql_nll_improved"]["observed"] == 16
    assert summary["checks"]["sql_greedy_token_gain"] == {
        "observed": 5,
        "required_min": 12,
        "passed": False,
    }
    assert summary["first_nongreedy_boundary_diagnostic"][
        "aggregation_original_barrier_cleared"
    ] == 0
    assert summary["gate_passed"] is False
    assert summary["decision"]["free_rollout_allowed"] is False
    assert summary["decision"]["additional_training_allowed"] is False
    assert summary["decision"]["promotion_allowed"] is False
    assert summary["contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths"] is False
    serialized = json.dumps(summary, sort_keys=True)
    assert "/workspace/" not in serialized
    assert "/data3/" not in serialized
