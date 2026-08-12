from __future__ import annotations

import pytest

from scripts.analyze_chosen_only_first_action_baseline import decide
from scripts.prepare_chosen_only_schema_action_sft import CONTRACT


def _dataset() -> dict:
    return {
        "contract": CONTRACT,
        "rows": 64,
        "train_rows": 48,
        "calibration_rows": 16,
        "training_allowed": False,
        "outputs": {"calibration": {"sha256": "cal"}},
    }


def _tokenization() -> dict:
    return {
        "contract": "chosen-only-schema-action-tokenization-gate-v1",
        "rows": 64,
        "all_rows_tokenize_without_truncation": True,
        "all_rows_loss_exactly_one_assistant_tool_action": True,
        "all_nonassistant_context_loss_zero": True,
        "all_tool_structure_and_sql_masks_nonempty_disjoint_and_complete": True,
        "training_allowed": False,
    }


def _baseline(all_greedy: int = 0) -> dict:
    return {
        "contract": "repair-sft-teacher-forced-component-diagnostic-v3",
        "model_label": "step120",
        "forward_only": True,
        "optimizer_initialized": False,
        "task_count": 16,
        "data_sha256": "cal",
        "official_assistant_loss": 1.4,
        "components": {
            "assistant": {"mean_nll": 1.4},
            "tool_turn": {"mean_nll": 1.4},
            "tool_structure": {"mean_nll": 1.42},
            "sql_shell": {"mean_nll": 1.29},
        },
        "sql_token_rank": {
            "token_count": 381,
            "greedy_token_count": 277,
            "top5_token_count": 344,
            "mean_rank": 18.8,
            "max_rank": 2806,
            "tasks_all_sql_tokens_greedy": all_greedy,
            "tasks_with_nongreedy_sql_token": 16 - all_greedy,
        },
    }


def test_baseline_authorizes_only_one_step_train48_canary() -> None:
    result = decide(
        _dataset(), _tokenization(), _baseline(), calibration_sha256="cal"
    )

    assert result["one_step_canary"]["allowed"] is True
    assert result["one_step_canary"]["training_steps"] == 1
    assert result["one_step_canary"]["calibration_rows_excluded_from_training"] == 16
    assert result["decision"]["training_scope"] == "one_step_train48_only"
    assert result["decision"]["free_rollout_allowed"] is False
    assert result["decision"]["promotion_allowed"] is False


def test_baseline_stops_training_when_correct_sql_is_already_sufficient() -> None:
    result = decide(
        _dataset(), _tokenization(), _baseline(all_greedy=12), calibration_sha256="cal"
    )
    assert result["one_step_canary"]["allowed"] is False
    assert result["decision"]["training_allowed"] is False


def test_baseline_fails_closed_on_calibration_hash_drift() -> None:
    with pytest.raises(ValueError, match="calibration parquet hash"):
        decide(
            _dataset(), _tokenization(), _baseline(), calibration_sha256="wrong"
        )
