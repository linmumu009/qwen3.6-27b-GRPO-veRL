#!/usr/bin/env python3
"""Authorize at most one chosen-only train48 canary from sealed baseline gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.prepare_chosen_only_schema_action_sft import CONTRACT
from scripts.prepare_repair_sft_dataset import sha256_file


RESULT_CONTRACT = "chosen-only-first-action-baseline-decision-v1"


def decide(
    dataset: dict[str, Any],
    tokenization: dict[str, Any],
    baseline: dict[str, Any],
    *,
    calibration_sha256: str,
) -> dict[str, Any]:
    if dataset.get("contract") != CONTRACT:
        raise ValueError("chosen-only baseline dataset contract mismatch")
    if (dataset.get("rows"), dataset.get("train_rows"), dataset.get("calibration_rows")) != (
        64,
        48,
        16,
    ):
        raise ValueError("chosen-only baseline split sizes drifted")
    if dataset.get("training_allowed") is not False:
        raise ValueError("source dataset unexpectedly authorizes training")
    expected_calibration = ((dataset.get("outputs") or {}).get("calibration") or {}).get(
        "sha256"
    )
    if calibration_sha256 != expected_calibration:
        raise ValueError("calibration parquet hash differs from dataset contract")

    required_token_flags = (
        "all_rows_tokenize_without_truncation",
        "all_rows_loss_exactly_one_assistant_tool_action",
        "all_nonassistant_context_loss_zero",
        "all_tool_structure_and_sql_masks_nonempty_disjoint_and_complete",
    )
    if tokenization.get("contract") != "chosen-only-schema-action-tokenization-gate-v1":
        raise ValueError("chosen-only tokenization contract mismatch")
    if int(tokenization.get("rows") or 0) != 64 or not all(
        tokenization.get(flag) is True for flag in required_token_flags
    ):
        raise ValueError("chosen-only tokenization gate did not pass")
    if tokenization.get("training_allowed") is not False:
        raise ValueError("tokenization gate unexpectedly authorizes training")

    if baseline.get("contract") != "repair-sft-teacher-forced-component-diagnostic-v3":
        raise ValueError("chosen-only teacher-forced baseline contract mismatch")
    if baseline.get("model_label") != "step120":
        raise ValueError("chosen-only baseline is not Step 120")
    if baseline.get("forward_only") is not True or baseline.get("optimizer_initialized") is not False:
        raise ValueError("chosen-only baseline was not forward-only")
    if int(baseline.get("task_count") or 0) != 16:
        raise ValueError("chosen-only baseline is not calibration16")
    if baseline.get("data_sha256") != calibration_sha256:
        raise ValueError("chosen-only baseline data hash differs from calibration16")
    components = baseline.get("components") or {}
    if set(components) != {"assistant", "tool_turn", "tool_structure", "sql_shell"}:
        raise ValueError("chosen-only baseline component set drifted")
    sql = components["sql_shell"]
    ranks = baseline.get("sql_token_rank") or {}
    sql_nll = float(sql.get("mean_nll"))
    token_count = int(ranks.get("token_count") or 0)
    greedy = int(ranks.get("greedy_token_count") or 0)
    top5 = int(ranks.get("top5_token_count") or 0)
    all_greedy_tasks = int(ranks.get("tasks_all_sql_tokens_greedy") or 0)
    if not math.isfinite(sql_nll) or sql_nll <= 0 or token_count <= 0:
        raise ValueError("chosen-only baseline SQL metrics are invalid")
    if not 0 <= greedy <= top5 <= token_count:
        raise ValueError("chosen-only baseline SQL rank counts are invalid")

    learnability_gap = all_greedy_tasks < 12 and sql_nll > 0.5
    canary_allowed = learnability_gap
    return {
        "contract": RESULT_CONTRACT,
        "scope": {
            "model": "step120",
            "train_rows": 48,
            "calibration_rows": 16,
            "baseline_forward_only": True,
            "optimizer_initialized": False,
            "checkpoint_saved": False,
            "oracle_relevant_table_selection": True,
            "deployment_ready": False,
        },
        "baseline": {
            "official_assistant_loss": float(baseline["official_assistant_loss"]),
            "tool_structure_mean_nll": float(components["tool_structure"]["mean_nll"]),
            "sql_mean_nll": sql_nll,
            "sql_token_count": token_count,
            "sql_greedy_token_count": greedy,
            "sql_top5_token_count": top5,
            "sql_mean_rank": float(ranks["mean_rank"]),
            "sql_max_rank": int(ranks["max_rank"]),
            "tasks_all_sql_tokens_greedy": all_greedy_tasks,
            "tasks_with_nongreedy_sql_token": int(
                ranks["tasks_with_nongreedy_sql_token"]
            ),
        },
        "one_step_canary": {
            "allowed": canary_allowed,
            "training_rows": 48,
            "training_steps": 1,
            "source_checkpoint": "step120_model_state",
            "optimizer_state": "new_cpu_offload_adam",
            "loss_weights": {"tool_structure": 0.25, "sql_payload": 8.0},
            "calibration_rows_excluded_from_training": 16,
            "save_model_checkpoint_only": True,
        },
        "post_canary_gates": {
            "calibration_sql_nll_relative_improvement_min": 0.05,
            "calibration_tasks_sql_nll_improved_min": 12,
            "calibration_sql_greedy_token_gain_min": 12,
            "calibration_sql_top5_token_count_min": top5,
            "calibration_sql_mean_rank_must_improve": True,
            "calibration_tool_structure_nll_relative_regression_max": 0.05,
            "earlier_template_or_sql_boundary_regressions_max": 0,
        },
        "decision": {
            "diagnosis": "correct_first_action_has_sparse_token_ranking_gap",
            "next_action": (
                "run_one_step_train48_then_forward_only_calibration16"
                if canary_allowed
                else "stop_chosen_only_training"
            ),
            "training_allowed": canary_allowed,
            "training_scope": "one_step_train48_only" if canary_allowed else "none",
            "free_rollout_allowed": False,
            "promotion_allowed": False,
        },
        "contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--tokenization-gate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--calibration-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decide(
        json.loads(args.dataset_contract.read_text(encoding="utf-8")),
        json.loads(args.tokenization_gate.read_text(encoding="utf-8")),
        json.loads(args.baseline.read_text(encoding="utf-8")),
        calibration_sha256=sha256_file(args.calibration_parquet),
    )
    result["source_sha256"] = {
        "dataset_contract": sha256_file(args.dataset_contract),
        "tokenization_gate": sha256_file(args.tokenization_gate),
        "baseline": sha256_file(args.baseline),
        "calibration_parquet": sha256_file(args.calibration_parquet),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
