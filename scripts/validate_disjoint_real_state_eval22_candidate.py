#!/usr/bin/env python3
"""Validate chosen-only eval22 comparison and emit a safe combined summary."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from scripts.compare_disjoint_real_state_evaluation import compare
from scripts.validate_disjoint_real_state_eval22_artifacts import sha256_file


MARGIN_CONTRACT = "disjoint-real-state-eval22-margin-baseline-v1"
COMPARISON_CONTRACT = "disjoint-real-state-eval22-future-candidate-comparison-v1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _index(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in result.get("per_task") or []:
        task = str(row.get("task_id") or "")
        if not task or task in output:
            raise ValueError(f"missing or duplicate candidate task: {task!r}")
        output[task] = row
    return output


def _safe_margin(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_delta": result["semantic_delta_margin"],
        "full_sql": result["full_sql_margin"],
    }


def validate(
    *,
    baseline_safe_summary_file: Path,
    baseline_margin_file: Path,
    candidate_token_gate_file: Path,
    candidate_diagnostic_file: Path,
    candidate_margin_file: Path,
    comparison_file: Path,
    candidate_experiment_contract_file: Path,
    candidate_exit_code_file: Path,
    candidate_started_at_file: Path,
    candidate_finished_at_file: Path,
    candidate_unused_checkpoint_dir: Path,
    npu_processes_after_completion: int,
) -> dict[str, Any]:
    baseline_safe = _load(baseline_safe_summary_file)
    baseline = _load(baseline_margin_file)
    token = _load(candidate_token_gate_file)
    diagnostic = _load(candidate_diagnostic_file)
    candidate = _load(candidate_margin_file)
    observed_comparison = _load(comparison_file)

    if baseline_safe.get("contract") != "disjoint-real-state-eval22-safe-summary-v1":
        raise ValueError("unexpected baseline safe summary")
    if baseline.get("contract") != MARGIN_CONTRACT or candidate.get("contract") != MARGIN_CONTRACT:
        raise ValueError("unexpected eval22 margin contract")
    if baseline.get("evaluated_model_label") != "step120_disjoint_real_state_eval22":
        raise ValueError("unexpected eval22 baseline model")
    if candidate.get("evaluated_model_label") != "chosen_only_train48_one_step":
        raise ValueError("unexpected eval22 candidate model")
    if baseline.get("selection_bias") != candidate.get("selection_bias"):
        raise ValueError("baseline/candidate selection contract differs")
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        if result.get("evaluation_only") is not True:
            raise ValueError(f"{label} is not evaluation-only")
        for key in ("may_be_used_as_training_data", "training_allowed", "promotion_allowed"):
            if result.get(key) is not False:
                raise ValueError(f"{label} is not fail closed: {key}")

    if token.get("contract") != "current-definition-disjoint-pair-evaluation-token-gate-v1":
        raise ValueError("candidate token gate contract mismatch")
    if token.get("evaluation_only") is not True or int(token.get("pairs") or 0) != 22:
        raise ValueError("candidate token gate identity differs")
    for key in (
        "all_delta_masks_nonempty",
        "all_delta_masks_subset_of_sql",
        "all_pairs_adjacent_chosen_then_rejected",
        "all_candidate_signs_and_pair_indices_match",
    ):
        if token.get(key) is not True:
            raise ValueError(f"candidate token gate failed: {key}")

    if diagnostic.get("contract") != "repair-sft-teacher-forced-component-diagnostic-v3":
        raise ValueError("candidate diagnostic contract mismatch")
    if diagnostic.get("forward_only") is not True or diagnostic.get("optimizer_initialized") is not False:
        raise ValueError("candidate diagnostic is not pure forward-only")
    if diagnostic.get("model_label") != "chosen_only_train48_one_step":
        raise ValueError("candidate diagnostic model label differs")
    if diagnostic.get("data_sha256") != baseline_safe["source_sha256"]["data_parquet"]:
        raise ValueError("candidate diagnostic does not use frozen eval22")

    baseline_rows = _index(baseline)
    candidate_rows = _index(candidate)
    if set(baseline_rows) != set(candidate_rows) or len(candidate_rows) != 22:
        raise ValueError("baseline/candidate eval22 task identities differ")
    semantic_margins = [
        float(row["semantic_delta_log_probability_margin_per_token"])
        for row in candidate_rows.values()
    ]
    full_sql_margins = [
        float(row["full_sql_log_probability_margin_per_token"])
        for row in candidate_rows.values()
    ]
    if not all(math.isfinite(value) for value in semantic_margins + full_sql_margins):
        raise ValueError("candidate margins contain non-finite values")
    if int(candidate["semantic_delta_margin"]["chosen_preferred"]) != sum(
        value > 0 for value in semantic_margins
    ):
        raise ValueError("candidate semantic chosen-preferred count differs")
    if int(candidate["full_sql_margin"]["chosen_preferred"]) != sum(
        value > 0 for value in full_sql_margins
    ):
        raise ValueError("candidate full-SQL chosen-preferred count differs")
    if not math.isclose(
        float(candidate["semantic_delta_margin"]["mean_margin"]),
        fmean(semantic_margins),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("candidate semantic mean margin differs")

    recomputed_comparison = compare(baseline, candidate)
    if recomputed_comparison != observed_comparison:
        raise ValueError("saved eval22 comparison differs from recomputation")
    if observed_comparison.get("contract") != COMPARISON_CONTRACT:
        raise ValueError("unexpected eval22 comparison contract")
    checks = observed_comparison["checks"]
    expected_checks = {
        "chosen_preferred": (3, 17, False),
        "per_task_margin_improved": (18, 18, True),
        "new_earlier_first_nongreedy_regressions": (0, 0, True),
    }
    for key, (observed, threshold, passed) in expected_checks.items():
        check = checks[key]
        actual_threshold = check.get("required_min", check.get("allowed_max"))
        if (int(check["observed"]), int(actual_threshold), bool(check["passed"])) != (
            observed,
            threshold,
            passed,
        ):
            raise ValueError(f"candidate comparison changed: {key}")
    if observed_comparison.get("gate_passed") is not False:
        raise ValueError("candidate unexpectedly passed eval22 gate")
    for key in ("full64_pareto_replay_allowed", "additional_training_allowed", "promotion_allowed"):
        if observed_comparison["decision"].get(key) is not False:
            raise ValueError(f"candidate comparison is not fail closed: {key}")

    if candidate_exit_code_file.read_text(encoding="utf-8").strip() != "0":
        raise ValueError("candidate forward-only run did not exit successfully")
    started = datetime.fromisoformat(candidate_started_at_file.read_text(encoding="utf-8").strip())
    finished = datetime.fromisoformat(candidate_finished_at_file.read_text(encoding="utf-8").strip())
    wall_seconds = int((finished - started).total_seconds())
    experiment = candidate_experiment_contract_file.read_text(encoding="utf-8")
    for line in (
        "evaluation_only=true",
        "may_be_used_as_training_data=false",
        "forward_only=true",
        "optimizer_initialized=false",
        "checkpoint_saved=false",
        "training_allowed=false",
        "promotion_allowed=false",
    ):
        if line not in experiment:
            raise ValueError(f"candidate experiment contract is missing {line}")
    checkpoint_files = (
        sum(1 for path in candidate_unused_checkpoint_dir.rglob("*") if path.is_file())
        if candidate_unused_checkpoint_dir.exists()
        else 0
    )
    if checkpoint_files:
        raise ValueError("candidate forward-only run wrote checkpoint files")
    if npu_processes_after_completion != 0:
        raise ValueError("candidate forward-only run left NPU processes")

    baseline_semantic = baseline["semantic_delta_margin"]
    candidate_semantic = candidate["semantic_delta_margin"]
    baseline_sql = baseline["full_sql_margin"]
    candidate_sql = candidate["full_sql_margin"]
    return {
        "contract": "disjoint-real-state-eval22-chosen-candidate-safe-summary-v2",
        "date": "2026-08-13",
        "scope": baseline_safe["scope"],
        "data_gate": baseline_safe["data_gate"],
        "token_gate": baseline_safe["token_gate"],
        "execution": {
            "step120": baseline_safe["execution"],
            "chosen_only_candidate": {
                "exit_code": 0,
                "wall_seconds": wall_seconds,
                "forward_only": True,
                "optimizer_initialized": False,
                "checkpoint_files": checkpoint_files,
                "npu_processes_after_completion": npu_processes_after_completion,
            },
        },
        "step120_margin": _safe_margin(baseline),
        "chosen_only_candidate_margin": _safe_margin(candidate),
        "paired_change": {
            "semantic_delta_chosen_preferred_change": int(
                candidate_semantic["chosen_preferred"]
            )
            - int(baseline_semantic["chosen_preferred"]),
            "semantic_delta_mean_margin_change": float(candidate_semantic["mean_margin"])
            - float(baseline_semantic["mean_margin"]),
            "semantic_delta_median_margin_change": float(candidate_semantic["median_margin"])
            - float(baseline_semantic["median_margin"]),
            "full_sql_chosen_preferred_change": int(candidate_sql["chosen_preferred"])
            - int(baseline_sql["chosen_preferred"]),
            "full_sql_mean_margin_change": float(candidate_sql["mean_margin"])
            - float(baseline_sql["mean_margin"]),
            "per_task_margin_improved": int(checks["per_task_margin_improved"]["observed"]),
            "new_earlier_first_nongreedy_regressions": int(
                checks["new_earlier_first_nongreedy_regressions"]["observed"]
            ),
        },
        "independent_replication": baseline_safe["independent_replication"],
        "candidate_gate": {
            "checks": checks,
            "gate_passed": False,
            "full64_pareto_replay_allowed": False,
            "additional_training_allowed": False,
            "promotion_allowed": False,
        },
        "decision": {
            "eval22_was_sufficient_to_reject_existing_candidate": True,
            "systematic_correct_vs_actual_wrong_misranking_remains": True,
            "expand_training_pair_data_to_at_least_48": True,
            "use_eval22_as_training_data": False,
            "training_now": False,
            "run_full64_now": False,
            "additional_chosen_only_steps": False,
            "promotion_allowed": False,
            "selected_next_action": (
                "freeze_eval22_and_expand_disjoint_real_state_training_pairs_to_at_least_48"
            ),
        },
        "caveats": [
            "eval22 is postselected on Step 120 failure and is not a population accuracy benchmark",
            "the chosen-only candidate improves margin on 18 of 22 states but does not increase the number of states preferring the correct candidate",
            "eval22 must remain evaluation-only; a separate disjoint pool is required for training",
        ],
        "source_sha256": {
            **baseline_safe["source_sha256"],
            "baseline_safe_summary": sha256_file(baseline_safe_summary_file),
            "candidate_token_gate": sha256_file(candidate_token_gate_file),
            "candidate_diagnostic": sha256_file(candidate_diagnostic_file),
            "candidate_margin": sha256_file(candidate_margin_file),
            "candidate_comparison": sha256_file(comparison_file),
            "candidate_experiment_contract": sha256_file(candidate_experiment_contract_file),
        },
        "contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-safe-summary", type=Path, required=True)
    parser.add_argument("--baseline-margin", type=Path, required=True)
    parser.add_argument("--candidate-token-gate", type=Path, required=True)
    parser.add_argument("--candidate-diagnostic", type=Path, required=True)
    parser.add_argument("--candidate-margin", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--candidate-experiment-contract", type=Path, required=True)
    parser.add_argument("--candidate-exit-code", type=Path, required=True)
    parser.add_argument("--candidate-started-at", type=Path, required=True)
    parser.add_argument("--candidate-finished-at", type=Path, required=True)
    parser.add_argument("--candidate-unused-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--npu-processes-after-completion", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        baseline_safe_summary_file=args.baseline_safe_summary,
        baseline_margin_file=args.baseline_margin,
        candidate_token_gate_file=args.candidate_token_gate,
        candidate_diagnostic_file=args.candidate_diagnostic,
        candidate_margin_file=args.candidate_margin,
        comparison_file=args.comparison,
        candidate_experiment_contract_file=args.candidate_experiment_contract,
        candidate_exit_code_file=args.candidate_exit_code,
        candidate_started_at_file=args.candidate_started_at,
        candidate_finished_at_file=args.candidate_finished_at,
        candidate_unused_checkpoint_dir=args.candidate_unused_checkpoint_dir,
        npu_processes_after_completion=args.npu_processes_after_completion,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
