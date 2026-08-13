#!/usr/bin/env python3
"""Validate eval22 artifacts and emit a sensitive-data-free decision summary."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import pandas as pd


DATA_CONTRACT = "current-definition-disjoint-first-error-evaluation-v1"
TOKEN_CONTRACT = "current-definition-disjoint-pair-evaluation-token-gate-v1"
MARGIN_CONTRACT = "disjoint-real-state-eval22-margin-baseline-v1"
FROZEN16_CONTRACT = "semantic-delta-pairwise-canary-safe-summary-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _index(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get(key) or "")
        if not identity or identity in output:
            raise ValueError(f"missing or duplicate {key}: {identity!r}")
        output[identity] = row
    return output


def _close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} differs: {observed} != {expected}")


def validate(
    *,
    data_file: Path,
    dataset_contract_file: Path,
    token_gate_file: Path,
    diagnostic_file: Path,
    margin_file: Path,
    experiment_contract_file: Path,
    exit_code_file: Path,
    started_at_file: Path,
    finished_at_file: Path,
    unused_checkpoint_dir: Path,
    frozen16_summary_file: Path,
    npu_processes_after_completion: int,
) -> dict[str, Any]:
    contract = _load(dataset_contract_file)
    token = _load(token_gate_file)
    diagnostic = _load(diagnostic_file)
    margin = _load(margin_file)
    frozen16 = _load(frozen16_summary_file)

    if contract.get("contract") != DATA_CONTRACT:
        raise ValueError("unexpected eval22 data contract")
    pairs = int(contract.get("pairs") or 0)
    rows = int(contract.get("rows") or 0)
    if pairs != 22 or rows != 44 or int(contract.get("expected_pairs") or 0) != 22:
        raise ValueError("eval22 pair/row identity changed")
    for key in ("pair_evaluation_gate_passed", "evaluation_only"):
        if contract.get(key) is not True:
            raise ValueError(f"eval22 data contract failed: {key}")
    for key in ("may_be_used_as_training_data", "training_allowed", "promotion_allowed"):
        if contract.get(key) is not False:
            raise ValueError(f"eval22 data contract is not fail closed: {key}")
    if contract.get("output_sha256") != sha256_file(data_file):
        raise ValueError("eval22 Parquet hash differs from contract")

    frame = pd.read_parquet(data_file)
    if len(frame) != rows:
        raise ValueError("eval22 Parquet row count differs from contract")
    required = {"source_task_id", "candidate_label", "pair_index"}
    if not required.issubset(frame.columns):
        raise ValueError("eval22 Parquet is missing pair identity columns")
    evidence = _index(list(contract.get("evidence") or []), "task_id")
    if len(evidence) != pairs:
        raise ValueError("eval22 evidence task count differs")
    observed_tasks = {str(value) for value in frame["source_task_id"]}
    if observed_tasks != set(evidence):
        raise ValueError("eval22 Parquet and evidence task identities differ")
    for position, row in frame.iterrows():
        expected_index = position // 2
        expected_label = "chosen" if position % 2 == 0 else "rejected"
        if int(row["pair_index"]) != expected_index or str(row["candidate_label"]) != expected_label:
            raise ValueError("eval22 Parquet pair order changed")

    if token.get("contract") != TOKEN_CONTRACT or token.get("evaluation_only") is not True:
        raise ValueError("eval22 token gate contract mismatch")
    if int(token.get("pairs") or 0) != pairs or int(token.get("rows") or 0) != rows:
        raise ValueError("eval22 token gate grain differs")
    for key in (
        "all_delta_masks_nonempty",
        "all_delta_masks_subset_of_sql",
        "all_pairs_adjacent_chosen_then_rejected",
        "all_candidate_signs_and_pair_indices_match",
    ):
        if token.get(key) is not True:
            raise ValueError(f"eval22 token gate failed: {key}")
    samples = list(token.get("samples") or [])
    if len(samples) != rows:
        raise ValueError("eval22 token sample count differs")
    if Counter(str(row["candidate_label"]) for row in samples) != Counter(
        {"chosen": pairs, "rejected": pairs}
    ):
        raise ValueError("eval22 token sample labels are imbalanced")

    if diagnostic.get("contract") != "repair-sft-teacher-forced-component-diagnostic-v3":
        raise ValueError("eval22 diagnostic contract mismatch")
    if diagnostic.get("forward_only") is not True or diagnostic.get("optimizer_initialized") is not False:
        raise ValueError("eval22 diagnostic is not pure forward-only")
    if diagnostic.get("data_sha256") != contract.get("output_sha256"):
        raise ValueError("eval22 diagnostic dataset identity differs")
    diagnostic_rows = _index(list(diagnostic.get("per_task") or []), "task_id")
    expected_candidates = {
        f"{task}::{label}" for task in evidence for label in ("chosen", "rejected")
    }
    if set(diagnostic_rows) != expected_candidates:
        raise ValueError("eval22 diagnostic candidate identities differ")

    if margin.get("contract") != MARGIN_CONTRACT or int(margin.get("task_count") or 0) != pairs:
        raise ValueError("eval22 margin contract mismatch")
    if margin.get("evaluation_only") is not True:
        raise ValueError("eval22 margin is not evaluation-only")
    for key in ("may_be_used_as_training_data", "training_allowed", "promotion_allowed"):
        if margin.get(key) is not False:
            raise ValueError(f"eval22 margin is not fail closed: {key}")
    margin_rows = _index(list(margin.get("per_task") or []), "task_id")
    if set(margin_rows) != set(evidence):
        raise ValueError("eval22 margin task identities differ")
    semantic_margins = [
        float(row["semantic_delta_log_probability_margin_per_token"])
        for row in margin_rows.values()
    ]
    full_sql_margins = [
        float(row["full_sql_log_probability_margin_per_token"])
        for row in margin_rows.values()
    ]
    if not all(math.isfinite(value) for value in semantic_margins + full_sql_margins):
        raise ValueError("eval22 margins contain non-finite values")
    semantic = margin["semantic_delta_margin"]
    full_sql = margin["full_sql_margin"]
    if int(semantic["chosen_preferred"]) != sum(value > 0 for value in semantic_margins):
        raise ValueError("eval22 semantic chosen-preferred count differs")
    if int(full_sql["chosen_preferred"]) != sum(value > 0 for value in full_sql_margins):
        raise ValueError("eval22 full-SQL chosen-preferred count differs")
    _close(float(semantic["mean_margin"]), fmean(semantic_margins), "semantic mean margin")
    _close(float(full_sql["mean_margin"]), fmean(full_sql_margins), "full-SQL mean margin")
    if sum(int(value["tasks"]) for value in semantic["by_critical_token_family"].values()) != pairs:
        raise ValueError("eval22 critical-token family counts do not sum to 22")

    if exit_code_file.read_text(encoding="utf-8").strip() != "0":
        raise ValueError("eval22 run did not exit successfully")
    started = datetime.fromisoformat(started_at_file.read_text(encoding="utf-8").strip())
    finished = datetime.fromisoformat(finished_at_file.read_text(encoding="utf-8").strip())
    wall_seconds = int((finished - started).total_seconds())
    experiment_contract = experiment_contract_file.read_text(encoding="utf-8")
    for line in (
        "evaluation_only=true",
        "may_be_used_as_training_data=false",
        "forward_only=true",
        "optimizer_initialized=false",
        "checkpoint_saved=false",
        "training_allowed=false",
        "promotion_allowed=false",
    ):
        if line not in experiment_contract:
            raise ValueError(f"eval22 experiment contract is missing {line}")
    checkpoint_files = (
        sum(1 for path in unused_checkpoint_dir.rglob("*") if path.is_file())
        if unused_checkpoint_dir.exists()
        else 0
    )
    if checkpoint_files:
        raise ValueError("eval22 forward-only run unexpectedly wrote checkpoint files")
    if npu_processes_after_completion != 0:
        raise ValueError("eval22 left active NPU processes")

    if frozen16.get("contract") != FROZEN16_CONTRACT:
        raise ValueError("unexpected frozen16 comparison source")
    frozen_gate = frozen16["probability_gate"]
    frozen_family = frozen16["by_critical_token_family"]["aggregation_function"]
    combined_tasks = pairs + int(frozen16["scope"]["task_count"])
    combined_chosen = int(semantic["chosen_preferred"]) + int(
        frozen_gate["baseline_chosen_preferred"]
    )
    combined_mean = (
        pairs * float(semantic["mean_margin"])
        + int(frozen16["scope"]["task_count"]) * float(frozen_gate["baseline_mean_margin"])
    ) / combined_tasks

    sequence_tokens = [int(row["sequence_tokens"]) for row in samples]
    sql_tokens = [int(row["sql_tokens"]) for row in samples]
    delta_tokens = [int(row["semantic_delta_tokens"]) for row in samples]
    return {
        "contract": "disjoint-real-state-eval22-safe-summary-v1",
        "date": "2026-08-13",
        "scope": {
            "source_checkpoint": "step120",
            "pairs": pairs,
            "rows": rows,
            "source_candidate_tasks": int(contract["source_candidate_rows"]),
            "evaluation_only": True,
            "postselected_on_step120_failure": True,
            "population_representative": False,
            "forbidden_frozen16_val20_test20_overlap": int(
                contract["forbidden_frozen16_val20_test20_overlap"]
            ),
        },
        "data_gate": {
            "unique_source_tasks": len(observed_tasks),
            "chosen_rows": pairs,
            "rejected_rows": pairs,
            "first_error_category_counts": contract["first_error_category_counts"],
            "exclusion_counts": contract["exclusion_counts"],
            "chosen_queries_mechanically_verified": True,
            "rejected_queries_are_actual_step120_first_errors": True,
            "all_first_error_tool_results_observed": True,
            "pair_prefix_identical_through_observed_error_result": True,
        },
        "token_gate": {
            "rows": rows,
            "pairs": pairs,
            "sequence_tokens_min": min(sequence_tokens),
            "sequence_tokens_max": max(sequence_tokens),
            "sql_tokens_min": min(sql_tokens),
            "sql_tokens_max": max(sql_tokens),
            "semantic_delta_tokens_min": min(delta_tokens),
            "semantic_delta_tokens_max": max(delta_tokens),
            "all_delta_masks_nonempty": True,
            "all_delta_masks_subset_of_sql": True,
            "all_pairs_adjacent_chosen_then_rejected": True,
            "truncation": "error",
        },
        "execution": {
            "exit_code": 0,
            "wall_seconds": wall_seconds,
            "forward_only": True,
            "optimizer_initialized": False,
            "checkpoint_files": checkpoint_files,
            "npu_processes_after_completion": npu_processes_after_completion,
        },
        "step120_margin": {
            "semantic_delta": semantic,
            "full_sql": full_sql,
        },
        "independent_replication": {
            "frozen16_tasks": int(frozen16["scope"]["task_count"]),
            "frozen16_baseline_chosen_preferred": int(frozen_gate["baseline_chosen_preferred"]),
            "frozen16_baseline_mean_margin": float(frozen_gate["baseline_mean_margin"]),
            "combined_disjoint_tasks": combined_tasks,
            "combined_chosen_preferred": combined_chosen,
            "combined_chosen_preferred_rate": combined_chosen / combined_tasks,
            "combined_task_weighted_mean_margin": combined_mean,
            "combined_aggregation_tasks": int(frozen_family["tasks"])
            + int(semantic["by_critical_token_family"]["aggregation_function"]["tasks"]),
            "combined_aggregation_chosen_preferred": int(
                semantic["by_critical_token_family"]["aggregation_function"]["chosen_preferred"]
            ),
        },
        "decision": {
            "systematic_correct_vs_actual_wrong_misranking_observed": True,
            "expand_training_pair_data_to_at_least_48": True,
            "use_eval22_as_training_data": False,
            "training_now": False,
            "promotion_allowed": False,
            "future_candidate_gate": margin["future_candidate_gate"],
        },
        "caveats": [
            "eval22 is postselected on Step 120 failure and is not a population accuracy benchmark",
            "eval22 establishes a baseline and training-target diagnosis; margin improvement requires a future candidate",
            "passing the future eval22 gate still requires the prespecified same64 full25 Pareto replay before promotion",
        ],
        "source_sha256": {
            "data_parquet": sha256_file(data_file),
            "dataset_contract": sha256_file(dataset_contract_file),
            "token_gate": sha256_file(token_gate_file),
            "diagnostic": sha256_file(diagnostic_file),
            "margin_baseline": sha256_file(margin_file),
            "experiment_contract": sha256_file(experiment_contract_file),
            "frozen16_summary": sha256_file(frozen16_summary_file),
        },
        "contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--token-gate", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--margin", type=Path, required=True)
    parser.add_argument("--experiment-contract", type=Path, required=True)
    parser.add_argument("--exit-code", type=Path, required=True)
    parser.add_argument("--started-at", type=Path, required=True)
    parser.add_argument("--finished-at", type=Path, required=True)
    parser.add_argument("--unused-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--frozen16-summary", type=Path, required=True)
    parser.add_argument("--npu-processes-after-completion", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        data_file=args.data_file,
        dataset_contract_file=args.dataset_contract,
        token_gate_file=args.token_gate,
        diagnostic_file=args.diagnostic,
        margin_file=args.margin,
        experiment_contract_file=args.experiment_contract,
        exit_code_file=args.exit_code,
        started_at_file=args.started_at,
        finished_at_file=args.finished_at,
        unused_checkpoint_dir=args.unused_checkpoint_dir,
        frozen16_summary_file=args.frozen16_summary,
        npu_processes_after_completion=args.npu_processes_after_completion,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
