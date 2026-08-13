#!/usr/bin/env python3
"""Validate native-pair screen artifacts and emit an aggregate-only summary."""

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


DATA_CONTRACT = "current-definition-native-first-error-training-candidates-v1"
TOKEN_CONTRACT = "current-definition-disjoint-pair-candidate-token-gate-v1"
MARGIN_CONTRACT = "native-disjoint-real-state-step120-margin-screen-v1"


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
    npu_processes_after_completion: int,
) -> dict[str, Any]:
    contract = _load(dataset_contract_file)
    token = _load(token_gate_file)
    diagnostic = _load(diagnostic_file)
    margin = _load(margin_file)
    if contract.get("contract") != DATA_CONTRACT:
        raise ValueError("unexpected native candidate data contract")
    pairs = int(contract.get("pairs") or 0)
    rows = int(contract.get("rows") or 0)
    if pairs <= 0 or pairs != int(contract.get("expected_pairs") or 0) or rows != 2 * pairs:
        raise ValueError("native candidate pair/row grain differs")
    for key in ("candidate_pair_gate_passed", "candidate_only"):
        if contract.get(key) is not True:
            raise ValueError(f"native candidate data gate failed: {key}")
    for key in ("may_be_used_as_training_data", "training_allowed", "promotion_allowed"):
        if contract.get(key) is not False:
            raise ValueError(f"native candidate data is not fail closed: {key}")
    if contract.get("output_sha256") != sha256_file(data_file):
        raise ValueError("native candidate Parquet hash differs")

    frame = pd.read_parquet(data_file)
    if len(frame) != rows:
        raise ValueError("native candidate Parquet row count differs")
    evidence = _index(list(contract.get("evidence") or []), "task_id")
    if len(evidence) != pairs or set(map(str, frame["source_task_id"])) != set(evidence):
        raise ValueError("native candidate Parquet/evidence identities differ")
    for position, row in frame.iterrows():
        if int(row["pair_index"]) != position // 2:
            raise ValueError("native candidate pair index changed")
        expected_label = "chosen" if position % 2 == 0 else "rejected"
        if str(row["candidate_label"]) != expected_label:
            raise ValueError("native candidate row order changed")

    if token.get("contract") != TOKEN_CONTRACT or token.get("candidate_only") is not True:
        raise ValueError("native candidate token contract mismatch")
    if int(token.get("pairs") or 0) != pairs or int(token.get("rows") or 0) != rows:
        raise ValueError("native candidate token grain differs")
    for key in (
        "all_delta_masks_nonempty",
        "all_delta_masks_subset_of_sql",
        "all_pairs_adjacent_chosen_then_rejected",
        "all_candidate_signs_and_pair_indices_match",
    ):
        if token.get(key) is not True:
            raise ValueError(f"native candidate token gate failed: {key}")
    samples = list(token.get("samples") or [])
    if len(samples) != rows or Counter(str(row["candidate_label"]) for row in samples) != Counter(
        {"chosen": pairs, "rejected": pairs}
    ):
        raise ValueError("native candidate token samples differ")

    if diagnostic.get("contract") != "repair-sft-teacher-forced-component-diagnostic-v3":
        raise ValueError("native candidate diagnostic contract mismatch")
    if diagnostic.get("forward_only") is not True or diagnostic.get("optimizer_initialized") is not False:
        raise ValueError("native candidate diagnostic is not pure forward-only")
    if diagnostic.get("data_sha256") != contract.get("output_sha256"):
        raise ValueError("native candidate diagnostic dataset identity differs")
    diagnostic_rows = _index(list(diagnostic.get("per_task") or []), "task_id")
    expected_candidates = {f"{task}::{label}" for task in evidence for label in ("chosen", "rejected")}
    if set(diagnostic_rows) != expected_candidates:
        raise ValueError("native candidate diagnostic identities differ")

    if margin.get("contract") != MARGIN_CONTRACT or int(margin.get("task_count") or 0) != pairs:
        raise ValueError("native candidate margin contract mismatch")
    for key in ("may_be_used_as_training_data", "training_allowed", "promotion_allowed"):
        if margin.get(key) is not False:
            raise ValueError(f"native candidate margin is not fail closed: {key}")
    margin_rows = _index(list(margin.get("per_task") or []), "task_id")
    if set(margin_rows) != set(evidence):
        raise ValueError("native candidate margin identities differ")
    semantic_margins = [
        float(row["semantic_delta_log_probability_margin_per_token"])
        for row in margin_rows.values()
    ]
    full_margins = [
        float(row["full_sql_log_probability_margin_per_token"])
        for row in margin_rows.values()
    ]
    if not all(math.isfinite(value) for value in semantic_margins + full_margins):
        raise ValueError("native candidate margin contains non-finite values")
    semantic = margin["semantic_delta_margin"]
    full_sql = margin["full_sql_margin"]
    if int(semantic["chosen_preferred"]) != sum(value > 0 for value in semantic_margins):
        raise ValueError("native semantic chosen-preferred count differs")
    if int(full_sql["chosen_preferred"]) != sum(value > 0 for value in full_margins):
        raise ValueError("native full-SQL chosen-preferred count differs")
    if not math.isclose(float(semantic["mean_margin"]), fmean(semantic_margins), abs_tol=1e-12):
        raise ValueError("native semantic mean margin differs")
    if not math.isclose(float(full_sql["mean_margin"]), fmean(full_margins), abs_tol=1e-12):
        raise ValueError("native full-SQL mean margin differs")

    if exit_code_file.read_text(encoding="utf-8").strip() != "0":
        raise ValueError("native candidate run did not exit successfully")
    started = datetime.fromisoformat(started_at_file.read_text(encoding="utf-8").strip())
    finished = datetime.fromisoformat(finished_at_file.read_text(encoding="utf-8").strip())
    wall_seconds = int((finished - started).total_seconds())
    experiment = experiment_contract_file.read_text(encoding="utf-8")
    for line in (
        "candidate_only=true",
        "may_be_used_as_training_data=false",
        "forward_only=true",
        "optimizer_initialized=false",
        "checkpoint_saved=false",
        "training_allowed=false",
        "promotion_allowed=false",
    ):
        if line not in experiment:
            raise ValueError(f"native experiment contract is missing {line}")
    checkpoint_files = (
        sum(path.is_file() for path in unused_checkpoint_dir.rglob("*"))
        if unused_checkpoint_dir.exists()
        else 0
    )
    if checkpoint_files:
        raise ValueError("native forward-only run unexpectedly wrote checkpoint files")
    if npu_processes_after_completion != 0:
        raise ValueError("native forward-only run left active NPU processes")

    sequence = [int(row["sequence_tokens"]) for row in samples]
    sql = [int(row["sql_tokens"]) for row in samples]
    delta = [int(row["semantic_delta_tokens"]) for row in samples]
    return {
        "contract": "native-disjoint-real-state-step120-margin-safe-summary-v1",
        "date": "2026-08-13",
        "scope": {
            "state_source_checkpoint": "native_base",
            "evaluated_model_checkpoint": "step120",
            "pairs": pairs,
            "rows": rows,
            "source_first_error_pairs_before_frozen_exclusion": int(
                contract["source_first_error_pairs_before_frozen_exclusion"]
            ),
            "frozen_first_error_pairs_excluded": int(contract["frozen_first_error_pairs_excluded"]),
            "candidate_only": True,
        },
        "data_gate": {
            "unique_source_tasks": len(evidence),
            "first_error_category_counts": contract["first_error_category_counts"],
            "forbidden_set_audits": contract["forbidden_set_audits"],
            "chosen_queries_mechanically_verified": True,
            "rejected_queries_are_actual_native_first_errors": True,
            "all_first_error_tool_results_observed": True,
            "pair_prefix_identical_through_observed_error_result": True,
        },
        "token_gate": {
            "sequence_tokens_min": min(sequence),
            "sequence_tokens_max": max(sequence),
            "sql_tokens_min": min(sql),
            "sql_tokens_max": max(sql),
            "semantic_delta_tokens_min": min(delta),
            "semantic_delta_tokens_max": max(delta),
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
        "step120_margin": {"semantic_delta": semantic, "full_sql": full_sql},
        "decision": margin["screening_decision"],
        "caveats": [
            "states were generated by the native model and scored under Step 120, so they are an off-policy source stratum",
            "seven pairs are a training-supply screen, not a population benchmark or sufficient standalone training set",
            "retention only reduces the future supply gap; it does not authorize training below the frozen 48-pair gate",
        ],
        "source_sha256": {
            "data_parquet": sha256_file(data_file),
            "dataset_contract": sha256_file(dataset_contract_file),
            "token_gate": sha256_file(token_gate_file),
            "diagnostic": sha256_file(diagnostic_file),
            "margin_screen": sha256_file(margin_file),
            "experiment_contract": sha256_file(experiment_contract_file),
        },
        "contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths": False,
        "may_be_used_as_training_data": False,
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
        npu_processes_after_completion=args.npu_processes_after_completion,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
