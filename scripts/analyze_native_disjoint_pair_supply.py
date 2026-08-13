#!/usr/bin/env python3
"""Audit whether existing native full25 errors can supply non-eval training states.

The output is deliberately aggregate-only.  It never emits task identities,
prompts, SQL, answers, or tool results, and it does not authorize training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from scripts.analyze_disjoint_first_query_outcomes import classify_first_query_outcomes
from scripts.analyze_repair_sft_free_run_divergence import read_openai
from scripts.prepare_repair_sft_dataset import load_parquet_rows, sha256_file, task_id


CONTRACT = "native-disjoint-real-state-supply-audit-v1"
EVAL_CONTRACT = "current-definition-disjoint-first-error-evaluation-v1"


def forbidden_task_id(row: dict[str, Any]) -> str:
    """Prefer explicit source identity for derived pair/calibration assets."""
    source = str(row.get("source_task_id") or "")
    if source:
        return source
    truth_identity = task_id(row)
    if truth_identity:
        return truth_identity
    display = str(row.get("task_id") or "")
    return display.split("::", 1)[0] if display else ""


def eval_task_ids(contract: dict[str, Any]) -> set[str]:
    if contract.get("contract") != EVAL_CONTRACT:
        raise ValueError("eval22 contract mismatch")
    if contract.get("evaluation_only") is not True:
        raise ValueError("eval22 is not evaluation-only")
    if contract.get("may_be_used_as_training_data") is not False:
        raise ValueError("eval22 training prohibition is missing")
    task_ids = {
        str(row.get("task_id") or "") for row in contract.get("evidence") or []
    }
    task_ids.discard("")
    if len(task_ids) != int(contract.get("pairs") or -1):
        raise ValueError("eval22 task identities do not match pair count")
    return task_ids


def summarize_supply(
    classified: dict[str, dict[str, str]],
    frozen_eval_task_ids: set[str],
    additional_forbidden_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    additional_forbidden_task_ids = additional_forbidden_task_ids or set()
    if not frozen_eval_task_ids.issubset(classified):
        raise ValueError("eval22 contains tasks absent from native full25")
    native_errors = {
        task_id
        for task_id, row in classified.items()
        if row.get("outcome") == "observed_first_query_error"
    }
    overlap = native_errors & frozen_eval_task_ids
    outside = native_errors - frozen_eval_task_ids
    additional_overlap = outside & additional_forbidden_task_ids
    recoverable = outside - additional_forbidden_task_ids
    categories = Counter(
        classified[current_task_id].get("first_error_category") or "missing"
        for current_task_id in recoverable
    )
    return {
        "contract": CONTRACT,
        "source_tasks": len(classified),
        "frozen_eval22_tasks": len(frozen_eval_task_ids),
        "native_observed_first_query_errors": len(native_errors),
        "native_error_overlap_with_eval22": len(overlap),
        "native_error_states_outside_eval22": len(outside),
        "additional_frozen_overlap_outside_eval22": len(additional_overlap),
        "native_error_states_outside_all_frozen_sets": len(recoverable),
        "outside_all_frozen_first_error_category_counts": dict(sorted(categories.items())),
        "all_reported_counts_are_unique_task_counts": True,
        "contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths": False,
        "evaluation_only_tasks_excluded_from_training_supply": True,
        "states_are_training_ready_pairs": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "next_action": (
            "mechanically_build_and_audit_native_sourced_pairs_outside_eval22"
            if recoverable
            else "expand_external_disjoint_task_supply"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--native-rollout-openai", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--eval22-contract", type=Path, required=True)
    parser.add_argument(
        "--additional-forbidden-parquet", type=Path, action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    classified = classify_first_query_outcomes(
        replay_rows=load_parquet_rows(args.replay_parquet),
        rollout_messages=read_openai(args.native_rollout_openai),
        database=args.database,
    )
    contract = json.loads(args.eval22_contract.read_text(encoding="utf-8"))
    additional_forbidden: set[str] = set()
    for path in args.additional_forbidden_parquet:
        additional_forbidden.update(
            current_task_id
            for row in load_parquet_rows(path)
            if (current_task_id := forbidden_task_id(row))
        )
    result = summarize_supply(
        classified, eval_task_ids(contract), additional_forbidden
    )
    result["source_sha256"] = {
        "replay_parquet": sha256_file(args.replay_parquet),
        "native_rollout_openai": sha256_file(args.native_rollout_openai),
        "database": sha256_file(args.database),
        "eval22_contract": sha256_file(args.eval22_contract),
        "additional_forbidden_parquet": {
            path.name: sha256_file(path) for path in args.additional_forbidden_parquet
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
