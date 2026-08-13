#!/usr/bin/env python3
"""Build native-model real first-error candidates outside every frozen set.

The resulting pairs are a source-stratum screening asset only.  They may be
tokenized and scored by a forward-only Step 120 diagnostic, but this
sub-threshold asset cannot authorize training or checkpoint promotion.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import pandas as pd

from llin_verl.boss_pi_contract import load_boss_pi_contract
from scripts.analyze_native_disjoint_pair_supply import eval_task_ids
from scripts.analyze_repair_sft_free_run_divergence import read_openai
from scripts.prepare_disjoint_first_error_pairs import CANDIDATES, build_pairs, source_contract
from scripts.prepare_repair_sft_dataset import load_parquet_rows, sha256_file, task_id


CONTRACT = "current-definition-native-first-error-training-candidates-v1"


def row_task_id(row: dict[str, Any]) -> str:
    """Resolve source identity without trusting display/sample identifiers."""
    source = str(row.get("source_task_id") or "")
    if source:
        return source
    truth_identity = task_id(row)
    if truth_identity:
        return truth_identity
    display = str(row.get("task_id") or "")
    return display.split("::", 1)[0] if display else ""


def forbidden_ids(
    eval_contract: dict[str, Any], additional_rows: list[list[dict[str, Any]]]
) -> tuple[set[str], list[dict[str, int]]]:
    frozen = eval_task_ids(eval_contract)
    audits: list[dict[str, int]] = []
    for rows in additional_rows:
        identities = {identity for row in rows if (identity := row_task_id(row))}
        frozen.update(identities)
        audits.append({"rows": len(rows), "unique_task_ids": len(identities)})
    return frozen, audits


def frozen_overlap_audit(
    evidence: list[dict[str, Any]],
    eval_contract: dict[str, Any],
    additional_rows: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    native_tasks = {str(row["task_id"]) for row in evidence}
    eval_ids = eval_task_ids(eval_contract)
    additional_ids = [
        {identity for row in rows if (identity := row_task_id(row))}
        for rows in additional_rows
    ]
    outside_eval = native_tasks - eval_ids
    return {
        "native_first_error_pairs": len(native_tasks),
        "eval22_overlap": len(native_tasks & eval_ids),
        "outside_eval22": len(outside_eval),
        "additional_overlap_outside_eval22": [
            len(outside_eval & identities) for identities in additional_ids
        ],
        "retained_after_union": len(native_tasks - eval_ids - set().union(*additional_ids)),
    }


def filter_pairs(
    pair_rows: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    forbidden: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    evidence_by_task = {str(row.get("task_id") or ""): row for row in evidence}
    if "" in evidence_by_task or len(evidence_by_task) != len(evidence):
        raise ValueError("native pair evidence has missing or duplicate task IDs")
    retained_evidence = [row for row in evidence if str(row["task_id"]) not in forbidden]
    retained_tasks = {str(row["task_id"]) for row in retained_evidence}
    retained_rows = [row for row in pair_rows if row_task_id(row) in retained_tasks]
    if len(retained_rows) != 2 * len(retained_evidence):
        raise ValueError("retained native candidates are not exactly two rows per pair")
    for pair_index, evidence_row in enumerate(retained_evidence):
        task = str(evidence_row["task_id"])
        chosen, rejected = retained_rows[2 * pair_index : 2 * pair_index + 2]
        if (
            row_task_id(chosen) != task
            or row_task_id(rejected) != task
            or chosen.get("candidate_label") != "chosen"
            or rejected.get("candidate_label") != "rejected"
            or chosen.get("messages", [])[:4] != rejected.get("messages", [])[:4]
        ):
            raise ValueError(f"native candidate pair identity/order differs: {task}")
        for row in (chosen, rejected):
            row["pair_index"] = pair_index
            row["purpose"] = "native_real_first_error_correct_vs_actual_wrong_sql_candidate"
            row["state_source_checkpoint"] = "native_base"
    return retained_rows, retained_evidence, len(evidence) - len(retained_evidence)


def build_candidates(
    *,
    replay_parquet: Path,
    native_rollout_openai: Path,
    rollout_candidate_contract: Path,
    database: Path,
    boss_contract: Path,
    eval22_contract: Path,
    additional_forbidden_parquets: list[Path],
    output_dir: Path,
    expected_pairs: int,
) -> dict[str, Any]:
    if expected_pairs <= 0:
        raise ValueError("expected_pairs must be positive")
    candidate_source = source_contract(rollout_candidate_contract, replay_parquet)
    replay_rows = load_parquet_rows(replay_parquet)
    pair_rows, evidence, exclusions = build_pairs(
        replay_rows=replay_rows,
        rollout_messages=read_openai(native_rollout_openai),
        database=database,
        boss_contract=load_boss_pi_contract(boss_contract),
        minimum_pairs=expected_pairs,
    )
    eval_contract = json.loads(eval22_contract.read_text(encoding="utf-8"))
    additional_rows = [load_parquet_rows(path) for path in additional_forbidden_parquets]
    overlap_audit = frozen_overlap_audit(evidence, eval_contract, additional_rows)
    forbidden, forbidden_audits = forbidden_ids(eval_contract, additional_rows)
    retained_rows, retained_evidence, excluded_frozen = filter_pairs(
        pair_rows, evidence, forbidden
    )
    pairs = len(retained_evidence)
    if pairs != expected_pairs:
        raise ValueError(
            f"native candidate pair count changed: {pairs} != {expected_pairs}; "
            f"aggregate frozen overlap audit={overlap_audit}"
        )
    retained_tasks = {str(row["task_id"]) for row in retained_evidence}
    if retained_tasks & forbidden:
        raise ValueError("native candidate tasks overlap a frozen set")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "native_disjoint_first_error_candidates.parquet"
    pd.DataFrame(retained_rows).to_parquet(output_path, index=False)
    return {
        "contract": CONTRACT,
        "source_checkpoint": "native_base",
        "candidate_role": "real_error_state_training_supply_screen_only",
        "selection_condition": "native_full25_observed_wrong_or_insufficient_first_readonly_query",
        "source_candidate_rows": len(replay_rows),
        "source_first_error_pairs_before_frozen_exclusion": len(evidence),
        "frozen_first_error_pairs_excluded": excluded_frozen,
        "expected_pairs": expected_pairs,
        "pairs": pairs,
        "rows": len(retained_rows),
        "candidate_pair_gate_passed": True,
        "candidate_labels": list(CANDIDATES),
        "first_error_category_counts": dict(
            sorted(Counter(row["first_error_category"] for row in retained_evidence).items())
        ),
        "source_exclusion_counts": exclusions,
        "chosen_queries_mechanically_verified": True,
        "rejected_queries_are_actual_native_first_errors": True,
        "rejected_queries_are_actual_model_first_errors": True,
        "all_first_error_tool_results_observed": True,
        "pair_prefix_identical_through_observed_error_result": True,
        "unique_source_task_ids": pairs,
        "forbidden_set_count": 1 + len(additional_forbidden_parquets),
        "forbidden_set_audits": {
            "eval22_unique_task_ids": len(eval_task_ids(eval_contract)),
            "additional_parquets": forbidden_audits,
            "union_unique_task_ids": len(forbidden),
            "retained_overlap": 0,
            "native_first_error_overlap": overlap_audit,
        },
        "output": output_path.name,
        "output_sha256": sha256_file(output_path),
        "source_sha256": {
            "replay_parquet": sha256_file(replay_parquet),
            "native_rollout_openai": sha256_file(native_rollout_openai),
            "rollout_candidate_contract": sha256_file(rollout_candidate_contract),
            "database": sha256_file(database),
            "boss_contract": sha256_file(boss_contract),
            "eval22_contract": sha256_file(eval22_contract),
            "additional_forbidden_parquets": {
                path.name: sha256_file(path) for path in additional_forbidden_parquets
            },
        },
        "evidence": retained_evidence,
        "contains_raw_prompts_sql_answers_or_tool_outputs_outside_parquet": False,
        "candidate_only": True,
        "evaluation_only": False,
        "may_be_used_as_training_data": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "next_action": "run_cpu_token_gate_then_step120_forward_only_margin_screen",
        "source_candidate_contract_rows": int(candidate_source["rows"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--native-rollout-openai", type=Path, required=True)
    parser.add_argument("--rollout-candidate-contract", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--boss-contract", type=Path, required=True)
    parser.add_argument("--eval22-contract", type=Path, required=True)
    parser.add_argument("--additional-forbidden-parquet", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=7)
    args = parser.parse_args()
    result = build_candidates(
        replay_parquet=args.replay_parquet,
        native_rollout_openai=args.native_rollout_openai,
        rollout_candidate_contract=args.rollout_candidate_contract,
        database=args.database,
        boss_contract=args.boss_contract,
        eval22_contract=args.eval22_contract,
        additional_forbidden_parquets=args.additional_forbidden_parquet,
        output_dir=args.output_dir,
        expected_pairs=args.expected_pairs,
    )
    contract_path = args.output_dir / "native_first_error_candidate_contract.json"
    contract_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "evidence"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
