#!/usr/bin/env python3
"""Freeze sub-threshold real Step 120 first-error pairs for evaluation only.

This path deliberately does not weaken the 48-pair training gate.  It writes
an exact, post-selected failure-case evaluation set whose rows may be used for
forward-only diagnostics and future before/after comparisons, but never as
training data or as a standalone promotion benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from llin_verl.boss_pi_contract import load_boss_pi_contract
from scripts.analyze_repair_sft_free_run_divergence import read_openai
from scripts.prepare_disjoint_first_error_pairs import (
    CANDIDATES,
    build_pairs,
    source_contract,
)
from scripts.prepare_repair_sft_dataset import load_parquet_rows, sha256_file


CONTRACT = "current-definition-disjoint-first-error-evaluation-v1"


def build_evaluation(
    *,
    replay_parquet: Path,
    rollout_openai: Path,
    rollout_candidate_contract: Path,
    database: Path,
    boss_contract: Path,
    output_dir: Path,
    expected_pairs: int,
) -> dict:
    if expected_pairs <= 0:
        raise ValueError("expected_pairs must be positive")
    candidate_contract = source_contract(rollout_candidate_contract, replay_parquet)
    replay_rows = load_parquet_rows(replay_parquet)
    rollout_messages = read_openai(rollout_openai)
    pair_rows, evidence, exclusions = build_pairs(
        replay_rows=replay_rows,
        rollout_messages=rollout_messages,
        database=database,
        boss_contract=load_boss_pi_contract(boss_contract),
        minimum_pairs=expected_pairs,
    )
    pairs = len(evidence)
    if pairs != expected_pairs:
        raise ValueError(f"frozen evaluation pair count changed: {pairs} != {expected_pairs}")
    if len(pair_rows) != 2 * expected_pairs:
        raise ValueError("frozen evaluation does not contain exactly two rows per pair")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "disjoint_first_error_evaluation.parquet"
    pd.DataFrame(pair_rows).to_parquet(output_path, index=False)
    return {
        "contract": CONTRACT,
        "source_checkpoint": "step120",
        "evaluation_role": "postselected_real_failure_state_diagnostic_only",
        "selection_condition": (
            "step120_full25_first_readonly_query_wrong_or_insufficient_"
            "with_observed_tool_result"
        ),
        "selection_bias": "conditioned_on_step120_failure_not_population_representative",
        "source_candidate_rows": len(replay_rows),
        "expected_pairs": expected_pairs,
        "pairs": pairs,
        "rows": len(pair_rows),
        "pair_evaluation_gate_passed": True,
        "candidate_labels": list(CANDIDATES),
        "first_error_category_counts": dict(
            sorted(Counter(item["first_error_category"] for item in evidence).items())
        ),
        "exclusion_counts": exclusions,
        "chosen_queries_mechanically_verified": True,
        "rejected_queries_are_actual_step120_first_errors": True,
        "all_first_error_tool_results_observed": True,
        "pair_prefix_identical_through_observed_error_result": True,
        "unique_source_task_ids": pairs,
        "forbidden_frozen16_val20_test20_overlap": 0,
        "output": output_path.name,
        "output_sha256": sha256_file(output_path),
        "source_sha256": {
            "replay_parquet": sha256_file(replay_parquet),
            "rollout_openai": sha256_file(rollout_openai),
            "rollout_candidate_contract": sha256_file(rollout_candidate_contract),
            "database": sha256_file(database),
            "boss_contract": sha256_file(boss_contract),
        },
        "evidence": evidence,
        "contains_raw_prompts_sql_answers_or_tool_outputs_outside_parquet": False,
        "evaluation_only": True,
        "may_be_used_as_training_data": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "future_candidate_gate": {
            "chosen_preferred_min": 17,
            "per_task_margin_improved_min": 18,
            "new_earlier_first_nongreedy_regressions_max": 0,
            "full64_pareto_required_after_pass": True,
        },
        "next_action": "run_step120_forward_only_eval22_baseline",
        "source_candidate_contract_rows": int(candidate_contract["rows"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--rollout-openai", type=Path, required=True)
    parser.add_argument("--rollout-candidate-contract", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--boss-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=22)
    args = parser.parse_args()
    result = build_evaluation(
        replay_parquet=args.replay_parquet,
        rollout_openai=args.rollout_openai,
        rollout_candidate_contract=args.rollout_candidate_contract,
        database=args.database,
        boss_contract=args.boss_contract,
        output_dir=args.output_dir,
        expected_pairs=args.expected_pairs,
    )
    contract_path = args.output_dir / "first_error_evaluation_contract.json"
    contract_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "evidence"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
