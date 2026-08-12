#!/usr/bin/env python3
"""Compare native and trained first-query outcomes at paired task grain."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.analyze_disjoint_first_query_outcomes import (
    classify_first_query_outcomes,
)
from scripts.analyze_repair_sft_free_run_divergence import read_openai
from scripts.prepare_disjoint_first_error_pairs import source_contract
from scripts.prepare_repair_sft_dataset import load_parquet_rows, sha256_file


CONTRACT = "disjoint-native-vs-step120-first-query-comparison-v1"
OBSERVED = frozenset(
    {"observed_first_query_error", "first_query_correct_or_equivalent"}
)


def compare_outcomes(
    native: dict[str, dict[str, str]], step120: dict[str, dict[str, str]]
) -> dict:
    if set(native) != set(step120):
        raise ValueError("native and Step 120 task IDs differ")
    transitions: Counter[str] = Counter()
    observed_presence: Counter[str] = Counter()
    for current_task_id in native:
        left = native[current_task_id]["outcome"]
        right = step120[current_task_id]["outcome"]
        transitions[f"{left} -> {right}"] += 1
        left_observed = left in OBSERVED
        right_observed = right in OBSERVED
        if left_observed and right_observed:
            observed_presence["both_observed"] += 1
        elif left_observed:
            observed_presence["native_only_observed"] += 1
        elif right_observed:
            observed_presence["step120_only_observed"] += 1
        else:
            observed_presence["neither_observed"] += 1
    return {
        "contract": CONTRACT,
        "rows": len(native),
        "outcome_transition_counts": dict(sorted(transitions.items())),
        "observed_first_query_presence": dict(sorted(observed_presence.items())),
        "contains_raw_prompts_sql_answers_task_ids_or_tool_outputs": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--rollout-candidate-contract", type=Path, required=True)
    parser.add_argument("--native-rollout-openai", type=Path, required=True)
    parser.add_argument("--step120-rollout-openai", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_contract = source_contract(
        args.rollout_candidate_contract, args.replay_parquet
    )
    replay_rows = load_parquet_rows(args.replay_parquet)
    native = classify_first_query_outcomes(
        replay_rows=replay_rows,
        rollout_messages=read_openai(args.native_rollout_openai),
        database=args.database,
    )
    step120 = classify_first_query_outcomes(
        replay_rows=replay_rows,
        rollout_messages=read_openai(args.step120_rollout_openai),
        database=args.database,
    )
    result = compare_outcomes(native, step120)
    result["source_candidate_contract_rows"] = int(candidate_contract["rows"])
    result["source_sha256"] = {
        "replay_parquet": sha256_file(args.replay_parquet),
        "rollout_candidate_contract": sha256_file(args.rollout_candidate_contract),
        "native_rollout_openai": sha256_file(args.native_rollout_openai),
        "step120_rollout_openai": sha256_file(args.step120_rollout_openai),
        "database": sha256_file(args.database),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
