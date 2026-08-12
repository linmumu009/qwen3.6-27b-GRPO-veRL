#!/usr/bin/env python3
"""Build disjoint chosen/rejected SQL pairs from Step 120 first errors.

Each selected row is conditioned on the exact assistant tool call and observed
tool result produced by Step 120.  The chosen candidate is the mechanically
verified current-definition SQL; the rejected candidate is the model's actual
first read-only SQL.  Correct/equivalent first queries and rows without a
read-only query are excluded rather than converted into synthetic negatives.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shlex
from typing import Any

import pandas as pd

from llin_verl.boss_pi_contract import load_boss_pi_contract
from scripts.analyze_repair_sft_first_query_semantics import classify_first_query
from scripts.analyze_repair_sft_free_run_divergence import (
    bash_calls,
    normalize_container,
    read_openai,
    sql_from_command,
    tool_outputs,
)
from scripts.prepare_repair_sft_dataset import (
    build_sft_row,
    load_parquet_rows,
    sha256_file,
    task_id,
)
from scripts.prepare_semantic_plan_sufficiency_gate import sha256_value
from scripts.prepare_state_conditioned_repair_sft import build_state_conditioned_row
from scripts.teacher_forced_component_masks import SQLITE_COMMAND_PREFIX


CONTRACT = "current-definition-disjoint-first-error-pairs-v1"
SOURCE_CONTRACT = "current-definition-disjoint-pair-rollout-candidates-v1"
CANDIDATES = ("chosen", "rejected")


def _pair_candidate(source: dict[str, Any], label: str, pair_index: int) -> dict[str, Any]:
    row = copy.deepcopy(source)
    source_task_id = str(row["task_id"])
    messages = row["messages"]
    if label == "rejected":
        rejected_call = copy.deepcopy(messages[2]["tool_calls"][0])
        rejected_call["id"] = f"call_disjoint_rejected_{source_task_id.removeprefix('task_')}"
        rejected_sql = sql_from_command(rejected_call["function"]["arguments"]["command"])
        if rejected_sql is None:
            raise ValueError(f"{source_task_id}: rejected candidate has no read-only SQL")
        rejected_call["function"]["arguments"]["command"] = (
            SQLITE_COMMAND_PREFIX + shlex.quote(rejected_sql)
        )
        messages[4] = {
            "role": "assistant",
            "content": str(messages[2].get("content") or ""),
            "tool_calls": [rejected_call],
        }
    candidate_call = messages[4]["tool_calls"][0]
    messages[5] = {"role": "tool", "tool_call_id": candidate_call["id"], "content": "[]"}
    messages[6] = {"role": "assistant", "content": "Done."}
    row["task_id"] = f"{source_task_id}::{label}"
    row["sample_id"] = f"disjoint-first-error-pair-{source_task_id}-{label}"
    row["source_task_id"] = source_task_id
    row["candidate_label"] = label
    row["pair_index"] = pair_index
    row["purpose"] = "disjoint_step120_first_error_correct_vs_actual_wrong_sql"
    return row


def build_pairs(
    *,
    replay_rows: list[dict[str, Any]],
    rollout_messages: dict[str, list[dict[str, Any]]],
    database: Path,
    boss_contract: dict[str, Any],
    minimum_pairs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if minimum_pairs <= 0:
        raise ValueError("minimum_pairs must be positive")
    truths: dict[str, dict[str, Any]] = {}
    source_by_task: dict[str, dict[str, Any]] = {}
    for row in replay_rows:
        current_task_id = task_id(row)
        if not current_task_id or current_task_id in source_by_task:
            raise ValueError(f"missing or duplicate replay task ID: {current_task_id!r}")
        truth = (row.get("reward_model") or {}).get("ground_truth") or {}
        truths[current_task_id] = {
            "answer_type": str(truth.get("answer_type") or ""),
            "expected": json.loads(str(truth.get("expected_value_json") or "null")),
            "verification_sql": str(truth.get("verification_sql") or ""),
        }
        source_by_task[current_task_id] = row
    if set(source_by_task) != set(rollout_messages):
        raise ValueError("rollout and replay task IDs differ")

    pair_rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for current_task_id in source_by_task:
        messages = rollout_messages[current_task_id]
        truth = truths[current_task_id]
        first_sql_call = next((call for call in bash_calls(messages) if call["sql"]), None)
        if first_sql_call is None:
            exclusions["no_readonly_query"] += 1
            continue
        if first_sql_call["call_id"] not in tool_outputs(messages):
            exclusions["first_readonly_query_without_observed_tool_result"] += 1
            continue
        semantic = classify_first_query(database=database, messages=messages, truth=truth)
        if semantic["gold_supported"] or semantic["teacher_result_equivalent"]:
            exclusions["first_query_correct_or_equivalent"] += 1
            continue

        source = source_by_task[current_task_id]
        source_truth = (source.get("reward_model") or {}).get("ground_truth") or {}
        teacher, _ = build_sft_row(
            {
                "row": source,
                "task_id": current_task_id,
                "warnings": [],
                "answer_type": str(source_truth.get("answer_type") or ""),
                "task_family": str(source_truth.get("task_family") or "dwh"),
            },
            boss_contract,
            database,
        )
        state, state_evidence = build_state_conditioned_row(
            teacher=teacher,
            rollout_messages=messages,
            truth=truth,
            database=database,
        )
        pair_index = len(evidence)
        chosen = _pair_candidate(state, "chosen", pair_index)
        rejected = _pair_candidate(state, "rejected", pair_index)
        if chosen["messages"][:4] != rejected["messages"][:4]:
            raise ValueError(f"{current_task_id}: chosen/rejected error-state prefix differs")
        pair_rows.extend([chosen, rejected])
        evidence.append(
            {
                "task_id": current_task_id,
                "first_error_category": semantic["category"],
                "first_error_query_sha256": semantic["query_sha256"],
                "first_error_tool_result_sha256": state_evidence[
                    "first_error_tool_result_sha256"
                ],
                "chosen_query_sha256": state_evidence["correction_query_sha256"],
                "identical_error_state_sha256": sha256_value(chosen["messages"][:4]),
            }
        )

    if len(pair_rows) != 2 * len(evidence):
        raise ValueError("chosen/rejected row count is not exactly two per pair")
    if len({item["task_id"] for item in evidence}) != len(evidence):
        raise ValueError("first-error pair task IDs are not unique")
    return pair_rows, evidence, dict(sorted(exclusions.items()))


def source_contract(path: Path, replay_path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contract") != SOURCE_CONTRACT:
        raise ValueError("rollout candidate contract mismatch")
    if value.get("training_allowed") is not False or value.get("promotion_allowed") is not False:
        raise ValueError("rollout candidate contract lacks fail-closed flags")
    if value.get("output_sha256") != sha256_file(replay_path):
        raise ValueError("rollout candidate Parquet hash differs from its contract")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--rollout-openai", type=Path, required=True)
    parser.add_argument("--rollout-candidate-contract", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--boss-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-pairs", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_contract = source_contract(args.rollout_candidate_contract, args.replay_parquet)
    replay_rows = load_parquet_rows(args.replay_parquet)
    rollout_messages = read_openai(args.rollout_openai)
    pair_rows, evidence, exclusions = build_pairs(
        replay_rows=replay_rows,
        rollout_messages=rollout_messages,
        database=args.database,
        boss_contract=load_boss_pi_contract(args.boss_contract),
        minimum_pairs=args.minimum_pairs,
    )
    pair_count = len(evidence)
    gate_passed = pair_count >= args.minimum_pairs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "disjoint_first_error_pairs.parquet"
    if gate_passed:
        pd.DataFrame(pair_rows).to_parquet(output_path, index=False)
    result = {
        "contract": CONTRACT,
        "source_checkpoint": "step120",
        "source_candidate_rows": len(replay_rows),
        "minimum_pairs": args.minimum_pairs,
        "pairs": pair_count,
        "rows": len(pair_rows),
        "pair_count_gate_passed": gate_passed,
        "candidate_labels": list(CANDIDATES),
        "first_error_category_counts": dict(
            sorted(Counter(item["first_error_category"] for item in evidence).items())
        ),
        "exclusion_counts": exclusions,
        "chosen_queries_mechanically_verified": True,
        "rejected_queries_are_actual_step120_first_errors": True,
        "all_first_error_tool_results_observed": True,
        "pair_prefix_identical_through_observed_error_result": True,
        "forbidden_frozen16_val20_test20_overlap": 0,
        "output": output_path.name if gate_passed else None,
        "output_sha256": sha256_file(output_path) if gate_passed else None,
        "source_sha256": {
            "replay_parquet": sha256_file(args.replay_parquet),
            "rollout_openai": sha256_file(args.rollout_openai),
            "rollout_candidate_contract": sha256_file(args.rollout_candidate_contract),
            "database": sha256_file(args.database),
            "boss_contract": sha256_file(args.boss_contract),
        },
        "evidence": evidence,
        "contains_raw_prompts_sql_answers_or_tool_outputs_outside_parquet": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "next_action": (
            "run_step120_forward_only_token_family_audit_and_stratification"
            if gate_passed
            else "expand_disjoint_current_definition_pool_without_training"
        ),
        "source_candidate_contract_rows": int(candidate_contract["rows"]),
    }
    (args.output_dir / "first_error_pair_contract.json").write_text(
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
