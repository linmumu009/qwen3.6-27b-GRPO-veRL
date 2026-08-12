#!/usr/bin/env python3
"""Build a no-answer-leakage query-initiation diagnostic subset.

The builder selects only tasks for which the baseline Step 120 full-budget
trajectory never issued a recognizable read-only query.  It appends one
task-agnostic execution constraint and preserves the hidden verifier solely so
the existing boss-aligned rollout runtime can execute the rows.  The output is
an inference-only diagnostic and never authorizes training or promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.analyze_disjoint_first_query_outcomes import (
    classify_first_query_outcomes,
)
from scripts.analyze_repair_sft_free_run_divergence import read_openai
from scripts.prepare_disjoint_first_error_pairs import source_contract
from scripts.prepare_repair_sft_dataset import load_parquet_rows, sha256_file, task_id


CONTRACT = "query-initiation-oracle-diagnostic-dataset-v1"
INTERVENTION = (
    "\n\n[诊断性执行约束：在进一步分析前，先定位当前任务可用的 SQLite 数据库，"
    "检查其 schema，并在最初 3 个助手回合内至少执行一条只读 SQLite 查询。"
    "不要猜测数据或最终答案。]"
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_oracle_rows(
    *,
    replay_rows: list[dict[str, Any]],
    baseline_messages: dict[str, list[dict[str, Any]]],
    database: Path,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    outcomes = classify_first_query_outcomes(
        replay_rows=replay_rows,
        rollout_messages=baseline_messages,
        database=database,
    )
    source_by_task: dict[str, dict[str, Any]] = {}
    for row in replay_rows:
        current_task_id = task_id(row)
        if not current_task_id or current_task_id in source_by_task:
            raise ValueError(f"missing or duplicate replay task ID: {current_task_id!r}")
        source_by_task[current_task_id] = row

    selected_ids = sorted(
        current_task_id
        for current_task_id, result in outcomes.items()
        if result["outcome"] == "no_readonly_query"
    )
    if len(selected_ids) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} no-query baseline rows, observed {len(selected_ids)}"
        )

    output: list[dict[str, Any]] = []
    evidence: list[dict[str, str]] = []
    for index, current_task_id in enumerate(selected_ids):
        row = copy.deepcopy(source_by_task[current_task_id])
        prompt = row.get("prompt") or []
        if [message.get("role") for message in prompt] != ["system", "user"]:
            raise ValueError(f"source prompt is not exactly system,user: {current_task_id}")
        instruction = prompt[1].get("content")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"source instruction is not a non-empty string: {current_task_id}")
        prompted_instruction = instruction + INTERVENTION
        row["prompt"] = [
            copy.deepcopy(prompt[0]),
            {**copy.deepcopy(prompt[1]), "content": prompted_instruction},
        ]
        row["data_source"] = "llin_pi_dwh_query_initiation_oracle_v1"
        extra = copy.deepcopy(row.get("extra_info") or {})
        extra.update(
            {
                "index": index,
                "split": "train_disjoint_query_initiation_diagnostic",
                "query_initiation_oracle_contract": CONTRACT,
                "query_initiation_source_instruction_sha256": sha256_text(instruction),
                "query_initiation_prompted_instruction_sha256": sha256_text(
                    prompted_instruction
                ),
                "query_initiation_baseline_outcome": "no_readonly_query",
            }
        )
        row["extra_info"] = extra
        output.append(row)
        evidence.append(
            {
                "task_id": current_task_id,
                "source_instruction_sha256": sha256_text(instruction),
                "prompted_instruction_sha256": sha256_text(prompted_instruction),
            }
        )
    return output, evidence


def build_contract(
    *,
    rows: list[dict[str, Any]],
    evidence: list[dict[str, str]],
    output_path: Path,
    recovery_floor: int,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    if not 1 <= recovery_floor <= len(rows):
        raise ValueError("recovery_floor must be within the selected row count")
    return {
        "contract": CONTRACT,
        "rows": len(rows),
        "selection": "step120_full25_no_readonly_query_only",
        "baseline_outcome_for_all_selected_rows": "no_readonly_query",
        "intervention_scope": "task_agnostic_query_start_policy_only",
        "intervention_sha256": sha256_text(INTERVENTION),
        "intervention_discloses_answer_table_field_sql_or_literal": False,
        "max_assistant_turns": 3,
        "max_tool_result_turns": 2,
        "observed_readonly_query_recovery_floor": recovery_floor,
        "unique_task_ids": len({item["task_id"] for item in evidence}),
        "unique_source_instruction_hashes": len(
            {item["source_instruction_sha256"] for item in evidence}
        ),
        "output": output_path.name,
        "output_sha256": sha256_file(output_path),
        "source_sha256": source_hashes,
        "optimizer_initialized": False,
        "checkpoint_saved": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "next_action": "run_step120_greedy_three_turn_query_initiation_diagnostic",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--baseline-openai", type=Path, required=True)
    parser.add_argument("--rollout-candidate-contract", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=41)
    parser.add_argument("--recovery-floor", type=int, default=31)
    args = parser.parse_args()

    candidate_contract = source_contract(
        args.rollout_candidate_contract, args.replay_parquet
    )
    if int(candidate_contract.get("rows") or 0) != 64:
        raise ValueError("query-initiation diagnostic requires the complete 64-row pool")
    rows, evidence = build_oracle_rows(
        replay_rows=load_parquet_rows(args.replay_parquet),
        baseline_messages=read_openai(args.baseline_openai),
        database=args.database,
        expected_rows=args.expected_rows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "query_initiation_oracle_candidates.parquet"
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    contract = build_contract(
        rows=rows,
        evidence=evidence,
        output_path=output_path,
        recovery_floor=args.recovery_floor,
        source_hashes={
            "replay_parquet": sha256_file(args.replay_parquet),
            "baseline_openai": sha256_file(args.baseline_openai),
            "rollout_candidate_contract": sha256_file(
                args.rollout_candidate_contract
            ),
            "database": sha256_file(args.database),
        },
    )
    contract_path = args.output_dir / "query_initiation_oracle_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
