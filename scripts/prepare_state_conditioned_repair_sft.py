#!/usr/bin/env python3
"""Build repair SFT rows conditioned on Step 120's first wrong SQL result."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from llin_verl.boss_pi_contract import canonical_json
from scripts.analyze_repair_sft_all_query_semantics import classify_query_sequence
from scripts.analyze_repair_sft_first_query_semantics import ground_truth_by_task
from scripts.analyze_repair_sft_free_run_divergence import (
    bash_calls,
    normalize_container,
    read_openai,
    sql_from_command,
    tool_outputs,
)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def teacher_rows(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    frame = pd.read_parquet(path)
    order: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    for _, series in frame.iterrows():
        row = normalize_container(series.to_dict())
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in rows:
            raise ValueError(f"missing or duplicate teacher task_id: {task_id!r}")
        order.append(task_id)
        rows[task_id] = row
    return order, rows


def _matching_tool_call(message: dict[str, Any], call_id: str) -> dict[str, Any]:
    matches = [
        item for item in message.get("tool_calls") or [] if str(item.get("id") or "") == call_id
    ]
    if len(matches) != 1:
        raise ValueError("first wrong SQL must map to exactly one tool call")
    return copy.deepcopy(matches[0])


def build_state_conditioned_row(
    *,
    teacher: dict[str, Any],
    rollout_messages: list[dict[str, Any]],
    truth: dict[str, Any],
    database: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = str(teacher.get("task_id") or "")
    messages = normalize_container(teacher.get("messages"))
    if [message.get("role") for message in messages] != [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]:
        raise ValueError(f"{task_id}: teacher row does not use the frozen two-turn repair shape")

    first_wrong = next((call for call in bash_calls(rollout_messages) if call["sql"]), None)
    if first_wrong is None:
        raise ValueError(f"{task_id}: Step 120 rollout has no read-only SQL state")
    sequence = classify_query_sequence(
        database=database,
        messages=rollout_messages,
        truth=truth,
    )
    first_result = sequence["queries"][0]
    if first_result["verified_or_equivalent"]:
        raise ValueError(f"{task_id}: first Step 120 SQL is already verified or equivalent")

    error_message = rollout_messages[first_wrong["message_index"]]
    error_call = _matching_tool_call(error_message, first_wrong["call_id"])
    observed_outputs = tool_outputs(rollout_messages)
    if first_wrong["call_id"] not in observed_outputs:
        raise ValueError(f"{task_id}: first wrong SQL has no observed tool result")
    error_output = observed_outputs[first_wrong["call_id"]]

    correction_message = copy.deepcopy(messages[2])
    correction_calls = correction_message.get("tool_calls") or []
    if len(correction_calls) != 1:
        raise ValueError(f"{task_id}: teacher correction must contain exactly one tool call")
    correction_command = correction_calls[0]["function"]["arguments"]["command"]
    correction_sql = sql_from_command(correction_command)
    if correction_sql is None or correction_sql == first_wrong["sql"]:
        raise ValueError(f"{task_id}: correction SQL is missing or identical to the error SQL")

    recovery_id = f"call_recover_{task_id.removeprefix('task_')}"
    correction_calls[0]["id"] = recovery_id
    correction_output = copy.deepcopy(messages[3])
    correction_output["tool_call_id"] = recovery_id
    correction_check = classify_query_sequence(
        database=database,
        messages=[correction_message],
        truth=truth,
    )
    if not correction_check["verified_or_equivalent_anywhere"]:
        raise ValueError(f"{task_id}: correction SQL is not mechanically verified")

    state_messages = [
        copy.deepcopy(messages[0]),
        copy.deepcopy(messages[1]),
        {
            "role": "assistant",
            "content": error_message.get("content"),
            "tool_calls": [error_call],
        },
        {
            "role": "tool",
            "tool_call_id": first_wrong["call_id"],
            "content": error_output,
        },
        correction_message,
        correction_output,
        copy.deepcopy(messages[4]),
    ]
    output = {
        **{key: copy.deepcopy(value) for key, value in teacher.items() if key != "messages"},
        "sample_id": f"state-repair-sft-{task_id}",
        "messages": state_messages,
        "purpose": "train236_step120_first_wrong_sql_state_recovery",
        "supervised_assistant_turn_indices": [1, 2],
        "error_context_assistant_turn_index": 0,
    }
    evidence = {
        "task_id": task_id,
        "first_error_category": first_result["category"],
        "first_error_query_sha256": first_result["query_sha256"],
        "first_error_tool_result_sha256": hashlib.sha256(error_output.encode("utf-8")).hexdigest(),
        "correction_query_sha256": sha256_value(correction_sql),
        "correction_verified_or_equivalent": True,
        "supervised_assistant_turn_indices": [1, 2],
    }
    return output, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-parquet", type=Path, required=True)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--step120-openai", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    order, teachers = teacher_rows(args.teacher_parquet)
    truth_order, truths = ground_truth_by_task(args.replay_parquet)
    rollouts = read_openai(args.step120_openai)
    if order != truth_order or set(order) != set(rollouts):
        raise ValueError("teacher, replay and Step 120 task IDs differ")

    output_rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for task_id in order:
        row, row_evidence = build_state_conditioned_row(
            teacher=teachers[task_id],
            rollout_messages=rollouts[task_id],
            truth=truths[task_id],
            database=args.database,
        )
        output_rows.append(row)
        evidence.append(row_evidence)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "state_conditioned_repair_sft_train.parquet"
    pd.DataFrame(output_rows).to_parquet(output_path, index=False)
    contract = {
        "contract": "train236-state-conditioned-repair-sft-dataset-v1",
        "rows": len(output_rows),
        "source_checkpoint": "step120",
        "source_split": "train236_same_task_development_gate",
        "heldout_overlap": 0,
        "first_error_assistant_context_loss_weight": 0,
        "supervised_assistant_turn_indices": [1, 2],
        "all_first_error_queries_not_verified_or_equivalent": True,
        "all_first_error_tool_results_observed": True,
        "all_correction_queries_verified_or_equivalent": True,
        "only_causal_change_vs_sql_weighted_canary": "condition_on_step120_first_wrong_sql_and_observed_tool_result",
        "output": output_path.name,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "evidence": evidence,
        "promotion_allowed": False,
    }
    contract_path = args.output_dir / "contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in contract.items() if key != "evidence"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
