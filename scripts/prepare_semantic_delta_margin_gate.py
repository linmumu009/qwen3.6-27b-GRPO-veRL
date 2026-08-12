#!/usr/bin/env python3
"""Build paired correct-vs-actual-wrong SQL rows at the identical first-error state."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any

import pandas as pd

from scripts.analyze_repair_sft_free_run_divergence import normalize_container, sql_from_command
from scripts.prepare_semantic_plan_sufficiency_gate import sha256_value
from scripts.teacher_forced_component_masks import SQLITE_COMMAND_PREFIX


CANDIDATES = ("chosen", "rejected")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_row(source: dict[str, Any], label: str) -> dict[str, Any]:
    row = copy.deepcopy(source)
    source_task_id = str(row["task_id"])
    messages = row["messages"]
    if label == "rejected":
        rejected_call = copy.deepcopy(messages[2]["tool_calls"][0])
        rejected_call["id"] = f"call_margin_rejected_{source_task_id.removeprefix('task_')}"
        rejected_sql = sql_from_command(rejected_call["function"]["arguments"]["command"])
        if rejected_sql is None:
            raise ValueError(f"{source_task_id}: rejected candidate has no SQL")
        rejected_call["function"]["arguments"]["command"] = (
            SQLITE_COMMAND_PREFIX + shlex.quote(rejected_sql)
        )
        messages[4] = {
            "role": "assistant",
            "content": str(messages[2].get("content") or ""),
            "tool_calls": [rejected_call],
        }
    candidate_call = messages[4]["tool_calls"][0]
    messages[5] = {
        "role": "tool",
        "tool_call_id": candidate_call["id"],
        "content": "[]",
    }
    messages[6] = {"role": "assistant", "content": "Done."}
    row["task_id"] = f"{source_task_id}::{label}"
    row["sample_id"] = f"semantic-delta-margin-{source_task_id}-{label}"
    row["source_task_id"] = source_task_id
    row["candidate_label"] = label
    row["purpose"] = "step120_correct_vs_actual_wrong_sql_semantic_delta_margin"
    return row


def build_rows(
    critical_rows: list[dict[str, Any]], critical_contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contract_evidence = {
        str(row["task_id"]): row for row in critical_contract.get("evidence") or []
    }
    output: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in critical_rows:
        source = normalize_container(raw)
        task_id = str(source.get("task_id") or "")
        if not task_id or task_id in seen or task_id not in contract_evidence:
            raise ValueError(f"missing, duplicate or unaudited critical task ID: {task_id!r}")
        seen.add(task_id)
        messages = source.get("messages") or []
        if [message.get("role") for message in messages] != [
            "system", "user", "assistant", "tool", "assistant", "tool", "assistant"
        ]:
            raise ValueError(f"{task_id}: unexpected state-conditioned message shape")
        if list(source.get("supervised_assistant_turn_indices") or []) != [1, 2]:
            raise ValueError(f"{task_id}: unexpected supervised assistant turns")
        error_sql = sql_from_command(messages[2]["tool_calls"][0]["function"]["arguments"]["command"])
        chosen_sql = sql_from_command(messages[4]["tool_calls"][0]["function"]["arguments"]["command"])
        if error_sql is None or chosen_sql is None or error_sql == chosen_sql:
            raise ValueError(f"{task_id}: invalid chosen/rejected SQL pair")
        prior = contract_evidence[task_id]
        if prior["critical_sql_token_offset"] != source["critical_sql_token_offset"]:
            raise ValueError(f"{task_id}: critical offset differs from audited contract")
        if prior["critical_sql_target_id"] != source["critical_sql_target_id"]:
            raise ValueError(f"{task_id}: critical target differs from audited contract")
        base_hash = sha256_value(messages[:4])
        for label in CANDIDATES:
            output.append(_candidate_row(source, label))
        evidence.append(
            {
                "task_id": task_id,
                "critical_token_family": str(source["critical_token_family"]),
                "critical_sql_token_offset": int(source["critical_sql_token_offset"]),
                "critical_sql_target_id": int(source["critical_sql_target_id"]),
                "identical_error_state_sha256": base_hash,
                "chosen_query_sha256": sha256_value(chosen_sql),
                "rejected_query_sha256": sha256_value(error_sql),
            }
        )
    if len(output) != 32 or len(evidence) != 16:
        raise ValueError(f"semantic-delta margin gate requires 16 pairs, got {len(evidence)}")
    return output, evidence


def prepare(critical_data: Path, critical_contract_path: Path, output_dir: Path) -> dict[str, Any]:
    critical_contract = json.loads(critical_contract_path.read_text(encoding="utf-8"))
    if critical_contract.get("contract") != "train236-critical-token-repair-sft-dataset-v1":
        raise ValueError("semantic-delta margin gate requires critical-token dataset v1")
    if critical_contract.get("output_sha256") != sha256_file(critical_data):
        raise ValueError("critical-token parquet hash differs from its contract")
    rows = [normalize_container(row) for row in pd.read_parquet(critical_data).to_dict("records")]
    output_rows, evidence = build_rows(rows, critical_contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "semantic_delta_margin_gate.parquet"
    pd.DataFrame(output_rows).to_parquet(output, index=False)
    family_counts = Counter(row["critical_token_family"] for row in evidence)
    result = {
        "contract": "semantic-delta-margin-gate-dataset-v1",
        "source_checkpoint": "step120",
        "pairs": 16,
        "rows": 32,
        "candidate_labels": list(CANDIDATES),
        "critical_token_family_counts": dict(sorted(family_counts.items())),
        "chosen_queries_mechanically_verified_by_source_contract": True,
        "rejected_queries_are_actual_step120_first_errors": True,
        "rejected_candidate_shell_wrapper_normalized_to_teacher_contract": True,
        "pair_prefix_is_identical_through_observed_error_result": True,
        "post_candidate_tail_is_fixed_non_scored_stub": True,
        "output": output.name,
        "output_sha256": sha256_file(output),
        "evidence": evidence,
        "promotion_allowed": False,
    }
    (output_dir / "contract.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critical-data", type=Path, required=True)
    parser.add_argument("--critical-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.critical_data, args.critical_contract, args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "evidence"}, indent=2))


if __name__ == "__main__":
    main()
