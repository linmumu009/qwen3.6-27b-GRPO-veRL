#!/usr/bin/env python3
"""Build disjoint chosen-only supervision for one correct SQLite action."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any

import pandas as pd

from llin_verl.boss_pi_contract import contract_hashes, load_boss_pi_contract
from scripts.prepare_disjoint_first_error_pairs import source_contract
from scripts.prepare_repair_sft_dataset import (
    execute_sql_with_columns,
    load_parquet_rows,
    sha256_file,
    sha256_value,
    task_id,
)
from scripts.prepare_pi_formal_dataset import gold_supported_by_rows
from scripts.prepare_schema_oracle_action_gate import schema_payload
from scripts.prepare_semantic_plan_sufficiency_gate import _table_aliases


CONTRACT = "chosen-only-schema-conditioned-first-action-sft-v1"
SOURCE_CONTRACT = "current-definition-disjoint-pair-rollout-candidates-v1"
PROMPT_PREFIX = "CHOSEN_ONLY_SCHEMA_ACTION_V1\n"
SQLITE_COMMAND_PREFIX = "sqlite3 -json /workspace/logistics.sqlite "


def prompted_instruction(instruction: str, schema: dict[str, Any]) -> str:
    payload = {
        "instruction": (
            "Return exactly one bash tool call. Run sqlite3 -json against "
            "/workspace/logistics.sqlite with one read-only SELECT or WITH query "
            "that answers the original question. Do not inspect schema, call any "
            "other tool, repeat a command, or provide a final answer."
        ),
        "schema_context": schema,
    }
    return instruction + "\n\n" + PROMPT_PREFIX + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def build_action_row(
    *,
    source: dict[str, Any],
    boss_contract: dict[str, Any],
    database: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_task_id = task_id(source)
    prompt = source.get("prompt") or []
    if [message.get("role") for message in prompt] != ["system", "user"]:
        raise ValueError(f"{current_task_id}: source prompt is not system,user")
    if prompt[0].get("content") != boss_contract.get("system_prompt"):
        raise ValueError(f"{current_task_id}: source system differs from boss contract")
    instruction = prompt[1].get("content")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{current_task_id}: source instruction is empty")

    truth = (source.get("reward_model") or {}).get("ground_truth") or {}
    sql = str(truth.get("verification_sql") or "").strip()
    tables, _ = _table_aliases(sql)
    if not tables:
        raise ValueError(f"{current_task_id}: verification SQL has no table")
    schema = schema_payload(database, tables)
    user_content = prompted_instruction(instruction, schema)
    if sql.casefold() in user_content.casefold():
        raise ValueError(f"{current_task_id}: gold SQL leaked into prompt")

    answer_type = str(truth.get("answer_type") or "")
    expected = json.loads(str(truth.get("expected_value_json") or "null"))
    _, rows = execute_sql_with_columns(database, sql)
    if not rows:
        raise ValueError(f"{current_task_id}: chosen SQL returned no rows")
    if not gold_supported_by_rows(
        {"answer_type": answer_type, "value": expected}, rows
    ):
        raise ValueError(f"{current_task_id}: chosen SQL does not support expected value")

    command = SQLITE_COMMAND_PREFIX + shlex.quote(sql)
    if shlex.split(command)[-1] != sql:
        raise ValueError(f"{current_task_id}: chosen SQL shell round trip failed")
    call_id = f"call_chosen_{current_task_id.removeprefix('task_')}"
    messages = [
        copy.deepcopy(prompt[0]),
        {**copy.deepcopy(prompt[1]), "content": user_content},
        {
            "role": "assistant",
            "content": "根据给定字段元数据，直接执行唯一的只读查询。",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": command},
                    },
                }
            ],
        },
    ]
    output = {
        "sample_id": f"chosen-schema-action-{current_task_id}",
        "task_id": current_task_id,
        "messages": messages,
        "tools": copy.deepcopy(boss_contract["tools"]),
        "enable_thinking": False,
        "tool_argument_storage": "mapping_for_qwen36_chat_template",
        "purpose": "disjoint_chosen_only_schema_conditioned_first_action",
        "supervised_assistant_turn_indices": [0],
    }
    evidence = {
        "task_id": current_task_id,
        "answer_type": answer_type,
        "task_family": str(truth.get("task_family") or "missing"),
        "table_count": len(tables),
        "column_count": sum(len(table["columns"]) for table in schema["tables"]),
        "source_prompt_sha256": sha256_value(prompt),
        "prompted_instruction_sha256": sha256_value(user_content),
        "schema_sha256": sha256_value(schema),
        "chosen_sql_sha256": sha256_value(sql),
        "chosen_sql_rows": len(rows),
    }
    return output, evidence


def _calibration_quotas(
    evidence: list[dict[str, Any]], calibration_rows: int
) -> dict[str, int]:
    counts = Counter(item["answer_type"] for item in evidence)
    total = len(evidence)
    raw = {
        answer_type: calibration_rows * count / total
        for answer_type, count in counts.items()
    }
    quotas = {answer_type: int(value) for answer_type, value in raw.items()}
    remaining = calibration_rows - sum(quotas.values())
    for answer_type in sorted(
        counts,
        key=lambda key: (-(raw[key] - quotas[key]), key),
    )[:remaining]:
        quotas[answer_type] += 1
    if sum(quotas.values()) != calibration_rows:
        raise ValueError("calibration quota allocation drifted")
    return quotas


def split_rows(
    *,
    rows: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    calibration_rows: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != len(evidence) or not 0 < calibration_rows < len(rows):
        raise ValueError("invalid chosen-only train/calibration split sizes")
    evidence_by_task = {item["task_id"]: item for item in evidence}
    if len(evidence_by_task) != len(evidence):
        raise ValueError("chosen-only evidence has duplicate task IDs")
    quotas = _calibration_quotas(evidence, calibration_rows)
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in evidence:
        grouped[item["answer_type"]].append(item["task_id"])
    calibration_ids: set[str] = set()
    for answer_type, ids in grouped.items():
        ordered = sorted(
            ids,
            key=lambda value: hashlib.sha256(
                f"{seed}:{value}".encode("utf-8")
            ).hexdigest(),
        )
        calibration_ids.update(ordered[: quotas[answer_type]])

    train: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for row in rows:
        destination = calibration if row["task_id"] in calibration_ids else train
        copied = copy.deepcopy(row)
        copied["source_split"] = (
            "calibration16" if destination is calibration else "train48"
        )
        destination.append(copied)
    if len(calibration) != calibration_rows:
        raise ValueError("chosen-only calibration row count drifted")
    return train, calibration, {
        "seed": seed,
        "calibration_answer_type_quotas": dict(sorted(quotas.items())),
        "train_answer_types": dict(
            sorted(Counter(evidence_by_task[row["task_id"]]["answer_type"] for row in train).items())
        ),
        "calibration_answer_types": dict(
            sorted(
                Counter(
                    evidence_by_task[row["task_id"]]["answer_type"]
                    for row in calibration
                ).items()
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-parquet", type=Path, required=True)
    parser.add_argument("--candidate-contract", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--boss-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=64)
    parser.add_argument("--calibration-rows", type=int, default=16)
    parser.add_argument("--seed", default="chosen-schema-action-v1")
    args = parser.parse_args()

    candidate_contract = source_contract(
        args.candidate_contract, args.candidate_parquet
    )
    if candidate_contract.get("contract") != SOURCE_CONTRACT:
        raise ValueError("chosen-only source contract mismatch")
    source_rows = load_parquet_rows(args.candidate_parquet)
    if len(source_rows) != args.expected_rows:
        raise ValueError("chosen-only source row count drifted")
    if candidate_contract.get("training_allowed") is not False:
        raise ValueError("chosen-only source unexpectedly authorizes training")
    boss_contract = load_boss_pi_contract(args.boss_contract)

    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for source in source_rows:
        row, row_evidence = build_action_row(
            source=source,
            boss_contract=boss_contract,
            database=args.database,
        )
        rows.append(row)
        evidence.append(row_evidence)
    if len({row["task_id"] for row in rows}) != args.expected_rows:
        raise ValueError("chosen-only source task IDs are not unique")

    train, calibration, split = split_rows(
        rows=rows,
        evidence=evidence,
        calibration_rows=args.calibration_rows,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "all": args.output_dir / "chosen_only_schema_action_all.parquet",
        "train": args.output_dir / "chosen_only_schema_action_train48.parquet",
        "calibration": args.output_dir
        / "chosen_only_schema_action_calibration16.parquet",
    }
    for key, values in (("all", rows), ("train", train), ("calibration", calibration)):
        pd.DataFrame(values).to_parquet(outputs[key], index=False)

    contract = {
        "contract": CONTRACT,
        "rows": len(rows),
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "source_split": "train236_current_definition_disjoint_from_frozen16_val20_test20",
        "oracle_relevant_table_selection": True,
        "deployment_ready": False,
        "messages_roles": ["system", "user", "assistant"],
        "supervised_assistant_turn_indices": [0],
        "all_targets_exactly_one_bash_tool_call": True,
        "target_contains_mechanically_verified_gold_sql": True,
        "prompt_contains_gold_sql": False,
        "prompt_contains_database_rows_tool_results_answers_or_expected_values": False,
        "all_chosen_sql_read_only_executable_nonempty": True,
        "all_expected_values_supported_by_chosen_sql": True,
        "contains_tool_results": False,
        "contains_final_answers": False,
        "heldout_frozen16_val20_test20_overlap": 0,
        "answer_types": dict(sorted(Counter(item["answer_type"] for item in evidence).items())),
        "task_families": dict(sorted(Counter(item["task_family"] for item in evidence).items())),
        "table_count_min": min(item["table_count"] for item in evidence),
        "table_count_max": max(item["table_count"] for item in evidence),
        "column_count_min": min(item["column_count"] for item in evidence),
        "column_count_max": max(item["column_count"] for item in evidence),
        "split": split,
        "boss_contract_hashes": contract_hashes(boss_contract),
        "source_sha256": {
            "candidate_parquet": sha256_file(args.candidate_parquet),
            "candidate_contract": sha256_file(args.candidate_contract),
            "database": sha256_file(args.database),
            "boss_contract": sha256_file(args.boss_contract),
        },
        "outputs": {
            key: {"file": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
        },
        "cpu_tokenization_gate_passed": False,
        "teacher_forced_baseline_allowed": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "next_action": "run_cpu_qwen36_tokenization_and_first_action_loss_mask_gate",
    }
    contract_path = args.output_dir / "contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
