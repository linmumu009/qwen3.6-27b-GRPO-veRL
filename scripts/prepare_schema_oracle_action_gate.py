#!/usr/bin/env python3
"""Build a one-turn, task-specific schema-oracle action diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import quote

import pandas as pd

from scripts.prepare_disjoint_first_error_pairs import source_contract
from scripts.prepare_query_initiation_oracle_candidates import sha256_text
from scripts.prepare_repair_sft_dataset import load_parquet_rows, sha256_file, task_id
from scripts.prepare_semantic_plan_sufficiency_gate import _table_aliases


CONTRACT = "task-specific-schema-oracle-action-dataset-v1"
SOURCE_CONTRACT = "current-definition-disjoint-pair-rollout-candidates-v1"
PROMPT_PREFIX = "SCHEMA_ORACLE_ACTION_GATE_V1\n"


def schema_payload(database: Path, tables: list[str]) -> dict[str, Any]:
    if not tables:
        raise ValueError("schema oracle requires at least one table")
    resolved = database.resolve(strict=True)
    uri = f"file:{quote(resolved.as_posix(), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        connection.execute("PRAGMA query_only=ON")
        output_tables: list[dict[str, Any]] = []
        bare_tables = {table.split(".")[-1].casefold() for table in tables}
        for table in tables:
            bare = table.split(".")[-1]
            escaped = bare.replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
            if not columns:
                raise ValueError(f"schema oracle references unknown table: {table}")
            foreign_keys = connection.execute(
                f'PRAGMA foreign_key_list("{escaped}")'
            ).fetchall()
            output_tables.append(
                {
                    "name": bare.casefold(),
                    "columns": [
                        {
                            "name": str(row[1]).casefold(),
                            "type": str(row[2] or "").upper(),
                            "not_null": bool(row[3]),
                            "primary_key_position": int(row[5] or 0),
                        }
                        for row in columns
                    ],
                    "foreign_keys": [
                        {
                            "from": str(row[3]).casefold(),
                            "to_table": str(row[2]).casefold(),
                            "to_column": str(row[4]).casefold(),
                        }
                        for row in foreign_keys
                        if str(row[2]).casefold() in bare_tables
                    ],
                }
            )
        return {"tables": output_tables}
    finally:
        connection.close()


def prompt_suffix(schema: dict[str, Any]) -> str:
    payload = {
        "instruction": (
            "Return exactly one bash tool call now. In that call, locate one .sqlite or .db "
            "file into a shell variable and immediately run sqlite3 -json non-interactively "
            "with one read-only SELECT or WITH query that answers the original question. "
            "Do not start an interactive shell, inspect schema again, repeat a command, call "
            "another tool, or provide a final answer."
        ),
        "schema_oracle": schema,
    }
    return "\n\n" + PROMPT_PREFIX + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def build_rows(
    *,
    candidate_rows: list[dict[str, Any]],
    candidate_contract: dict[str, Any],
    database: Path,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if candidate_contract.get("contract") != SOURCE_CONTRACT:
        raise ValueError("source rollout candidate contract mismatch")
    if int(candidate_contract.get("rows") or 0) != expected_rows:
        raise ValueError("source rollout candidate contract row count drifted")
    if len(candidate_rows) != expected_rows:
        raise ValueError("source rollout candidate parquet row count drifted")
    if candidate_contract.get("training_allowed") is not False:
        raise ValueError("source rollout candidates unexpectedly authorize training")

    output: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(candidate_rows):
        row = copy.deepcopy(source)
        current_task_id = task_id(row)
        if not current_task_id or current_task_id in seen:
            raise ValueError(f"missing or duplicate task ID: {current_task_id!r}")
        seen.add(current_task_id)
        prompt = row.get("prompt") or []
        if [message.get("role") for message in prompt] != ["system", "user"]:
            raise ValueError(f"source prompt is not exactly system,user: {current_task_id}")
        instruction = prompt[1].get("content")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"source instruction is empty: {current_task_id}")
        truth = (row.get("reward_model") or {}).get("ground_truth") or {}
        verification_sql = str(truth.get("verification_sql") or "").strip()
        tables, _ = _table_aliases(verification_sql)
        if not tables:
            raise ValueError(f"verified SQL has no FROM/JOIN table: {current_task_id}")
        schema = schema_payload(database, tables)
        suffix = prompt_suffix(schema)
        prompted_instruction = instruction + suffix
        if verification_sql.casefold() in prompted_instruction.casefold():
            raise ValueError(f"schema oracle leaked verification SQL: {current_task_id}")
        row["prompt"] = [
            copy.deepcopy(prompt[0]),
            {**copy.deepcopy(prompt[1]), "content": prompted_instruction},
        ]
        row["data_source"] = "llin_pi_dwh_task_specific_schema_oracle_action_v1"
        extra = copy.deepcopy(row.get("extra_info") or {})
        extra.update(
            {
                "index": index,
                "split": "train_disjoint_schema_oracle_action_diagnostic",
                "schema_oracle_action_contract": CONTRACT,
                "schema_oracle_sha256": sha256_text(
                    json.dumps(schema, sort_keys=True, separators=(",", ":"))
                ),
                "schema_oracle_prompted_instruction_sha256": sha256_text(
                    prompted_instruction
                ),
            }
        )
        row["extra_info"] = extra
        output.append(row)
        evidence.append(
            {
                "task_id": current_task_id,
                "answer_type": str(truth.get("answer_type") or "missing"),
                "table_count": len(tables),
                "column_count": sum(
                    len(table["columns"]) for table in schema["tables"]
                ),
                "schema_sha256": extra["schema_oracle_sha256"],
                "prompted_instruction_sha256": extra[
                    "schema_oracle_prompted_instruction_sha256"
                ],
            }
        )
    return output, evidence


def build_contract(
    *,
    rows: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    output_path: Path,
    correct_floor: int,
    wrong_pair_floor: int,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    if not 1 <= correct_floor <= len(rows):
        raise ValueError("correct_floor must be within the row count")
    if not 1 <= wrong_pair_floor <= len(rows):
        raise ValueError("wrong_pair_floor must be within the row count")
    return {
        "contract": CONTRACT,
        "rows": len(rows),
        "source_split": "train236_current_definition_disjoint_from_frozen16_val20_test20",
        "intervention_scope": "task_specific_relevant_schema_plus_generic_dynamic_db_template",
        "schema_derived_only_from_sqlite_metadata": True,
        "schema_contains_database_rows_tool_results_answers_or_expected_values": False,
        "prompt_contains_gold_sql": False,
        "max_assistant_turns": 1,
        "max_tool_result_turns": 1,
        "correct_or_equivalent_runtime_floor": correct_floor,
        "observed_wrong_pair_floor": wrong_pair_floor,
        "unique_task_ids": len({item["task_id"] for item in evidence}),
        "unique_prompted_instruction_hashes": len(
            {item["prompted_instruction_sha256"] for item in evidence}
        ),
        "answer_types": dict(
            sorted(Counter(item["answer_type"] for item in evidence).items())
        ),
        "table_count_min": min(item["table_count"] for item in evidence),
        "table_count_max": max(item["table_count"] for item in evidence),
        "column_count_min": min(item["column_count"] for item in evidence),
        "column_count_max": max(item["column_count"] for item in evidence),
        "output": output_path.name,
        "output_sha256": sha256_file(output_path),
        "source_sha256": source_hashes,
        "optimizer_initialized": False,
        "checkpoint_saved": False,
        "pair_construction_allowed": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "next_action": "run_step120_greedy_one_turn_schema_oracle_action_diagnostic",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-parquet", type=Path, required=True)
    parser.add_argument("--candidate-contract", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=64)
    parser.add_argument("--correct-floor", type=int, default=32)
    parser.add_argument("--wrong-pair-floor", type=int, default=48)
    args = parser.parse_args()

    candidate_contract = source_contract(
        args.candidate_contract, args.candidate_parquet
    )
    rows, evidence = build_rows(
        candidate_rows=load_parquet_rows(args.candidate_parquet),
        candidate_contract=candidate_contract,
        database=args.database,
        expected_rows=args.expected_rows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "schema_oracle_action_candidates.parquet"
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    contract = build_contract(
        rows=rows,
        evidence=evidence,
        output_path=output_path,
        correct_floor=args.correct_floor,
        wrong_pair_floor=args.wrong_pair_floor,
        source_hashes={
            "candidate_parquet": sha256_file(args.candidate_parquet),
            "candidate_contract": sha256_file(args.candidate_contract),
            "database": sha256_file(args.database),
        },
    )
    contract_path = args.output_dir / "schema_oracle_action_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
