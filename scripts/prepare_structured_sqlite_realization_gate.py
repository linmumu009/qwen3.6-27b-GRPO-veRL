#!/usr/bin/env python3
"""Build a structured, non-interactive SQLite realization diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.prepare_query_initiation_oracle_candidates import (
    CONTRACT as QUERY_ORACLE_CONTRACT,
    INTERVENTION as QUERY_ORACLE_INTERVENTION,
    sha256_text,
)
from scripts.prepare_repair_sft_dataset import load_parquet_rows, sha256_file, task_id


CONTRACT = "structured-noninteractive-sqlite-realization-dataset-v1"
INTERVENTION = (
    "\n\n[诊断性非交互 SQLite 工作流：第 1 个助手回合最多执行一次路径定位，选定一个"
    " SQLite 数据库文件；第 2 个助手回合只执行一次非交互 sqlite3 .tables/.schema"
    " 结构检查，不启动交互 shell；第 3 个助手回合必须用 sqlite3 非交互执行一条与原问题"
    "相关的只读 SELECT 或 WITH 并获取结果。禁止重复完全相同的命令，禁止猜测最终答案。]"
)


def build_rows(
    *,
    query_oracle_rows: list[dict[str, Any]],
    query_oracle_contract: dict[str, Any],
    expected_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if query_oracle_contract.get("contract") != QUERY_ORACLE_CONTRACT:
        raise ValueError("source query-initiation contract mismatch")
    if int(query_oracle_contract.get("rows") or 0) != expected_rows:
        raise ValueError("source query-initiation contract row count drifted")
    if len(query_oracle_rows) != expected_rows:
        raise ValueError("source query-initiation parquet row count drifted")
    if query_oracle_contract.get("training_allowed") is not False:
        raise ValueError("source query-initiation dataset unexpectedly authorizes training")

    output: list[dict[str, Any]] = []
    evidence: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, source in enumerate(query_oracle_rows):
        row = copy.deepcopy(source)
        current_task_id = task_id(row)
        if not current_task_id or current_task_id in seen:
            raise ValueError(f"missing or duplicate task ID: {current_task_id!r}")
        seen.add(current_task_id)
        prompt = row.get("prompt") or []
        if [message.get("role") for message in prompt] != ["system", "user"]:
            raise ValueError(f"source prompt is not exactly system,user: {current_task_id}")
        source_instruction = prompt[1].get("content")
        if not isinstance(source_instruction, str) or not source_instruction.endswith(
            QUERY_ORACLE_INTERVENTION
        ):
            raise ValueError(
                f"source query-initiation intervention missing: {current_task_id}"
            )
        base_instruction = source_instruction[: -len(QUERY_ORACLE_INTERVENTION)]
        structured_instruction = base_instruction + INTERVENTION
        row["prompt"] = [
            copy.deepcopy(prompt[0]),
            {**copy.deepcopy(prompt[1]), "content": structured_instruction},
        ]
        row["data_source"] = "llin_pi_dwh_structured_sqlite_realization_v1"
        extra = copy.deepcopy(row.get("extra_info") or {})
        extra.update(
            {
                "index": index,
                "split": "train_disjoint_structured_sqlite_diagnostic",
                "structured_sqlite_realization_contract": CONTRACT,
                "structured_sqlite_base_instruction_sha256": sha256_text(
                    base_instruction
                ),
                "structured_sqlite_prompted_instruction_sha256": sha256_text(
                    structured_instruction
                ),
            }
        )
        row["extra_info"] = extra
        output.append(row)
        evidence.append(
            {
                "task_id": current_task_id,
                "base_instruction_sha256": sha256_text(base_instruction),
                "prompted_instruction_sha256": sha256_text(structured_instruction),
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
        raise ValueError("recovery_floor must be within the row count")
    return {
        "contract": CONTRACT,
        "rows": len(rows),
        "source_selection": "step120_full25_no_readonly_query_only",
        "intervention_scope": "generic_noninteractive_path_schema_select_workflow",
        "intervention_sha256": sha256_text(INTERVENTION),
        "intervention_discloses_task_specific_answer_schema_query_or_literal": False,
        "intervention_discloses_generic_sqlite_method_keywords": True,
        "max_assistant_turns": 3,
        "max_tool_result_turns": 3,
        "observed_readonly_query_recovery_floor": recovery_floor,
        "unique_task_ids": len({item["task_id"] for item in evidence}),
        "unique_base_instruction_hashes": len(
            {item["base_instruction_sha256"] for item in evidence}
        ),
        "output": output_path.name,
        "output_sha256": sha256_file(output_path),
        "source_sha256": source_hashes,
        "optimizer_initialized": False,
        "checkpoint_saved": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "next_action": "run_step120_greedy_structured_sqlite_realization_diagnostic",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-oracle-parquet", type=Path, required=True)
    parser.add_argument("--query-oracle-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=41)
    parser.add_argument("--recovery-floor", type=int, default=31)
    args = parser.parse_args()

    source_contract = json.loads(
        args.query_oracle_contract.read_text(encoding="utf-8")
    )
    if sha256_file(args.query_oracle_parquet) != source_contract.get("output_sha256"):
        raise ValueError("source query-initiation parquet hash differs from contract")
    rows, evidence = build_rows(
        query_oracle_rows=load_parquet_rows(args.query_oracle_parquet),
        query_oracle_contract=source_contract,
        expected_rows=args.expected_rows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "structured_sqlite_realization_candidates.parquet"
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    contract = build_contract(
        rows=rows,
        evidence=evidence,
        output_path=output_path,
        recovery_floor=args.recovery_floor,
        source_hashes={
            "query_oracle_parquet": sha256_file(args.query_oracle_parquet),
            "query_oracle_contract": sha256_file(args.query_oracle_contract),
        },
    )
    contract_path = args.output_dir / "structured_sqlite_realization_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
