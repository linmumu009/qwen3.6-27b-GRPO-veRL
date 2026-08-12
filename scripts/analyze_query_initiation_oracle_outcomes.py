#!/usr/bin/env python3
"""Audit query-initiation oracle outcomes under the diagnostic contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_disjoint_first_query_outcomes import audit_first_query_outcomes
from scripts.analyze_repair_sft_free_run_divergence import read_openai
from scripts.prepare_query_initiation_oracle_candidates import CONTRACT
from scripts.prepare_repair_sft_dataset import load_parquet_rows, sha256_file
from scripts.prepare_structured_sqlite_realization_gate import (
    CONTRACT as STRUCTURED_SQLITE_CONTRACT,
)


def audit_oracle_outcomes(
    *,
    replay_rows: list[dict[str, Any]],
    rollout_messages: dict[str, list[dict[str, Any]]],
    database: Path,
    dataset_contract: dict[str, Any],
) -> dict[str, Any]:
    source_contract = dataset_contract.get("contract")
    if source_contract not in {CONTRACT, STRUCTURED_SQLITE_CONTRACT}:
        raise ValueError("query diagnostic dataset contract mismatch")
    rows = int(dataset_contract.get("rows") or 0)
    if rows <= 0 or len(replay_rows) != rows or len(rollout_messages) != rows:
        raise ValueError("dataset contract, parquet, and rollout row counts differ")
    if dataset_contract.get("max_assistant_turns") != 3:
        raise ValueError("query-initiation assistant-turn contract drifted")
    if dataset_contract.get("max_tool_result_turns") != 3:
        raise ValueError("query-initiation tool-result contract drifted")
    if dataset_contract.get("training_allowed") is not False:
        raise ValueError("query-initiation dataset unexpectedly authorizes training")

    result = audit_first_query_outcomes(
        replay_rows=replay_rows,
        rollout_messages=rollout_messages,
        database=database,
        model_source=(
            "step120_query_initiation_oracle"
            if source_contract == CONTRACT
            else "step120_structured_sqlite_realization"
        ),
    )
    result["source_query_initiation_contract_rows"] = rows
    result["source_query_initiation_contract"] = source_contract
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--rollout-openai", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.dataset_contract.read_text(encoding="utf-8"))
    if sha256_file(args.replay_parquet) != contract.get("output_sha256"):
        raise ValueError("query-initiation parquet hash differs from dataset contract")
    result = audit_oracle_outcomes(
        replay_rows=load_parquet_rows(args.replay_parquet),
        rollout_messages=read_openai(args.rollout_openai),
        database=args.database,
        dataset_contract=contract,
    )
    result["source_sha256"] = {
        "replay_parquet": sha256_file(args.replay_parquet),
        "rollout_openai": sha256_file(args.rollout_openai),
        "dataset_contract": sha256_file(args.dataset_contract),
        "database": sha256_file(args.database),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
