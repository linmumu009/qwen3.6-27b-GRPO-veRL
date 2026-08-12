#!/usr/bin/env python3
"""Attach v3 semantic critical-token targets to state-conditioned SFT rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.analyze_repair_sft_free_run_divergence import normalize_container
from scripts.analyze_state_recovery_semantics import critical_token_family


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-parquet", type=Path, required=True)
    parser.add_argument("--step120-diagnostic", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    diagnostic = json.loads(args.step120_diagnostic.read_text(encoding="utf-8"))
    if diagnostic.get("contract") != "repair-sft-teacher-forced-component-diagnostic-v3":
        raise ValueError("critical-token dataset requires semantic diagnostic v3")
    audit = json.loads(args.semantic_audit.read_text(encoding="utf-8"))
    if audit.get("contract") != "repair-sft-state-recovery-semantic-audit-v1":
        raise ValueError("critical-token dataset requires state recovery semantic audit v1")
    rank_by_task = {str(row["task_id"]): row["sql_token_rank"] for row in diagnostic["per_task"]}
    audited_ids = {str(row["task_id"]) for row in audit["per_task"]}

    frame = pd.read_parquet(args.state_parquet)
    output_rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    evidence: list[dict[str, Any]] = []
    for _, series in frame.iterrows():
        row = normalize_container(series.to_dict())
        task_id = str(row.get("task_id") or "")
        if task_id not in rank_by_task or task_id not in audited_ids:
            raise ValueError(f"missing critical-token evidence for {task_id!r}")
        rank = rank_by_task[task_id]
        offset = rank.get("first_nongreedy_offset")
        target_id = rank.get("first_nongreedy_target_id")
        if offset is None or target_id is None:
            raise ValueError(f"{task_id}: semantic SQL sequence has no critical token")
        if not 0 <= int(offset) < int(rank["token_count"]):
            raise ValueError(f"{task_id}: critical SQL offset is out of range")
        family = critical_token_family(rank.get("first_nongreedy_target_token"))
        row["critical_sql_token_offset"] = int(offset)
        row["critical_sql_target_id"] = int(target_id)
        row["critical_token_family"] = family
        row["purpose"] = "train236_step120_semantic_critical_token_recovery"
        output_rows.append(row)
        family_counts[family] += 1
        evidence.append(
            {
                "task_id": task_id,
                "critical_sql_token_offset": int(offset),
                "critical_sql_target_id": int(target_id),
                "critical_token_family": family,
                "rank_step120": int(rank["first_nongreedy_rank"]),
                "probability_step120": rank["first_nongreedy_target_probability"],
            }
        )

    query_plan_count = family_counts["aggregation_function"] + family_counts["query_start"]
    if query_plan_count < 12:
        raise ValueError("fewer than 12 tasks have aggregation/query-start critical tokens")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "critical_token_repair_sft_train.parquet"
    pd.DataFrame(output_rows).to_parquet(output_path, index=False)
    contract = {
        "contract": "train236-critical-token-repair-sft-dataset-v1",
        "rows": len(output_rows),
        "source_checkpoint": "step120",
        "source_split": "train236_same_task_development_gate",
        "heldout_overlap": 0,
        "first_error_assistant_context_loss_weight": 0,
        "all_first_error_queries_not_verified_or_equivalent": True,
        "all_first_error_tool_results_observed": True,
        "all_correction_queries_verified_or_equivalent": True,
        "all_critical_offsets_valid": True,
        "all_critical_target_ids_present": True,
        "critical_token_family_counts": dict(sorted(family_counts.items())),
        "aggregation_or_query_start_tasks": query_plan_count,
        "only_causal_change_vs_state_conditioned_canary": "critical_semantic_sql_token_weight_8_to_32",
        "source_sha256": {
            "state_parquet": sha256_file(args.state_parquet),
            "step120_diagnostic": sha256_file(args.step120_diagnostic),
            "semantic_audit": sha256_file(args.semantic_audit),
        },
        "output": output_path.name,
        "output_sha256": sha256_file(output_path),
        "evidence": evidence,
        "promotion_allowed": False,
    }
    contract_path = args.output_dir / "contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in contract.items() if key != "evidence"}, indent=2))


if __name__ == "__main__":
    main()
