#!/usr/bin/env python3
"""Build leakage-free current-definition rows for Step 120 first-error collection.

The source train Parquet preserves historical prompts by design.  This builder
accepts only ``strict_available`` identities from the mechanically verified
pool audit and rewrites the user instruction and hidden verifier label from the
current authoritative task manifest.  It does not build a training dataset;
the output is only an inference input used to observe Step 120's first query.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.analyze_disjoint_pair_candidate_pool import (
    current_instruction,
    load_manifest,
    sha256_text,
)
from scripts.prepare_repair_sft_dataset import load_parquet_rows, sha256_file, task_id


CONTRACT = "current-definition-disjoint-pair-rollout-candidates-v1"
POOL_CONTRACT = "current-definition-disjoint-pair-pool-audit-v1"


def _index_source(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        current_task_id = task_id(row)
        if not current_task_id or current_task_id in output:
            raise ValueError(f"missing or duplicate source task ID: {current_task_id!r}")
        output[current_task_id] = row
    return output


def _stable_key(task: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{task}".encode()).hexdigest()


def _strict_records(audit: dict[str, Any]) -> list[dict[str, Any]]:
    if audit.get("contract") != POOL_CONTRACT:
        raise ValueError("candidate audit contract mismatch")
    if audit.get("data_gate_passed") is not True:
        raise ValueError("candidate pool data gate did not pass")
    records = [row for row in audit.get("records") or [] if row.get("tier") == "strict_available"]
    if len(records) != int(audit.get("strict_available") or -1):
        raise ValueError("strict candidate count differs from audit summary")
    return records


def build_candidate_rows(
    *,
    train_rows: list[dict[str, Any]],
    manifest_by_task: dict[str, dict[str, Any]],
    audit: dict[str, Any],
    row_count: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if row_count < 48 or row_count > 64:
        raise ValueError("row_count must be between 48 and 64")
    strict = _strict_records(audit)
    if len(strict) < row_count:
        raise ValueError(f"only {len(strict)} strict candidates for requested {row_count}")

    source_by_task = _index_source(train_rows)
    selected = sorted(strict, key=lambda row: _stable_key(str(row.get("task_id") or ""), seed))[
        :row_count
    ]
    output: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        current_task_id = str(record.get("task_id") or "")
        if current_task_id not in source_by_task or current_task_id not in manifest_by_task:
            raise ValueError(f"strict candidate missing source/current definition: {current_task_id}")
        source = copy.deepcopy(source_by_task[current_task_id])
        manifest = manifest_by_task[current_task_id]
        instruction = current_instruction(manifest)
        gold = manifest.get("gold_answer") or {}
        answer_type = str(gold.get("answer_type") or "")
        sql = str(gold.get("verification_sql") or "").strip()
        if answer_type not in {"numeric", "table"} or not instruction or not sql:
            raise ValueError(f"current definition became incomplete: {current_task_id}")
        if sha256_text(instruction) != record.get("current_instruction_sha256"):
            raise ValueError(f"instruction hash changed after pool audit: {current_task_id}")
        if sha256_text(sql) != record.get("current_verification_sql_sha256"):
            raise ValueError(f"verification SQL hash changed after pool audit: {current_task_id}")

        prompt = source.get("prompt") or []
        if [message.get("role") for message in prompt] != ["system", "user"]:
            raise ValueError(f"source prompt is not exactly system,user: {current_task_id}")
        source["prompt"] = [copy.deepcopy(prompt[0]), {"role": "user", "content": instruction}]
        source["data_source"] = "llin_pi_dwh_current_definition_pair_acquisition_v1"

        reward_model = copy.deepcopy(source.get("reward_model") or {})
        truth = copy.deepcopy(reward_model.get("ground_truth") or {})
        truth.update(
            {
                "task_id": current_task_id,
                "answer_type": answer_type,
                "expected_value_json": json.dumps(
                    gold.get("value"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "verification_sql": sql,
                "required_tables": sorted(
                    {str(value).casefold() for value in manifest.get("expected_tables") or []}
                ),
                "must_use_fields": sorted(
                    {
                        str(value).casefold()
                        for value in (manifest.get("verification_criteria") or {}).get(
                            "must_use_fields"
                        )
                        or []
                    }
                ),
            }
        )
        reward_model["ground_truth"] = truth
        source["reward_model"] = reward_model

        extra = copy.deepcopy(source.get("extra_info") or {})
        extra.update(
            {
                "index": index,
                "split": "train_disjoint_pair_acquisition",
                "instruction_sha256": sha256_text(instruction),
                "pair_acquisition_contract": CONTRACT,
                "source_instruction_rebuilt": bool(record.get("source_instruction_rebuilt")),
            }
        )
        source["extra_info"] = extra
        output.append(source)
        evidence.append(
            {
                "task_id": current_task_id,
                "answer_type": answer_type,
                "semantic_warnings": list(record.get("semantic_warnings") or []),
                "source_instruction_rebuilt": bool(record.get("source_instruction_rebuilt")),
                "instruction_sha256": sha256_text(instruction),
                "verification_sql_sha256": sha256_text(sql),
            }
        )

    if len({item["task_id"] for item in evidence}) != row_count:
        raise ValueError("selected task IDs are not unique")
    if len({item["instruction_sha256"] for item in evidence}) != row_count:
        raise ValueError("selected current instructions are not unique")
    return output, evidence


def build_contract(
    *,
    rows: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    audit: dict[str, Any],
    seed: str,
    output_path: Path,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "contract": CONTRACT,
        "rows": len(rows),
        "source_split": "train236_current_definition_disjoint_from_frozen16_val20_test20",
        "purpose": "step120_greedy_first_query_error_state_acquisition_only",
        "selection_seed": seed,
        "strict_pool_rows": int(audit["strict_available"]),
        "selected_task_ids": [item["task_id"] for item in evidence],
        "answer_types": dict(sorted(Counter(item["answer_type"] for item in evidence).items())),
        "source_instruction_rebuilt_rows": sum(
            bool(item["source_instruction_rebuilt"]) for item in evidence
        ),
        "semantic_warning_rows": sum(bool(item["semantic_warnings"]) for item in evidence),
        "unique_task_ids": len({item["task_id"] for item in evidence}),
        "unique_instruction_hashes": len({item["instruction_sha256"] for item in evidence}),
        "unique_verification_sql_hashes": len(
            {item["verification_sql_sha256"] for item in evidence}
        ),
        "forbidden_task_instruction_or_sql_overlap": 0,
        "current_instruction_and_gold_rebuilt_from_authority": True,
        "all_candidates_previously_mechanically_verified": True,
        "parquet_contains_hidden_verifier_labels": True,
        "contract_contains_prompts_sql_expected_values_or_tool_outputs": False,
        "output": output_path.name,
        "output_sha256": sha256_file(output_path),
        "source_sha256": source_hashes,
        "training_allowed": False,
        "promotion_allowed": False,
        "next_action": "run_step120_greedy_first_query_collection_then_filter_mechanical_first_errors",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--current-task-manifest", type=Path, required=True)
    parser.add_argument("--pool-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--seed", default="disjoint-pair-acquisition-20260812-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = json.loads(args.pool_audit.read_text(encoding="utf-8"))
    expected_sources = audit.get("source_sha256") or {}
    actual_sources = {
        "train": sha256_file(args.train_file),
        "current_task_manifest": sha256_file(args.current_task_manifest),
        "pool_audit": sha256_file(args.pool_audit),
    }
    for key in ("train", "current_task_manifest"):
        if actual_sources[key] != expected_sources.get(key):
            raise ValueError(f"{key} hash differs from pool audit")

    rows, evidence = build_candidate_rows(
        train_rows=load_parquet_rows(args.train_file),
        manifest_by_task=load_manifest(args.current_task_manifest),
        audit=audit,
        row_count=args.rows,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "disjoint_pair_rollout_candidates.parquet"
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    contract = build_contract(
        rows=rows,
        evidence=evidence,
        audit=audit,
        seed=args.seed,
        output_path=output_path,
        source_hashes=actual_sources,
    )
    contract_path = args.output_dir / "rollout_candidate_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
