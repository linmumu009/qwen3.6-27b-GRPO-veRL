#!/usr/bin/env python3
"""Apply frozen Qwen3.8 candidate-review decisions on the source host."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.audit_qwen38_grpo_candidates import CONTRACT


def candidate_key(row: dict[str, Any]) -> tuple[str, str]:
    extra = row["extra_info"]
    return str(extra["instruction_sha256"]), str(extra["gold_sha256"])


def write_private_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def apply_review(
    candidate_paths: Sequence[Path],
    decisions_path: Path,
    output_dir: Path,
    *,
    host_label: str,
) -> dict[str, Any]:
    payload = json.loads(decisions_path.read_text(encoding="utf-8"))
    if payload.get("contract") != CONTRACT:
        raise ValueError("unsupported Qwen3.8 review decision contract")
    decisions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in payload["decisions"]:
        if row["host"] != host_label:
            continue
        key = (str(row["batch"]), str(row["instruction_sha256"]), str(row["gold_sha256"]))
        if key in decisions:
            raise ValueError(f"duplicate review decision for host {host_label}")
        decisions[key] = row

    approved_rows: list[dict[str, Any]] = []
    source_schema: pa.Schema | None = None
    reviewed = rejected = pending = 0
    seen: set[tuple[str, str, str]] = set()
    by_batch: dict[str, dict[str, int]] = {}
    for path in candidate_paths:
        table = pq.read_table(path)
        if source_schema is None:
            source_schema = table.schema
        elif table.schema != source_schema:
            raise ValueError("candidate Parquet schemas differ")
        batch = path.parent.name
        batch_counts = {"candidates": 0, "approved": 0, "rejected": 0, "needs_manual_review": 0}
        for row in table.to_pylist():
            instruction_hash, gold_hash = candidate_key(row)
            key = (batch, instruction_hash, gold_hash)
            if key in seen:
                raise ValueError("duplicate candidate identity during review application")
            seen.add(key)
            decision = decisions.get(key)
            if decision is None:
                raise ValueError("candidate is missing a frozen review decision")
            reviewed += 1
            batch_counts["candidates"] += 1
            outcome = str(decision["decision"])
            if outcome == "approved_candidate":
                required_checks = (
                    decision["instruction_unambiguously_entails_gold"],
                    decision["verification_sql_fully_answers_instruction"],
                    decision["expected_value_supported_by_query_result"],
                    decision["final_outcome_routing_trustworthy"],
                )
                if not all(required_checks):
                    raise ValueError("approved candidate failed a required semantic check")
                approved = deepcopy(row)
                approved["extra_info"]["explicit_semantic_reviewed"] = True
                approved["extra_info"]["semantic_review_decision"] = "approved_candidate"
                approved["extra_info"]["training_allowed"] = False
                approved["extra_info"]["promotion_allowed"] = False
                approved_rows.append(approved)
                batch_counts["approved"] += 1
            elif outcome == "rejected":
                rejected += 1
                batch_counts["rejected"] += 1
            elif outcome == "needs_manual_review":
                pending += 1
                batch_counts["needs_manual_review"] += 1
            else:
                raise ValueError(f"unsupported review decision: {outcome}")
        by_batch[batch] = batch_counts

    missing = set(decisions) - seen
    if missing:
        raise ValueError("review decisions contain candidates absent from host Parquets")
    if source_schema is None:
        raise ValueError("no candidate Parquets supplied")
    approved_table = (
        pa.Table.from_pylist(approved_rows, schema=source_schema)
        if approved_rows
        else pa.Table.from_batches([], schema=source_schema)
    )
    write_private_table(
        output_dir / "semantic_approved_candidates.sensitive.parquet",
        approved_table,
    )
    summary = {
        "contract": "llin-qwen38-grpo-candidate-review-application-v1",
        "source_review_contract": CONTRACT,
        "host": host_label,
        "reviewed_candidates": reviewed,
        "approved_candidates": len(approved_rows),
        "rejected_candidates": rejected,
        "needs_manual_review": pending,
        "all_candidates_accounted_for": reviewed == len(approved_rows) + rejected + pending,
        "by_batch": by_batch,
        "explicit_semantic_review_completed_for_approved_rows": True,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_sql_gold_values_task_ids_hashes_server_paths_or_tool_outputs": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "safe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host-label", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            apply_review(
                args.candidate,
                args.decisions,
                args.output_dir,
                host_label=args.host_label,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
