#!/usr/bin/env python3
"""Prepare a fail-closed 42-task semantic-review pilot on the server.

The sensitive review packet contains current instructions and gold definitions
and must remain server-side.  The emitted contract contains aggregates only.
No task is approved automatically; explicit semantic adjudication is required.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.analyze_disjoint_pair_candidate_pool import (
    current_instruction,
    load_manifest,
    sha256_text,
)
from scripts.prepare_repair_sft_dataset import sha256_file


POOL_CONTRACT = "current-definition-disjoint-pair-pool-audit-v1"
CONTRACT = "disjoint-pair-semantic-review-pilot-v1"
WARNING_WEIGHTS = {
    "latest_instruction_without_temporal_sql": 0,
    "limit_without_order_by": 1,
    "numeric_gold_without_aggregation": 2,
    "broad_instruction_exact_hidden_target": 3,
    "broad_instruction_reduced_to_row_count": 4,
}


def risk_key(record: dict[str, Any], seed: str) -> tuple[int, int, str]:
    warnings = list(record.get("semantic_warnings") or [])
    score = sum(WARNING_WEIGHTS.get(str(warning), 10) for warning in warnings)
    identity = str(record.get("task_id") or "")
    tie_break = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()
    return score, len(warnings), tie_break


def select_records(
    records: list[dict[str, Any]], review_count: int, seed: str
) -> list[dict[str, Any]]:
    if review_count <= 0:
        raise ValueError("review_count must be positive")
    review_required = [row for row in records if row.get("tier") == "review_required"]
    if len(review_required) < review_count:
        raise ValueError(
            f"only {len(review_required)} review-required records for requested {review_count}"
        )
    selected = sorted(review_required, key=lambda row: risk_key(row, seed))[:review_count]
    identities = [str(row.get("task_id") or "") for row in selected]
    if "" in identities or len(set(identities)) != len(identities):
        raise ValueError("selected review identities are missing or duplicated")
    return selected


def build_pilot(
    *,
    pool_audit_file: Path,
    current_task_manifest: Path,
    output_dir: Path,
    review_count: int = 42,
    seed: str = "llin-review-pilot42-20260813-v1",
) -> dict[str, Any]:
    pool = json.loads(pool_audit_file.read_text(encoding="utf-8"))
    if pool.get("contract") != POOL_CONTRACT:
        raise ValueError("candidate pool contract mismatch")
    if pool.get("promotion_allowed") is not False:
        raise ValueError("candidate pool audit is not fail closed")
    if pool.get("source_sha256", {}).get("current_task_manifest") != sha256_file(
        current_task_manifest
    ):
        raise ValueError("current task manifest differs from candidate pool audit")
    records = list(pool.get("records") or [])
    if len(records) != int(pool.get("train_rows") or 0):
        raise ValueError("candidate pool record grain differs")
    selected = select_records(records, review_count, seed)
    manifest = load_manifest(current_task_manifest)

    packet: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    combinations: Counter[tuple[str, ...]] = Counter()
    answer_types: Counter[str] = Counter()
    task_families: Counter[str] = Counter()
    risk_scores: list[int] = []
    for position, record in enumerate(selected):
        identity = str(record["task_id"])
        source = manifest.get(identity)
        if source is None:
            raise ValueError(f"selected review task absent from current manifest: {identity}")
        instruction = current_instruction(source)
        gold = source.get("gold_answer") or {}
        sql = str(gold.get("verification_sql") or "").strip()
        if (
            sha256_text(instruction) != record.get("current_instruction_sha256")
            or sha256_text(sql) != record.get("current_verification_sql_sha256")
        ):
            raise ValueError(f"selected review identity hashes differ: {identity}")
        warnings = [str(value) for value in record.get("semantic_warnings") or []]
        score = risk_key(record, seed)[0]
        warning_counts.update(warnings)
        combinations[tuple(sorted(warnings))] += 1
        answer_types[str(record.get("answer_type") or "missing")] += 1
        task_families[str(record.get("task_family") or "missing")] += 1
        risk_scores.append(score)
        packet.append(
            {
                "review_index": position,
                "task_id": identity,
                "natural_language_instruction": instruction,
                "gold_answer": gold,
                "semantic_warnings": warnings,
                "mechanically_verified": True,
                "forbidden_identity_overlap": False,
                "adjudication": {
                    "decision": None,
                    "instruction_unambiguously_entails_gold": None,
                    "verification_sql_answers_instruction": None,
                    "expected_value_supported_by_query_result": True,
                    "notes": None,
                    "reviewer": None,
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / "semantic_review_pilot42.sensitive.jsonl"
    packet_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet),
        encoding="utf-8",
    )
    packet_path.chmod(0o600)
    return {
        "contract": CONTRACT,
        "date": "2026-08-13",
        "selection_role": "lowest_mechanical_risk_semantic_approval_rate_pilot",
        "review_required_pool": sum(row.get("tier") == "review_required" for row in records),
        "selected_tasks": len(packet),
        "selection_seed": seed,
        "selection_order": "ascending_warning_weight_then_warning_count_then_seeded_sha256",
        "warning_weights": WARNING_WEIGHTS,
        "selected_warning_counts": dict(sorted(warning_counts.items())),
        "selected_warning_combinations": [
            {"warnings": list(warnings), "count": count}
            for warnings, count in sorted(combinations.items(), key=lambda item: (-item[1], item[0]))
        ],
        "selected_answer_types": dict(sorted(answer_types.items())),
        "selected_task_families": dict(sorted(task_families.items())),
        "risk_score_min": min(risk_scores),
        "risk_score_max": max(risk_scores),
        "all_current_instruction_and_sql_hashes_match_pool_audit": True,
        "all_mechanical_and_forbidden_identity_gates_passed": True,
        "automatic_semantic_approvals": 0,
        "completed_semantic_reviews": 0,
        "review_packet": packet_path.name,
        "review_packet_sha256": sha256_file(packet_path),
        "review_packet_permissions": "0600",
        "review_packet_contains_sensitive_task_content": True,
        "review_packet_must_remain_server_side": True,
        "source_sha256": {
            "pool_audit": sha256_file(pool_audit_file),
            "current_task_manifest": sha256_file(current_task_manifest),
        },
        "next_action": "complete_explicit_semantic_adjudication_then_measure_approval_rate",
        "may_be_used_as_training_or_rollout_data": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-audit", type=Path, required=True)
    parser.add_argument("--current-task-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-count", type=int, default=42)
    parser.add_argument("--seed", default="llin-review-pilot42-20260813-v1")
    args = parser.parse_args()
    result = build_pilot(
        pool_audit_file=args.pool_audit,
        current_task_manifest=args.current_task_manifest,
        output_dir=args.output_dir,
        review_count=args.review_count,
        seed=args.seed,
    )
    contract_path = args.output_dir / "semantic_review_pilot42_contract.json"
    contract_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
