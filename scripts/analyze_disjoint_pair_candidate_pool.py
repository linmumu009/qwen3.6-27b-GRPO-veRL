#!/usr/bin/env python3
"""Audit current-definition train tasks for disjoint repair-pair acquisition.

The historical boss-aligned Parquet intentionally preserves the instruction
that appeared in the source trajectory.  Many of those instructions no longer
belong to the current task definition.  This audit does not silently approve
that drift.  It rebuilds only an in-memory candidate identity from the current
authoritative task manifest, executes the current verification SQL against the
immutable database, and reports whether enough leakage-free tasks exist for a
new pairwise training set.

No prompt, SQL, expected value, or tool output is emitted in the report.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from scripts.audit_formal_instruction_gold_alignment import classify
from scripts.prepare_pi_formal_dataset import gold_supported_by_rows
from scripts.prepare_repair_sft_dataset import (
    ALLOWED_SEMANTIC_WARNINGS,
    execute_sql_with_columns,
    load_parquet_rows,
    read_jsonl,
    sha256_file,
    task_id,
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def user_instruction(row: dict[str, Any]) -> str:
    prompt = row.get("prompt") or []
    users = [str(message.get("content") or "") for message in prompt if message.get("role") == "user"]
    return users[-1].strip() if users else ""


def verification_sql(row: dict[str, Any]) -> str:
    truth = (row.get("reward_model") or {}).get("ground_truth") or {}
    return str(truth.get("verification_sql") or "").strip()


def current_instruction(manifest_row: dict[str, Any]) -> str:
    instruction = str(manifest_row.get("natural_language_instruction") or "").strip()
    if instruction:
        return instruction
    for variant in manifest_row.get("instruction_variants") or []:
        if isinstance(variant, str) and variant.strip():
            return variant.strip()
        if isinstance(variant, dict):
            for key in ("instruction", "content", "text"):
                value = str(variant.get(key) or "").strip()
                if value:
                    return value
    return ""


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        current_task_id = str(row.get("task_id") or "")
        if not current_task_id or current_task_id in output:
            raise ValueError(f"missing or duplicate current task ID: {current_task_id!r}")
        output[current_task_id] = row
    return output


def identity_sets(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    task_ids: set[str] = set()
    instruction_hashes: set[str] = set()
    sql_hashes: set[str] = set()
    for row in rows:
        current_task_id = task_id(row)
        instruction = user_instruction(row)
        sql = verification_sql(row)
        if current_task_id:
            task_ids.add(current_task_id)
        if instruction:
            instruction_hashes.add(sha256_text(instruction))
        if sql:
            sql_hashes.add(sha256_text(sql))
    return {
        "task_ids": task_ids,
        "instruction_hashes": instruction_hashes,
        "sql_hashes": sql_hashes,
    }


def merge_identities(*items: dict[str, set[str]]) -> dict[str, set[str]]:
    merged = {"task_ids": set(), "instruction_hashes": set(), "sql_hashes": set()}
    for item in items:
        for key in merged:
            merged[key].update(item.get(key) or set())
    return merged


def audit_pool(
    *,
    train_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    manifest_by_task: dict[str, dict[str, Any]],
    database: Path,
    forbidden: dict[str, set[str]],
    minimum_available: int = 48,
) -> dict[str, Any]:
    review_by_task = {str(row.get("task_id") or ""): row for row in review_rows}
    tier_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    warning_combinations: Counter[tuple[str, ...]] = Counter()
    mechanical_failures: Counter[str] = Counter()
    source_alignment: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    task_families: Counter[str] = Counter()
    records: list[dict[str, Any]] = []

    resolved_database = database.resolve(strict=True)
    for source in train_rows:
        current_task_id = task_id(source)
        review = review_by_task.get(current_task_id)
        manifest = manifest_by_task.get(current_task_id)
        truth = (source.get("reward_model") or {}).get("ground_truth") or {}
        failure: str | None = None
        instruction = ""
        sql = ""
        answer_type = "missing"
        warnings: list[str] = []

        if not current_task_id:
            failure = "missing_task_id"
        elif review is None:
            failure = "missing_alignment_review"
        elif review.get("split") != "train":
            failure = "not_train_review"
        elif review.get("approved_for_grpo") is not True or review.get("review_status") != "approved":
            failure = "not_approved"
        elif manifest is None:
            failure = "missing_current_definition"
        else:
            source_alignment[
                "source_instruction_current"
                if review.get("source_instruction_in_current_task_definition") is True
                else "source_instruction_drift_rebuilt"
            ] += 1
            instruction = current_instruction(manifest)
            gold = manifest.get("gold_answer") or {}
            answer_type = str(gold.get("answer_type") or "missing")
            sql = str(gold.get("verification_sql") or "").strip()
            if not instruction:
                failure = "current_instruction_missing"
            elif answer_type not in {"numeric", "table"}:
                failure = "unsupported_answer_type"
            elif not sql:
                failure = "verification_sql_missing"
            else:
                try:
                    _, rows = execute_sql_with_columns(resolved_database, sql)
                except Exception:
                    failure = "verification_sql_failed"
                else:
                    if not rows:
                        failure = "verification_sql_empty"
                    elif not gold_supported_by_rows(
                        {"answer_type": answer_type, "value": gold.get("value")}, rows
                    ):
                        failure = "gold_result_mismatch"
            if failure is None:
                warnings = classify(
                    {
                        "instruction": instruction,
                        "gold": {"answer_type": answer_type, "verification_sql": sql},
                    }
                )

        identity_overlap: list[str] = []
        if failure is None:
            if current_task_id in forbidden["task_ids"]:
                identity_overlap.append("task_id")
            if sha256_text(instruction) in forbidden["instruction_hashes"]:
                identity_overlap.append("instruction")
            if sha256_text(sql) in forbidden["sql_hashes"]:
                identity_overlap.append("verification_sql")

        if failure is not None:
            tier = "mechanically_blocked"
            mechanical_failures[failure] += 1
        elif identity_overlap:
            tier = "forbidden_overlap"
        elif set(warnings).issubset(ALLOWED_SEMANTIC_WARNINGS):
            tier = "strict_available"
        else:
            tier = "review_required"
        tier_counts[tier] += 1
        if failure is None:
            warning_combinations[tuple(sorted(warnings))] += 1
            warning_counts.update(warnings)
            answer_types[answer_type] += 1
            task_families[str(truth.get("task_family") or "missing")] += 1
        records.append(
            {
                "task_id": current_task_id,
                "tier": tier,
                "semantic_warnings": sorted(warnings),
                "identity_overlap": identity_overlap,
                "mechanical_failure": failure,
                "source_instruction_rebuilt": bool(
                    review
                    and review.get("source_instruction_in_current_task_definition") is not True
                ),
                "answer_type": answer_type,
                "task_family": str(truth.get("task_family") or "missing"),
                "current_instruction_sha256": sha256_text(instruction) if instruction else None,
                "current_verification_sql_sha256": sha256_text(sql) if sql else None,
            }
        )

    strict_available = tier_counts["strict_available"]
    return {
        "contract": "current-definition-disjoint-pair-pool-audit-v1",
        "train_rows": len(train_rows),
        "minimum_strict_available": minimum_available,
        "strict_available": strict_available,
        "data_gate_passed": strict_available >= minimum_available,
        "tier_counts": dict(sorted(tier_counts.items())),
        "source_alignment": dict(sorted(source_alignment.items())),
        "semantic_warning_counts": dict(sorted(warning_counts.items())),
        "semantic_warning_combinations": [
            {"warnings": list(warnings), "count": count}
            for warnings, count in sorted(
                warning_combinations.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "mechanical_failure_counts": dict(sorted(mechanical_failures.items())),
        "answer_types_mechanically_valid": dict(sorted(answer_types.items())),
        "task_families_mechanically_valid": dict(sorted(task_families.items())),
        "forbidden_identity_counts": {
            key: len(value) for key, value in sorted(forbidden.items())
        },
        "records": records,
        "contains_prompts_sql_expected_values_or_tool_outputs": False,
        "next_action": (
            "prepare_current_definition_disjoint_rollout_candidates"
            if strict_available >= minimum_available
            else "expand_or_review_current_definition_training_pool_before_npu"
        ),
        "promotion_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--current-task-manifest", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--forbidden-parquet", type=Path, action="append", default=[])
    parser.add_argument("--minimum-available", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = load_parquet_rows(args.train_file)
    val_rows = load_parquet_rows(args.val_file)
    test_rows = load_parquet_rows(args.test_file)
    forbidden_sources = [identity_sets(val_rows), identity_sets(test_rows)]
    forbidden_sources.extend(
        identity_sets(load_parquet_rows(path)) for path in args.forbidden_parquet
    )
    result = audit_pool(
        train_rows=train_rows,
        review_rows=read_jsonl(args.review_queue),
        manifest_by_task=load_manifest(args.current_task_manifest),
        database=args.database,
        forbidden=merge_identities(*forbidden_sources),
        minimum_available=args.minimum_available,
    )
    result["source_sha256"] = {
        "train": sha256_file(args.train_file),
        "val": sha256_file(args.val_file),
        "test": sha256_file(args.test_file),
        "review_queue": sha256_file(args.review_queue),
        "current_task_manifest": sha256_file(args.current_task_manifest),
        "database": sha256_file(args.database),
        "forbidden_parquet": {
            path.name: sha256_file(path) for path in args.forbidden_parquet
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
