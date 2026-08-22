#!/usr/bin/env python3
"""Build and validate the exact approved 21-task, four-exposure train view."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CONTRACT = "v15-mixed21-four-exposure-strict-gated-v1"
APPROVAL_FILENAME = "mixed_approved_candidates.sensitive.parquet"
EXPECTED_TASKS = 21
EXPOSURES = 4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def identity(row: dict[str, Any]) -> tuple[str, str]:
    extra = row.get("extra_info") or {}
    instruction_hash = str(extra.get("instruction_sha256") or "")
    gold_hash = str(extra.get("gold_sha256") or "")
    if not instruction_hash:
        instruction_hash = canonical_hash(row.get("prompt"))
    if not gold_hash:
        gold_hash = canonical_hash((row.get("reward_model") or {}).get("ground_truth"))
    return instruction_hash, gold_hash


def _extra(row: dict[str, Any]) -> dict[str, Any]:
    extra = row.get("extra_info")
    if not isinstance(extra, dict):
        raise ValueError("every approved row must contain mapping extra_info")
    return extra


def _ground_truth(row: dict[str, Any]) -> dict[str, Any]:
    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict) or not isinstance(reward_model.get("ground_truth"), dict):
        raise ValueError("every approved row must contain reward_model.ground_truth")
    return reward_model["ground_truth"]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _write_private(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    os.chmod(path, 0o600)


def _assert_source_contract(source: Path, audit_summary: Path) -> list[dict[str, Any]]:
    if source.name != APPROVAL_FILENAME:
        raise ValueError(f"training source must be the exact independent approval file {APPROVAL_FILENAME}")
    summary = json.loads(audit_summary.read_text(encoding="utf-8"))
    mixed_review = summary.get("mixed_review") or {}
    if int(mixed_review.get("approved_candidates", -1)) != EXPECTED_TASKS:
        raise ValueError("audit summary does not approve exactly 21 original mixed tasks")
    if int(mixed_review.get("reviewed", -1)) != EXPECTED_TASKS:
        raise ValueError("audit summary does not review all 21 original mixed tasks")
    if summary.get("promotion_allowed") is not False:
        raise ValueError("audit summary must keep promotion_allowed=false")

    rows = _read_rows(source)
    if len(rows) != EXPECTED_TASKS or len({identity(row) for row in rows}) != EXPECTED_TASKS:
        raise ValueError("approval parquet must contain exactly 21 unique tasks")
    if any(bool(_extra(row).get("training_allowed")) for row in rows):
        raise ValueError("the independent approval parquet must preserve original training_allowed=false")
    if any(bool(_extra(row).get("promotion_allowed")) for row in rows):
        raise ValueError("approved tasks must preserve promotion_allowed=false")
    if any(str(_ground_truth(row).get("answer_type")) != "numeric" for row in rows):
        raise ValueError("this run is restricted to the 21 numeric mixed approvals")
    return rows


def _authorized_row(row: dict[str, Any], exposure: int | None = None) -> dict[str, Any]:
    current = copy.deepcopy(row)
    extra = _extra(current)
    extra["training_allowed"] = True
    extra["promotion_allowed"] = False
    extra["training_authorization_contract"] = CONTRACT
    extra["training_authorized_by_independent_approval_list"] = True
    extra["reward_contract"] = "strict-correctness-gated-v3"
    if exposure is not None:
        extra["training_exposure_index"] = exposure
    return current


def build(
    source: Path,
    audit_summary: Path,
    canonical: Path,
    schedule: Path,
    validation: Path,
    safe_summary: Path,
    seed: int,
) -> dict[str, Any]:
    approved = _assert_source_contract(source, audit_summary)
    train_ids = {identity(row) for row in approved}
    validation_rows = _read_rows(validation)
    validation_ids = {identity(row) for row in validation_rows}
    if len(validation_ids) != len(validation_rows):
        raise ValueError("validation file contains duplicate task identities")
    if train_ids & validation_ids:
        raise ValueError("approved training tasks overlap the sealed validation file")

    canonical_rows = [_authorized_row(row) for row in approved]
    schedule_rows: list[dict[str, Any]] = []
    for exposure in range(1, EXPOSURES + 1):
        current = [_authorized_row(row, exposure) for row in approved]
        random.Random(f"{seed}:{exposure}").shuffle(current)
        schedule_rows.extend(current)

    if len(schedule_rows) != EXPECTED_TASKS * EXPOSURES:
        raise AssertionError("unexpected training schedule length")
    _write_private(canonical, canonical_rows)
    _write_private(schedule, schedule_rows)
    summary = {
        "contract": CONTRACT,
        "status": "passed",
        "approved_source_rows": len(approved),
        "approved_unique_tasks": len(train_ids),
        "answer_type_counts": {"numeric": EXPECTED_TASKS},
        "exposures_per_task": EXPOSURES,
        "schedule_groups": len(schedule_rows),
        "responses_per_group": 8,
        "online_trajectories": len(schedule_rows) * 8,
        "groups_per_optimizer_step": 2,
        "optimizer_steps": len(schedule_rows) // 2,
        "validation_rows": len(validation_rows),
        "train_validation_overlap": 0,
        "source_training_allowed_true": 0,
        "schedule_training_allowed_true": len(schedule_rows),
        "promotion_allowed": False,
        "conditional_reward_repaired_candidates_included": 0,
        "source_file_sha256": file_sha256(source),
        "audit_summary_sha256": file_sha256(audit_summary),
        "canonical_file_sha256": file_sha256(canonical),
        "schedule_file_sha256": file_sha256(schedule),
        "validation_file_sha256": file_sha256(validation),
        "seed": seed,
        "contains_prompts_gold_sql_outputs_or_task_ids": False,
    }
    safe_summary.parent.mkdir(parents=True, exist_ok=True)
    safe_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(safe_summary, 0o600)
    return summary


def validate(
    source: Path,
    audit_summary: Path,
    canonical: Path,
    schedule: Path,
    validation: Path,
    safe_summary: Path,
) -> dict[str, Any]:
    approved = _assert_source_contract(source, audit_summary)
    canonical_rows = _read_rows(canonical)
    schedule_rows = _read_rows(schedule)
    validation_rows = _read_rows(validation)
    summary = json.loads(safe_summary.read_text(encoding="utf-8"))
    approved_ids = {identity(row) for row in approved}
    canonical_ids = {identity(row) for row in canonical_rows}
    validation_ids = {identity(row) for row in validation_rows}
    if summary.get("contract") != CONTRACT or summary.get("status") != "passed":
        raise ValueError("safe summary contract is invalid")
    if len(canonical_rows) != EXPECTED_TASKS or canonical_ids != approved_ids:
        raise ValueError("canonical train view is not the exact 21-task approval set")
    if len(schedule_rows) != EXPECTED_TASKS * EXPOSURES:
        raise ValueError("schedule must contain exactly 84 groups")
    if {identity(row) for row in schedule_rows} != approved_ids:
        raise ValueError("schedule contains a task outside the approval set")
    if {sum(identity(row) == task for row in schedule_rows) for task in approved_ids} != {EXPOSURES}:
        raise ValueError("every approved task must appear exactly four times")
    if any(not bool(_extra(row).get("training_allowed")) for row in schedule_rows):
        raise ValueError("every schedule row must carry the derived authorization")
    if canonical_ids & validation_ids:
        raise ValueError("sealed validation overlaps training")
    expected_hashes = {
        "source_file_sha256": file_sha256(source),
        "audit_summary_sha256": file_sha256(audit_summary),
        "canonical_file_sha256": file_sha256(canonical),
        "schedule_file_sha256": file_sha256(schedule),
        "validation_file_sha256": file_sha256(validation),
    }
    if any(summary.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("safe summary file hash mismatch")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "validate"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--safe-summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    function = build if args.mode == "build" else validate
    kwargs = {
        "source": args.source,
        "audit_summary": args.audit_summary,
        "canonical": args.canonical,
        "schedule": args.schedule,
        "validation": args.validation,
        "safe_summary": args.safe_summary,
    }
    if args.mode == "build":
        kwargs["seed"] = args.seed
    print(json.dumps(function(**kwargs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
