#!/usr/bin/env python3
"""Build the frozen 43x4 schedule without scanning the source 100-task pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.outcome_gated_contract import evidence_binding_hash, stable_json_hash


PARQUET_SHA256 = "d86b53d906806b150d43a508dce9b0dd6d05105c07e03961e8e7bf9439ccd944"
MANIFEST_SHA256 = "1426bc09a3dbaf4709fd89227790603afb7a2bf11beeba80946057d490e0f424"
TASKS = 43
EXPOSURES = 4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(
    approved_path: Path,
    manifest_path: Path,
    tasks_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    if file_sha256(approved_path) != PARQUET_SHA256:
        raise ValueError("approved43 Parquet hash mismatch")
    if file_sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError("approved43 manifest hash mismatch")
    approved = pq.read_table(approved_path).to_pylist()
    manifest = read_jsonl(manifest_path)
    tasks = read_jsonl(tasks_path)
    if len(approved) != TASKS or len(manifest) != TASKS:
        raise ValueError("approved43 package must contain exactly 43 members")
    manifest_by_instruction = {
        str(row["instruction_sha256"]): row for row in manifest
    }
    instructions = [str(row["extra_info"]["instruction_sha256"]) for row in approved]
    if len(set(instructions)) != TASKS or set(instructions) != set(manifest_by_instruction):
        raise ValueError("approved43 identities are not exact and unique")
    if any(bool((row.get("extra_info") or {}).get("training_allowed")) for row in approved):
        raise ValueError("source training_allowed must remain false")

    canonical: list[dict[str, Any]] = []
    binding_hashes: list[str] = []
    for row in approved:
        item = json.loads(json.dumps(row, ensure_ascii=False))
        instruction = str(item["extra_info"]["instruction_sha256"])
        source_index = int(item["extra_info"]["global_index"])
        if not 0 <= source_index < len(tasks):
            raise ValueError("approved43 global_index is outside frozen tasks file")
        task = tasks[source_index]
        truth = item["reward_model"]["ground_truth"]
        criteria = task.get("verification_criteria") or {}
        truth["evidence_plan"] = task.get("evidence_plan") or {}
        truth["required_tables"] = task.get("expected_tables") or truth.get("required_tables", [])
        truth["must_use_fields"] = criteria.get("must_use_fields") or truth.get("must_use_fields", [])
        binding = evidence_binding_hash(truth)
        truth["process_evidence_binding_sha256"] = binding
        item["extra_info"]["process_evidence_binding_sha256"] = binding
        item["extra_info"]["approved43_authorization"] = True
        item["extra_info"]["approved43_manifest_identity_sha256"] = stable_json_hash(
            manifest_by_instruction[instruction]
        )
        canonical.append(item)
        binding_hashes.append(binding)

    schedule: list[dict[str, Any]] = []
    for exposure in range(EXPOSURES):
        for task_index, row in enumerate(canonical):
            item = json.loads(json.dumps(row, ensure_ascii=False))
            item["extra_info"]["exposure_index"] = exposure
            item["extra_info"]["nominal_group_index"] = exposure * TASKS + task_index
            schedule.append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pq.write_table(pa.Table.from_pylist(schedule), output_path)
    os.chmod(output_path, 0o600)
    summary = {
        "contract": "qwen38-approved43-four-exposure-outcome-gated-v5",
        "approved_members": TASKS,
        "exposures_per_task": EXPOSURES,
        "nominal_groups": len(schedule),
        "responses_per_group": 8,
        "accepted_online_trajectories": len(schedule) * 8,
        "approved43_parquet_sha256": PARQUET_SHA256,
        "approved43_manifest_sha256": MANIFEST_SHA256,
        "unique_evidence_binding_hashes": len(set(binding_hashes)),
        "schedule_sha256": file_sha256(output_path),
        "source_training_allowed_all_false": True,
        "membership_authority": "exact_approved43_package_only",
        "excluded_source_pool_scanning": True,
        "formal_training_allowed": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved43", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.approved43, args.manifest, args.tasks, args.output, args.safe_summary), sort_keys=True))


if __name__ == "__main__":
    main()
