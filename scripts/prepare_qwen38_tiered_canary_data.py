#!/usr/bin/env python3
"""Freeze at most 20 approved43 groups for the five-update tiered canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.outcome_gated_contract import evidence_binding_hash, stable_json_hash
from scripts import prepare_qwen38_approved43_outcome_training as approved_prep


CANARY_GROUPS = 20


def _answer_type(row: dict) -> str:
    truth = (row.get("reward_model") or {}).get("ground_truth") or {}
    value = str(truth.get("answer_type") or "").casefold()
    return "table" if value == "table" else "numeric"


def build(
    approved: Path,
    manifest: Path,
    tasks: Path,
    output: Path,
    summary: Path,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if approved_prep.file_sha256(approved) != approved_prep.PARQUET_SHA256:
        raise ValueError("approved43 Parquet hash mismatch")
    if approved_prep.file_sha256(manifest) != approved_prep.MANIFEST_SHA256:
        raise ValueError("approved43 manifest hash mismatch")
    approved_rows = pq.read_table(approved).to_pylist()
    manifest_rows = approved_prep.read_jsonl(manifest)
    task_rows = approved_prep.read_jsonl(tasks)
    if len(approved_rows) != 43 or len(manifest_rows) != 43:
        raise ValueError("approved43 package must contain exactly 43 members")
    manifest_by_instruction = {
        str(row["instruction_sha256"]): row for row in manifest_rows
    }
    instructions = [str(row["extra_info"]["instruction_sha256"]) for row in approved_rows]
    if len(set(instructions)) != 43 or set(instructions) != set(manifest_by_instruction):
        raise ValueError("approved43 identities are not exact and unique")
    if any(bool((row.get("extra_info") or {}).get("training_allowed")) for row in approved_rows):
        raise ValueError("source training_allowed must remain false")

    canonical: list[dict] = []
    for row in approved_rows:
        item = json.loads(json.dumps(row, ensure_ascii=False))
        instruction = str(item["extra_info"]["instruction_sha256"])
        source_index = int(item["extra_info"]["global_index"])
        if not 0 <= source_index < len(task_rows):
            raise ValueError("approved43 global_index is outside frozen tasks file")
        task = task_rows[source_index]
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

    by_type = {"numeric": [], "table": []}
    for row in canonical:
        by_type[_answer_type(row)].append(row)
    for values in by_type.values():
        values.sort(key=lambda row: str(row["extra_info"]["instruction_sha256"]))
    if min(map(len, by_type.values())) < CANARY_GROUPS // 2:
        raise ValueError("approved43 cannot provide ten numeric and ten table canary groups")

    selected: list[dict] = []
    for index in range(CANARY_GROUPS // 2):
        selected.extend((by_type["numeric"][index], by_type["table"][index]))
    identities: list[str] = []
    for nominal_index, row in enumerate(selected):
        extra = row["extra_info"]
        extra["canary_training_authorized"] = True
        extra["canary_nominal_group_index"] = nominal_index
        extra["canary_batch_index"] = nominal_index // 2
        identities.append(str(extra["instruction_sha256"]))

    if len(selected) != CANARY_GROUPS or len(set(identities)) != CANARY_GROUPS:
        raise ValueError("canary groups are not 20 exact unique approved43 members")
    pq.write_table(pa.Table.from_pylist(selected), output)
    os.chmod(output, 0o600)
    identity_set_sha256 = hashlib.sha256(
        "\n".join(sorted(identities)).encode("ascii")
    ).hexdigest()
    result = {
        "contract": "qwen38-approved43-tiered-query-cost-canary-data-v1",
        "approved43_parquet_sha256": approved_prep.PARQUET_SHA256,
        "approved43_manifest_sha256": approved_prep.MANIFEST_SHA256,
        "membership_authority": "exact_approved43_package_only",
        "pool_scanning_for_training": False,
        "nominal_groups": CANARY_GROUPS,
        "groups_per_nominal_batch": 2,
        "maximum_nominal_batches": 10,
        "target_actual_optimizer_steps": 5,
        "responses_per_group": 8,
        "answer_type_counts": {key: len([row for row in selected if _answer_type(row) == key]) for key in by_type},
        "unique_instruction_count": len(set(identities)),
        "selected_identity_set_sha256": identity_set_sha256,
        "schedule_sha256": approved_prep.file_sha256(output),
        "source_training_allowed_all_false": all(
            (row.get("extra_info") or {}).get("training_allowed") is False for row in selected
        ),
        "formal_full_training_allowed": False,
    }
    summary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved43", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.approved43, args.manifest, args.tasks, args.output, args.safe_summary), sort_keys=True))


if __name__ == "__main__":
    main()
