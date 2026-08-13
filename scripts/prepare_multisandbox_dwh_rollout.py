#!/usr/bin/env python3
"""Build the strict non-v15 multi-sandbox DWH rollout-screening dataset.

The input and parquet outputs contain prompts, hidden gold, SQL, and task
identities.  Keep them in a permission-restricted server run directory.  The
safe manifest contains aggregate counts and hashes only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.boss_pi_contract import canonical_json, contract_hashes, load_boss_pi_contract
from scripts.profile_boss_sandbox_catalog import canonical_hash


CONTRACT = "boss-multisandbox-dwh-rollout-screening-v1"
DIRECT_TASK_TYPES = {"aggregate_query", "single_metric_query", "comparison_analysis"}
TOOL_NAMES = ["bash", "read", "write", "edit"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def strict_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("version") or "") == "20260628_v15":
            continue
        if row.get("globally_unique_instruction") is not True:
            continue
        if str(row.get("task_type") or "") not in DIRECT_TASK_TYPES:
            continue
        gold = row.get("gold") or {}
        if (
            str(row.get("task_category") or "") != "answerable"
            or row.get("mechanical_sql_verified") is not True
            or list(row.get("semantic_review_flags") or [])
            or str(gold.get("answer_type") or "") not in {"numeric", "table"}
            or not str(gold.get("verification_sql") or "").strip()
        ):
            raise ValueError("strict candidate violates the profiler contract")
        instruction = str(row.get("instruction") or "").strip()
        environment_id = str(row.get("environment_id") or "").strip()
        version = str(row.get("version") or "").strip()
        if not instruction or environment_id != f"sft/{version}":
            raise ValueError("strict candidate has an invalid instruction or environment")
        if canonical_hash(instruction) != row.get("instruction_sha256"):
            raise ValueError("strict candidate instruction hash mismatch")
        if canonical_hash(gold) != row.get("gold_sha256"):
            raise ValueError("strict candidate gold hash mismatch")
        selected.append(row)

    selected.sort(key=lambda row: (str(row["version"]), str(row["instruction_sha256"])))
    if len({str(row["instruction_sha256"]) for row in selected}) != len(selected):
        raise ValueError("strict candidate instructions are not globally unique")
    verifier_ids = {
        f"{row['environment_id']}:{row['task_id']}" for row in selected
    }
    if len(verifier_ids) != len(selected):
        raise ValueError("strict candidate environment/task identities are not unique")
    return selected


def stable_key(row: dict[str, Any], seed: str) -> str:
    return hashlib.sha256(f"{seed}:{row['instruction_sha256']}".encode()).hexdigest()


def select_probe(rows: list[dict[str, Any]], count: int, seed: str) -> list[dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise ValueError("probe task count must be within the strict pool")
    ordered = sorted(rows, key=lambda row: stable_key(row, seed))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    strata = [
        (answer_type, task_type)
        for answer_type in ("numeric", "table")
        for task_type in sorted(DIRECT_TASK_TYPES)
    ]
    for answer_type, task_type in strata:
        match = next(
            (
                row
                for row in ordered
                if str((row.get("gold") or {}).get("answer_type")) == answer_type
                and str(row.get("task_type")) == task_type
                and str(row["instruction_sha256"]) not in seen
            ),
            None,
        )
        if match is not None and len(selected) < count:
            selected.append(match)
            seen.add(str(match["instruction_sha256"]))
    for row in ordered:
        identity = str(row["instruction_sha256"])
        if len(selected) == count:
            break
        if identity not in seen:
            selected.append(row)
            seen.add(identity)
    return sorted(selected, key=lambda row: (str(row["version"]), str(row["instruction_sha256"])))


def build_record(
    row: dict[str, Any],
    *,
    system_prompt: str,
    guidance_prefix: str,
    index: int,
) -> dict[str, Any]:
    gold = row["gold"]
    verifier_id = f"{row['environment_id']}:{row['task_id']}"
    ground_truth = {
        "verifier_id": verifier_id,
        "task_id": str(row["task_id"]),
        "environment_id": str(row["environment_id"]),
        "answer_type": str(gold["answer_type"]),
        "expected_value_json": canonical_json(gold.get("value")),
        "verification_sql": str(gold["verification_sql"]),
        "required_tables": sorted(
            {str(value).casefold() for value in row.get("expected_tables") or []}
        ),
        "must_use_fields": [],
        "task_family": "dwh",
        "reward_contract": "final-outcome-screening-only-v1",
        "abs_tol": 1e-3,
        "rel_tol": 1e-5,
    }
    tools_kwargs = {
        name: {"create_kwargs": {"environment_id": str(row["environment_id"])}}
        for name in TOOL_NAMES
    }
    return {
        "data_source": "llin_boss_multisandbox_dwh_rollout_v1",
        "agent_name": "pi_agent",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": guidance_prefix + str(row["instruction"])},
        ],
        "ability": "boss_pi_dwh",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": index,
            "split": "multisandbox_rollout_screening",
            "source_version": str(row["version"]),
            "verifier_id": verifier_id,
            "environment_id": str(row["environment_id"]),
            "instruction_sha256": str(row["instruction_sha256"]),
            "gold_sha256": str(row["gold_sha256"]),
            "mechanical_screen_passed": True,
            "explicit_semantic_reviewed": False,
            "training_allowed": False,
            "response_messages_in_grpo_input": 0,
            "need_tools_kwargs": True,
            "tool_selection": TOOL_NAMES,
            "tools_kwargs": tools_kwargs,
        },
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def count_values(rows: list[dict[str, Any]], key) -> dict[str, int]:
    return dict(sorted(Counter(str(key(row)) for row in rows).items()))


def prepare(
    candidates_path: Path,
    output_dir: Path,
    *,
    expected_rows: int,
    probe_tasks: int,
    seed: str,
) -> dict[str, Any]:
    strict = strict_candidates(read_jsonl(candidates_path))
    if len(strict) != expected_rows:
        raise ValueError(f"expected {expected_rows} strict candidates, got {len(strict)}")
    probe = select_probe(strict, probe_tasks, seed)
    contract = load_boss_pi_contract()
    guidance = str((contract.get("runtime") or {}).get("guidance_prefix") or "")
    if not guidance:
        raise ValueError("boss PI guidance prefix is missing")

    full_records = [
        build_record(
            row,
            system_prompt=str(contract["system_prompt"]),
            guidance_prefix=guidance,
            index=index,
        )
        for index, row in enumerate(strict)
    ]
    probe_ids = {str(row["instruction_sha256"]) for row in probe}
    probe_records = [
        record
        for row, record in zip(strict, full_records, strict=True)
        if str(row["instruction_sha256"]) in probe_ids
    ]
    full_path = output_dir / "boss_multisandbox_dwh_281.sensitive.parquet"
    probe_path = output_dir / f"boss_multisandbox_dwh_probe{probe_tasks}.sensitive.parquet"
    write_parquet(full_path, full_records)
    write_parquet(probe_path, probe_records)

    manifest = {
        "contract": CONTRACT,
        "source_candidate_rows": len(read_jsonl(candidates_path)),
        "strict_rows": len(strict),
        "probe_rows": len(probe),
        "samples_per_task": 8,
        "versions": count_values(strict, lambda row: row["version"]),
        "task_types": count_values(strict, lambda row: row["task_type"]),
        "answer_types": count_values(strict, lambda row: row["gold"]["answer_type"]),
        "context_contract": {
            "max_prompt_tokens": 4096,
            "max_response_tokens": 45056,
            "max_context_tokens": 49152,
        },
        "sampling_contract": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "boss_contract_hashes": contract_hashes(contract),
        "source_candidates_sha256": file_sha256(candidates_path),
        "full_dataset_sha256": file_sha256(full_path),
        "probe_dataset_sha256": file_sha256(probe_path),
        "sensitive_artifacts_permissions": "0600",
        "contains_prompts_gold_sql_task_ids_or_server_paths": False,
        "explicit_semantic_review_completed": False,
        "rollout_screening_allowed": True,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "boss_multisandbox_dwh_rollout_safe_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=281)
    parser.add_argument("--probe-tasks", type=int, default=8)
    parser.add_argument("--seed", default="boss-multisandbox-dwh-rollout-probe-20260813-v1")
    args = parser.parse_args()
    result = prepare(
        args.candidates,
        args.output_dir,
        expected_rows=args.expected_rows,
        probe_tasks=args.probe_tasks,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
