#!/usr/bin/env python3
"""Freeze v22 evaluation and prepare fresh v23-v26 three-host acquisition shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_open_multisandbox_dwh_rollout import prepare
from scripts.prepare_plan_first_dwh_model_comparison import file_sha256


CONTRACT = "llin-qwen38-fresh-v22-v26-acquisition-freeze-v1"
HOSTS = ("m05", "m06", "m00")
VERSIONS = ("v22", "v23", "v24", "v25", "v26")
SOURCE_TEMPLATE = "20260815_llin_dwh_open_api_v3_{version}"
PILOT_TASKS = 100


def _identity(row: dict[str, Any]) -> str:
    return str((row.get("extra_info") or {}).get("instruction_sha256") or "")


def _difficulty(row: dict[str, Any]) -> int:
    return int((row.get("extra_info") or {}).get("difficulty_level", 0))


def _stable(identity: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def _write_private(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist([], schema=schema)
    pq.write_table(table, temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(
        sorted(Counter(str((row.get("extra_info") or {}).get(field, "unknown")) for row in rows).items())
    )


def _balanced_assign(
    arms: dict[str, list[dict[str, Any]]],
    *,
    seed: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    assigned = {host: {arm: [] for arm in arms} for host in HOSTS}
    host_totals: Counter[str] = Counter()
    difficulty_totals: dict[str, Counter[int]] = defaultdict(Counter)
    for arm, rows in arms.items():
        by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_level[_difficulty(row)].append(row)
        for level in sorted(by_level):
            ordered = sorted(
                by_level[level], key=lambda row: _stable(_identity(row), f"{seed}:{arm}:L{level}")
            )
            tie_order = sorted(HOSTS, key=lambda host: _stable(host, f"{seed}:{arm}:L{level}:hosts"))
            tie_rank = {host: index for index, host in enumerate(tie_order)}
            for row in ordered:
                host = min(
                    HOSTS,
                    key=lambda item: (
                        difficulty_totals[item][level],
                        host_totals[item],
                        len(assigned[item][arm]),
                        tie_rank[item],
                    ),
                )
                assigned[host][arm].append(row)
                host_totals[host] += 1
                difficulty_totals[host][level] += 1
    if max(host_totals.values()) - min(host_totals.values()) > 1:
        raise ValueError("three-host acquisition totals are not balanced")
    return assigned


def build(
    source_root: Path,
    runtime_root: Path,
    output_root: Path,
    *,
    seed: str,
) -> dict[str, Any]:
    prepared: dict[str, list[dict[str, Any]]] = {}
    schemas: dict[str, pa.Schema] = {}
    prepare_manifests: dict[str, dict[str, Any]] = {}
    runtime_environment_ids: dict[str, str] = {}
    all_identities: set[str] = set()
    for version in VERSIONS:
        source_dir = source_root / SOURCE_TEMPLATE.format(version=version)
        output_dir = output_root / "prepared" / version
        prepare_manifest = prepare(
            source_dir,
            runtime_root,
            output_dir,
            seed=f"{seed}:{version}",
        )
        full_path = output_dir / "open_multisandbox_dwh_full500.sensitive.parquet"
        table = pq.read_table(full_path)
        rows = table.to_pylist()
        identities = {_identity(row) for row in rows}
        if "" in identities or len(identities) != 500:
            raise ValueError(f"{version} identities are missing or duplicated")
        if all_identities & identities:
            raise ValueError("v22-v26 identities overlap")
        all_identities.update(identities)
        prepared[version] = rows
        schemas[version] = table.schema
        prepare_manifests[version] = prepare_manifest
        environment_ids = {
            str((row.get("extra_info") or {}).get("environment_id") or "") for row in rows
        }
        if "" in environment_ids or len(environment_ids) != 1:
            raise ValueError(f"{version} runtime environment identity mismatch")
        runtime_environment_ids[version] = next(iter(environment_ids))

    v22_eval_path = output_root / "frozen_eval" / "v22_full500.sensitive.parquet"
    _write_private(v22_eval_path, prepared["v22"], schemas["v22"])

    v23_by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in prepared["v23"]:
        v23_by_level[_difficulty(row)].append(row)
    pilot: list[dict[str, Any]] = []
    for level in range(1, 6):
        ordered = sorted(
            v23_by_level[level], key=lambda row: _stable(_identity(row), f"{seed}:pilot:L{level}")
        )
        pilot.extend(ordered[: PILOT_TASKS // 5])
    pilot_ids = {_identity(row) for row in pilot}
    if len(pilot_ids) != PILOT_TASKS:
        raise ValueError("pilot100 identity count mismatch")

    arms = {
        "v23_pilot100": pilot,
        "v23_rest400": [row for row in prepared["v23"] if _identity(row) not in pilot_ids],
        "v24": prepared["v24"],
        "v25": prepared["v25"],
        "v26": prepared["v26"],
    }
    assigned = _balanced_assign(arms, seed=seed)
    partition_root = output_root / "partitions"
    partition_hashes: dict[str, dict[str, str]] = {}
    host_totals: dict[str, int] = {}
    host_difficulties: dict[str, dict[str, int]] = {}
    host_arms: dict[str, dict[str, int]] = {}
    for host in HOSTS:
        partition_hashes[host] = {}
        host_rows: list[dict[str, Any]] = []
        host_arms[host] = {}
        for arm, rows in assigned[host].items():
            source_version = "v23" if arm.startswith("v23_") else arm
            path = partition_root / f"{arm}_{host}.sensitive.parquet"
            _write_private(path, rows, schemas[source_version])
            partition_hashes[host][arm] = file_sha256(path)
            host_arms[host][arm] = len(rows)
            host_rows.extend(rows)
        host_totals[host] = len(host_rows)
        host_difficulties[host] = _counts(host_rows, "difficulty_level")

    acquisition_ids = {
        _identity(row)
        for host in HOSTS
        for arm_rows in assigned[host].values()
        for row in arm_rows
    }
    v22_ids = {_identity(row) for row in prepared["v22"]}
    checks = {
        "all_five_versions_prepared_500": all(len(prepared[version]) == 500 for version in VERSIONS),
        "all_2500_identities_unique": len(all_identities) == 2500,
        "v22_eval_has_500": len(v22_ids) == 500,
        "acquisition_has_2000": len(acquisition_ids) == 2000,
        "v22_and_acquisition_disjoint": not (v22_ids & acquisition_ids),
        "pilot_has_100": len(pilot_ids) == 100,
        "pilot_levels_20_each": _counts(pilot, "difficulty_level")
        == {str(level): 20 for level in range(1, 6)},
        "host_totals_balanced": sorted(host_totals.values()) == [666, 667, 667],
        "all_source_training_disabled": all(
            not bool((row.get("extra_info") or {}).get("training_allowed"))
            for version in VERSIONS
            for row in prepared[version]
        ),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"fresh acquisition freeze checks failed: {failed}")

    summary = {
        "contract": CONTRACT,
        "model_label": "qwen38-27b-native-hf",
        "policy_step": 0,
        "reasoning_effort": "medium",
        "frozen_eval": {
            "version": "v22",
            "tasks": 500,
            "difficulty_counts": _counts(prepared["v22"], "difficulty_level"),
            "dataset_sha256": file_sha256(v22_eval_path),
        },
        "acquisition": {
            "versions": ["v23", "v24", "v25", "v26"],
            "tasks": 2000,
            "pilot_tasks": 100,
            "pilot_difficulty_counts": _counts(pilot, "difficulty_level"),
            "host_task_counts": host_totals,
            "host_difficulty_counts": host_difficulties,
            "host_arm_task_counts": host_arms,
            "partition_sha256": partition_hashes,
        },
        "runtime_environment_ids": runtime_environment_ids,
        "source_prepare_contracts": {
            version: str(prepare_manifests[version]["contract"]) for version in VERSIONS
        },
        "sampling_contract": {
            "adaptive_sampling": "banded_v2_strict_2_plus_2_plus_2_max_6_then_candidate_plus_2",
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_context_tokens": 94208,
            "trajectory_timeout_seconds": 1800,
            "queue_wait_counts_toward_timeout": False,
        },
        "checks": checks,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_hashes_server_paths_final_answers_or_tool_outputs": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "freeze.safe.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", default="qwen38-fresh-v22-v26-20260820-v1")
    args = parser.parse_args()
    result = build(args.source_root, args.runtime_root, args.output_root, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
