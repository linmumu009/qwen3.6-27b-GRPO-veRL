#!/usr/bin/env python3
"""Freeze the exact Qwen3.8 v15/v20/v21 holdout and two-host partitions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_qwen38_three_host_rerun import file_sha256, partition_version


CONTRACT = "llin-qwen38-grpo-step70-heldout-eval-v1"
VERSIONS = ("v15", "v20", "v21")
EXPECTED_FULL_TASKS = {version: 500 for version in VERSIONS}
EXPECTED_TRAIN_TASKS = {"v15": 12, "v20": 39, "v21": 19}
EXPECTED_HOLDOUT_TASKS = {"v15": 488, "v20": 461, "v21": 481}


def training_identity(row: dict[str, Any]) -> tuple[str, str]:
    extra = row.get("extra_info") or {}
    value = (str(extra.get("instruction_sha256") or ""), str(extra.get("gold_sha256") or ""))
    if not all(value):
        raise ValueError("instruction/gold identity is incomplete")
    return value


def source_version(row: dict[str, Any]) -> str:
    raw = str((row.get("extra_info") or {}).get("source_version") or "")
    for version in VERSIONS:
        if raw == version or raw.endswith(f"_{version}"):
            return version
    raise ValueError(f"unsupported source version: {raw}")


def parse_key_value(value: str) -> tuple[str, Path]:
    key, separator, raw = value.partition("=")
    if not separator or key not in VERSIONS or not raw:
        raise argparse.ArgumentTypeError("expected v15|v20|v21=PARQUET")
    return key, Path(raw)


def parse_allocation(value: str) -> tuple[str, dict[str, int]]:
    version, separator, raw = value.partition("=")
    if not separator or version not in VERSIONS:
        raise argparse.ArgumentTypeError("expected VERSION=HOST:COUNT,...")
    result: dict[str, int] = {}
    for item in raw.split(","):
        host, item_separator, count = item.partition(":")
        if not item_separator or not host:
            raise argparse.ArgumentTypeError("expected VERSION=HOST:COUNT,...")
        result[host] = int(count)
    return version, result


def write_private(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def build(
    sources: list[tuple[str, Path]],
    *,
    training_pool: Path,
    allocations: dict[str, dict[str, int]],
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for version, path in sources:
        grouped[version].append(path.resolve(strict=True))
    if set(grouped) != set(VERSIONS) or set(allocations) != set(VERSIONS):
        raise ValueError("all and only v15/v20/v21 sources and allocations are required")

    training_rows = pq.read_table(training_pool.resolve(strict=True)).to_pylist()
    training_ids = {training_identity(row) for row in training_rows}
    expected_training_total = sum(EXPECTED_TRAIN_TASKS.values())
    if len(training_rows) != expected_training_total or len(training_ids) != expected_training_total:
        raise ValueError("training pool does not contain the exact expected unique identities")
    training_counts = Counter(source_version(row) for row in training_rows)
    if dict(training_counts) != EXPECTED_TRAIN_TASKS:
        raise ValueError(f"training version counts mismatch: {dict(training_counts)}")

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    observed_training_ids: set[tuple[str, str]] = set()
    observed_full_ids: set[tuple[str, str]] = set()
    versions: list[dict[str, Any]] = []
    for version in VERSIONS:
        rows: list[dict[str, Any]] = []
        for path in grouped[version]:
            rows.extend(pq.read_table(path).to_pylist())
        identities = [training_identity(row) for row in rows]
        if len(rows) != EXPECTED_FULL_TASKS[version] or len(set(identities)) != len(rows):
            raise ValueError(f"{version} source is not exactly 500 unique tasks")
        if observed_full_ids & set(identities):
            raise ValueError("full version sources overlap")
        observed_full_ids.update(identities)

        excluded = {identity for identity in identities if identity in training_ids}
        if len(excluded) != EXPECTED_TRAIN_TASKS[version]:
            raise ValueError(f"{version} training exclusion count mismatch: {len(excluded)}")
        observed_training_ids.update(excluded)
        holdout: list[dict[str, Any]] = []
        for row, identity in zip(rows, identities):
            if identity in training_ids:
                continue
            record = deepcopy(row)
            extra = dict(record.get("extra_info") or {})
            extra.update(
                {
                    "qwen38_heldout_contract": CONTRACT,
                    "qwen38_heldout_from_training": True,
                    "training_allowed": False,
                    "promotion_allowed": False,
                }
            )
            record["extra_info"] = extra
            holdout.append(record)
        if len(holdout) != EXPECTED_HOLDOUT_TASKS[version]:
            raise AssertionError("heldout count mismatch")
        if sum(allocations[version].values()) != len(holdout):
            raise ValueError(f"{version} allocation does not cover heldout tasks")
        holdout_path = output_dir / f"{version}_heldout.sensitive.parquet"
        write_private(holdout_path, holdout)
        partition = partition_version(
            holdout_path,
            output_dir / "partitions",
            version=version,
            allocation=allocations[version],
        )
        versions.append(
            {
                "version": version,
                "full_tasks": len(rows),
                "excluded_training_tasks": len(excluded),
                "heldout_tasks": len(holdout),
                "heldout_difficulty_counts": dict(
                    sorted(Counter(str((row.get("extra_info") or {}).get("difficulty_level")) for row in holdout).items())
                ),
                "heldout_dataset_sha256": file_sha256(holdout_path),
                "partitions": partition["partitions"],
            }
        )
    if observed_training_ids != training_ids:
        raise ValueError("some training identities were not found exactly once in the full sources")

    manifest = {
        "contract": CONTRACT,
        "full_tasks": sum(EXPECTED_FULL_TASKS.values()),
        "excluded_training_tasks": expected_training_total,
        "heldout_tasks": sum(EXPECTED_HOLDOUT_TASKS.values()),
        "training_overlap_tasks": 0,
        "host_task_counts": {
            host: sum(item["partitions"][host]["tasks"] for item in versions)
            for host in sorted(next(iter(allocations.values())))
        },
        "versions": versions,
        "adaptive_sampling": "strict_2_plus_2_plus_2_max_6",
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_key_value, required=True)
    parser.add_argument("--training-pool", type=Path, required=True)
    parser.add_argument("--allocation", action="append", type=parse_allocation, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    allocations = dict(args.allocation)
    result = build(
        args.source,
        training_pool=args.training_pool,
        allocations=allocations,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
