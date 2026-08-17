#!/usr/bin/env python3
"""Build deterministic, difficulty-balanced private Qwen3.8 rerun partitions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


CONTRACT = "llin-qwen38-three-host-rerun-partition-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(row: dict) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError("row prompt must be a chat-message list")
    payload = json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def difficulty(row: dict) -> str:
    extra = row.get("extra_info") or {}
    return str(extra.get("difficulty_level", "unknown"))


def allocate_group(size: int, remaining: dict[str, int]) -> dict[str, int]:
    remaining_total = sum(remaining.values())
    if size > remaining_total:
        raise ValueError("difficulty group exceeds remaining capacity")
    if size == remaining_total:
        return dict(remaining)
    ideals = {host: size * count / remaining_total for host, count in remaining.items()}
    assigned = {host: min(int(ideals[host]), remaining[host]) for host in remaining}
    for host in sorted(
        remaining,
        key=lambda item: (-(ideals[item] - assigned[item]), -remaining[item], item),
    ):
        if sum(assigned.values()) == size:
            break
        if assigned[host] < remaining[host]:
            assigned[host] += 1
    if sum(assigned.values()) != size:
        for host in sorted(remaining, key=lambda item: (-remaining[item], item)):
            while assigned[host] < remaining[host] and sum(assigned.values()) < size:
                assigned[host] += 1
    if sum(assigned.values()) != size:
        raise AssertionError("could not allocate exact difficulty group")
    return assigned


def partition_version(
    source: Path,
    output_dir: Path,
    *,
    version: str,
    allocation: dict[str, int],
) -> dict:
    rows = pq.read_table(source).to_pylist()
    if sum(allocation.values()) != len(rows):
        raise ValueError("host allocation must cover the source exactly")
    if any(count < 0 for count in allocation.values()):
        raise ValueError("host allocation counts must be non-negative")
    identities = [identity(row) for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("source contains duplicate prompt identities")

    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for prompt_sha256, row in zip(identities, rows):
        groups[difficulty(row)].append((prompt_sha256, row))
    remaining = dict(allocation)
    partitions: dict[str, list[dict]] = {host: [] for host in allocation}
    for level in sorted(groups, key=lambda item: (item == "unknown", item)):
        ranked = sorted(groups[level], key=lambda item: item[0])
        level_allocation = allocate_group(len(ranked), remaining)
        cursor = 0
        for host in sorted(allocation):
            stop = cursor + level_allocation[host]
            for prompt_sha256, row in ranked[cursor:stop]:
                record = deepcopy(row)
                extra = dict(record.get("extra_info") or {})
                extra.update(
                    {
                        "qwen38_rerun_contract": CONTRACT,
                        "qwen38_rerun_version": version,
                        "qwen38_rerun_host": host,
                        "qwen38_rerun_prompt_sha256": prompt_sha256,
                        "training_allowed": False,
                        "promotion_allowed": False,
                    }
                )
                record["extra_info"] = extra
                partitions[host].append(record)
            remaining[host] -= level_allocation[host]
            cursor = stop
    if any(remaining.values()):
        raise AssertionError("host allocation was not exhausted")

    output_dir.mkdir(parents=True, exist_ok=True)
    partition_manifest = {}
    persisted_identities: set[str] = set()
    for host, host_rows in partitions.items():
        if len(host_rows) != allocation[host]:
            raise AssertionError("persisted host count differs from allocation")
        output = output_dir / f"{version}_{host}.sensitive.parquet"
        temporary = output.with_suffix(output.suffix + ".tmp")
        pq.write_table(pa.Table.from_pylist(host_rows), temporary)
        os.chmod(temporary, 0o600)
        temporary.replace(output)
        host_identities = {identity(row) for row in host_rows}
        if persisted_identities & host_identities:
            raise AssertionError("host partitions overlap")
        persisted_identities |= host_identities
        partition_manifest[host] = {
            "tasks": len(host_rows),
            "difficulty_level_counts": dict(sorted(Counter(difficulty(row) for row in host_rows).items())),
            "dataset_sha256": file_sha256(output),
        }
    if persisted_identities != set(identities):
        raise AssertionError("host partitions do not cover source")
    return {
        "version": version,
        "source_tasks": len(rows),
        "source_dataset_sha256": file_sha256(source),
        "partitions": partition_manifest,
    }


def parse_key_value(value: str) -> tuple[str, str]:
    key, separator, raw = value.partition("=")
    if not separator or not key or not raw:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    return key, raw


def parse_allocation(value: str) -> tuple[str, dict[str, int]]:
    version, raw = parse_key_value(value)
    result = {}
    for item in raw.split(","):
        host, separator, count = item.partition(":")
        if not separator or not host:
            raise argparse.ArgumentTypeError("expected VERSION=HOST:COUNT,...")
        result[host] = int(count)
    return version, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_key_value, required=True)
    parser.add_argument("--allocation", action="append", type=parse_allocation, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    sources = dict(args.source)
    allocations = dict(args.allocation)
    if set(sources) != set(allocations):
        raise ValueError("source versions and allocation versions must match")
    versions = [
        partition_version(
            Path(sources[version]),
            args.output_dir,
            version=version,
            allocation=allocations[version],
        )
        for version in sorted(sources)
    ]
    manifest = {
        "contract": CONTRACT,
        "versions": versions,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
