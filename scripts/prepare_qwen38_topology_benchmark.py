#!/usr/bin/env python3
"""Select a deterministic private task subset for Qwen3.8 topology benchmarks."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


CONTRACT = "llin-qwen38-topology-benchmark-dataset-v1"


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


def prepare(source: Path, output: Path, manifest_path: Path, *, tasks: int) -> dict:
    table = pq.read_table(source)
    rows = table.to_pylist()
    identities = [identity(row) for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("benchmark source contains duplicate prompt identities")
    if not 0 < tasks <= len(rows):
        raise ValueError("benchmark task count is outside source bounds")
    ranked = sorted(zip(identities, rows), key=lambda item: item[0])[:tasks]
    selected = []
    for prompt_sha256, row in ranked:
        record = deepcopy(row)
        extra = dict(record.get("extra_info") or {})
        extra.update(
            {
                "qwen38_topology_benchmark_contract": CONTRACT,
                "qwen38_topology_benchmark_prompt_sha256": prompt_sha256,
                "training_allowed": False,
                "promotion_allowed": False,
            }
        )
        record["extra_info"] = extra
        selected.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    # Infer the updated nested extra_info schema so the benchmark contract
    # fields are not silently dropped by the source struct schema.
    pq.write_table(pa.Table.from_pylist(selected), temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(output)
    manifest = {
        "contract": CONTRACT,
        "source_tasks": len(rows),
        "selected_tasks": len(selected),
        "selection": "lowest_prompt_sha256",
        "source_dataset_sha256": file_sha256(source),
        "selected_dataset_sha256": file_sha256(output),
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source, args.output, args.manifest, tasks=args.tasks),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
