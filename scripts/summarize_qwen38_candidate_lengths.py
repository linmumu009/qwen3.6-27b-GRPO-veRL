#!/usr/bin/env python3
"""Emit aggregate token-length evidence for approved Qwen3.8 candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _identity(row: dict[str, Any]) -> str:
    return str((row.get("extra_info") or {}).get("instruction_sha256") or "")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _describe(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": 0.0, "p50": 0, "p95": 0, "p99": 0, "max": 0}

    def percentile(fraction: float) -> int:
        return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))]

    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 2),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def summarize(
    approved: Path,
    waves: list[tuple[Path, Path]],
    *,
    expected_tasks: int,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    approved_rows = pq.read_table(approved.resolve(strict=True)).to_pylist()
    approved_ids = {_identity(row) for row in approved_rows}
    if len(approved_rows) != expected_tasks or "" in approved_ids or len(approved_ids) != expected_tasks:
        raise ValueError("approved task identity/count mismatch")
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    completed_lengths: list[int] = []
    timeout_lengths: list[int] = []
    observed_tasks: set[str] = set()
    tokenizer = None
    if tokenizer_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    for dataset_path, run_dir in waves:
        dataset = pq.read_table(dataset_path.resolve(strict=True)).to_pylist()
        for shard in sorted((run_dir / "shards").glob("tasks_*.jsonl")):
            for trajectory in _read_jsonl(shard):
                index = int(trajectory["source_task_index"])
                if index < 0 or index >= len(dataset):
                    raise ValueError("source_task_index out of range")
                task_identity = _identity(dataset[index])
                if task_identity not in approved_ids:
                    continue
                observed_tasks.add(task_identity)
                if tokenizer is not None:
                    prompt_lengths.append(len(tokenizer.encode(str(trajectory.get("input") or ""))))
                response = max(0, int(trajectory.get("response_tokens", 0)))
                response_lengths.append(response)
                if trajectory.get("trajectory_timeout"):
                    timeout_lengths.append(response)
                elif not trajectory.get("runtime_error"):
                    completed_lengths.append(response)
    if observed_tasks != approved_ids:
        raise ValueError("not every approved task has a trajectory")
    return {
        "contract": "llin-qwen38-approved-candidate-length-summary-v1",
        "approved_tasks": len(approved_ids),
        "observed_trajectories": len(response_lengths),
        "prompt_tokens": _describe(prompt_lengths),
        "response_tokens_all": _describe(response_lengths),
        "response_tokens_completed": _describe(completed_lengths),
        "response_tokens_timeout": _describe(timeout_lengths),
        "contains_prompts_gold_sql_task_ids_server_paths_or_tool_outputs": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved", type=Path, required=True)
    parser.add_argument("--wave", action="append", nargs=2, metavar=("DATASET", "RUN_DIR"), required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    result = summarize(
        args.approved,
        [(Path(dataset), Path(run_dir)) for dataset, run_dir in args.wave],
        expected_tasks=args.expected_tasks,
        tokenizer_path=args.tokenizer,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
