#!/usr/bin/env python3
"""Pure helpers for resumable standalone rollout shards."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


def shard_ranges(total_tasks: int, task_batch_size: int) -> list[tuple[int, int]]:
    if total_tasks <= 0:
        raise ValueError("total_tasks must be positive")
    if task_batch_size <= 0:
        raise ValueError("task_batch_size must be positive")
    return [
        (start, min(start + task_batch_size, total_tasks))
        for start in range(0, total_tasks, task_batch_size)
    ]


def shard_path(root: Path, start: int, stop: int) -> Path:
    return root / "shards" / f"tasks_{start:05d}_{stop:05d}.jsonl"


def padded_rows_for_equal_chunks(rows: int, chunks: int) -> int:
    """Return the smallest row count at least ``rows`` divisible by ``chunks``."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    return ((rows + chunks - 1) // chunks) * chunks


def completed_shard_rows(
    path: Path,
    *,
    start: int,
    stop: int,
    samples_per_task: int,
) -> int:
    """Return the validated row count, or zero for absent/incomplete shards."""

    if not path.is_file():
        return 0
    expected = (stop - start) * samples_per_task
    counts = {index: 0 for index in range(start, stop)}
    rows = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                task_index = int(row["source_task_index"])
                sample_index = int(row["sample_index"])
                if task_index not in counts or not 0 <= sample_index < samples_per_task:
                    return 0
                counts[task_index] += 1
                rows += 1
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return 0
    if rows != expected or any(value != samples_per_task for value in counts.values()):
        return 0
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count
