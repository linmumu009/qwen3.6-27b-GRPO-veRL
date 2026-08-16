#!/usr/bin/env python3
"""Pure helpers for resumable standalone rollout shards."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable


def trajectory_admission_contract(
    *,
    task_batch_size: int,
    samples_per_task: int,
    max_num_seqs_per_dp_engine: int,
    data_parallel_size: int,
) -> dict[str, int | bool | str]:
    """Fail closed when one shard starts more trajectories than vLLM can admit.

    AgentLoopWorker starts every row in its chunk concurrently, and the PI
    trajectory timeout starts before vLLM admission.  Allowing a shard to
    exceed the aggregate DP sequence capacity therefore turns queue wait into
    false trajectory timeouts.  Keep the invariant in a dependency-free helper
    so launchers and tests can audit it without importing veRL.
    """

    values = {
        "task_batch_size": task_batch_size,
        "samples_per_task": samples_per_task,
        "max_num_seqs_per_dp_engine": max_num_seqs_per_dp_engine,
        "data_parallel_size": data_parallel_size,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError("trajectory admission inputs must be positive")
    requested = task_batch_size * samples_per_task
    capacity = max_num_seqs_per_dp_engine * data_parallel_size
    if requested > capacity:
        raise ValueError(
            "trajectory admission overflow: "
            f"requested={requested}, capacity={capacity}; "
            "reduce task_batch_size or samples_per_task"
        )
    return {
        "contract": "verl-standalone-trajectory-admission-v1",
        "valid": True,
        "timeout_starts_before_vllm_admission": True,
        "requested_trajectories_per_shard": requested,
        "aggregate_sequence_capacity": capacity,
        "unused_sequence_capacity": capacity - requested,
    }


def rolling_admission_contract(
    *,
    enabled: bool,
    requested_window_trajectories: int,
    aggregate_sequence_capacity: int,
    max_window_multiplier: float = 1.0,
) -> dict[str, int | float | bool | str]:
    """Validate the refill window used by the per-trajectory scheduler.

    ``aggregate_sequence_capacity`` is the physical vLLM sequence limit.  A
    larger logical window can be useful for tool-using agents because some
    trajectories are executing tools while other trajectories are generating.
    Oversubscription is therefore allowed only when the launcher explicitly
    raises ``max_window_multiplier``; the historical default remains 1.0x.
    """

    if requested_window_trajectories < 0:
        raise ValueError("rolling admission window cannot be negative")
    if aggregate_sequence_capacity <= 0:
        raise ValueError("aggregate sequence capacity must be positive")
    if not math.isfinite(max_window_multiplier) or not 1.0 <= max_window_multiplier <= 2.0:
        raise ValueError("rolling admission max window multiplier must be within [1.0, 2.0]")
    effective_window = (
        requested_window_trajectories or aggregate_sequence_capacity if enabled else 0
    )
    max_allowed_window = math.floor(aggregate_sequence_capacity * max_window_multiplier)
    if effective_window > max_allowed_window:
        raise ValueError(
            "rolling admission overflow: "
            f"requested={effective_window}, capacity={aggregate_sequence_capacity}, "
            f"max_allowed={max_allowed_window}"
        )
    return {
        "contract": "verl-standalone-rolling-admission-v2",
        "enabled": enabled,
        "valid": True,
        "requested_window_trajectories": requested_window_trajectories,
        "effective_window_trajectories": effective_window,
        "aggregate_sequence_capacity": aggregate_sequence_capacity,
        "max_window_multiplier": max_window_multiplier,
        "max_allowed_window_trajectories": max_allowed_window,
        "logical_to_physical_ratio": (
            effective_window / aggregate_sequence_capacity if enabled else 0.0
        ),
        "logical_oversubscription_enabled": bool(
            enabled and effective_window > aggregate_sequence_capacity
        ),
        "refill_on_each_trajectory_completion": enabled,
        "atomic_persistence_scope": "configured_task_shard",
    }


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
