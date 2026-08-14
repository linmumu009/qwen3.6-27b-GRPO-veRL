#!/usr/bin/env python3
"""Prepare and reconcile timeout-only retries for the plan-first comparison.

The retry dataset contains exactly one row per timed-out trajectory slot.  A
successful retry is written back to that original ``task x sample`` position,
so every task still has exactly eight observations after reconciliation.
Sensitive prompts, answers, SQL, and task identities remain in 0600 artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.analyze_multisandbox_dwh_rollout import analyze
from scripts.standalone_rollout_shards import completed_shard_rows, write_jsonl_atomic


CONTRACT = "llin-plan-first-dwh-timeout-retry-v1"
MERGE_CONTRACT = "llin-plan-first-dwh-timeout-retry-merge-v1"
TIMEOUT_SECONDS = 1800
MAX_PROMPT_TOKENS = 4096
MAX_RESPONSE_TOKENS = 90112
MAX_CONTEXT_TOKENS = 94208
_SHARD_RE = re.compile(r"^tasks_(\d+)_(\d+)\.jsonl$")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_private_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("timeout retry dataset cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def load_complete_shards(
    shards_dir: Path,
    *,
    expected_tasks: int,
    samples_per_task: int,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[tuple[int, int]]]:
    observations: dict[tuple[int, int], dict[str, Any]] = {}
    ranges: list[tuple[int, int]] = []
    for path in sorted(shards_dir.glob("tasks_*.jsonl")):
        match = _SHARD_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid shard filename: {path.name}")
        start, stop = (int(value) for value in match.groups())
        if not 0 <= start < stop <= expected_tasks:
            raise ValueError(f"shard range outside dataset: {path.name}")
        rows = completed_shard_rows(
            path,
            start=start,
            stop=stop,
            samples_per_task=samples_per_task,
        )
        if rows != (stop - start) * samples_per_task:
            raise ValueError(f"incomplete or invalid shard: {path.name}")
        ranges.append((start, stop))
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (int(row["source_task_index"]), int(row["sample_index"]))
                if key in observations:
                    raise ValueError(f"duplicate task/sample observation: {key}")
                observations[key] = row
    expected = {
        (task_index, sample_index)
        for task_index in range(expected_tasks)
        for sample_index in range(samples_per_task)
    }
    if set(observations) != expected:
        raise ValueError(
            "original rollout is not complete: "
            f"missing={len(expected - set(observations))}, "
            f"extra={len(set(observations) - expected)}"
        )
    covered = [task for start, stop in ranges for task in range(start, stop)]
    if sorted(covered) != list(range(expected_tasks)) or len(covered) != len(set(covered)):
        raise ValueError("original shard ranges do not form an exact task partition")
    return observations, ranges


def timeout_keys(observations: dict[tuple[int, int], dict[str, Any]]) -> list[tuple[int, int]]:
    return sorted(
        key for key, row in observations.items() if bool(row.get("trajectory_timeout"))
    )


def prepare_retry_dataset(
    dataset_path: Path,
    original_shards_dir: Path,
    output_dataset: Path,
    safe_manifest_path: Path,
    *,
    arm_label: str,
    expected_tasks: int = 300,
    samples_per_task: int = 8,
    expected_timeouts: int | None = None,
) -> dict[str, Any]:
    dataset = pq.read_table(dataset_path).to_pylist()
    if len(dataset) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} dataset rows, got {len(dataset)}")
    observations, _ = load_complete_shards(
        original_shards_dir,
        expected_tasks=expected_tasks,
        samples_per_task=samples_per_task,
    )
    keys = timeout_keys(observations)
    if expected_timeouts is not None and len(keys) != expected_timeouts:
        raise ValueError(f"expected {expected_timeouts} timeouts, got {len(keys)}")
    records: list[dict[str, Any]] = []
    for retry_index, (task_index, sample_index) in enumerate(keys):
        record = deepcopy(dataset[task_index])
        extra = dict(record["extra_info"])
        extra.update(
            {
                "retry_contract": CONTRACT,
                "retry_index": retry_index,
                "retry_original_task_index": task_index,
                "retry_original_sample_index": sample_index,
                "retry_source_row_sha256": canonical_hash(observations[(task_index, sample_index)]),
                "retry_timeout_seconds": TIMEOUT_SECONDS,
                "retry_max_response_tokens": MAX_RESPONSE_TOKENS,
                "retry_max_context_tokens": MAX_CONTEXT_TOKENS,
                "training_allowed": False,
            }
        )
        record["extra_info"] = extra
        records.append(record)
    write_private_parquet(output_dataset, records)
    answer_types = Counter(
        str(record["reward_model"]["ground_truth"]["answer_type"]) for record in records
    )
    unique_tasks = len({task for task, _ in keys})
    manifest = {
        "contract": CONTRACT,
        "arm_label": arm_label,
        "retry_rows": len(records),
        "unique_source_tasks": unique_tasks,
        "original_tasks": expected_tasks,
        "original_samples_per_task": samples_per_task,
        "retry_samples_per_dataset_row": 1,
        "answer_type_counts": dict(sorted(answer_types.items())),
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "source_dataset_sha256": file_sha256(dataset_path),
        "retry_dataset_sha256": file_sha256(output_dataset),
        "slot_mapping_is_exact_timeout_set": True,
        "completed_trajectories_retried": 0,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_outputs_or_server_paths": False,
    }
    safe_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    safe_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def retry_mapping(
    retry_dataset: list[dict[str, Any]],
    original_observations: dict[tuple[int, int], dict[str, Any]],
) -> dict[int, tuple[int, int]]:
    mapping: dict[int, tuple[int, int]] = {}
    for retry_index, record in enumerate(retry_dataset):
        extra = record["extra_info"]
        if str(extra.get("retry_contract")) != CONTRACT:
            raise ValueError(f"retry row {retry_index} has the wrong contract")
        if int(extra.get("retry_index", -1)) != retry_index:
            raise ValueError(f"retry row {retry_index} has a mismatched retry index")
        key = (
            int(extra["retry_original_task_index"]),
            int(extra["retry_original_sample_index"]),
        )
        if key not in original_observations or not bool(
            original_observations[key].get("trajectory_timeout")
        ):
            raise ValueError(f"retry row {retry_index} does not map to a timeout")
        if str(extra["retry_source_row_sha256"]) != canonical_hash(original_observations[key]):
            raise ValueError(f"retry row {retry_index} source hash mismatch")
        if key in mapping.values():
            raise ValueError(f"duplicate retry mapping: {key}")
        mapping[retry_index] = key
    if set(mapping.values()) != set(timeout_keys(original_observations)):
        raise ValueError("retry mapping is not the exact original timeout set")
    return mapping


def merge_retries(
    original_shards_dir: Path,
    retry_dataset_path: Path,
    retry_shards_dir: Path,
    output_root: Path,
    *,
    expected_tasks: int = 300,
    samples_per_task: int = 8,
) -> dict[str, Any]:
    original, ranges = load_complete_shards(
        original_shards_dir,
        expected_tasks=expected_tasks,
        samples_per_task=samples_per_task,
    )
    retry_dataset = pq.read_table(retry_dataset_path).to_pylist()
    mapping = retry_mapping(retry_dataset, original)
    retries, _ = load_complete_shards(
        retry_shards_dir,
        expected_tasks=len(retry_dataset),
        samples_per_task=1,
    )
    if len(retries) != len(mapping):
        raise ValueError("retry output count does not match retry dataset")
    merged = {key: dict(row) for key, row in original.items()}
    for retry_index, original_key in mapping.items():
        retry = dict(retries[(retry_index, 0)])
        retry["source_task_index"] = original_key[0]
        retry["sample_index"] = original_key[1]
        retry["retry_contract"] = CONTRACT
        retry["retry_attempt"] = 1
        retry["replaced_original_timeout"] = True
        retry["retry_timeout_limit_seconds"] = TIMEOUT_SECONDS
        retry["retry_max_response_tokens"] = MAX_RESPONSE_TOKENS
        retry["retry_max_context_tokens"] = MAX_CONTEXT_TOKENS
        merged[original_key] = retry
    for start, stop in ranges:
        rows = [
            merged[(task_index, sample_index)]
            for task_index in range(start, stop)
            for sample_index in range(samples_per_task)
        ]
        write_jsonl_atomic(
            output_root / "shards" / f"tasks_{start:05d}_{stop:05d}.jsonl", rows
        )
    remaining_timeouts = sum(
        bool(merged[key].get("trajectory_timeout")) for key in mapping.values()
    )
    retry_runtime_errors = sum(
        bool(merged[key].get("runtime_error")) for key in mapping.values()
    )
    summary = {
        "contract": MERGE_CONTRACT,
        "tasks": expected_tasks,
        "samples_per_task": samples_per_task,
        "original_timeout_slots": len(mapping),
        "retry_outputs": len(retries),
        "resolved_timeout_slots": len(mapping) - remaining_timeouts,
        "remaining_timeout_slots": remaining_timeouts,
        "retry_runtime_error_slots": retry_runtime_errors,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "every_retry_replaced_exactly_one_original_slot": True,
        "completed_original_slots_unchanged": True,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_outputs_or_server_paths": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "retry_merge_safe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def wait_for_exit(run_dir: Path, timeout_seconds: float, poll_seconds: float) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        path = run_dir / "exit_code"
        if path.is_file():
            return int(path.read_text(encoding="utf-8").strip())
        time.sleep(poll_seconds)
    return 124


def finalize_arm(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retry_exit = wait_for_exit(args.retry_run_dir, args.wait_timeout_seconds, args.poll_seconds)
    if retry_exit != 0:
        raise RuntimeError(f"retry arm exited with {retry_exit}")
    merge_summary = merge_retries(
        args.original_shards_dir,
        args.retry_dataset,
        args.retry_run_dir / "shards",
        args.output_dir,
        expected_tasks=args.expected_tasks,
        samples_per_task=args.samples_per_task,
    )
    outcome = analyze(
        args.original_dataset,
        args.output_dir / "shards",
        args.output_dir / "outcomes",
        expected_tasks=args.expected_tasks,
        samples_per_task=args.samples_per_task,
    )
    status = {
        "contract": "llin-plan-first-dwh-timeout-retry-arm-finalizer-v1",
        "retry_exit_code": retry_exit,
        "tasks": outcome["tasks"],
        "trajectories": outcome["trajectories"],
        "original_timeout_slots": merge_summary["original_timeout_slots"],
        "resolved_timeout_slots": merge_summary["resolved_timeout_slots"],
        "remaining_timeout_slots": merge_summary["remaining_timeout_slots"],
        "correct_trajectories": outcome["correct_trajectories"],
        "training_allowed": False,
        "contains_prompts_gold_sql_task_ids_outputs_or_server_paths": False,
    }
    (args.output_dir / "retry_arm_safe_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset", type=Path, required=True)
    prepare_parser.add_argument("--original-shards-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-dataset", type=Path, required=True)
    prepare_parser.add_argument("--safe-manifest", type=Path, required=True)
    prepare_parser.add_argument("--arm-label", required=True)
    prepare_parser.add_argument("--expected-tasks", type=int, default=300)
    prepare_parser.add_argument("--samples-per-task", type=int, default=8)
    prepare_parser.add_argument("--expected-timeouts", type=int)

    finalizer_parser = subparsers.add_parser("finalize-arm")
    finalizer_parser.add_argument("--original-dataset", type=Path, required=True)
    finalizer_parser.add_argument("--original-shards-dir", type=Path, required=True)
    finalizer_parser.add_argument("--retry-dataset", type=Path, required=True)
    finalizer_parser.add_argument("--retry-run-dir", type=Path, required=True)
    finalizer_parser.add_argument("--output-dir", type=Path, required=True)
    finalizer_parser.add_argument("--expected-tasks", type=int, default=300)
    finalizer_parser.add_argument("--samples-per-task", type=int, default=8)
    finalizer_parser.add_argument("--wait-timeout-seconds", type=float, default=172800)
    finalizer_parser.add_argument("--poll-seconds", type=float, default=30)

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_retry_dataset(
            args.dataset,
            args.original_shards_dir,
            args.output_dataset,
            args.safe_manifest,
            arm_label=args.arm_label,
            expected_tasks=args.expected_tasks,
            samples_per_task=args.samples_per_task,
            expected_timeouts=args.expected_timeouts,
        )
    else:
        result = finalize_arm(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
