#!/usr/bin/env python3
"""Emit a content-free health and capacity summary for a standalone rollout."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics

from scripts.standalone_rollout_shards import completed_shard_rows, shard_path, shard_ranges


SCHEDULER_RE = re.compile(r"Running:\s*(\d+)\s*reqs,\s*Waiting:\s*(\d+)\s*reqs")
ERROR_PATTERNS = {
    "traceback": re.compile(r"Traceback", re.IGNORECASE),
    "out_of_memory": re.compile(r"out of memory|OutOfMemory", re.IGNORECASE),
    "context_or_truncation": re.compile(
        r"(?:ERROR|Exception|ValueError|RuntimeError).*?(?:context length|too long|truncat)",
        re.IGNORECASE,
    ),
}


def percentile(values: list[int | float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def parse_time(path: Path) -> datetime | None:
    if not path.is_file():
        return None
    return datetime.fromisoformat(path.read_text(encoding="utf-8").strip())


def summarize(run_dir: Path) -> dict:
    contract = read_json(run_dir / "standalone_contract.json")
    tasks = int(contract.get("tasks", 0))
    samples = int(contract.get("samples_per_task", 0))
    batch_size = int(contract.get("task_batch_size", tasks or 1))
    completed_tasks = completed_rows = runtime_errors = timeout_rows = at_response_budget = 0
    timeout_abort_acks = timeout_abort_physical = timeout_abort_errors = 0
    response_lengths: list[int] = []
    complete_shards = 0
    if tasks and samples:
        for start, stop in shard_ranges(tasks, batch_size):
            path = shard_path(run_dir, start, stop)
            validated = completed_shard_rows(
                path, start=start, stop=stop, samples_per_task=samples
            )
            if not validated:
                continue
            complete_shards += 1
            completed_tasks += stop - start
            completed_rows += validated
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    runtime_errors += bool(row.get("runtime_error"))
                    timeout_rows += bool(row.get("trajectory_timeout"))
                    timeout_abort_acks += int(
                        row.get("trajectory_abort_acknowledged_count", 0) or 0
                    )
                    timeout_abort_physical += int(
                        row.get("trajectory_abort_physical_request_count", 0) or 0
                    )
                    timeout_abort_errors += int(
                        row.get("trajectory_abort_error_count", 0) or 0
                    )
                    length = int(row.get("response_tokens", 0))
                    response_lengths.append(length)
                    at_response_budget += length >= int(contract.get("max_response_tokens", 0))

    driver_text = (run_dir / "driver.log").read_text(
        encoding="utf-8", errors="ignore"
    ) if (run_dir / "driver.log").is_file() else ""
    scheduler = [(int(a), int(b)) for a, b in SCHEDULER_RE.findall(driver_text)]
    error_counts = {name: len(regex.findall(driver_text)) for name, regex in ERROR_PATTERNS.items()}

    npu_rows = []
    npu_path = run_dir / "npu_utilization.csv"
    if npu_path.is_file():
        with npu_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    npu_rows.append({
                        key: int(row[key])
                        for key in ("aicore_pct", "npu_util_pct", "hbm_usage_pct", "hbm_bandwidth_pct")
                    })
                except (KeyError, ValueError):
                    continue
    recent_npu_rows = npu_rows[-16:]

    started = parse_time(run_dir / "started_at")
    finished = parse_time(run_dir / "finished_at")
    exit_code = None
    if (run_dir / "exit_code").is_file():
        exit_code = int((run_dir / "exit_code").read_text(encoding="utf-8").strip())
    max_seqs = int(contract.get("max_num_seqs_per_dp_engine", 0))
    running = [row[0] for row in scheduler]
    waiting = [row[1] for row in scheduler]
    return {
        "contract": "verl-standalone-rollout-safe-summary-v1",
        "finished": finished is not None,
        "exit_code": exit_code,
        "wall_seconds": (finished - started).total_seconds() if started and finished else None,
        "tasks": tasks,
        "samples_per_task": samples,
        "completed_tasks": completed_tasks,
        "completed_rows": completed_rows,
        "complete_shards": complete_shards,
        "runtime_error_rows": runtime_errors,
        "trajectory_timeout_rows": timeout_rows,
        "trajectory_timeout_seconds": contract.get("trajectory_timeout_seconds"),
        "timeout_abort_acknowledged_count": timeout_abort_acks,
        "timeout_abort_physical_request_count": timeout_abort_physical,
        "timeout_abort_error_count": timeout_abort_errors,
        "sampling": {
            "temperature": contract.get("temperature"),
            "top_p": contract.get("top_p"),
            "top_k": contract.get("top_k"),
        },
        "context_tokens": contract.get("context_tokens"),
        "max_num_seqs_per_dp_engine": max_seqs,
        "scheduler": {
            "samples": len(scheduler),
            "running_mean": statistics.fmean(running) if running else None,
            "running_max": max(running) if running else None,
            "running_latest": running[-1] if running else None,
            "waiting_mean": statistics.fmean(waiting) if waiting else None,
            "waiting_max": max(waiting) if waiting else None,
            "waiting_latest": waiting[-1] if waiting else None,
            "at_sequence_cap_samples": sum(value >= max_seqs for value in running) if max_seqs else 0,
        },
        "response_tokens": {
            "p50": percentile(response_lengths, 0.50),
            "p95": percentile(response_lengths, 0.95),
            "max": max(response_lengths) if response_lengths else None,
            "at_budget_rows": at_response_budget,
        },
        "npu": {
            "rows": len(npu_rows),
            "aicore_mean": statistics.fmean(row["aicore_pct"] for row in npu_rows) if npu_rows else None,
            "aicore_p95": percentile([row["aicore_pct"] for row in npu_rows], 0.95),
            "npu_util_mean": statistics.fmean(row["npu_util_pct"] for row in npu_rows) if npu_rows else None,
            "hbm_usage_max": max((row["hbm_usage_pct"] for row in npu_rows), default=None),
            "hbm_bandwidth_mean": statistics.fmean(row["hbm_bandwidth_pct"] for row in npu_rows) if npu_rows else None,
            "recent_aicore_mean": statistics.fmean(
                row["aicore_pct"] for row in recent_npu_rows
            ) if recent_npu_rows else None,
            "recent_npu_util_mean": statistics.fmean(
                row["npu_util_pct"] for row in recent_npu_rows
            ) if recent_npu_rows else None,
            "recent_hbm_usage_max": max(
                (row["hbm_usage_pct"] for row in recent_npu_rows), default=None
            ),
        },
        "driver_error_markers": error_counts,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
