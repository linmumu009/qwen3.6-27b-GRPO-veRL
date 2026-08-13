#!/usr/bin/env python3
"""Run the boss-native PI Agent arm over a frozen task list."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    runner = load_module(args.runner, "boss_batch_run_pi_runtime_parity")
    tasks = read_jsonl(args.tasks)
    if len(tasks) != args.expected_tasks:
        raise ValueError(f"expected {args.expected_tasks} tasks, got {len(tasks)}")
    if len({row["task_key"] for row in tasks}) != len(tasks):
        raise ValueError("duplicate task_key")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    work = []
    for task in tasks:
        for sample_index in range(args.samples_per_task):
            sample_id = f"{task['task_key'][:20]}_s{sample_index:02d}"
            work.append(
                {
                    "task_key": task["task_key"],
                    "sample_index": sample_index,
                    "runner_task": {
                        "v": task["version"],
                        "type": task["task_family"],
                        "group": task["group"],
                        "task_id": sample_id,
                        "instruction": task["instruction_without_guidance"],
                    },
                }
            )

    def one(item: dict[str, Any]) -> dict[str, Any]:
        task, status, duration = runner.run_one(
            item["runner_task"],
            args.timeout,
            str(args.output_dir),
            args.served_model,
            use_runner=True,
        )
        filename = (
            f"{task['group']}_{task['v']}_{task['type']}_{task['task_id']}.jsonl"
        )
        return {
            "task_key": item["task_key"],
            "sample_index": item["sample_index"],
            "status": status,
            "duration_seconds": duration,
            "trajectory_file": filename,
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(one, item) for item in work]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    results.sort(key=lambda row: (row["task_key"], row["sample_index"]))
    manifest_path = args.output_dir / "pi_arm_manifest.sensitive.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.chmod(manifest_path, 0o600)
    summary = {
        "contract": "boss-native-pi-runtime-parity-arm-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": len(tasks),
        "samples_per_task": args.samples_per_task,
        "rows": len(results),
        "max_workers": args.max_workers,
        "served_model": args.served_model,
        "status_counts": {
            status: sum(row["status"] == status for row in results)
            for status in sorted({row["status"] for row in results})
        },
    }
    (args.output_dir / "pi_arm_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--served-model", default="Qwen3.6-27B")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

