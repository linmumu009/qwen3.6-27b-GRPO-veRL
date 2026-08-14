#!/usr/bin/env python3
"""Wait for both model arms, recover the remote aggregate, and compare them."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from scripts.compare_plan_first_dwh_model_rollouts import compare


def wait_for_arm(run_dir: str, timeout_seconds: float, poll_seconds: float) -> dict:
    root = Path(run_dir)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        exit_path = root / "exit_code"
        if exit_path.is_file():
            code = int(exit_path.read_text(encoding="utf-8").strip())
            if code != 0:
                return {"exit_code": code}
            summary = root / "outcomes" / "safe_summary.json"
            per_task = root / "outcomes" / "per_task.sensitive.jsonl"
            if summary.is_file() and per_task.is_file():
                return {
                    "exit_code": 0,
                    "safe_summary": summary.read_bytes(),
                    "per_task": per_task.read_bytes(),
                }
        time.sleep(poll_seconds)
    return {"exit_code": 124, "timed_out_waiting_for_arm": True}


def node_for_resource(resource: str) -> str:
    matches = [
        str(node["NodeID"])
        for node in ray.nodes()
        if node.get("Alive") and float(node.get("Resources", {}).get(resource, 0)) > 0
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one live node with {resource}, got {len(matches)}")
    return matches[0]


def write_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def finalize(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ray.init(address=args.ray_address, ignore_reinit_error=True)
    try:
        remote_wait = ray.remote(num_cpus=0.1)(wait_for_arm)
        remote_ref = remote_wait.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_for_resource(args.remote_resource), soft=False
            )
        ).remote(str(args.step120_arm_dir), args.timeout_seconds, args.poll_seconds)
        native = wait_for_arm(str(args.native_arm_dir), args.timeout_seconds, args.poll_seconds)
        step120 = ray.get(remote_ref)
    finally:
        ray.shutdown()
    status = {
        "contract": "llin-plan-first-dwh-base-step120-finalizer-v1",
        "native_exit_code": int(native["exit_code"]),
        "step120_exit_code": int(step120["exit_code"]),
        "compared": False,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
    }
    if native["exit_code"] != 0 or step120["exit_code"] != 0:
        (args.output_dir / "finalizer_safe_status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(status)
    remote_dir = args.output_dir / "step120_remote"
    write_private(remote_dir / "safe_summary.json", step120["safe_summary"])
    write_private(remote_dir / "per_task.sensitive.jsonl", step120["per_task"])
    comparison = compare(
        args.dataset,
        args.native_arm_dir / "outcomes" / "per_task.sensitive.jsonl",
        remote_dir / "per_task.sensitive.jsonl",
        args.output_dir / "comparison",
        samples_per_task=args.samples_per_task,
    )
    status.update(
        {
            "compared": True,
            "tasks": int(comparison["tasks"]),
            "trajectories": int(comparison["tasks"]) * args.samples_per_task * 2,
            "step120_mixed_training_candidates": int(
                comparison["step120_mixed_training_candidates"]
            ),
        }
    )
    (args.output_dir / "finalizer_safe_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="192.168.202.5:26379")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--native-arm-dir", type=Path, required=True)
    parser.add_argument("--step120-arm-dir", type=Path, required=True)
    parser.add_argument("--remote-resource", default="llin_rollout_m06")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=172800)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    print(json.dumps(finalize(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
