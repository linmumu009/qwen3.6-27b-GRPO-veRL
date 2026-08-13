#!/usr/bin/env python3
"""Wait for both rollout arms, copy the remote result, and merge automatically."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from scripts.merge_multisandbox_dwh_rollout_analysis import merge


def wait_for_arm(run_dir: str, timeout_seconds: float, poll_seconds: float) -> dict:
    root = Path(run_dir)
    deadline = time.monotonic() + timeout_seconds
    exit_path = root / "exit_code"
    while time.monotonic() < deadline:
        if exit_path.is_file():
            code = int(exit_path.read_text(encoding="utf-8").strip())
            if code != 0:
                return {"exit_code": code, "run_dir": run_dir}
            summary_path = root / "outcomes" / "safe_summary.json"
            mixed_path = root / "outcomes" / "mixed_groups.sensitive.parquet"
            if summary_path.is_file() and mixed_path.is_file():
                return {
                    "exit_code": 0,
                    "run_dir": run_dir,
                    "safe_summary": summary_path.read_bytes(),
                    "mixed_groups": mixed_path.read_bytes(),
                }
        time.sleep(poll_seconds)
    return {"exit_code": 124, "run_dir": run_dir, "timed_out_waiting_for_arm": True}


def node_for_resource(resource: str) -> str:
    matches = [
        str(node["NodeID"])
        for node in ray.nodes()
        if node.get("Alive") and float(node.get("Resources", {}).get(resource, 0)) > 0
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one live node with {resource}, got {len(matches)}")
    return matches[0]


def write_bytes_private(path: Path, payload: bytes) -> None:
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
        ).remote(str(args.remote_arm_dir), args.timeout_seconds, args.poll_seconds)
        local = wait_for_arm(
            str(args.local_arm_dir), args.timeout_seconds, args.poll_seconds
        )
        remote = ray.get(remote_ref)
    finally:
        ray.shutdown()

    status = {
        "contract": "boss-multisandbox-dwh-dual-server-finalizer-v1",
        "local_exit_code": int(local["exit_code"]),
        "remote_exit_code": int(remote["exit_code"]),
        "merged": False,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
    }
    if local["exit_code"] != 0 or remote["exit_code"] != 0:
        (args.output_dir / "finalizer_safe_status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(status)

    remote_copy = args.output_dir / "remote_arm"
    remote_summary = remote_copy / "safe_summary.json"
    remote_mixed = remote_copy / "mixed_groups.sensitive.parquet"
    write_bytes_private(remote_summary, remote["safe_summary"])
    write_bytes_private(remote_mixed, remote["mixed_groups"])
    merged = merge(
        [
            args.local_arm_dir / "outcomes" / "safe_summary.json",
            remote_summary,
        ],
        [
            args.local_arm_dir / "outcomes" / "mixed_groups.sensitive.parquet",
            remote_mixed,
        ],
        args.output_dir / "merged",
    )
    status.update(
        {
            "merged": True,
            "tasks": int(merged["tasks"]),
            "trajectories": int(merged["trajectories"]),
            "timeout_trajectories": int(merged.get("timeout_trajectories", 0)),
            "mixed_screening_rows": int(merged["mixed_screening_rows"]),
        }
    )
    (args.output_dir / "finalizer_safe_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="192.168.202.5:26379")
    parser.add_argument("--local-arm-dir", type=Path, required=True)
    parser.add_argument("--remote-arm-dir", type=Path, required=True)
    parser.add_argument("--remote-resource", default="llin_rollout_m06")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=172800)
    parser.add_argument("--poll-seconds", type=float, default=30)
    return parser.parse_args()


def main() -> None:
    result = finalize(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
