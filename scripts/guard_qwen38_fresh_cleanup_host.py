#!/usr/bin/env python3
"""Run a second, independent cleanup pass after a fresh Qwen3.8 acquisition ends."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from scripts import run_qwen38_fresh_acquisition_host as acquisition


CONTRACT = "llin-qwen38-fresh-acquisition-cleanup-guardian-v1"


def wait_until_idle(args: argparse.Namespace, host: str, timeout_seconds: float = 60) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            acquisition.assert_idle(
                acquisition.on_host(args, host, ["npu-smi", "info"], capture=True),
                host,
            )
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--supervisor-dir", type=Path, required=True)
    parser.add_argument("--host-project", type=Path, default=Path("/data3/llin/qwen3.6-27b-verl-grpo"))
    parser.add_argument("--container-project", default="/workspace/llin-verl-grpo")
    parser.add_argument("--container", default="llin-verl-qwen38-smoke-m05-20260817")
    parser.add_argument("--remote-container", default="llin-verl-qwen38-smoke-m06-20260817")
    parser.add_argument("--remote-host", default="192.168.202.4")
    parser.add_argument("--m00-container", default="llin-verl-qwen38-bench-m00-20260817")
    parser.add_argument("--m00-host", default="10.10.2.2")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-wait-seconds", type=float, default=1814400)
    args = parser.parse_args()
    args.source_root = Path("/unused")
    args.runtime_root = Path("/unused")
    args.m00_runtime_root = Path("/unused")
    return args


def main() -> int:
    args = parse_args()
    deadline = time.monotonic() + args.max_wait_seconds
    finished = args.supervisor_dir / "finished_at"
    while not finished.is_file():
        if time.monotonic() >= deadline:
            acquisition.write_json(
                args.supervisor_dir / "cleanup_guardian.safe.json",
                {
                    "contract": CONTRACT,
                    "stage": "failed",
                    "error_type": "WaitTimeout",
                    "contains_prompts_gold_sql_task_ids_hashes_tool_outputs_or_server_paths": False,
                },
            )
            return 1
        time.sleep(args.poll_seconds)

    try:
        acquisition.stop(args)
        idle_hosts: list[str] = []
        for host in acquisition.specs(args):
            wait_until_idle(args, host)
            idle_hosts.append(host)
        acquisition.write_json(
            args.supervisor_dir / "cleanup_guardian.safe.json",
            {
                "contract": CONTRACT,
                "stage": "complete",
                "independent_second_cleanup_pass": True,
                "idle_hosts": idle_hosts,
                "contains_prompts_gold_sql_task_ids_hashes_tool_outputs_or_server_paths": False,
            },
        )
        return 0
    except BaseException as exc:
        acquisition.write_json(
            args.supervisor_dir / "cleanup_guardian.safe.json",
            {
                "contract": CONTRACT,
                "stage": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
                "contains_prompts_gold_sql_task_ids_hashes_tool_outputs_or_server_paths": False,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
