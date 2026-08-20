#!/usr/bin/env python3
"""Wait for the three-host replay and aggregate banded-v2 strict variance."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any


CONTRACT = "llin-qwen38-step70-original70-banded-v2-strict-replay-v1"
REPLAY_CONTRACT = "llin-banded-v2-strict-table-replay-gate-v1"
ARM = "original70"
HOST_TASKS = {"m05": 20, "m06": 29, "m00": 21}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_text(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in parts)


def run(parts: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(parts, text=True, capture_output=capture, check=check)


def host_specs(args: argparse.Namespace) -> dict[str, dict[str, str | None]]:
    return {
        "m05": {"ssh": None, "container": args.container},
        "m06": {"ssh": args.remote_host, "container": args.remote_container},
        "m00": {"ssh": args.m00_host, "container": args.m00_container},
    }


def on_host(
    args: argparse.Namespace,
    host: str,
    parts: list[str],
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    ssh_host = host_specs(args)[host]["ssh"]
    if ssh_host is None:
        return run(parts, capture=capture, check=check)
    return run(
        ["ssh", "-o", "BatchMode=yes", f"root@{ssh_host}", command_text(parts)],
        capture=capture,
        check=check,
    )


def status(args: argparse.Namespace, stage: str, **fields: Any) -> None:
    write_json(
        args.output_dir / "strict_replay.safe.json",
        {
            "contract": CONTRACT,
            "stage": stage,
            "updated_at": utc_now(),
            "model_label": "qwen38-27b-grpo-step70",
            "policy_step": 70,
            "tasks": 70,
            "reward_contract": "banded-v2-strict-table-v1",
            "training_allowed": False,
            "promotion_allowed": False,
            "contains_prompts_gold_sql_task_ids_hashes_final_answers_tool_outputs_or_server_paths": False,
            **fields,
        },
    )


def wait_primary(args: argparse.Namespace) -> None:
    exit_path = args.primary_supervisor / "exit_code"
    while not exit_path.is_file():
        status(args, "waiting_for_sampling")
        time.sleep(args.poll_seconds)
    if int(exit_path.read_text(encoding="utf-8").strip()) != 0:
        raise RuntimeError("three-host sampling failed; strict replay will not run")


def replay_command(args: argparse.Namespace, host: str) -> list[str]:
    project = args.container_project
    root = f"{project}/runs/{args.eval_name}/{host}"
    approved = (
        f"{project}/runs/llin-qwen38-grpo-audit-70-20260818-01/"
        f"applied/{host}/semantic_approved_candidates.sensitive.parquet"
    )
    output = f"{root}/strict_reward"
    command = [
        "python3",
        f"{project}/scripts/replay_strict_table_reward_gate.py",
        "--approved",
        approved,
        "--output-safe-json",
        f"{output}/safe_summary.json",
        "--output-qualified-parquet",
        f"{output}/strict_qualified_candidates.sensitive.parquet",
        "--expected-approved",
        str(HOST_TASKS[host]),
        "--host-label",
        host,
    ]
    for wave, dataset in (
        (2, "initial.sensitive.parquet"),
        (4, "unresolved2.sensitive.parquet"),
        (6, "unresolved4.sensitive.parquet"),
    ):
        command.extend(
            [
                "--wave",
                f"{ARM}-wave{wave}",
                f"{root}/state/{ARM}/{dataset}",
                f"{root}/waves/{args.eval_name}-{host}-{ARM}-wave{wave}",
            ]
        )
    return command


def run_replays(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for host, spec in host_specs(args).items():
        command = ["docker", "exec", str(spec["container"]), *replay_command(args, host)]
        result = on_host(args, host, command, capture=True, check=False)
        # The replay gate intentionally returns 2 when not all 70 remain strict-mixed.
        if result.returncode not in {0, 2}:
            raise RuntimeError(f"{host} strict replay failed with {result.returncode}")
        path = (
            args.host_project
            / "runs"
            / args.eval_name
            / host
            / "strict_reward"
            / "safe_summary.json"
        )
        if host == "m05":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            remote_result = on_host(args, host, ["cat", str(path)], capture=True)
            payload = json.loads(remote_result.stdout)
        if payload.get("contract") != REPLAY_CONTRACT:
            raise ValueError(f"{host} strict replay contract mismatch")
        if int(payload.get("approved_tasks", -1)) != HOST_TASKS[host]:
            raise ValueError(f"{host} strict replay task count mismatch")
        summaries[host] = payload
    return summaries


def aggregate(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    numeric = (
        "approved_tasks",
        "covered_tasks",
        "observed_trajectories",
        "completed_trajectories",
        "legacy_correct_trajectories",
        "strict_correct_trajectories",
        "legacy_mixed_tasks",
        "strict_mixed_tasks",
        "tasks_lost_to_strict_judge",
        "qualified_private_rows",
    )
    sources: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    for payload in summaries.values():
        sources.update({str(key): int(value) for key, value in payload["strict_mixed_by_source_version"].items()})
        difficulties.update({str(key): int(value) for key, value in payload["strict_mixed_by_difficulty"].items()})
    return {
        "contract": CONTRACT,
        "stage": "complete",
        "model_label": "qwen38-27b-grpo-step70",
        "policy_step": 70,
        "reward_contract": "banded-v2-strict-table-v1",
        **{
            field: sum(int(payload[field]) for payload in summaries.values())
            for field in numeric
        },
        "strict_mixed_by_source_version": dict(sorted(sources.items())),
        "strict_mixed_by_difficulty": dict(sorted(difficulties.items())),
        "per_host": {
            host: {
                "approved_tasks": int(payload["approved_tasks"]),
                "observed_trajectories": int(payload["observed_trajectories"]),
                "legacy_mixed_tasks": int(payload["legacy_mixed_tasks"]),
                "strict_mixed_tasks": int(payload["strict_mixed_tasks"]),
            }
            for host, payload in summaries.items()
        },
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_hashes_final_answers_tool_outputs_or_server_paths": False,
    }


def execute(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("exit_code", "finished_at"):
        path = args.output_dir / stale
        if path.is_file():
            path.unlink()
    (args.output_dir / "finalizer.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        wait_primary(args)
        status(args, "replaying_banded_v2_strict_reward")
        result = aggregate(run_replays(args))
        write_json(args.output_dir / "strict_replay_result.safe.json", result)
        status(args, "complete", result_summary=result)
        (args.output_dir / "exit_code").write_text("0\n", encoding="utf-8")
    except BaseException as exc:
        status(args, "failed", error_type=type(exc).__name__)
        (args.output_dir / "exit_code").write_text("1\n", encoding="utf-8")
        raise
    finally:
        (args.output_dir / "finished_at").write_text(utc_now() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-project", type=Path, default=Path("/data3/llin/qwen3.6-27b-verl-grpo"))
    parser.add_argument("--container-project", default="/workspace/llin-verl-grpo")
    parser.add_argument("--container", default="llin-verl-qwen38-smoke-m05-20260817")
    parser.add_argument("--remote-container", default="llin-verl-qwen38-smoke-m06-20260817")
    parser.add_argument("--remote-host", default="192.168.202.4")
    parser.add_argument("--m00-container", default="llin-verl-rollout-m00-20260817")
    parser.add_argument("--m00-host", default="10.10.2.2")
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--primary-supervisor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    execute(parse_args())
