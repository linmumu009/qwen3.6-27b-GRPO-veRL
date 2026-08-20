#!/usr/bin/env python3
"""Run the original 70 approved tasks on final Qwen3.8 Step70 across three hosts.

This host-side supervisor starts three independent TP4xDP4 Ray clusters, launches
strict 2+2+2 queues, aggregates only safe counts, and always tears down queues
before Ray.  Private prompts and verifier material never enter supervisor output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any


CONTRACT = "llin-qwen38-step70-original70-replay-threehost-v1"
MODEL_LABEL = "qwen38-27b-grpo-step70"
POLICY_STEP = 70
ARM_LABEL = "original70"
HOST_TASKS = {"m05": 20, "m06": 29, "m00": 21}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_text(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in parts)


def run(parts: list[str], *, capture: bool = False, log: Path | None = None) -> str:
    if log is None:
        result = subprocess.run(parts, check=True, text=True, capture_output=capture)
        return result.stdout if capture else ""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(parts, check=True, text=True, stdout=handle, stderr=subprocess.STDOUT)
    return ""


def remote(host: str, parts: list[str], *, capture: bool = False, log: Path | None = None) -> str:
    return run(
        ["ssh", "-o", "BatchMode=yes", f"root@{host}", command_text(parts)],
        capture=capture,
        log=log,
    )


def host_specs(args: argparse.Namespace) -> dict[str, dict[str, str | None]]:
    return {
        "m05": {
            "ssh": None,
            "container": args.container,
            "node_ip": "192.168.202.5",
            "ifname": "eno0",
        },
        "m06": {
            "ssh": args.remote_host,
            "container": args.remote_container,
            "node_ip": "192.168.202.4",
            "ifname": "eno0",
        },
        "m00": {
            "ssh": args.m00_host,
            "container": args.m00_container,
            "node_ip": "10.10.2.2",
            "ifname": "enp196s0f0",
        },
    }


def on_host(
    args: argparse.Namespace,
    host: str,
    parts: list[str],
    *,
    capture: bool = False,
    log: Path | None = None,
) -> str:
    ssh_host = host_specs(args)[host]["ssh"]
    if ssh_host is None:
        return run(parts, capture=capture, log=log)
    return remote(str(ssh_host), parts, capture=capture, log=log)


def status(args: argparse.Namespace, stage: str, **fields: Any) -> None:
    write_json(
        args.supervisor_dir / "replay70.safe.json",
        {
            "contract": CONTRACT,
            "stage": stage,
            "updated_at": utc_now(),
            "model_label": MODEL_LABEL,
            "policy_step": POLICY_STEP,
            "tasks": 70,
            "host_task_counts": HOST_TASKS,
            "adaptive_sampling": "strict_2_plus_2_plus_2_max_6",
            "reasoning_effort": "medium",
            "training_allowed": False,
            "promotion_allowed": False,
            "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
            **fields,
        },
    )


def assert_idle(output: str, host: str) -> None:
    process = re.compile(r"^\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|\s*[A-Za-z0-9_]", re.MULTILINE)
    if not output.strip() or process.search(output):
        raise RuntimeError(f"{host} NPUs are not idle")


def container_path(args: argparse.Namespace, host_path: Path) -> str:
    return str(host_path).replace(str(args.host_project), args.container_project, 1)


def dataset_host_path(args: argparse.Namespace, host: str) -> Path:
    return (
        args.host_project
        / "runs"
        / "llin-qwen38-grpo-audit-70-20260818-01"
        / "applied"
        / host
        / "semantic_approved_candidates.sensitive.parquet"
    )


def preflight(args: argparse.Namespace) -> None:
    specs = host_specs(args)
    local_manifest = args.model / "llin_export_manifest.json"
    payload = json.loads(local_manifest.read_text(encoding="utf-8"))
    if (payload.get("verification") or {}).get("valid") is not True:
        raise ValueError("Step70 export verification is not valid")
    if "global_step_70" not in str(payload.get("actor_checkpoint") or ""):
        raise ValueError("model export is not from global_step_70")

    for host, spec in specs.items():
        assert_idle(on_host(args, host, ["npu-smi", "info"], capture=True), host)
        dataset = dataset_host_path(args, host)
        dataset_in_container = container_path(args, dataset)
        code = (
            "import pyarrow.parquet as pq; "
            f"p={dataset_in_container!r}; "
            "print(pq.read_metadata(p).num_rows)"
        )
        raw = on_host(
            args,
            host,
            ["docker", "exec", str(spec["container"]), "python3", "-c", code],
            capture=True,
        ).strip()
        if int(raw) != HOST_TASKS[host]:
            raise ValueError(f"{host} expected {HOST_TASKS[host]} tasks, observed {raw}")
        model_manifest = container_path(args, local_manifest)
        on_host(
            args,
            host,
            ["docker", "exec", str(spec["container"]), "test", "-s", model_manifest],
        )


def start_ray(args: argparse.Namespace) -> None:
    specs = host_specs(args)
    for host, spec in specs.items():
        on_host(
            args,
            host,
            ["docker", "exec", str(spec["container"]), "bash", "-lc", "ray stop --force"],
            log=args.supervisor_dir / f"ray_cleanup_before_{host}.log",
        )

    common = f"bash {args.container_project}/scripts/start_ray_qwen38_topology_benchmark.sh"
    base_ports = {"m05": 60000, "m06": 61000, "m00": 62000}
    for host, spec in specs.items():
        port = base_ports[host]
        environment = " ".join(
            [
                f"NODE_IP={spec['node_ip']}",
                "RAY_PORT=56379",
                f"RAY_RESOURCE=q38_replay70_{host}",
                "EXPECTED_NPUS=16",
                "RAY_MIN_WORKER_PORT=57000",
                "RAY_MAX_WORKER_PORT=57999",
                f"RAY_TEMP_DIR=/tmp/q38-step70-replay70-{host}",
                f"HCCL_IF_IP={spec['node_ip']}",
                f"HCCL_SOCKET_IFNAME={spec['ifname']}",
                f"HCCL_IF_BASE_PORT={port}",
                f"HCCL_HOST_SOCKET_PORT_RANGE={port + 100}-{port + 163}",
                f"HCCL_NPU_SOCKET_PORT_RANGE={port + 200}-{port + 263}",
                common,
            ]
        )
        on_host(
            args,
            host,
            ["docker", "exec", str(spec["container"]), "bash", "-lc", environment],
            log=args.supervisor_dir / f"ray_{host}.log",
        )


def queue_command(args: argparse.Namespace, host: str) -> list[str]:
    root = f"{args.container_project}/runs/{args.eval_name}/{host}"
    dataset = container_path(args, dataset_host_path(args, host))
    model = container_path(args, args.model)
    node_ip = str(host_specs(args)[host]["node_ip"])
    return [
        "env",
        f"PYTHONPATH={args.container_project}",
        "python3",
        f"{args.container_project}/scripts/run_qwen38_host_rerun_queue.py",
        "--project-root",
        args.container_project,
        "--model",
        model,
        "--model-label",
        MODEL_LABEL,
        "--policy-step",
        str(POLICY_STEP),
        "--host-label",
        host,
        "--arm",
        f"{ARM_LABEL}={dataset}",
        "--rollout-resource",
        f"q38_replay70_{host}",
        "--tensor-parallel-size",
        "4",
        "--data-parallel-size",
        "4",
        "--rollout-npus",
        "16",
        "--max-num-seqs",
        "16",
        "--task-batch-size",
        "32",
        "--rolling-window-trajectories",
        "80",
        "--monitor-first-card",
        "0",
        "--monitor-num-cards",
        "16",
        "--runs-dir",
        f"{root}/waves",
        "--run-prefix",
        f"{args.eval_name}-{host}",
        "--state-root",
        f"{root}/state",
        "--final-root",
        f"{root}/final",
        "--queue-dir",
        f"{root}/queue",
        "--ray-address",
        f"{node_ip}:56379",
        "--poll-seconds",
        "30",
        "--stage-timeout-seconds",
        "1209600",
    ]


def launch_queues(args: argparse.Namespace) -> None:
    for host, spec in host_specs(args).items():
        root = f"{args.container_project}/runs/{args.eval_name}/{host}"
        shell = (
            f"mkdir -p {shlex.quote(root)} && "
            f"{command_text(queue_command(args, host))} "
            f"> {shlex.quote(root + '/host_queue.log')} 2>&1"
        )
        on_host(
            args,
            host,
            ["docker", "exec", "-d", str(spec["container"]), "bash", "-lc", shell],
        )


def read_host_exit(args: argparse.Namespace, host: str) -> int | None:
    path = args.host_project / "runs" / args.eval_name / host / "queue" / "exit_code"
    if host == "m05":
        return int(path.read_text().strip()) if path.is_file() else None
    raw = on_host(
        args,
        host,
        ["bash", "-lc", f"test -f {shlex.quote(str(path))} && cat {shlex.quote(str(path))} || true"],
        capture=True,
    ).strip()
    return int(raw) if raw else None


def wait_queues(args: argparse.Namespace) -> None:
    while True:
        exit_codes = {host: read_host_exit(args, host) for host in HOST_TASKS}
        status(args, "evaluating", host_exit_codes=exit_codes)
        failures = {host: code for host, code in exit_codes.items() if code not in (None, 0)}
        if failures:
            raise RuntimeError(f"replay queues failed: {failures}")
        if all(code == 0 for code in exit_codes.values()):
            return
        time.sleep(args.poll_seconds)


def read_summary(args: argparse.Namespace, host: str) -> dict[str, Any]:
    path = (
        args.host_project
        / "runs"
        / args.eval_name
        / host
        / "final"
        / ARM_LABEL
        / "adaptive_final_safe_summary.json"
    )
    if host == "m05":
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(on_host(args, host, ["cat", str(path)], capture=True))


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    per_host = {host: read_summary(args, host) for host in HOST_TASKS}
    for host, payload in per_host.items():
        if int(payload.get("initial_tasks", -1)) != HOST_TASKS[host]:
            raise ValueError(f"{host} final task count mismatch")
    numeric = (
        "initial_tasks",
        "mixed_after_two_tasks",
        "mixed_after_four_tasks",
        "mixed_after_six_tasks",
        "variance_candidate_tasks",
        "unresolved_after_six_tasks",
        "actual_sampling_trajectories",
        "avoided_trajectories_vs_full_six",
    )
    difficulty: dict[str, int] = {}
    for payload in per_host.values():
        for level, count in (payload.get("candidate_difficulty_counts") or {}).items():
            difficulty[str(level)] = difficulty.get(str(level), 0) + int(count)
    result = {
        "contract": CONTRACT,
        "stage": "complete",
        "model_label": MODEL_LABEL,
        "policy_step": POLICY_STEP,
        **{key: sum(int(per_host[host][key]) for host in per_host) for key in numeric},
        "candidate_difficulty_counts": dict(sorted(difficulty.items())),
        "per_host": {
            host: {
                "initial_tasks": int(payload["initial_tasks"]),
                "variance_candidate_tasks": int(payload["variance_candidate_tasks"]),
                "actual_sampling_trajectories": int(payload["actual_sampling_trajectories"]),
            }
            for host, payload in per_host.items()
        },
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
    }
    write_json(args.supervisor_dir / "replay70_result.safe.json", result)
    return result


def stop_queues(args: argparse.Namespace) -> None:
    pattern = f"[{args.eval_name[0]}]{args.eval_name[1:]}"
    shell = (
        command_text(["pkill", "-TERM", "-f", "--", pattern])
        + "; sleep 2; "
        + command_text(["pkill", "-KILL", "-f", "--", pattern])
        + " || true"
    )
    for host, spec in host_specs(args).items():
        try:
            on_host(
                args,
                host,
                ["docker", "exec", str(spec["container"]), "bash", "-lc", shell],
            )
        except subprocess.CalledProcessError:
            pass


def stop_ray(args: argparse.Namespace) -> None:
    for host, spec in host_specs(args).items():
        try:
            on_host(
                args,
                host,
                ["docker", "exec", str(spec["container"]), "bash", "-lc", "ray stop --force"],
                log=args.supervisor_dir / f"ray_cleanup_after_{host}.log",
            )
        except subprocess.CalledProcessError:
            pass


def execute(args: argparse.Namespace) -> None:
    args.supervisor_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.host_project / "runs" / ".qwen38-step70-replay70.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for stale in ("exit_code", "finished_at"):
            path = args.supervisor_dir / stale
            if path.is_file():
                path.unlink()
        (args.supervisor_dir / "supervisor.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
        result: dict[str, Any] | None = None
        error_type: str | None = None
        try:
            status(args, "preflighting")
            preflight(args)
            status(args, "starting_three_independent_tp4dp4_clusters")
            start_ray(args)
            status(args, "launching_queues")
            launch_queues(args)
            wait_queues(args)
            result = aggregate(args)
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            stop_queues(args)
            stop_ray(args)
            if result is not None:
                status(args, "complete", result_summary=result, cleanup_complete=True)
                (args.supervisor_dir / "exit_code").write_text("0\n", encoding="utf-8")
            else:
                status(args, "failed", error_type=error_type or "UnknownError", cleanup_complete=True)
                (args.supervisor_dir / "exit_code").write_text("1\n", encoding="utf-8")
            (args.supervisor_dir / "finished_at").write_text(utc_now() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-project", type=Path, default=Path("/data3/llin/qwen3.6-27b-verl-grpo"))
    parser.add_argument("--container-project", default="/workspace/llin-verl-grpo")
    parser.add_argument("--container", default="llin-verl-qwen38-smoke-m05-20260817")
    parser.add_argument("--remote-container", default="llin-verl-qwen38-smoke-m06-20260817")
    parser.add_argument("--remote-host", default="192.168.202.4")
    parser.add_argument("--m00-container", default="llin-verl-rollout-m00-20260817")
    parser.add_argument("--m00-host", default="10.10.2.2")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--supervisor-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    execute(parse_args())
