#!/usr/bin/env python3
"""Prepare and run the fresh Qwen3.8 v23-v26 acquisition on three hosts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CONTRACT = "llin-qwen38-fresh-v23-v26-threehost-acquisition-v1"
MODEL_LABEL = "qwen38-27b-native-hf"
ARMS = ("v23_pilot100", "v23_rest400", "v24", "v25", "v26")
PREPARED_VERSIONS = ("v22", "v23", "v24", "v25", "v26")
RUNTIME_VERSIONS = ("v23", "v24", "v25", "v26")
RUNTIME_FILES = (
    "adaptive_dwh_wave_earlystop.py",
    "analyze_multisandbox_dwh_rollout.py",
    "confirm_qwen38_strict_candidates.py",
    "launch_multisandbox_dwh_standalone.sh",
    "monitor_npu_utilization.py",
    "patch_verl_abort_partial_tokens.py",
    "patch_verl_agent_loop_continuous_token.py",
    "patch_verl_fastest_k_abort_observability.py",
    "patch_verl_fastest_k_abort_retry.py",
    "patch_verl_fastest_k_oversampling.py",
    "patch_verl_force_final_config.py",
    "patch_verl_none_rollout_logprobs.py",
    "patch_verl_vllm_abort_api.py",
    "patch_verl_vllm_dp_weight_sync.py",
    "pi_runtime_preflight.py",
    "run_adaptive_dwh_wave_earlystop_queue.py",
    "run_qwen38_adaptive_dwh_three_wave_queue.py",
    "run_qwen38_host_rerun_queue.py",
    "run_runtime_parity_verl_standalone.py",
    "standalone_rollout_shards.py",
    "start_ray_qwen38_topology_benchmark.sh",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def specs(args: argparse.Namespace) -> dict[str, dict[str, str | None]]:
    return {
        "m05": {"ssh": None, "container": args.container, "node_ip": "192.168.202.5", "ifname": "eno0"},
        "m06": {"ssh": args.remote_host, "container": args.remote_container, "node_ip": "192.168.202.4", "ifname": "eno0"},
        "m00": {"ssh": args.m00_host, "container": args.m00_container, "node_ip": "10.10.2.2", "ifname": "enp196s0f0"},
    }


def on_host(
    args: argparse.Namespace,
    host: str,
    parts: list[str],
    *,
    capture: bool = False,
    log: Path | None = None,
) -> str:
    target = specs(args)[host]["ssh"]
    return run(parts, capture=capture, log=log) if target is None else remote(str(target), parts, capture=capture, log=log)


def status(args: argparse.Namespace, stage: str, **fields: Any) -> None:
    write_json(
        args.supervisor_dir / "acquisition.safe.json",
        {
            "contract": CONTRACT,
            "stage": stage,
            "updated_at": utc_now(),
            "model_label": MODEL_LABEL,
            "policy_step": 0,
            "reasoning_effort": "medium",
            "topology": "three_independent_tp4_dp4_clusters",
            "physical_npus": 48,
            "sampling": "strict_2_plus_2_plus_2_max_6_then_candidate_plus_2",
            "reward_contract": "banded-v2-strict-table-v1",
            "trajectory_timeout_seconds": 1800,
            "queue_wait_counts_toward_timeout": False,
            "training_allowed": False,
            "promotion_allowed": False,
            "contains_prompts_gold_sql_task_ids_hashes_tool_outputs_or_server_paths": False,
            **fields,
        },
    )


def assert_idle(output: str, host: str) -> None:
    process = re.compile(r"^\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|\s*[A-Za-z0-9_]", re.MULTILINE)
    if not output.strip() or process.search(output):
        raise RuntimeError(f"{host} NPUs are not idle")


def prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    safe = args.data_root / "freeze.safe.json"
    if not safe.is_file():
        container_source = str(args.source_root).replace(
            str(args.host_project), args.container_project, 1
        )
        container_output = str(args.data_root).replace(
            str(args.host_project), args.container_project, 1
        )
        run(
            [
                "docker",
                "exec",
                "-e",
                f"PYTHONPATH={args.container_project}",
                args.container,
                "python3",
                f"{args.container_project}/scripts/prepare_qwen38_fresh_v22_v26.py",
                "--source-root",
                container_source,
                "--runtime-root",
                f"{container_output}/runtime_projection",
                "--output-root",
                container_output,
                "--seed",
                args.seed,
            ],
            log=args.supervisor_dir / "prepare_data.log",
        )
    freeze = json.loads(safe.read_text(encoding="utf-8"))
    if freeze.get("contract") != "llin-qwen38-fresh-v22-v26-acquisition-freeze-v1":
        raise ValueError("existing freeze contract mismatch")
    if not all((freeze.get("checks") or {}).values()):
        raise ValueError("existing freeze checks are not all true")
    staged_runtime = args.data_root / "runtime_projection" / "sft"
    for version in PREPARED_VERSIONS:
        name = f"20260815_llin_dwh_open_api_v3_{version}_runtime"
        source = staged_runtime / name
        destination = args.runtime_root / "sft" / name
        destination.mkdir(parents=True, exist_ok=True)
        run(["rsync", "--archive", "--partial", str(source) + "/", str(destination) + "/"])
        expected_runtime = {
            path.relative_to(source).as_posix(): sha256(path)
            for path in sorted(source.rglob("*"))
            if path.is_file()
        }
        observed_runtime = {
            path.relative_to(destination).as_posix(): sha256(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        }
        if observed_runtime != expected_runtime:
            raise RuntimeError(f"m05 {version} runtime projection hash mismatch")
    for host in specs(args):
        for arm in ARMS:
            path = args.data_root / "partitions" / f"{arm}_{host}.sensitive.parquet"
            expected = str(freeze["acquisition"]["partition_sha256"][host][arm])
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"local frozen partition mismatch: {host}/{arm}")
    return freeze


def sync_runtime(args: argparse.Namespace) -> None:
    sources = [args.host_project / "scripts" / name for name in RUNTIME_FILES]
    sources.append(args.host_project / "runtime" / "sitecustomize.py")
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"runtime bundle is incomplete: {missing}")
    expected = {path.relative_to(args.host_project).as_posix(): sha256(path) for path in sources}
    for host in ("m06", "m00"):
        target = str(specs(args)[host]["ssh"])
        remote(target, ["mkdir", "-p", str(args.host_project / "scripts"), str(args.host_project / "runtime"), str(args.host_project / "llin_verl")])
        run(["scp", "-q", *[str(path) for path in sources[:-1]], f"root@{target}:{args.host_project / 'scripts'}/"])
        run(["scp", "-q", str(sources[-1]), f"root@{target}:{args.host_project / 'runtime'}/sitecustomize.py"])
        run(["rsync", "--archive", "--partial", str(args.host_project / "llin_verl") + "/", f"root@{target}:{args.host_project / 'llin_verl'}/"])
        remote_paths = [str(args.host_project / relative) for relative in expected]
        observed_raw = remote(target, ["sha256sum", *remote_paths], capture=True)
        observed = {
            Path(line.split(maxsplit=1)[1]).relative_to(args.host_project).as_posix(): line.split(maxsplit=1)[0]
            for line in observed_raw.splitlines()
            if len(line.split(maxsplit=1)) == 2
        }
        if observed != expected:
            raise RuntimeError(f"{host} runtime bundle hash mismatch")
    write_json(
        args.supervisor_dir / "runtime_bundle.safe.json",
        {
            "contract": "llin-qwen38-fresh-acquisition-runtime-bundle-v1",
            "status": "passed",
            "file_count": len(expected),
            "files": expected,
            "remote_hosts_verified": ["m06", "m00"],
            "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
        },
    )


def _verify_remote_hashes(target: str, paths: list[Path], expected: dict[str, str]) -> None:
    raw = remote(target, ["sha256sum", *[str(path) for path in paths]], capture=True)
    observed = {Path(line.split(maxsplit=1)[1]).name: line.split(maxsplit=1)[0] for line in raw.splitlines()}
    if observed != expected:
        raise RuntimeError("remote private data SHA256 mismatch")


def distribute_data(args: argparse.Namespace, freeze: dict[str, Any]) -> None:
    for host in ("m06", "m00"):
        target = str(specs(args)[host]["ssh"])
        remote_partition_root = args.data_root / "partitions"
        remote(target, ["mkdir", "-p", str(remote_partition_root)])
        files = [args.data_root / "partitions" / f"{arm}_{host}.sensitive.parquet" for arm in ARMS]
        run(["scp", "-q", *[str(path) for path in files], f"root@{target}:{remote_partition_root}/"])
        run(["scp", "-q", str(args.data_root / "freeze.safe.json"), f"root@{target}:{args.data_root}/freeze.safe.json"])
        expected = {
            path.name: str(freeze["acquisition"]["partition_sha256"][host][path.name.split(f"_{host}.")[0]])
            for path in files
        }
        _verify_remote_hashes(target, [remote_partition_root / path.name for path in files], expected)

        destination_root = args.runtime_root if host == "m06" else args.m00_runtime_root
        for version in RUNTIME_VERSIONS:
            name = f"20260815_llin_dwh_open_api_v3_{version}_runtime"
            source = args.runtime_root / "sft" / name
            destination = destination_root / "sft" / name
            remote(target, ["mkdir", "-p", str(destination)])
            run(["rsync", "--archive", "--partial", str(source) + "/", f"root@{target}:{destination}/"])
            local_files = sorted(path for path in source.rglob("*") if path.is_file())
            relative_hashes = {path.relative_to(source).as_posix(): sha256(path) for path in local_files}
            remote_files = [destination / relative for relative in relative_hashes]
            raw = remote(target, ["sha256sum", *[str(path) for path in remote_files]], capture=True)
            observed = {
                Path(line.split(maxsplit=1)[1]).relative_to(destination).as_posix(): line.split(maxsplit=1)[0]
                for line in raw.splitlines()
            }
            if observed != relative_hashes:
                raise RuntimeError(f"{host} {version} runtime projection hash mismatch")


def start_ray(args: argparse.Namespace) -> None:
    for host, spec in specs(args).items():
        assert_idle(on_host(args, host, ["npu-smi", "info"], capture=True), host)
        on_host(args, host, ["docker", "exec", str(spec["container"]), "bash", "-lc", "ray stop --force"], log=args.supervisor_dir / f"ray_cleanup_{host}.log")
    base_ports = {"m05": 63000, "m06": 64000, "m00": 65000}
    for host, spec in specs(args).items():
        port = base_ports[host]
        environment = " ".join(
            [
                f"NODE_IP={spec['node_ip']}", "RAY_PORT=46379", f"RAY_RESOURCE=q38_{host}", "EXPECTED_NPUS=16",
                "RAY_MIN_WORKER_PORT=48000", "RAY_MAX_WORKER_PORT=48999", f"RAY_TEMP_DIR=/tmp/q38-fresh-acq-{host}",
                f"HCCL_IF_IP={spec['node_ip']}", f"HCCL_SOCKET_IFNAME={spec['ifname']}", f"HCCL_IF_BASE_PORT={port}",
                f"HCCL_HOST_SOCKET_PORT_RANGE={port + 100}-{port + 163}",
                f"HCCL_NPU_SOCKET_PORT_RANGE={port + 200}-{port + 263}",
                f"bash {args.container_project}/scripts/start_ray_qwen38_topology_benchmark.sh",
            ]
        )
        on_host(args, host, ["docker", "exec", str(spec["container"]), "bash", "-lc", environment], log=args.supervisor_dir / f"ray_{host}.log")


def queue_command(args: argparse.Namespace, host: str, freeze: dict[str, Any]) -> list[str]:
    root = f"{args.container_project}/runs/{args.run_name}"
    parts = [
        "env", f"PYTHONPATH={args.container_project}", "python3",
        f"{args.container_project}/scripts/run_qwen38_host_rerun_queue.py",
        "--project-root", args.container_project, "--model", "/models/Qwen3.8-27B",
        "--model-label", MODEL_LABEL, "--policy-step", "0", "--host-label", host,
        "--confirm-candidates",
    ]
    for arm in ARMS:
        tasks = int(freeze["acquisition"]["host_arm_task_counts"][host][arm])
        if tasks:
            parts.extend(["--arm", f"{arm}={root}/data/partitions/{arm}_{host}.sensitive.parquet"])
    parts.extend(
        [
            "--rollout-resource", f"q38_{host}", "--tensor-parallel-size", "4", "--data-parallel-size", "4",
            "--rollout-npus", "16", "--max-num-seqs", "16", "--task-batch-size", "32",
            "--rolling-window-trajectories", "80", "--monitor-first-card", "0", "--monitor-num-cards", "16",
            "--runs-dir", f"{root}/{host}/waves", "--run-prefix", f"{args.run_name}-{host}",
            "--state-root", f"{root}/{host}/state", "--final-root", f"{root}/{host}/final",
            "--queue-dir", f"{root}/{host}/queue", "--ray-address", f"{specs(args)[host]['node_ip']}:46379",
            "--poll-seconds", "30", "--stage-timeout-seconds", "1209600",
        ]
    )
    return parts


def launch_queues(args: argparse.Namespace, freeze: dict[str, Any]) -> None:
    for host, spec in specs(args).items():
        root = f"{args.container_project}/runs/{args.run_name}/{host}"
        shell = f"mkdir -p {shlex.quote(root)} && {command_text(queue_command(args, host, freeze))} > {shlex.quote(root + '/host_queue.log')} 2>&1"
        on_host(args, host, ["docker", "exec", "-d", str(spec["container"]), "bash", "-lc", shell])


def remote_json(args: argparse.Namespace, host: str, path: Path) -> dict[str, Any] | None:
    if host == "m05":
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    raw = on_host(
        args,
        host,
        ["bash", "-lc", f"test -f {shlex.quote(str(path))} && cat {shlex.quote(str(path))} || true"],
        capture=True,
    ).strip()
    return json.loads(raw) if raw else None


def wait_queues(args: argparse.Namespace, started: float) -> None:
    pilot_written = (args.supervisor_dir / "pilot100.safe.json").is_file()
    while True:
        exits: dict[str, int | None] = {}
        pilots: dict[str, dict[str, Any] | None] = {}
        for host in specs(args):
            queue = args.host_project / "runs" / args.run_name / host / "queue"
            exit_payload = remote_json(args, host, queue / "host_queue.safe.json")
            stage = str((exit_payload or {}).get("stage") or "")
            exits[host] = 0 if stage == "complete" else (1 if stage == "failed" else None)
            pilots[host] = remote_json(
                args,
                host,
                args.host_project / "runs" / args.run_name / host / "final" / "v23_pilot100" / "robust_confirmation" / "confirmation.safe.json",
            )
        if not pilot_written and all(payload is not None for payload in pilots.values()):
            elapsed = time.monotonic() - started
            robust = sum(int(payload["robust_candidates"]) for payload in pilots.values() if payload)
            provisional = sum(int(payload["provisional_candidates"]) for payload in pilots.values() if payload)
            projected_total_seconds = elapsed * 20
            write_json(
                args.supervisor_dir / "pilot100.safe.json",
                {
                    "contract": "llin-qwen38-fresh-acquisition-pilot100-v1",
                    "tasks": 100,
                    "elapsed_seconds": elapsed,
                    "provisional_candidates": provisional,
                    "robust_candidates": robust,
                    "projected_total_seconds_at_pilot_rate": projected_total_seconds,
                    "projected_remaining_seconds_at_pilot_rate": max(0.0, projected_total_seconds - elapsed),
                    "projection_is_rough_and_includes_cold_start": True,
                    "training_allowed": False,
                    "contains_prompts_gold_sql_task_ids_hashes_tool_outputs_or_server_paths": False,
                },
            )
            pilot_written = True
        status(args, "sampling", host_exit_codes=exits, pilot100_complete=pilot_written)
        failures = {host: code for host, code in exits.items() if code not in (None, 0)}
        if failures:
            raise RuntimeError(f"acquisition queues failed: {failures}")
        if all(code == 0 for code in exits.values()):
            return
        time.sleep(args.poll_seconds)


def collect_private(args: argparse.Namespace, host: str, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if host == "m05":
        destination.write_bytes(source.read_bytes())
    else:
        target = str(specs(args)[host]["ssh"])
        run(["scp", "-q", f"root@{target}:{source}", str(destination)])
    os.chmod(destination, 0o600)


def aggregate(args: argparse.Namespace, freeze: dict[str, Any]) -> dict[str, Any]:
    staging = args.supervisor_dir / "private_staging"
    rows: list[dict[str, Any]] = []
    schema: pa.Schema | None = None
    arms: list[dict[str, Any]] = []
    for arm in ARMS:
        for host in specs(args):
            final = args.host_project / "runs" / args.run_name / host / "final" / arm
            confirmation = remote_json(args, host, final / "robust_confirmation" / "confirmation.safe.json")
            adaptive = remote_json(args, host, final / "adaptive_final_safe_summary.json")
            if confirmation is None or adaptive is None:
                raise FileNotFoundError(f"missing final summary for {host}/{arm}")
            expected_tasks = int(freeze["acquisition"]["host_arm_task_counts"][host][arm])
            if int(adaptive["initial_tasks"]) != expected_tasks:
                raise ValueError("final task count does not match frozen partition")
            private = staging / f"{arm}_{host}.parquet"
            collect_private(args, host, final / "robust_confirmation" / "robust_candidates.sensitive.parquet", private)
            table = pq.read_table(private)
            schema = schema or table.schema
            rows.extend(table.to_pylist())
            arms.append(
                {
                    "arm": arm,
                    "host": host,
                    "tasks": expected_tasks,
                    "sampling_trajectories": int(adaptive["actual_sampling_trajectories"]),
                    "provisional_candidates": int(confirmation["provisional_candidates"]),
                    "confirmation_trajectories": int(confirmation["confirmation_trajectories"]),
                    "robust_candidates": int(confirmation["robust_candidates"]),
                }
            )
    identities = [str((row.get("extra_info") or {}).get("instruction_sha256") or "") for row in rows]
    if "" in identities or len(identities) != len(set(identities)):
        raise ValueError("robust candidate identities are missing or duplicated")
    if schema is None:
        raise ValueError("no robust candidate schema available")
    output = args.supervisor_dir / "robust_candidates_all.sensitive.parquet"
    table = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist([], schema=schema)
    pq.write_table(table, output)
    os.chmod(output, 0o600)
    for path in staging.glob("*.parquet"):
        path.unlink()
    staging.rmdir()
    difficulty = dict(sorted(Counter(str((row.get("extra_info") or {}).get("difficulty_level", "unknown")) for row in rows).items()))
    source = dict(sorted(Counter(str((row.get("extra_info") or {}).get("source_version", "unknown")) for row in rows).items()))
    result = {
        "contract": CONTRACT,
        "stage": "complete",
        "acquisition_tasks": 2000,
        "arms": arms,
        "sampling_trajectories": sum(item["sampling_trajectories"] for item in arms),
        "confirmation_trajectories": sum(item["confirmation_trajectories"] for item in arms),
        "provisional_candidates": sum(item["provisional_candidates"] for item in arms),
        "robust_candidates": len(rows),
        "robust_by_difficulty": difficulty,
        "robust_by_source_version": source,
        "robust_dataset_sha256": sha256(output),
        "minimum_robust_candidates_for_canary": 24,
        "canary_data_gate_passed": len(rows) >= 24,
        "needs_v27_plus": len(rows) < 24,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_hashes_tool_outputs_or_server_paths": False,
    }
    write_json(args.supervisor_dir / "result.safe.json", result)
    return result


def stop(args: argparse.Namespace) -> None:
    pattern = f"[{args.run_name[0]}]{args.run_name[1:]}"
    shell = command_text(["pkill", "-TERM", "-f", "--", pattern]) + "; sleep 2; " + command_text(["pkill", "-KILL", "-f", "--", pattern]) + " || true"
    for host, spec in specs(args).items():
        on_host(args, host, ["docker", "exec", str(spec["container"]), "bash", "-lc", shell])
        on_host(args, host, ["docker", "exec", str(spec["container"]), "bash", "-lc", "ray stop --force"])


def execute(args: argparse.Namespace) -> None:
    args.supervisor_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.host_project / "runs" / ".qwen38-fresh-acquisition.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.supervisor_dir / "supervisor.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    ray_started = False
    try:
        status(args, "preparing_and_freezing_v22_v26")
        freeze = prepare_data(args)
        status(args, "syncing_runtime_bundle")
        sync_runtime(args)
        status(args, "distributing_private_partitions_and_runtime_projections")
        distribute_data(args, freeze)
        status(args, "starting_three_tp4_dp4_clusters")
        start_ray(args)
        ray_started = True
        status(args, "launching_three_host_queues")
        started = time.monotonic()
        launch_queues(args, freeze)
        wait_queues(args, started)
        result = aggregate(args, freeze)
        status(args, "complete", result_summary=result)
        (args.supervisor_dir / "exit_code").write_text("0\n", encoding="utf-8")
    except BaseException as exc:
        status(args, "failed", error_type=type(exc).__name__, error_message=str(exc)[:500])
        (args.supervisor_dir / "exit_code").write_text("1\n", encoding="utf-8")
        raise
    finally:
        if ray_started:
            stop(args)
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
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path("/data/renjunxiang/pi/sandbox"))
    parser.add_argument("--m00-runtime-root", type=Path, default=Path("/data3/llin/pi-sandbox"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--supervisor-dir", type=Path, required=True)
    parser.add_argument("--seed", default="qwen38-fresh-v22-v26-20260820-v1")
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    args.data_root = args.host_project / "runs" / args.run_name / "data"
    return args


if __name__ == "__main__":
    execute(parse_args())
