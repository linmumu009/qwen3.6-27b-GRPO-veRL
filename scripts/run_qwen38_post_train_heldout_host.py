#!/usr/bin/env python3
"""Wait for formal Step70, export it, then run two-host heldout 2+2+2 evaluation."""

from __future__ import annotations

import argparse
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


CONTRACT = "llin-qwen38-step70-post-train-heldout-supervisor-v1"
MODEL_LABEL = "qwen38-27b-grpo-step70"
POLICY_STEP = 70
VERSIONS = ("v15", "v20", "v21")
HOST_TASKS = {
    "m05": {"v15": 244, "v20": 230, "v21": 241},
    "m06": {"v15": 244, "v20": 231, "v21": 240},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_text(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in parts)


def run(parts: list[str], *, log: Path | None = None, capture: bool = False) -> str:
    if log is None:
        result = subprocess.run(parts, check=True, text=True, capture_output=capture)
        return result.stdout if capture else ""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(parts, check=True, text=True, stdout=handle, stderr=subprocess.STDOUT)
    return ""


def remote(host: str, parts: list[str], *, capture: bool = False, log: Path | None = None) -> str:
    return run(["ssh", "-o", "BatchMode=yes", f"root@{host}", command_text(parts)], log=log, capture=capture)


def status(supervisor_dir: Path, stage: str, **fields: Any) -> None:
    write_json(
        supervisor_dir / "post_train_eval.safe.json",
        {
            "contract": CONTRACT,
            "stage": stage,
            "updated_at": utc_now(),
            "model_label": MODEL_LABEL,
            "policy_step": POLICY_STEP,
            "heldout_tasks": 1430,
            "host_task_counts": {"m05": 715, "m06": 715},
            "adaptive_sampling": "strict_2_plus_2_plus_2_max_6",
            "reasoning_effort": "medium",
            "training_allowed": False,
            "promotion_allowed": False,
            "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
            **fields,
        },
    )


def process_alive(pid_path: Path) -> bool:
    if not pid_path.is_file():
        return False
    try:
        os.kill(int(pid_path.read_text().strip()), 0)
    except (ValueError, ProcessLookupError):
        return False
    return True


def wait_for_training(args: argparse.Namespace) -> None:
    while True:
        state_path = args.training_supervisor / "state"
        state = state_path.read_text(encoding="utf-8").strip() if state_path.is_file() else "missing"
        exit_path = args.training_supervisor / "exit_code"
        exit_code = int(exit_path.read_text().strip()) if exit_path.is_file() else None
        if state == "failed" or (exit_code is not None and exit_code != 0):
            raise RuntimeError("formal training failed; heldout evaluation will not start")
        cleanup = args.training_supervisor / "ray_cleanup_finished_at"
        if state == "complete" and exit_code == 0 and cleanup.is_file() and not process_alive(args.training_supervisor / "supervisor.pid"):
            return
        if state not in {"missing", "validating_assets", "checking_resources", "starting_ray", "checking_hccl_fanout", "training", "complete"}:
            raise RuntimeError(f"unexpected training supervisor state: {state}")
        if not process_alive(args.training_supervisor / "supervisor.pid") and state not in {"missing", "complete"}:
            raise RuntimeError("training supervisor stopped before a successful completion")
        status(args.supervisor_dir, "waiting_for_training", training_stage=state)
        time.sleep(args.poll_seconds)


def verify_checkpoint(args: argparse.Namespace) -> Path:
    checkpoint_root = args.training_run / "checkpoints"
    steps = sorted(path for path in checkpoint_root.glob("global_step_*") if path.is_dir())
    expected = checkpoint_root / f"global_step_{POLICY_STEP}"
    if steps != [expected]:
        raise RuntimeError(f"expected only global_step_{POLICY_STEP}, observed {[p.name for p in steps]}")
    actor = expected / "actor"
    if not (actor / "ckpt_contents.json").is_file():
        raise FileNotFoundError("final actor checkpoint manifest is missing")
    return actor


def verify_export_manifest(model: Path) -> None:
    manifest = model / "llin_export_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("export manifest is missing")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if (payload.get("verification") or {}).get("valid") is not True:
        raise ValueError("HF export verification is not valid")
    if f"global_step_{POLICY_STEP}" not in str(payload.get("actor_checkpoint") or ""):
        raise ValueError("HF export is not from formal Step70")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_heldout_data(args: argparse.Namespace) -> None:
    root = args.host_project / "runs" / args.eval_name / "data"
    manifest = json.loads((root / "heldout.safe.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != "llin-qwen38-grpo-step70-heldout-eval-v1":
        raise ValueError("heldout manifest contract mismatch")
    if manifest.get("heldout_tasks") != 1430 or manifest.get("training_overlap_tasks") != 0:
        raise ValueError("heldout count or leakage gate failed")
    if manifest.get("host_task_counts") != {"m05": 715, "m06": 715}:
        raise ValueError("heldout host balance gate failed")
    by_version = {item["version"]: item for item in manifest["versions"]}
    for host in ("m05", "m06"):
        for version in VERSIONS:
            expected = by_version[version]["partitions"][host]
            if int(expected["tasks"]) != HOST_TASKS[host][version]:
                raise ValueError("heldout partition task count mismatch")
            path = root / "partitions" / f"{version}_{host}.sensitive.parquet"
            if host == "m05":
                observed = file_sha256(path)
            else:
                observed = remote(args.remote_host, ["sha256sum", str(path)], capture=True).split()[0]
            if observed != expected["dataset_sha256"]:
                raise ValueError("heldout partition SHA256 mismatch")


def export_model(args: argparse.Namespace, actor: Path) -> None:
    if args.export_model.exists():
        verify_export_manifest(args.export_model)
        return
    run(
        [
            "docker", "exec", args.container,
            "python3", f"{args.container_project}/scripts/export_megatron_dist_to_hf.py",
            "--actor-checkpoint", str(actor).replace(str(args.host_project), args.container_project),
            "--base-model", "/models/Qwen3.8-27B",
            "--output-dir", str(args.export_model).replace(str(args.host_project), args.container_project),
        ],
        log=args.supervisor_dir / "export.log",
    )
    verify_export_manifest(args.export_model)


def transfer_model(args: argparse.Namespace) -> None:
    manifest = args.supervisor_dir / "model_transfer.safe.json"
    run(
        ["python3", str(args.host_project / "scripts/verify_model_transfer.py"), "build", "--model-dir", str(args.export_model), "--manifest", str(manifest)],
        log=args.supervisor_dir / "model_transfer_hash.log",
    )
    remote_supervisor = str(args.supervisor_dir)
    remote(args.remote_host, ["mkdir", "-p", remote_supervisor])
    run(["scp", "-q", str(manifest), f"root@{args.remote_host}:{remote_supervisor}/model_transfer.safe.json"])
    remote_final = str(args.export_model)
    remote_partial = remote_final + ".incomplete"
    # `test` with check=True cannot represent absence, so query via a shell-safe marker.
    present = remote(args.remote_host, ["bash", "-lc", f"test -d {shlex.quote(remote_final)} && echo yes || echo no"], capture=True).strip() == "yes"
    if not present:
        remote(args.remote_host, ["mkdir", "-p", remote_partial])
        run(
            ["rsync", "--archive", "--partial", "--human-readable", str(args.export_model) + "/", f"root@{args.remote_host}:{remote_partial}/"],
            log=args.supervisor_dir / "model_rsync.log",
        )
        remote(
            args.remote_host,
            ["python3", str(args.host_project / "scripts/verify_model_transfer.py"), "verify", "--model-dir", remote_partial, "--manifest", f"{remote_supervisor}/model_transfer.safe.json"],
            log=args.supervisor_dir / "remote_model_verify.log",
        )
        remote(args.remote_host, ["mv", remote_partial, remote_final])
    else:
        remote(
            args.remote_host,
            ["python3", str(args.host_project / "scripts/verify_model_transfer.py"), "verify", "--model-dir", remote_final, "--manifest", f"{remote_supervisor}/model_transfer.safe.json"],
            log=args.supervisor_dir / "remote_model_verify.log",
        )


def assert_idle(output: str, host: str) -> None:
    pattern = re.compile(r"^\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|\s*[A-Za-z0-9_]", re.MULTILINE)
    if not output.strip() or pattern.search(output):
        raise RuntimeError(f"{host} NPUs are not idle")


def start_ray(args: argparse.Namespace) -> None:
    assert_idle(run(["npu-smi", "info"], capture=True), "m05")
    assert_idle(remote(args.remote_host, ["npu-smi", "info"], capture=True), "m06")
    run(["docker", "exec", args.container, "bash", "-lc", "ray stop --force"], log=args.supervisor_dir / "ray_cleanup.log")
    remote(args.remote_host, ["docker", "exec", args.remote_container, "bash", "-lc", "ray stop --force"], log=args.supervisor_dir / "ray_cleanup.log")
    common = f"bash {args.container_project}/scripts/start_ray_qwen38_topology_benchmark.sh"
    local_env = " ".join(
        [
            "NODE_IP=192.168.202.5", "RAY_PORT=46379", "RAY_RESOURCE=q38_m05", "EXPECTED_NPUS=16",
            "RAY_MIN_WORKER_PORT=47000", "RAY_MAX_WORKER_PORT=47999", "RAY_TEMP_DIR=/tmp/q38-step70-heldout-m05",
            "HCCL_IF_IP=192.168.202.5", "HCCL_SOCKET_IFNAME=eno0", "HCCL_IF_BASE_PORT=63000",
            "HCCL_HOST_SOCKET_PORT_RANGE=63100-63163", "HCCL_NPU_SOCKET_PORT_RANGE=63200-63263", common,
        ]
    )
    remote_env = local_env.replace("192.168.202.5", "192.168.202.4").replace("q38_m05", "q38_m06").replace("m05", "m06").replace("63000", "64000").replace("63100-63163", "64100-64163").replace("63200-63263", "64200-64263")
    run(["docker", "exec", args.container, "bash", "-lc", local_env], log=args.supervisor_dir / "ray_m05.log")
    remote(args.remote_host, ["docker", "exec", args.remote_container, "bash", "-lc", remote_env], log=args.supervisor_dir / "ray_m06.log")


def queue_command(args: argparse.Namespace, host: str) -> list[str]:
    root = f"{args.container_project}/runs/{args.eval_name}"
    parts = [
        "python3", f"{args.container_project}/scripts/run_qwen38_host_rerun_queue.py",
        "--project-root", args.container_project,
        "--model", str(args.export_model).replace(str(args.host_project), args.container_project),
        "--model-label", MODEL_LABEL, "--policy-step", str(POLICY_STEP),
        "--host-label", host,
    ]
    for version in VERSIONS:
        parts.extend(["--arm", f"{version}={root}/data/partitions/{version}_{host}.sensitive.parquet"])
    parts.extend(
        [
            "--rollout-resource", f"q38_{host}", "--tensor-parallel-size", "4", "--data-parallel-size", "4",
            "--rollout-npus", "16", "--max-num-seqs", "16", "--task-batch-size", "32",
            "--rolling-window-trajectories", "80", "--monitor-first-card", "0", "--monitor-num-cards", "16",
            "--runs-dir", f"{root}/{host}/waves", "--run-prefix", f"{args.eval_name}-{host}",
            "--state-root", f"{root}/{host}/state", "--final-root", f"{root}/{host}/final",
            "--queue-dir", f"{root}/{host}/queue", "--ray-address", "192.168.202.5:46379" if host == "m05" else "192.168.202.4:46379",
            "--poll-seconds", "30", "--stage-timeout-seconds", "1209600",
        ]
    )
    return parts


def launch_queues(args: argparse.Namespace) -> None:
    for host, container in (("m05", args.container), ("m06", args.remote_container)):
        command = command_text(queue_command(args, host))
        root = f"{args.container_project}/runs/{args.eval_name}/{host}"
        shell = f"mkdir -p {shlex.quote(root)} && {command} > {shlex.quote(root + '/host_queue.log')} 2>&1"
        parts = ["docker", "exec", "-d", container, "bash", "-lc", shell]
        if host == "m05":
            run(parts)
        else:
            remote(args.remote_host, parts)


def read_exit(path: Path) -> int | None:
    return int(path.read_text().strip()) if path.is_file() else None


def wait_queues(args: argparse.Namespace) -> None:
    local_exit = args.host_project / "runs" / args.eval_name / "m05" / "queue" / "exit_code"
    remote_exit = str(args.host_project / "runs" / args.eval_name / "m06" / "queue" / "exit_code")
    while True:
        m05 = read_exit(local_exit)
        raw = remote(args.remote_host, ["bash", "-lc", f"test -f {shlex.quote(remote_exit)} && cat {shlex.quote(remote_exit)} || true"], capture=True).strip()
        m06 = int(raw) if raw else None
        status(args.supervisor_dir, "evaluating", host_exit_codes={"m05": m05, "m06": m06})
        if m05 is not None and m05 != 0:
            raise RuntimeError(f"m05 heldout queue failed with {m05}")
        if m06 is not None and m06 != 0:
            raise RuntimeError(f"m06 heldout queue failed with {m06}")
        if m05 == 0 and m06 == 0:
            return
        time.sleep(args.poll_seconds)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    versions = []
    for version in VERSIONS:
        per_host = {}
        for host in ("m05", "m06"):
            path = args.host_project / "runs" / args.eval_name / host / "final" / version / "adaptive_final_safe_summary.json"
            if host == "m05":
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                payload = json.loads(remote(args.remote_host, ["cat", str(path)], capture=True))
            if int(payload.get("initial_tasks", -1)) != HOST_TASKS[host][version]:
                raise ValueError("host final summary task count mismatch")
            per_host[host] = payload
        numeric = ("initial_tasks", "mixed_after_two_tasks", "mixed_after_four_tasks", "mixed_after_six_tasks", "variance_candidate_tasks", "unresolved_after_six_tasks", "actual_sampling_trajectories", "avoided_trajectories_vs_full_six")
        versions.append({"version": version, **{key: sum(int(per_host[h][key]) for h in per_host) for key in numeric}})
    result = {
        "contract": CONTRACT,
        "stage": "complete",
        "model_label": MODEL_LABEL,
        "policy_step": POLICY_STEP,
        "heldout_tasks": sum(item["initial_tasks"] for item in versions),
        "versions": versions,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
    }
    write_json(args.supervisor_dir / "heldout_result.safe.json", result)
    return result


def stop_ray(args: argparse.Namespace) -> None:
    subprocess.run(["docker", "exec", args.container, "bash", "-lc", "ray stop --force"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ssh", "-o", "BatchMode=yes", f"root@{args.remote_host}", command_text(["docker", "exec", args.remote_container, "bash", "-lc", "ray stop --force"])], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def execute(args: argparse.Namespace) -> None:
    args.supervisor_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.host_project / "runs" / ".qwen38-step70-heldout.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for stale in ("exit_code", "finished_at"):
        path = args.supervisor_dir / stale
        if path.is_file():
            path.unlink()
    (args.supervisor_dir / "supervisor.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    ray_started = False
    try:
        wait_for_training(args)
        status(args.supervisor_dir, "verifying_frozen_heldout_data")
        verify_heldout_data(args)
        status(args.supervisor_dir, "verifying_final_checkpoint")
        actor = verify_checkpoint(args)
        status(args.supervisor_dir, "exporting_final_model")
        export_model(args, actor)
        status(args.supervisor_dir, "copying_verified_model_to_m06")
        transfer_model(args)
        status(args.supervisor_dir, "starting_two_independent_tp4dp4_clusters")
        start_ray(args)
        ray_started = True
        status(args.supervisor_dir, "launching_heldout_queues")
        launch_queues(args)
        wait_queues(args)
        result = aggregate(args)
        status(args.supervisor_dir, "complete", result_summary=result)
        (args.supervisor_dir / "exit_code").write_text("0\n", encoding="utf-8")
    except BaseException as exc:
        status(args.supervisor_dir, "failed", error_type=type(exc).__name__)
        (args.supervisor_dir / "exit_code").write_text("1\n", encoding="utf-8")
        raise
    finally:
        if ray_started:
            stop_ray(args)
        (args.supervisor_dir / "finished_at").write_text(utc_now() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-project", type=Path, default=Path("/data3/llin/qwen3.6-27b-verl-grpo"))
    parser.add_argument("--container-project", default="/workspace/llin-verl-grpo")
    parser.add_argument("--container", default="llin-verl-qwen38-smoke-m05-20260817")
    parser.add_argument("--remote-container", default="llin-verl-qwen38-smoke-m06-20260817")
    parser.add_argument("--remote-host", default="192.168.202.4")
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--training-supervisor", type=Path, required=True)
    parser.add_argument("--export-model", type=Path, required=True)
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--supervisor-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    execute(parse_args())
