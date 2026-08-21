#!/usr/bin/env python3
"""Evaluate the mixed27 Step54 checkpoint on the six frozen sealed tasks.

The supervisor runs on machine 5.  It exports the final Megatron actor to an
exactly verified HF model, copies the immutable model and sealed Parquet to
machine 6, runs strict 2+2+2 TP4xDP4 evaluation, writes only aggregate safe
results, and releases Ray/NPU resources.  Sealed rows and sampled trajectories
remain private and are never made training-eligible.
"""

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


CONTRACT = "llin-qwen38-step70-mixed27-sealed6-eval-supervisor-v1"
RESULT_CONTRACT = "llin-qwen38-step70-mixed27-sealed6-eval-result-v1"
MODEL_LABEL = "qwen38-27b-step70-mixed27-4x-step54"
POLICY_STEP = 54
SEALED_TASKS = 6
RUNTIME_FILES = (
    "adaptive_dwh_wave_earlystop.py",
    "analyze_multisandbox_dwh_rollout.py",
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_text(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def run(
    parts: list[str],
    *,
    capture: bool = False,
    log: Path | None = None,
    input_text: str | None = None,
) -> str:
    if log is None:
        result = subprocess.run(
            parts,
            check=True,
            text=True,
            capture_output=capture,
            input=input_text,
        )
        return result.stdout if capture else ""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(
            parts,
            check=True,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            input=input_text,
        )
    return ""


def remote(
    args: argparse.Namespace,
    parts: list[str],
    *,
    capture: bool = False,
    log: Path | None = None,
    input_text: str | None = None,
) -> str:
    return run(
        ["ssh", "-o", "BatchMode=yes", f"root@{args.remote_host}", command_text(parts)],
        capture=capture,
        log=log,
        input_text=input_text,
    )


def status(args: argparse.Namespace, stage: str, **fields: Any) -> None:
    write_json(
        args.supervisor_dir / "sealed6_eval.safe.json",
        {
            "contract": CONTRACT,
            "stage": stage,
            "updated_at": utc_now(),
            "model_label": MODEL_LABEL,
            "policy_step": POLICY_STEP,
            "sealed_tasks": SEALED_TASKS,
            "sampling": "strict_2_plus_2_plus_2_max_6",
            "reasoning_effort": "medium",
            "max_context_tokens": 94208,
            "trajectory_timeout_seconds": 1800,
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
        os.kill(int(pid_path.read_text(encoding="utf-8").strip()), 0)
    except (ValueError, ProcessLookupError):
        return False
    return True


def verify_frozen_inputs(args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads(args.pool_summary.read_text(encoding="utf-8"))
    if int(summary.get("sealed_tasks", -1)) != SEALED_TASKS:
        raise ValueError("sealed task count gate failed")
    if summary.get("sealed_training_allowed") is not False:
        raise ValueError("sealed training gate is not closed")
    observed_sha256 = file_sha256(args.sealed_dataset)
    if observed_sha256 != str(summary.get("sealed_sha256") or ""):
        raise ValueError("sealed dataset SHA256 gate failed")
    checkpoint_root = args.actor_checkpoint.parent.parent
    checkpoint_steps = sorted(path.name for path in checkpoint_root.glob("global_step_*") if path.is_dir())
    if checkpoint_steps != [f"global_step_{POLICY_STEP}"]:
        raise ValueError(f"expected only global_step_{POLICY_STEP}, observed {checkpoint_steps}")
    if not (args.actor_checkpoint / "ckpt_contents.json").is_file():
        raise FileNotFoundError("actor checkpoint manifest is missing")
    return {
        "sealed_sha256": observed_sha256,
        "sealed_source_version_counts": summary.get("sealed_source_version_counts"),
        "sealed_difficulty_counts": summary.get("sealed_difficulty_counts"),
    }


def export_command(args: argparse.Namespace) -> list[str]:
    pythonpath = ":".join(
        [
            "/vllm",
            f"{args.container_project}/reference/Megatron-Bridge-de93536e/src",
            f"{args.container_project}/runtime",
            args.container_project,
        ]
    )
    return [
        "docker",
        "exec",
        "-e",
        f"PYTHONPATH={pythonpath}",
        args.local_container,
        "python3",
        f"{args.container_project}/scripts/export_megatron_dist_to_hf.py",
        "--actor-checkpoint",
        str(args.actor_checkpoint).replace(str(args.host_project), args.container_project),
        "--base-model",
        "/models/Qwen3.8-27B",
        "--output-dir",
        str(args.export_model).replace(str(args.host_project), args.container_project),
    ]


def wait_for_export(args: argparse.Namespace) -> None:
    manifest = args.export_model / "llin_export_manifest.json"
    export_pid = args.supervisor_dir / "export.pid"
    if not manifest.is_file() and not process_alive(export_pid):
        if args.export_model.exists():
            raise FileExistsError("partial export exists without a live exporter")
        log = (args.supervisor_dir / "export.log").open("a", encoding="utf-8")
        process = subprocess.Popen(export_command(args), stdout=log, stderr=subprocess.STDOUT, text=True)
        export_pid.write_text(f"{process.pid}\n", encoding="utf-8")
    while not manifest.is_file():
        if not process_alive(export_pid):
            raise RuntimeError("model export stopped before publishing a verified manifest")
        status(args, "exporting_final_checkpoint")
        time.sleep(args.poll_seconds)
    gate = args.supervisor_dir / "export_origin.safe.json"
    run(
        [
            "python3",
            str(args.host_project / "scripts" / "check_qwen38_export_origin.py"),
            "--model",
            str(args.export_model),
            "--expected-policy-step",
            str(POLICY_STEP),
            "--output",
            str(gate),
        ],
        log=args.supervisor_dir / "export_origin.log",
    )


def transfer_model(args: argparse.Namespace) -> None:
    manifest = args.supervisor_dir / "model_transfer.safe.json"
    verifier = args.host_project / "scripts" / "verify_model_transfer.py"
    run(
        ["python3", str(verifier), "build", "--model-dir", str(args.export_model), "--manifest", str(manifest)],
        log=args.supervisor_dir / "model_transfer_build.log",
    )
    remote(args, ["mkdir", "-p", str(args.supervisor_dir), str(args.export_model.parent)])
    run(["scp", "-q", str(manifest), f"root@{args.remote_host}:{args.supervisor_dir}/model_transfer.safe.json"])
    run(["scp", "-q", str(verifier), f"root@{args.remote_host}:{args.supervisor_dir}/verify_model_transfer.py"])
    present = remote(
        args,
        ["bash", "-lc", f"test -d {shlex.quote(str(args.export_model))} && echo yes || echo no"],
        capture=True,
    ).strip() == "yes"
    if not present:
        partial = Path(str(args.export_model) + ".incomplete")
        remote(args, ["mkdir", str(partial)])
        run(
            ["rsync", "--archive", "--partial", str(args.export_model) + "/", f"root@{args.remote_host}:{partial}/"],
            log=args.supervisor_dir / "model_rsync.log",
        )
        remote(
            args,
            [
                "python3",
                str(args.supervisor_dir / "verify_model_transfer.py"),
                "verify",
                "--model-dir",
                str(partial),
                "--manifest",
                str(args.supervisor_dir / "model_transfer.safe.json"),
            ],
            log=args.supervisor_dir / "remote_model_verify.log",
        )
        remote(args, ["mv", str(partial), str(args.export_model)])
    else:
        remote(
            args,
            [
                "python3",
                str(args.supervisor_dir / "verify_model_transfer.py"),
                "verify",
                "--model-dir",
                str(args.export_model),
                "--manifest",
                str(args.supervisor_dir / "model_transfer.safe.json"),
            ],
            log=args.supervisor_dir / "remote_model_verify.log",
        )


def transfer_sealed_data(args: argparse.Namespace, frozen: dict[str, Any]) -> None:
    remote(args, ["mkdir", "-p", str(args.sealed_dataset.parent)])
    remote_sha = remote(
        args,
        ["bash", "-lc", f"test -f {shlex.quote(str(args.sealed_dataset))} && sha256sum {shlex.quote(str(args.sealed_dataset))} | cut -d' ' -f1 || true"],
        capture=True,
    ).strip()
    if remote_sha != frozen["sealed_sha256"]:
        temporary = Path(str(args.sealed_dataset) + ".incomplete")
        run(["scp", "-q", str(args.sealed_dataset), f"root@{args.remote_host}:{temporary}"])
        observed = remote(args, ["sha256sum", str(temporary)], capture=True).split()[0]
        if observed != frozen["sealed_sha256"]:
            raise ValueError("remote sealed dataset SHA256 mismatch")
        remote(args, ["chmod", "600", str(temporary)])
        remote(args, ["mv", str(temporary), str(args.sealed_dataset)])


def sync_runtime(args: argparse.Namespace) -> None:
    scripts = [args.host_project / "scripts" / name for name in RUNTIME_FILES]
    sources = scripts + [args.host_project / "runtime" / "sitecustomize.py"]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"runtime files missing: {missing}")
    expected = {path.relative_to(args.host_project).as_posix(): file_sha256(path) for path in sources}
    remote(args, ["mkdir", "-p", str(args.host_project / "scripts"), str(args.host_project / "runtime")])
    run(["scp", "-q", *[str(path) for path in scripts], f"root@{args.remote_host}:{args.host_project}/scripts/"])
    run(["scp", "-q", str(sources[-1]), f"root@{args.remote_host}:{args.host_project}/runtime/sitecustomize.py"])
    paths = [str(args.host_project / relative) for relative in expected]
    raw = remote(args, ["sha256sum", *paths], capture=True)
    observed = {
        Path(line.split(maxsplit=1)[1]).relative_to(args.host_project).as_posix(): line.split(maxsplit=1)[0]
        for line in raw.splitlines()
        if len(line.split(maxsplit=1)) == 2
    }
    if observed != expected:
        raise RuntimeError("remote frozen runtime SHA256 mismatch")
    write_json(
        args.supervisor_dir / "runtime_bundle.safe.json",
        {
            "contract": "llin-qwen38-mixed27-sealed6-runtime-bundle-v1",
            "status": "passed",
            "file_count": len(expected),
            "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
        },
    )


def assert_remote_idle(args: argparse.Namespace) -> None:
    output = remote(args, ["npu-smi", "info"], capture=True)
    process_pattern = re.compile(r"^\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|\s*[A-Za-z0-9_]", re.MULTILINE)
    if not output.strip() or process_pattern.search(output):
        raise RuntimeError("machine 6 NPUs are not idle")


def start_ray(args: argparse.Namespace) -> None:
    assert_remote_idle(args)
    remote(args, ["docker", "exec", args.remote_container, "bash", "-lc", "ray stop --force"])
    environment = " ".join(
        [
            "NODE_IP=192.168.202.4",
            "RAY_PORT=46379",
            "RAY_RESOURCE=q38_m06",
            "EXPECTED_NPUS=16",
            "RAY_MIN_WORKER_PORT=47000",
            "RAY_MAX_WORKER_PORT=47999",
            "RAY_TEMP_DIR=/tmp/q38-mixed27-sealed6-m06",
            "HCCL_IF_IP=192.168.202.4",
            "HCCL_SOCKET_IFNAME=eno0",
            "HCCL_IF_BASE_PORT=64000",
            "HCCL_HOST_SOCKET_PORT_RANGE=64100-64163",
            "HCCL_NPU_SOCKET_PORT_RANGE=64200-64263",
            f"bash {args.container_project}/scripts/start_ray_qwen38_topology_benchmark.sh",
        ]
    )
    remote(
        args,
        ["docker", "exec", args.remote_container, "bash", "-lc", environment],
        log=args.supervisor_dir / "ray_m06.log",
    )


def queue_command(args: argparse.Namespace) -> list[str]:
    root = f"{args.container_project}/runs/{args.eval_name}"
    model = str(args.export_model).replace(str(args.host_project), args.container_project)
    sealed = str(args.sealed_dataset).replace(str(args.host_project), args.container_project)
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
        "m06",
        "--arm",
        f"sealed6={sealed}",
        "--rollout-resource",
        "q38_m06",
        "--tensor-parallel-size",
        "4",
        "--data-parallel-size",
        "4",
        "--rollout-npus",
        "16",
        "--max-num-seqs",
        "16",
        "--task-batch-size",
        "6",
        "--rolling-window-trajectories",
        "80",
        "--monitor-first-card",
        "0",
        "--monitor-num-cards",
        "16",
        "--runs-dir",
        f"{root}/waves",
        "--run-prefix",
        f"{args.eval_name}-m06",
        "--state-root",
        f"{root}/state",
        "--final-root",
        f"{root}/final",
        "--queue-dir",
        f"{root}/queue",
        "--ray-address",
        "192.168.202.4:46379",
        "--poll-seconds",
        "10",
        "--stage-timeout-seconds",
        "86400",
    ]


def run_queue(args: argparse.Namespace) -> None:
    remote(
        args,
        ["docker", "exec", args.remote_container, *queue_command(args)],
        log=args.supervisor_dir / "queue_m06.log",
    )


def aggregate_code(args: argparse.Namespace) -> str:
    root = f"{args.container_project}/runs/{args.eval_name}/final/sealed6"
    sealed = str(args.sealed_dataset).replace(str(args.host_project), args.container_project)
    return f'''import collections, json\nimport pyarrow.parquet as pq\n\ndef summarize(rows):\n    result = {{\n        "tasks": len(rows),\n        "tasks_with_any_correct": 0,\n        "mixed_tasks": 0,\n        "all_correct_tasks": 0,\n        "all_wrong_or_no_complete_tasks": 0,\n        "correct_trajectories": 0,\n        "completed_trajectories": 0,\n        "timeout_trajectories": 0,\n        "runtime_error_trajectories": 0,\n        "sample_count_distribution": {{}},\n        "source_version_counts": {{}},\n        "difficulty_counts": {{}},\n    }}\n    samples = collections.Counter()\n    versions = collections.Counter()\n    difficulties = collections.Counter()\n    for row in rows:\n        extra = row.get("extra_info") or {{}}\n        correct = int(extra.get("adaptive_correct_count") or 0)\n        completed = int(extra.get("adaptive_completed_count") or 0)\n        observed = int(extra.get("adaptive_samples_observed") or 0)\n        result["correct_trajectories"] += correct\n        result["completed_trajectories"] += completed\n        result["timeout_trajectories"] += int(extra.get("adaptive_timeout_count") or 0)\n        result["runtime_error_trajectories"] += int(extra.get("adaptive_runtime_error_count") or 0)\n        result["tasks_with_any_correct"] += int(correct > 0)\n        result["mixed_tasks"] += int(correct > 0 and completed > correct)\n        result["all_correct_tasks"] += int(completed > 0 and correct == completed)\n        result["all_wrong_or_no_complete_tasks"] += int(correct == 0)\n        samples[str(observed)] += 1\n        raw_version = str(extra.get("source_version") or extra.get("qwen38_rerun_version") or "unknown")\n        version = next((item for item in ("v15", "v20", "v21") if raw_version == item or raw_version.endswith("_" + item)), raw_version)\n        versions[version] += 1\n        difficulties[str(extra.get("difficulty_level"))] += 1\n    result["sample_count_distribution"] = dict(sorted(samples.items()))\n    result["source_version_counts"] = dict(sorted(versions.items()))\n    result["difficulty_counts"] = dict(sorted(difficulties.items()))\n    return result\n\nbaseline = pq.read_table({sealed!r}).to_pylist()\ncandidates = pq.read_table({(root + "/grpo_variance_candidates.sensitive.parquet")!r}).to_pylist()\nunresolved = pq.read_table({(root + "/unresolved_after_six.sensitive.parquet")!r}).to_pylist()\nprint(json.dumps({{"baseline": summarize(baseline), "post_training": summarize(candidates + unresolved)}}, sort_keys=True))\n'''


def aggregate(args: argparse.Namespace, frozen: dict[str, Any]) -> dict[str, Any]:
    root = args.host_project / "runs" / args.eval_name / "final" / "sealed6"
    summary = json.loads(remote(args, ["cat", str(root / "adaptive_final_safe_summary.json")], capture=True))
    raw = remote(
        args,
        ["docker", "exec", "-i", args.remote_container, "python3", "-"],
        capture=True,
        input_text=aggregate_code(args),
    )
    aggregates = json.loads(raw)
    baseline = aggregates["baseline"]
    post = aggregates["post_training"]
    if baseline["tasks"] != SEALED_TASKS or post["tasks"] != SEALED_TASKS:
        raise ValueError("sealed aggregate task count mismatch")
    if baseline["source_version_counts"] != frozen["sealed_source_version_counts"]:
        raise ValueError("sealed source-version aggregate changed")
    result = {
        "contract": RESULT_CONTRACT,
        "completed_at": utc_now(),
        "model_label": MODEL_LABEL,
        "policy_step": POLICY_STEP,
        "sealed_tasks": SEALED_TASKS,
        "sampling": {
            "adaptive": "strict_2_plus_2_plus_2_max_6",
            "reasoning_effort": "medium",
            "max_prompt_tokens": 4096,
            "max_response_tokens": 90112,
            "max_context_tokens": 94208,
            "trajectory_timeout_seconds": 1800,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "topology": "16 NPU, TP4xDP4",
            "max_num_seqs_per_replica": 16,
        },
        "baseline_before_mixed27_training": baseline,
        "post_training": post,
        "adaptive_final": {
            key: summary[key]
            for key in (
                "mixed_after_two_tasks",
                "mixed_after_four_tasks",
                "mixed_after_six_tasks",
                "variance_candidate_tasks",
                "unresolved_after_six_tasks",
                "actual_sampling_trajectories",
                "avoided_trajectories_vs_full_six",
            )
        },
        "comparison": {
            "tasks_with_any_correct_delta": post["tasks_with_any_correct"] - baseline["tasks_with_any_correct"],
            "mixed_tasks_delta": post["mixed_tasks"] - baseline["mixed_tasks"],
            "correct_trajectories_delta": post["correct_trajectories"] - baseline["correct_trajectories"],
            "completed_trajectories_delta": post["completed_trajectories"] - baseline["completed_trajectories"],
        },
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
    }
    write_json(args.supervisor_dir / "sealed6_result.safe.json", result)
    return result


def stop_remote(args: argparse.Namespace) -> None:
    pattern = f"[{args.eval_name[0]}]{args.eval_name[1:]}"
    shell = command_text(["pkill", "-TERM", "-f", "--", pattern]) + "; sleep 2; " + command_text(
        ["pkill", "-KILL", "-f", "--", pattern]
    ) + " || true"
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            f"root@{args.remote_host}",
            command_text(["docker", "exec", args.remote_container, "bash", "-lc", shell]),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            f"root@{args.remote_host}",
            command_text(["docker", "exec", args.remote_container, "bash", "-lc", "ray stop --force"]),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def execute(args: argparse.Namespace) -> None:
    args.supervisor_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.host_project / "runs" / ".qwen38-mixed27-sealed6-eval.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for stale in ("exit_code", "finished_at"):
        path = args.supervisor_dir / stale
        if path.is_file():
            path.unlink()
    (args.supervisor_dir / "supervisor.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    ray_started = False
    try:
        status(args, "validating_frozen_inputs")
        frozen = verify_frozen_inputs(args)
        status(args, "waiting_for_verified_hf_export")
        wait_for_export(args)
        status(args, "copying_verified_model_to_m06")
        transfer_model(args)
        status(args, "copying_sealed6_to_m06")
        transfer_sealed_data(args, frozen)
        status(args, "syncing_frozen_runtime_to_m06")
        sync_runtime(args)
        status(args, "starting_m06_tp4dp4")
        start_ray(args)
        ray_started = True
        status(args, "evaluating_sealed6")
        run_queue(args)
        status(args, "aggregating_safe_result")
        result = aggregate(args, frozen)
        status(args, "complete", result_summary=result)
        (args.supervisor_dir / "exit_code").write_text("0\n", encoding="utf-8")
    except BaseException as exc:
        status(args, "failed", error_type=type(exc).__name__)
        (args.supervisor_dir / "exit_code").write_text("1\n", encoding="utf-8")
        raise
    finally:
        if ray_started:
            stop_remote(args)
        (args.supervisor_dir / "finished_at").write_text(utc_now() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-project", type=Path, default=Path("/data3/llin/qwen3.6-27b-verl-grpo"))
    parser.add_argument("--container-project", default="/workspace/llin-verl-grpo")
    parser.add_argument("--local-container", default="llin-verl-qwen38-smoke-m05-20260817")
    parser.add_argument("--remote-container", default="llin-verl-qwen38-smoke-m06-20260817")
    parser.add_argument("--remote-host", default="192.168.202.4")
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--pool-summary", type=Path, required=True)
    parser.add_argument("--sealed-dataset", type=Path, required=True)
    parser.add_argument("--export-model", type=Path, required=True)
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--supervisor-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    execute(parse_args())
