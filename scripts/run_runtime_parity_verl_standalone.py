#!/usr/bin/env python3
"""Run the veRL PI agent loop without creating an actor/trainer worker.

The standalone server manager loads the frozen Step120 HF export directly on
the rollout node.  No optimizer, actor, checkpoint sync, or training step is
created.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import hydra
import numpy as np
import ray
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from transformers import AutoTokenizer

from llin_verl.trajectory_telemetry import ENQUEUED_EPOCH_NS_KEY, TELEMETRY_CONTRACT
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.workers.rollout.llm_server import LLMServerManager

from scripts.pi_runtime_preflight import validate_dataset_runtime_environments
from scripts.standalone_rollout_shards import (
    completed_shard_rows,
    padded_rows_for_equal_chunks,
    rolling_admission_contract,
    shard_path,
    shard_ranges,
    trajectory_admission_contract,
    write_jsonl_atomic,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(model: Path, policy_step: int) -> dict:
    """Verify either an exact llin export or the frozen native HF layout."""

    config = model / "config.json"
    index = model / "model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        raise FileNotFoundError("model config or safetensor index is missing")
    export_manifest = model / "llin_export_manifest.json"
    if export_manifest.is_file():
        payload = json.loads(export_manifest.read_text(encoding="utf-8"))
        verification = payload.get("verification") or {}
        actor_checkpoint = str(payload.get("actor_checkpoint") or "")
        if verification.get("valid") is not True:
            raise ValueError("llin export manifest verification is not valid")
        if f"global_step_{policy_step}" not in actor_checkpoint:
            raise ValueError("llin export manifest policy step mismatch")
        kind = "llin_megatron_to_hf_export"
        manifest_sha256 = file_sha256(export_manifest)
    else:
        if policy_step != 0:
            raise ValueError("only the native policy step may omit llin_export_manifest.json")
        kind = "native_hf_checkpoint"
        manifest_sha256 = None
    return {
        "valid": True,
        "kind": kind,
        "policy_step": policy_step,
        "config_sha256": file_sha256(config),
        "safetensor_index_sha256": file_sha256(index),
        "export_manifest_sha256": manifest_sha256,
    }


def build_config(args: argparse.Namespace):
    config_dir = "/verl/verl/experimental/one_step_off_policy/config"
    overrides = [
        f"data.train_files={args.dataset}",
        f"data.val_files={args.dataset}",
        f"data.max_prompt_length={args.max_prompt_tokens}",
        f"data.max_response_length={args.max_response_tokens}",
        "data.filter_overlong_prompts=True",
        "data.filter_overlong_prompts_workers=4",
        "data.return_raw_chat=True",
        "data.return_multi_modal_inputs=False",
        "data.truncation=error",
        "data.continuous_token.enable=True",
        "data.continuous_token.model_family=qwen35",
        f"actor_rollout_ref.model.path={args.model}",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
        "actor_rollout_ref.rollout.data_parallel_size=2",
        "actor_rollout_ref.rollout.pipeline_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={args.max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_model_len={args.max_context_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={args.max_num_seqs}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.enable_prefix_caching=True",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.calculate_log_probs=False",
        "actor_rollout_ref.rollout.disable_log_stats=False",
        f"actor_rollout_ref.rollout.n={args.samples_per_task}",
        "actor_rollout_ref.rollout.nnodes=1",
        "actor_rollout_ref.rollout.n_gpus_per_node=16",
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={args.project_root}/configs/pi_workspace_tools.yaml",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={args.project_root}/configs/pi_agent_loops.yaml",
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=26",
        "actor_rollout_ref.rollout.multi_turn.max_user_turns=25",
        "actor_rollout_ref.rollout.multi_turn.max_parallel_calls=4",
        "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=32768",
        "actor_rollout_ref.rollout.multi_turn.format=qwen3_coder",
        "actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable",
        f"+actor_rollout_ref.rollout.multi_turn.agent_timeout_seconds={args.trajectory_timeout_seconds}",
        f"actor_rollout_ref.rollout.agent.num_workers={args.agent_workers}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=pi_agent",
        f"actor_rollout_ref.rollout.val_kwargs.n={args.samples_per_task}",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_p=0.95",
        "actor_rollout_ref.rollout.val_kwargs.top_k=20",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "rollout.nnodes=1",
        "rollout.n_gpus_per_node=16",
        "trainer.n_gpus_per_node=16",
    ]
    previous_cwd = Path.cwd()
    try:
        # The upstream config declares ``file://verl/trainer/config`` relative
        # to the veRL checkout root, just like ``python -m ...`` does.
        os.chdir("/verl")
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            config = hydra.compose(
                config_name="one_step_off_ppo_megatron_trainer",
                overrides=overrides,
            )
    finally:
        os.chdir(previous_cwd)
    OmegaConf.resolve(config)
    return config


def safe_contract(config, args: argparse.Namespace) -> dict:
    rollout = config.actor_rollout_ref.rollout
    identity = model_identity(args.model, args.policy_step)
    return {
        "contract": "verl-standalone-runtime-parity-arm-v2",
        "model_label": args.model_label,
        "policy_step": args.policy_step,
        "model_identity": identity,
        "model_manifest_exists": (args.model / "llin_export_manifest.json").is_file(),
        "dataset_exists": args.dataset.is_file(),
        "tasks": args.expected_tasks,
        "samples_per_task": int(rollout.val_kwargs.n),
        "temperature": float(rollout.val_kwargs.temperature),
        "top_p": float(rollout.val_kwargs.top_p),
        "top_k": int(rollout.val_kwargs.top_k),
        "tensor_parallel_size": int(rollout.tensor_model_parallel_size),
        "data_parallel_size": int(rollout.data_parallel_size),
        "rollout_npus": int(rollout.n_gpus_per_node),
        "agent_workers": int(rollout.agent.num_workers),
        "max_num_seqs_per_dp_engine": int(rollout.max_num_seqs),
        "max_num_batched_tokens_per_dp_engine": int(rollout.max_num_batched_tokens),
        "gpu_memory_utilization": float(rollout.gpu_memory_utilization),
        "max_prompt_tokens": int(config.data.max_prompt_length),
        "max_response_tokens": int(config.data.max_response_length),
        "context_tokens": int(rollout.max_model_len),
        "trajectory_timeout_seconds": args.trajectory_timeout_seconds,
        "trajectory_telemetry": {
            "contract": TELEMETRY_CONTRACT,
            "queue_clock": "unix_epoch_ns_cross_process",
            "queue_scope": "model_service_ready_to_agent_execution_start",
            "generation_scope": "agent_generating_state_wall",
            "tool_scope": "agent_processing_tools_state_wall",
            "total_scope": "enqueue_to_agent_completion",
            "timeout_partial_tokens": "completed_turn_buffer_plus_active_vllm_request",
        },
        "task_batch_size": args.task_batch_size,
        "rolling_admission_enabled": bool(args.rolling_admission),
        "rolling_window_trajectories_requested": args.rolling_window_trajectories,
        "tail_batch_padding_policy": "duplicate_last_rows_then_trim_before_persisting",
        "batch_validate_mode": True,
        "batch_do_sample": True,
        "effective_sampling_source": "validation_val_kwargs",
        "default_validation_temperature_zero_overridden": True,
        "training_rollout_temperature": float(rollout.temperature),
        "native_pi_per_request_max_tokens": 8192,
        "verl_per_request_limit": "dynamic_remaining_context_up_to_response_budget",
        "strict_runtime_configuration_matched": False,
        "optimizer_initialized": False,
        "actor_worker_created": False,
        "checkpoint_saved": False,
    }


def build_dataset(config, args: argparse.Namespace, tokenizer) -> RLHFDataset:
    dataset = RLHFDataset(
        data_files=str(args.dataset),
        tokenizer=tokenizer,
        config=config.data,
        processor=None,
    )
    if len(dataset) != args.expected_tasks:
        raise ValueError(f"expected {args.expected_tasks} tasks, got {len(dataset)}")
    return dataset


def build_batch(
    config,
    args: argparse.Namespace,
    tokenizer,
    dataset: RLHFDataset,
    start: int,
    stop: int,
) -> DataProto:
    loader = DataLoader(
        Subset(dataset, range(start, stop)),
        batch_size=stop - start,
        shuffle=False,
        collate_fn=collate_fn,
    )
    batch = DataProto.from_single_dict(next(iter(loader)))
    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    batch = batch.repeat(repeat_times=args.samples_per_task, interleave=True)
    batch.meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": True,
        "validate": True,
        "global_steps": args.policy_step,
    }
    return batch


def safe_progress(args: argparse.Namespace, completed_tasks: int, completed_rows: int) -> dict:
    return {
        "contract": "verl-standalone-runtime-parity-progress-v1",
        "expected_tasks": args.expected_tasks,
        "samples_per_task": args.samples_per_task,
        "completed_tasks": completed_tasks,
        "completed_rows": completed_rows,
        "remaining_tasks": args.expected_tasks - completed_tasks,
    }


def write_progress(args: argparse.Namespace, completed_tasks: int, completed_rows: int) -> None:
    path = args.output_dir / "progress.safe.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(safe_progress(args, completed_tasks, completed_rows), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def stamp_trajectory_enqueue(batch: DataProto, *, epoch_ns: int | None = None) -> None:
    """Attach a per-row enqueue timestamp without mutating dataset-owned dicts."""

    timestamp = time.time_ns() if epoch_ns is None else int(epoch_ns)
    existing = batch.non_tensor_batch.get("extra_info")
    if existing is None:
        existing = np.array([{} for _ in range(len(batch))], dtype=object)
    if len(existing) != len(batch):
        raise ValueError("extra_info length does not match trajectory batch")
    stamped = np.empty(len(batch), dtype=object)
    for index, value in enumerate(existing):
        extra_info = dict(value) if isinstance(value, dict) else {}
        extra_info[ENQUEUED_EPOCH_NS_KEY] = timestamp
        stamped[index] = extra_info
    batch.non_tensor_batch["extra_info"] = stamped


def _non_tensor_scalar(output: DataProto, key: str, index: int, default):
    values = output.non_tensor_batch.get(key)
    if values is None or index >= len(values) or values[index] is None:
        return default
    return values[index]


def trajectory_telemetry_row(output: DataProto, index: int) -> dict:
    """Extract stable scalar telemetry columns from one generated row."""

    float_fields = (
        "trajectory_queue_wait_seconds",
        "trajectory_generation_seconds",
        "trajectory_tool_seconds",
        "trajectory_execution_seconds",
        "trajectory_total_seconds",
        "trajectory_overhead_seconds",
    )
    int_fields = (
        "trajectory_generation_calls",
        "trajectory_tool_calls",
        "trajectory_assistant_turns",
        "trajectory_user_turns",
        "trajectory_response_tokens_observed",
        "trajectory_generated_tokens_observed",
        "trajectory_timeout_partial_response_tokens",
        "trajectory_timeout_partial_generation_tokens",
    )
    row = {
        field: float(_non_tensor_scalar(output, field, index, -1.0))
        for field in float_fields
    }
    row.update(
        {
            field: int(_non_tensor_scalar(output, field, index, 0))
            for field in int_fields
        }
    )
    row["trajectory_queue_wait_available"] = bool(
        _non_tensor_scalar(output, "trajectory_queue_wait_available", index, False)
    )
    row["trajectory_telemetry_contract"] = str(
        _non_tensor_scalar(output, "trajectory_telemetry_contract", index, "")
    )
    return row


def decode_single_trajectory(
    output: DataProto,
    tokenizer,
    *,
    task_index: int,
    sample_index: int,
) -> dict:
    """Convert one completed worker result into the existing shard row schema."""

    if len(output) != 1:
        raise ValueError(f"rolling worker returned {len(output)} rows instead of one")
    prompt = tokenizer.decode(output.batch["prompts"][0], skip_special_tokens=True)
    solution = tokenizer.decode(output.batch["responses"][0], skip_special_tokens=True)
    response_tokens = int(
        output.batch["responses"][0].ne(tokenizer.pad_token_id).sum().cpu().item()
    )
    turns = output.non_tensor_batch.get(
        "__num_turns__",
        output.non_tensor_batch.get("num_turns", np.zeros(1, dtype=int)),
    )
    timeouts = output.non_tensor_batch.get(
        "trajectory_timeout", np.zeros(1, dtype=bool)
    )
    timeout_seconds = output.non_tensor_batch.get(
        "trajectory_timeout_seconds", np.zeros(1, dtype=float)
    )
    abort_acknowledged = output.non_tensor_batch.get(
        "trajectory_abort_acknowledged_count", np.zeros(1, dtype=int)
    )
    abort_physical = output.non_tensor_batch.get(
        "trajectory_abort_physical_request_count", np.zeros(1, dtype=int)
    )
    abort_errors = output.non_tensor_batch.get(
        "trajectory_abort_error_count", np.zeros(1, dtype=int)
    )
    row = {
        "source_task_index": task_index,
        "sample_index": sample_index,
        "input": prompt,
        "output": solution,
        "num_turns": int(turns[0]),
        "response_tokens": response_tokens,
        "trajectory_timeout": bool(timeouts[0]),
        "trajectory_timeout_seconds": float(timeout_seconds[0]),
        "trajectory_abort_acknowledged_count": int(abort_acknowledged[0]),
        "trajectory_abort_physical_request_count": int(abort_physical[0]),
        "trajectory_abort_error_count": int(abort_errors[0]),
        "runtime_error": False,
    }
    row.update(trajectory_telemetry_row(output, 0))
    return row


def run_rolling_pending_shards(
    *,
    config,
    args: argparse.Namespace,
    tokenizer,
    dataset: RLHFDataset,
    agent_manager: AgentLoopManager,
    pending: list[tuple[int, int, Path, int]],
    completed_tasks: int,
    completed_rows: int,
    window_trajectories: int,
) -> tuple[int, int]:
    """Keep the vLLM admission window full while retaining atomic shard files."""

    buffers: dict[tuple[int, int, Path], dict] = {}

    def units():
        for start, stop, result_path, _ in pending:
            batch = build_batch(config, args, tokenizer, dataset, start, stop)
            expected = (stop - start) * args.samples_per_task
            if len(batch) != expected:
                raise ValueError(f"expected {expected} rolling rows, got {len(batch)}")
            key = (start, stop, result_path)
            buffers[key] = {
                "expected": expected,
                "rows": {},
                "started": time.monotonic(),
            }
            for offset in range(expected):
                task_index = start + offset // args.samples_per_task
                sample_index = offset % args.samples_per_task
                unit = batch[offset : offset + 1]
                priority = task_index * args.samples_per_task + sample_index
                unit.non_tensor_batch["priority"] = np.array([priority], dtype=np.int64)
                yield key, task_index, sample_index, unit

    iterator = iter(units())
    workers = agent_manager.agent_loop_workers
    if not workers:
        raise RuntimeError("rolling admission requires at least one agent loop worker")
    inflight: dict[object, tuple[tuple[int, int, Path], int, int]] = {}
    worker_cursor = 0
    exhausted = False

    def refill() -> None:
        nonlocal worker_cursor, exhausted
        while not exhausted and len(inflight) < window_trajectories:
            try:
                key, task_index, sample_index, unit = next(iterator)
            except StopIteration:
                exhausted = True
                break
            worker = workers[worker_cursor % len(workers)]
            worker_cursor += 1
            # Stamp at the actual logical scheduler admission time.  Reusing
            # the run-start clock for later refills inflates queue-wait
            # telemetry by all prior shard wall time.
            stamp_trajectory_enqueue(unit)
            reference = worker.generate_sequences.remote(unit)
            inflight[reference] = (key, task_index, sample_index)

    refill()
    while inflight:
        ready, _ = ray.wait(list(inflight), num_returns=1)
        reference = ready[0]
        key, task_index, sample_index = inflight.pop(reference)
        output = ray.get(reference)
        state = buffers[key]
        identity = (task_index, sample_index)
        if identity in state["rows"]:
            raise ValueError(f"duplicate rolling trajectory identity {identity}")
        state["rows"][identity] = decode_single_trajectory(
            output,
            tokenizer,
            task_index=task_index,
            sample_index=sample_index,
        )
        if len(state["rows"]) == state["expected"]:
            start, stop, result_path = key
            ordered = [
                state["rows"][(task, sample)]
                for task in range(start, stop)
                for sample in range(args.samples_per_task)
            ]
            written = write_jsonl_atomic(result_path, ordered)
            if written != state["expected"]:
                raise ValueError(
                    f"expected to write {state['expected']} rolling rows, wrote {written}"
                )
            completed_tasks += stop - start
            completed_rows += written
            write_progress(args, completed_tasks, completed_rows)
            print(
                json.dumps(
                    {
                        "event": "rolling_shard_complete",
                        "start": start,
                        "stop": stop,
                        "rows": written,
                        "wall_seconds": time.monotonic() - state["started"],
                        "completed_tasks": completed_tasks,
                        "rolling_window_trajectories": window_trajectories,
                    }
                ),
                flush=True,
            )
            del buffers[key]
        refill()
    if buffers:
        raise RuntimeError("rolling scheduler exhausted with incomplete shard buffers")
    return completed_tasks, completed_rows


def run(args: argparse.Namespace) -> dict:
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    runtime_preflight = validate_dataset_runtime_environments(
        args.dataset,
        args.tool_sandbox_root,
    )
    config = build_config(args)
    contract = safe_contract(config, args)
    contract["runtime_environment_preflight"] = runtime_preflight
    contract["trajectory_admission"] = trajectory_admission_contract(
        task_batch_size=args.task_batch_size,
        samples_per_task=args.samples_per_task,
        max_num_seqs_per_dp_engine=int(config.actor_rollout_ref.rollout.max_num_seqs),
        data_parallel_size=int(config.actor_rollout_ref.rollout.data_parallel_size),
    )
    contract["rolling_admission"] = rolling_admission_contract(
        enabled=bool(args.rolling_admission),
        requested_window_trajectories=args.rolling_window_trajectories,
        aggregate_sequence_capacity=int(
            contract["trajectory_admission"]["aggregate_sequence_capacity"]
        ),
        max_window_multiplier=args.rolling_window_max_multiplier,
    )
    if not contract["dataset_exists"] or not contract["model_identity"]["valid"]:
        raise FileNotFoundError(contract)
    if args.max_prompt_tokens + args.max_response_tokens != args.max_context_tokens:
        raise ValueError("prompt + response token budgets must equal max context")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "standalone_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    dataset = build_dataset(config, args, tokenizer)
    ranges = shard_ranges(args.expected_tasks, args.task_batch_size)
    completed = []
    pending = []
    for start, stop in ranges:
        path = shard_path(args.output_dir, start, stop)
        rows = completed_shard_rows(
            path,
            start=start,
            stop=stop,
            samples_per_task=args.samples_per_task,
        )
        (completed if rows else pending).append((start, stop, path, rows))
    completed_tasks = sum(stop - start for start, stop, _, _ in completed)
    completed_rows = sum(rows for _, _, _, rows in completed)
    write_progress(args, completed_tasks, completed_rows)
    if args.preflight_only:
        return {
            **contract,
            "preflight_rows": args.expected_tasks * args.samples_per_task,
            "pending_shards": len(pending),
        }
    if not pending:
        return {
            **contract,
            "started_at": started_at.isoformat(),
            "wall_seconds": time.monotonic() - started_monotonic,
            "rows": completed_rows,
            "completed_tasks": completed_tasks,
            "resumed_without_model_load": True,
        }

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    try:
        server_manager = LLMServerManager.create(config=config)
        agent_manager = AgentLoopManager.create(config=config, llm_client=server_manager.get_client())
        enqueued_epoch_ns = time.time_ns()
        if args.rolling_admission:
            completed_tasks, completed_rows = run_rolling_pending_shards(
                config=config,
                args=args,
                tokenizer=tokenizer,
                dataset=dataset,
                agent_manager=agent_manager,
                pending=pending,
                completed_tasks=completed_tasks,
                completed_rows=completed_rows,
                window_trajectories=int(
                    contract["rolling_admission"]["effective_window_trajectories"]
                ),
            )
            pending = []
        for start, stop, result_path, _ in pending:
            shard_started = time.monotonic()
            batch = build_batch(config, args, tokenizer, dataset, start, stop)
            expected = (stop - start) * args.samples_per_task
            generated_rows = padded_rows_for_equal_chunks(expected, args.agent_workers)
            padding_rows = generated_rows - expected
            if padding_rows:
                # AgentLoopManager splits a DataProto evenly across workers.  A
                # remainder shard can therefore need temporary duplicate rows;
                # they are generated only to satisfy that runtime invariant and
                # are removed before task/sample identities are assigned or any
                # shard is persisted.
                batch.padding(padding_rows, padding_candidate="last")
            stamp_trajectory_enqueue(batch, epoch_ns=enqueued_epoch_ns)
            output = agent_manager.generate_sequences(batch)
            if len(output) != generated_rows:
                raise ValueError(
                    f"expected {generated_rows} generated rows, got {len(output)}"
                )
            if padding_rows:
                output = output[:expected]
            if len(output) != expected:
                raise ValueError(f"expected {expected} persisted rows, got {len(output)}")

            prompt_texts = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in output.batch["prompts"]
            ]
            output_texts = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in output.batch["responses"]
            ]
            response_lengths = (
                output.batch["responses"].ne(tokenizer.pad_token_id).sum(dim=-1).cpu().tolist()
            )
            turns = output.non_tensor_batch.get(
                "__num_turns__",
                output.non_tensor_batch.get("num_turns", np.zeros(len(output), dtype=int)),
            )
            timeouts = output.non_tensor_batch.get(
                "trajectory_timeout", np.zeros(len(output), dtype=bool)
            )
            timeout_seconds = output.non_tensor_batch.get(
                "trajectory_timeout_seconds", np.zeros(len(output), dtype=float)
            )
            abort_acknowledged = output.non_tensor_batch.get(
                "trajectory_abort_acknowledged_count", np.zeros(len(output), dtype=int)
            )
            abort_physical = output.non_tensor_batch.get(
                "trajectory_abort_physical_request_count", np.zeros(len(output), dtype=int)
            )
            abort_errors = output.non_tensor_batch.get(
                "trajectory_abort_error_count", np.zeros(len(output), dtype=int)
            )
            task_indices = np.repeat(np.arange(start, stop), args.samples_per_task)
            sample_indices = np.tile(np.arange(args.samples_per_task), stop - start)
            rows = []
            for offset in range(expected):
                row = {
                    "source_task_index": int(task_indices[offset]),
                    "sample_index": int(sample_indices[offset]),
                    "input": prompt_texts[offset],
                    "output": output_texts[offset],
                    "num_turns": int(turns[offset]),
                    "response_tokens": int(response_lengths[offset]),
                    "trajectory_timeout": bool(timeouts[offset]),
                    "trajectory_timeout_seconds": float(timeout_seconds[offset]),
                    "trajectory_abort_acknowledged_count": int(abort_acknowledged[offset]),
                    "trajectory_abort_physical_request_count": int(abort_physical[offset]),
                    "trajectory_abort_error_count": int(abort_errors[offset]),
                    "runtime_error": False,
                }
                row.update(trajectory_telemetry_row(output, offset))
                rows.append(row)
            written = write_jsonl_atomic(result_path, rows)
            if written != expected:
                raise ValueError(f"expected to write {expected} rows, wrote {written}")
            completed_tasks += stop - start
            completed_rows += written
            write_progress(args, completed_tasks, completed_rows)
            print(
                json.dumps(
                    {
                        "event": "shard_complete",
                        "start": start,
                        "stop": stop,
                        "rows": written,
                        "temporary_padding_rows": padding_rows,
                        "wall_seconds": time.monotonic() - shard_started,
                        "completed_tasks": completed_tasks,
                    }
                ),
                flush=True,
            )
    finally:
        # Standalone parity runs must not strand placement groups or hold NPUs
        # after an exception during model startup or agent generation.
        ray.shutdown()
    return {
        **contract,
        "started_at": started_at.isoformat(),
        "wall_seconds": time.monotonic() - started_monotonic,
        "rows": completed_rows,
        "completed_tasks": completed_tasks,
        "shards": len(ranges),
        "output_dir": str(args.output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/llin-verl-grpo"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", default="step120")
    parser.add_argument("--policy-step", type=int, default=120)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tool-sandbox-root", type=Path, default=Path("/pi_sandbox"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ray-address", default="192.168.202.5:26379")
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--task-batch-size", type=int, default=10)
    parser.add_argument("--max-num-seqs", type=int, default=24)
    parser.add_argument("--agent-workers", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-response-tokens", type=int, default=45056)
    parser.add_argument("--max-context-tokens", type=int, default=49152)
    parser.add_argument("--trajectory-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--rolling-admission", type=int, choices=(0, 1), default=0)
    parser.add_argument("--rolling-window-trajectories", type=int, default=0)
    parser.add_argument("--rolling-window-max-multiplier", type=float, default=1.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
