#!/usr/bin/env python3
"""Run the veRL PI agent loop without creating an actor/trainer worker.

The standalone server manager loads the frozen Step120 HF export directly on
the rollout node.  No optimizer, actor, checkpoint sync, or training step is
created.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from transformers import AutoTokenizer

from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.workers.rollout.llm_server import LLMServerManager


def build_config(args: argparse.Namespace):
    config_dir = "/verl/verl/experimental/one_step_off_policy/config"
    overrides = [
        f"data.train_files={args.dataset}",
        f"data.val_files={args.dataset}",
        "data.max_prompt_length=4096",
        "data.max_response_length=45056",
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
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.80",
        "actor_rollout_ref.rollout.max_num_batched_tokens=16384",
        "actor_rollout_ref.rollout.max_model_len=49152",
        "actor_rollout_ref.rollout.max_num_seqs=24",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.enable_prefix_caching=True",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.calculate_log_probs=False",
        "actor_rollout_ref.rollout.disable_log_stats=False",
        "actor_rollout_ref.rollout.n=8",
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
        "actor_rollout_ref.rollout.agent.num_workers=16",
        "actor_rollout_ref.rollout.agent.default_agent_loop=pi_agent",
        "actor_rollout_ref.rollout.val_kwargs.n=8",
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
    return {
        "contract": "verl-standalone-runtime-parity-arm-v1",
        "model_manifest_exists": (args.model / "llin_export_manifest.json").is_file(),
        "dataset_exists": args.dataset.is_file(),
        "tasks": 10,
        "samples_per_task": int(rollout.val_kwargs.n),
        "temperature": float(rollout.val_kwargs.temperature),
        "top_p": float(rollout.val_kwargs.top_p),
        "top_k": int(rollout.val_kwargs.top_k),
        "tensor_parallel_size": int(rollout.tensor_model_parallel_size),
        "data_parallel_size": int(rollout.data_parallel_size),
        "rollout_npus": int(rollout.n_gpus_per_node),
        "agent_workers": int(rollout.agent.num_workers),
        "context_tokens": int(rollout.max_model_len),
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


def build_batch(config, args: argparse.Namespace, tokenizer) -> DataProto:
    dataset = RLHFDataset(
        data_files=str(args.dataset),
        tokenizer=tokenizer,
        config=config.data,
        processor=None,
    )
    if len(dataset) != 10:
        raise ValueError(f"expected 10 tasks, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=collate_fn)
    batch = DataProto.from_single_dict(next(iter(loader)))
    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    batch = batch.repeat(repeat_times=8, interleave=True)
    batch.meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": True,
        "validate": True,
        "global_steps": 120,
    }
    return batch


def run(args: argparse.Namespace) -> dict:
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    config = build_config(args)
    contract = safe_contract(config, args)
    if not contract["dataset_exists"] or (
        not args.preflight_only and not contract["model_manifest_exists"]
    ):
        raise FileNotFoundError(contract)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "standalone_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    batch = build_batch(config, args, tokenizer)
    if args.preflight_only:
        return {**contract, "preflight_rows": len(batch)}

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    try:
        server_manager = LLMServerManager.create(config=config)
        agent_manager = AgentLoopManager.create(config=config, llm_client=server_manager.get_client())
        output = agent_manager.generate_sequences(batch)
        if len(output) != 80:
            raise ValueError(f"expected 80 output rows, got {len(output)}")

        prompt_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output.batch["prompts"]]
        output_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output.batch["responses"]]
        turns = output.non_tensor_batch.get("num_turns", np.zeros(len(output), dtype=int))
        result_path = args.output_dir / "0.jsonl"
        with result_path.open("w", encoding="utf-8", newline="\n") as handle:
            for prompt, solution, num_turns in zip(prompt_texts, output_texts, turns, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "input": prompt,
                            "output": solution,
                            "num_turns": int(num_turns),
                            "runtime_error": False,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.chmod(result_path, 0o600)
    finally:
        # Standalone parity runs must not strand placement groups or hold NPUs
        # after an exception during model startup or agent generation.
        ray.shutdown()
    return {
        **contract,
        "started_at": started_at.isoformat(),
        "wall_seconds": time.monotonic() - started_monotonic,
        "rows": len(output),
        "output": str(result_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/llin-verl-grpo"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ray-address", default="192.168.202.5:26379")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
