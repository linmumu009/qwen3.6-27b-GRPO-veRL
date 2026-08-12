#!/usr/bin/env python3
"""Run a forward-only, component-level SFT diagnostic on one Megatron checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import hydra
from tensordict.tensorclass import NonTensorData
import torch
from omegaconf import DictConfig, OmegaConf

from verl.trainer.sft_trainer import SFTTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.device import auto_set_device
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.workers.utils.losses import sft_loss

from scripts.teacher_forced_token_ranks import (
    resolve_model_vocab_size,
    sql_token_rank_metrics,
    summarize_sql_token_ranks,
    vocab_parallel_target_ranks,
)


COMPONENT_MASKS = {
    "assistant": "loss_mask",
    "tool_turn": "tool_turn_mask",
    "tool_structure": "tool_structure_mask",
    "sql_shell": "sql_shell_mask",
    "final_answer": "final_answer_mask",
}


def component_sft_loss(
    model_output=None,
    data=None,
    dp_group=None,
    config=None,
    student_logits=None,
    data_format=None,
):
    """Return the official SFT loss plus additive per-component statistics."""

    if student_logits is not None:
        return vocab_parallel_target_ranks(student_logits, data, data_format)
    if model_output is None:
        raise ValueError("component diagnostic received neither logits nor model output")

    loss, _ = sft_loss(config=config, model_output=model_output, data=data, dp_group=dp_group)
    log_prob = model_output["log_probs"].values()
    metrics: dict[str, torch.Tensor] = {}
    for component, mask_key in COMPONENT_MASKS.items():
        mask = torch.roll(data[mask_key].values(), shifts=-1, dims=0).to(log_prob.dtype)
        token_count = mask.sum()
        if token_count.item() <= 0:
            raise ValueError(f"empty shifted component mask: {component}")
        metrics[f"component/{component}/nll_sum"] = (-(log_prob * mask).sum()).detach()
        metrics[f"component/{component}/target_probability_sum"] = (
            (log_prob.exp() * mask).sum()
        ).detach()
        metrics[f"component/{component}/token_count"] = token_count.detach()
    metrics.update(sql_token_rank_metrics(model_output, data))
    return loss, metrics


def _numbers(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        output: list[float] = []
        for item in value:
            output.extend(_numbers(item))
        return output
    if isinstance(value, (int, float)):
        return [float(value)]
    raise TypeError(f"unsupported metric value: {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summarize(
    metrics: dict[str, Any], task_ids: list[str]
) -> tuple[dict, dict, list[dict]]:
    values: dict[str, dict[str, list[float]]] = {}
    for component in COMPONENT_MASKS:
        values[component] = {}
        for statistic in ("nll_sum", "target_probability_sum", "token_count"):
            key = f"component/{component}/{statistic}"
            series = _numbers(metrics[key])
            if len(series) != len(task_ids):
                raise ValueError(f"{key} has {len(series)} values for {len(task_ids)} tasks")
            values[component][statistic] = series

    aggregates: dict[str, dict[str, float | int]] = {}
    for component, statistics in values.items():
        tokens = sum(statistics["token_count"])
        nll = sum(statistics["nll_sum"]) / tokens
        aggregates[component] = {
            "token_count": int(round(tokens)),
            "mean_nll": nll,
            "perplexity": math.exp(min(nll, 50.0)),
            "arithmetic_mean_target_probability": sum(statistics["target_probability_sum"]) / tokens,
            "geometric_mean_target_probability": math.exp(-nll),
        }

    sql_token_rank, per_task_rank = summarize_sql_token_ranks(
        metrics, task_ids, numbers=_numbers
    )

    per_task: list[dict] = []
    for index, task_id in enumerate(task_ids):
        components: dict[str, dict[str, float | int]] = {}
        for component, statistics in values.items():
            tokens = statistics["token_count"][index]
            nll = statistics["nll_sum"][index] / tokens
            components[component] = {
                "token_count": int(round(tokens)),
                "mean_nll": nll,
                "geometric_mean_target_probability": math.exp(-nll),
                "arithmetic_mean_target_probability": statistics["target_probability_sum"][index] / tokens,
            }
        per_task.append(
            {"task_id": task_id, "components": components, "sql_token_rank": per_task_rank[index]}
        )
    return aggregates, sql_token_rank, per_task


def run(config: DictConfig) -> None:
    initialize_global_process_group()
    started = time.monotonic()
    try:
        trainer = SFTTrainer(config=config)
        if not trainer.engine_config.forward_only:
            raise ValueError("teacher-forced diagnostic requires engine.forward_only=true")
        trainer.training_client.set_loss_fn(component_sft_loss)
        if trainer.val_dataloader is None:
            raise ValueError("teacher-forced diagnostic requires data.val_files")

        meta_info = {
            "use_remove_padding": config.model.use_remove_padding,
            "use_dynamic_bsz": config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "distillation_use_topk": True,
            "distillation_only": False,
            "model_vocab_size": resolve_model_vocab_size(trainer.model_config.hf_config),
            "global_batch_size": trainer.global_batch_size,
            "pad_mode": config.data.pad_mode,
            "pad_token_id": trainer.model_config.tokenizer.pad_token_id,
        }
        outputs: list[dict[str, Any]] = []
        for val_data in trainer.val_dataloader:
            batch_seqlens = trainer._get_batch_seqlens(val_data)
            val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
            tu.assign_non_tensor(val_data, global_token_num=NonTensorData(batch_seqlens))
            output = trainer.training_client.infer_batch(val_data)
            if trainer.engine.is_mp_src_rank_with_outputs():
                outputs.append(tu.get(output, "metrics"))

        is_writer = trainer.engine.is_mp_src_rank_with_outputs() and trainer.engine.get_data_parallel_rank() == 0
        if is_writer:
            if len(outputs) != 1:
                raise ValueError(f"expected one validation batch, got {len(outputs)}")
            metrics = outputs[0]
            task_ids = trainer.val_dataset.dataframe["task_id"].astype(str).tolist()
            aggregates, sql_token_rank, per_task = _summarize(metrics, task_ids)
            for row in per_task:
                token_id = row["sql_token_rank"]["first_nongreedy_target_id"]
                row["sql_token_rank"]["first_nongreedy_target_token"] = (
                    None
                    if token_id is None
                    else trainer.model_config.tokenizer.convert_ids_to_tokens(token_id)
                )
            data_path = Path(str(config.data.val_files[0] if isinstance(config.data.val_files, list) else config.data.val_files))
            output_path = Path(str(config.diagnostic.output_path))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "contract": "repair-sft-teacher-forced-component-diagnostic-v2",
                "model_label": str(config.diagnostic.model_label),
                "source_model_dist_checkpoint": str(config.engine.dist_checkpointing_path),
                "forward_only": True,
                "optimizer_initialized": False,
                "task_count": len(task_ids),
                "task_ids": task_ids,
                "data_sha256": _sha256(data_path),
                "official_assistant_loss": _numbers(metrics["loss"])[0],
                "components": aggregates,
                "sql_token_rank": sql_token_rank,
                "per_task": per_task,
                "runtime_seconds": time.monotonic() - started,
                "peak_memory": {
                    key: _numbers(value)[0]
                    for key, value in metrics.items()
                    if key.startswith("perf/")
                },
                "config": {
                    "topology": "tp4_pp2_cp2",
                    "max_length": int(config.data.max_length),
                    "micro_batch_size_per_gpu": int(config.data.micro_batch_size_per_gpu),
                    "use_dynamic_bsz": bool(config.data.use_dynamic_bsz),
                },
            }
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"LLIN_TEACHER_FORCED_RESULT={output_path}")
        torch.distributed.barrier()
    finally:
        destroy_global_process_group()


@hydra.main(config_path="pkg://verl.trainer.config", config_name="sft_trainer_engine", version_base=None)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    OmegaConf.resolve(config)
    run(config)


if __name__ == "__main__":
    main()
