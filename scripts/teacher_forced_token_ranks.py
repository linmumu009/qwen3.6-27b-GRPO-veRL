#!/usr/bin/env python3
"""Exact teacher-token ranks for vocab-parallel Megatron forward diagnostics."""

from __future__ import annotations

from typing import Any, Callable

import torch


def resolve_model_vocab_size(hf_config) -> int:
    """Resolve vocabulary size from flat or multimodal Hugging Face configs."""

    candidates = (
        hf_config,
        getattr(hf_config, "text_config", None),
        getattr(hf_config, "language_config", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        value = (
            candidate.get("vocab_size")
            if isinstance(candidate, dict)
            else getattr(candidate, "vocab_size", None)
        )
        if value is not None and int(value) > 0:
            return int(value)
    raise ValueError(
        f"cannot resolve vocab_size from Hugging Face config {type(hf_config).__name__}"
    )


def ranks_from_full_logits(
    logits: torch.Tensor, labels: torch.Tensor, *, vocab_size: int | None = None
) -> torch.Tensor:
    """Reference rank implementation used by tests and single-shard checks."""

    if logits.shape[:-1] != labels.shape:
        raise ValueError("logit and label shapes differ")
    effective_vocab = int(vocab_size or logits.shape[-1])
    if not 0 < effective_vocab <= logits.shape[-1]:
        raise ValueError("invalid effective vocab size")
    if torch.any(labels < 0) or torch.any(labels >= effective_vocab):
        raise ValueError("label is outside the effective vocabulary")
    target_logits = torch.gather(logits[..., :effective_vocab], -1, labels.unsqueeze(-1)).squeeze(-1)
    return 1 + (logits[..., :effective_vocab] > target_logits.unsqueeze(-1)).sum(dim=-1)


def vocab_parallel_target_ranks(
    student_logits: torch.Tensor,
    data: Any,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Return exact 1-based target ranks without gathering the full TP vocabulary."""

    from megatron.core.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from megatron.core.tensor_parallel.utils import VocabUtility
    from verl.models.mcore.util import preprocess_bshd_engine, preprocess_thd_engine
    from verl.utils import tensordict_utils as tu

    input_ids = data["input_ids"]
    if data_format == "thd":
        labels, *_ = preprocess_thd_engine(input_ids, pre_process=True, need_roll=True)
    elif data_format == "bshd":
        labels, *_ = preprocess_bshd_engine(
            input_ids,
            pre_process=True,
            need_roll=True,
            forced_max_seqlen=tu.get_non_tensor_data(
                data=data, key="forced_max_seqlen", default=None
            ),
        )
    else:
        raise ValueError(f"unsupported data format: {data_format}")
    if student_logits.shape[:2] != labels.shape[:2]:
        raise ValueError(
            f"student logits and aligned labels differ: {student_logits.shape[:2]} != {labels.shape[:2]}"
        )

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    shard_size = student_logits.size(-1)
    vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(
        shard_size, tp_rank, tp_size
    )
    vocab_size = int(tu.get_non_tensor_data(data=data, key="model_vocab_size"))
    if torch.any(labels < 0) or torch.any(labels >= vocab_size):
        raise ValueError("aligned label is outside the model vocabulary")

    in_shard = (labels >= vocab_start) & (labels < vocab_end)
    local_labels = (labels - vocab_start).clamp(min=0, max=shard_size - 1)
    local_target = torch.gather(student_logits.float(), -1, local_labels.unsqueeze(-1)).squeeze(-1)
    local_target = torch.where(in_shard, local_target, torch.full_like(local_target, float("-inf")))
    torch.distributed.all_reduce(
        local_target,
        op=torch.distributed.ReduceOp.MAX,
        group=get_tensor_model_parallel_group(),
    )

    global_ids = torch.arange(vocab_start, vocab_end, device=student_logits.device)
    valid_vocab = global_ids < vocab_size
    local_higher = (
        (student_logits.float() > local_target.unsqueeze(-1)) & valid_vocab.view(1, 1, -1)
    ).sum(dim=-1, dtype=torch.int64)
    torch.distributed.all_reduce(
        local_higher,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group(),
    )
    return {"teacher_token_rank": (local_higher + 1).detach()}


def sql_token_rank_metrics(model_output: dict[str, Any], data: Any) -> dict[str, torch.Tensor]:
    """Produce one scalar metric set for a single-example diagnostic micro-batch."""

    if len(data) != 1:
        raise ValueError("SQL token-rank diagnostics require micro_batch_size_per_gpu=1")
    ranks = model_output["teacher_token_rank"].values().to(torch.int64)
    sql_mask = torch.roll(data["sql_shell_mask"].values(), shifts=-1, dims=0).bool()
    target_ids = torch.roll(data["input_ids"].values(), shifts=-1, dims=0).to(torch.int64)
    log_probs = model_output["log_probs"].values()
    sql_ranks = ranks[sql_mask]
    sql_target_ids = target_ids[sql_mask]
    sql_log_probs = log_probs[sql_mask]
    if sql_ranks.numel() <= 0:
        raise ValueError("SQL token-rank mask is empty")

    nongreedy = torch.nonzero(sql_ranks > 1, as_tuple=False).flatten()
    if nongreedy.numel():
        first = nongreedy[0]
        first_offset = first.to(torch.float64)
        first_rank = sql_ranks[first].to(torch.float64)
        first_target_id = sql_target_ids[first].to(torch.float64)
        first_probability = sql_log_probs[first].exp().to(torch.float64)
    else:
        sentinel = torch.tensor(-1.0, device=sql_ranks.device, dtype=torch.float64)
        first_offset = sentinel
        first_rank = sentinel
        first_target_id = sentinel
        first_probability = sentinel

    return {
        "sql_rank/token_count": torch.tensor(
            float(sql_ranks.numel()), device=sql_ranks.device, dtype=torch.float64
        ),
        "sql_rank/rank_sum": sql_ranks.sum(dtype=torch.float64),
        "sql_rank/greedy_count": (sql_ranks == 1).sum(dtype=torch.float64),
        "sql_rank/top5_count": (sql_ranks <= 5).sum(dtype=torch.float64),
        "sql_rank/max_rank": sql_ranks.max().to(torch.float64),
        "sql_rank/first_nongreedy_offset": first_offset,
        "sql_rank/first_nongreedy_rank": first_rank,
        "sql_rank/first_nongreedy_target_id": first_target_id,
        "sql_rank/first_nongreedy_target_probability": first_probability,
    }


def summarize_sql_token_ranks(
    metrics: dict[str, Any],
    task_ids: list[str],
    *,
    numbers: Callable[[Any], list[float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize scalar micro-batch rank metrics without importing the veRL runner."""

    rank_keys = (
        "token_count",
        "rank_sum",
        "greedy_count",
        "top5_count",
        "max_rank",
        "first_nongreedy_offset",
        "first_nongreedy_rank",
        "first_nongreedy_target_id",
        "first_nongreedy_target_probability",
    )
    values = {key: numbers(metrics[f"sql_rank/{key}"]) for key in rank_keys}
    if any(len(series) != len(task_ids) for series in values.values()):
        raise ValueError("SQL token-rank metrics do not align with task IDs")

    per_task: list[dict[str, Any]] = []
    for index, task_id in enumerate(task_ids):
        token_count = int(round(values["token_count"][index]))
        if token_count <= 0:
            raise ValueError(f"{task_id}: SQL token count is not positive")
        first_offset = int(round(values["first_nongreedy_offset"][index]))
        per_task.append(
            {
                "task_id": task_id,
                "token_count": token_count,
                "greedy_token_count": int(round(values["greedy_count"][index])),
                "top5_token_count": int(round(values["top5_count"][index])),
                "mean_rank": values["rank_sum"][index] / token_count,
                "max_rank": int(round(values["max_rank"][index])),
                "all_tokens_greedy": first_offset < 0,
                "first_nongreedy_offset": None if first_offset < 0 else first_offset,
                "first_nongreedy_rank": None
                if first_offset < 0
                else int(round(values["first_nongreedy_rank"][index])),
                "first_nongreedy_target_id": None
                if first_offset < 0
                else int(round(values["first_nongreedy_target_id"][index])),
                "first_nongreedy_target_probability": None
                if first_offset < 0
                else values["first_nongreedy_target_probability"][index],
            }
        )

    total_tokens = int(round(sum(values["token_count"])))
    aggregate = {
        "token_count": total_tokens,
        "greedy_token_count": int(round(sum(values["greedy_count"]))),
        "top5_token_count": int(round(sum(values["top5_count"]))),
        "greedy_token_rate": sum(values["greedy_count"]) / total_tokens,
        "top5_token_rate": sum(values["top5_count"]) / total_tokens,
        "mean_rank": sum(values["rank_sum"]) / total_tokens,
        "max_rank": int(round(max(values["max_rank"]))),
        "tasks_all_sql_tokens_greedy": sum(row["all_tokens_greedy"] for row in per_task),
        "tasks_with_nongreedy_sql_token": sum(not row["all_tokens_greedy"] for row in per_task),
    }
    return aggregate, per_task
