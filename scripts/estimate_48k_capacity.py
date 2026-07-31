#!/usr/bin/env python3
"""Estimate 48K-context HBM capacity from the validated two-node run.

The estimate separates directly calculated tensors from an explicit planning
allowance. It is a sizing gate, not a substitute for an 8K -> 16K -> 32K ->
48K allocation probe on the actual Ascend image.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

GIB = 1024**3


@dataclass(frozen=True)
class CapacityInputs:
    usable_hbm_gib: float = 61.27
    target_context_tokens: int = 49_152
    observed_context_tokens: int = 4_509
    observed_actor_peak_gib: float = 29.63
    hidden_size: int = 5_120
    vocab_size: int = 248_320
    num_layers: int = 64
    full_attention_layers: int = 16
    gdn_layers: int = 48
    num_kv_heads: int = 4
    head_dim: int = 256
    gdn_key_dim: int = 128
    gdn_value_dim: int = 128
    gdn_value_heads: int = 48
    gdn_page_copies: int = 2
    train_tp: int = 4
    train_pp: int = 2
    train_cp: int = 2
    rollout_tp: int = 8
    rollout_max_seqs_per_replica: int = 16
    bytes_per_element: int = 2
    training_workspace_allowance_gib: float = 10.0
    rollout_runtime_allowance_gib: float = 12.0
    rollout_gpu_memory_utilization: float = 0.60
    model_parameters: float = 27e9


def gib(value: float) -> float:
    return value / GIB


def checkpoint_inputs_gib(inputs: CapacityInputs, tokens: int) -> float:
    """Full-recompute checkpoint inputs, sharded by TP/PP/CP."""
    elements = (
        tokens
        / inputs.train_cp
        * inputs.hidden_size
        * (inputs.num_layers / inputs.train_pp)
        / inputs.train_tp
    )
    return gib(elements * inputs.bytes_per_element)


def vocab_parallel_logits_gib(inputs: CapacityInputs, tokens: int) -> float:
    """One vocab-parallel logits shard for the local CP rank."""
    elements = (
        tokens
        / inputs.train_cp
        * (inputs.vocab_size / inputs.train_tp)
    )
    return gib(elements * inputs.bytes_per_element)


def rollout_cache_per_sequence_gib(inputs: CapacityInputs) -> dict[str, float]:
    # When KV heads < TP size, vLLM replicates one KV head on each TP rank.
    kv_heads_per_rank = max(1, inputs.num_kv_heads // inputs.rollout_tp)
    attention_bytes = (
        inputs.target_context_tokens
        * inputs.full_attention_layers
        * 2  # K and V
        * kv_heads_per_rank
        * inputs.head_dim
        * inputs.bytes_per_element
    )
    # vLLM align-mode keeps two recurrent GDN state pages per active sequence.
    gdn_bytes = (
        inputs.gdn_layers
        * inputs.gdn_page_copies
        * inputs.gdn_value_heads
        * inputs.gdn_key_dim
        * inputs.gdn_value_dim
        * inputs.bytes_per_element
    )
    return {
        "full_attention_gib": gib(attention_bytes),
        "gdn_recurrent_state_gib": gib(gdn_bytes),
        "total_gib": gib(attention_bytes + gdn_bytes),
    }


def estimate(inputs: CapacityInputs) -> dict[str, object]:
    observed_direct = checkpoint_inputs_gib(inputs, inputs.observed_context_tokens) + (
        vocab_parallel_logits_gib(inputs, inputs.observed_context_tokens)
    )
    target_direct = checkpoint_inputs_gib(inputs, inputs.target_context_tokens) + (
        vocab_parallel_logits_gib(inputs, inputs.target_context_tokens)
    )
    direct_increment = target_direct - observed_direct
    training_planning_peak = (
        inputs.observed_actor_peak_gib
        + direct_increment
        + inputs.training_workspace_allowance_gib
    )

    per_sequence = rollout_cache_per_sequence_gib(inputs)
    rollout_cache_all_sequences = (
        per_sequence["total_gib"] * inputs.rollout_max_seqs_per_replica
    )
    rollout_budget = (
        inputs.usable_hbm_gib * inputs.rollout_gpu_memory_utilization
    )
    model_weight_shard = gib(
        inputs.model_parameters
        * inputs.bytes_per_element
        / inputs.rollout_tp
    )
    rollout_planning_total = (
        model_weight_shard
        + inputs.rollout_runtime_allowance_gib
        + rollout_cache_all_sequences
    )

    return {
        "inputs": asdict(inputs),
        "training": {
            "observed_peak_gib": inputs.observed_actor_peak_gib,
            "directly_accounted_increment_gib": direct_increment,
            "planning_workspace_allowance_gib": inputs.training_workspace_allowance_gib,
            "planning_peak_gib": training_planning_peak,
            "headroom_gib": inputs.usable_hbm_gib - training_planning_peak,
            "expected_to_fit": training_planning_peak < inputs.usable_hbm_gib,
        },
        "rollout_per_tp_rank": {
            "cache_per_48k_sequence": per_sequence,
            "cache_for_max_active_sequences_gib": rollout_cache_all_sequences,
            "model_weight_shard_gib": model_weight_shard,
            "runtime_allowance_gib": inputs.rollout_runtime_allowance_gib,
            "planning_total_gib": rollout_planning_total,
            "vllm_60pct_budget_gib": rollout_budget,
            "budget_headroom_gib": rollout_budget - rollout_planning_total,
            "expected_to_fit": rollout_planning_total < rollout_budget,
        },
        "verdict": {
            "expected_to_fit": (
                training_planning_peak < inputs.usable_hbm_gib
                and rollout_planning_total < rollout_budget
            ),
            "requires_staircase_probe": True,
            "probe_sequence_tokens": [8192, 16384, 32768, 49152],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-tokens", type=int, default=49_152)
    args = parser.parse_args()
    inputs = CapacityInputs(target_context_tokens=args.context_tokens)
    print(json.dumps(estimate(inputs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
