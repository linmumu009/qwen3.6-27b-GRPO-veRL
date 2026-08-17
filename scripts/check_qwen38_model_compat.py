#!/usr/bin/env python3
"""Fail-closed compatibility gate for replacing Qwen3.6 with Qwen3.8.

The gate compares architecture-critical config fields and the complete HF
safetensors key set, then asks the installed Transformers runtime to load the
candidate config and tokenizer from local files.  It never loads tensor data.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


TEXT_FIELDS = (
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "max_position_embeddings",
    "full_attention_interval",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "mtp_num_hidden_layers",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def architecture_signature(model_dir: Path) -> dict[str, Any]:
    config = read_json(model_dir / "config.json")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("config.json has no text_config object")
    layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        raise ValueError("text_config.layer_types is missing or empty")
    layer_payload = json.dumps(layer_types, separators=(",", ":")).encode()
    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "language_model_only": config.get("language_model_only"),
        "text": {field: text_config.get(field) for field in TEXT_FIELDS},
        "layer_type_count": len(layer_types),
        "layer_types_sha256": hashlib.sha256(layer_payload).hexdigest(),
        "full_attention_layers": sum(value == "full_attention" for value in layer_types),
        "linear_attention_layers": sum(value == "linear_attention" for value in layer_types),
    }


def weight_keys(model_dir: Path) -> tuple[set[str], int | None]:
    index = read_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index has no weight_map")
    metadata = index.get("metadata")
    total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
    return set(weight_map), int(total_size) if total_size is not None else None


def validate_static_compatibility(reference: Path, candidate: Path) -> dict[str, Any]:
    reference_signature = architecture_signature(reference)
    candidate_signature = architecture_signature(candidate)
    if candidate_signature["architectures"] != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("candidate is not Qwen3_5ForConditionalGeneration")
    if candidate_signature != reference_signature:
        raise ValueError("architecture-critical Qwen3.8 config differs from Qwen3.6")

    reference_keys, reference_size = weight_keys(reference)
    candidate_keys, candidate_size = weight_keys(candidate)
    missing = reference_keys - candidate_keys
    extra = candidate_keys - reference_keys
    if missing or extra:
        raise ValueError(
            f"HF tensor key set differs: missing={len(missing)} extra={len(extra)}"
        )
    return {
        "contract": "llin-qwen38-replacement-compat-v1",
        "architecture": candidate_signature,
        "hf_tensor_keys": len(candidate_keys),
        "reference_weight_bytes": reference_size,
        "candidate_weight_bytes": candidate_size,
        "tensor_key_set_equal": True,
        "runtime_config_loaded": False,
        "runtime_tokenizer_loaded": False,
        "training_checkpoint_reuse_allowed": False,
        "initialization": "candidate_hf_weights",
    }


def validate_runtime(reference: Path, candidate: Path, summary: dict[str, Any]) -> None:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(candidate, local_files_only=True)
    reference_tokenizer = AutoTokenizer.from_pretrained(reference, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=True)
    if getattr(config, "model_type", None) != "qwen3_5":
        raise ValueError("Transformers resolved an unexpected model_type")
    if len(tokenizer) != len(reference_tokenizer):
        raise ValueError("Qwen3.8 and Qwen3.6 runtime tokenizer lengths differ")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Return a concise engineering check."}],
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort="medium",
    )
    if not rendered:
        raise ValueError("Qwen3.8 chat template rendered no tokens")
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids")
    if not isinstance(rendered, list) or not rendered:
        raise ValueError("Qwen3.8 chat template did not return input_ids")
    if isinstance(rendered[0], list):
        rendered = rendered[0]
    if not rendered or not all(isinstance(token_id, int) for token_id in rendered):
        raise ValueError("Qwen3.8 chat template returned invalid token IDs")
    vocab_size = summary["architecture"]["text"]["vocab_size"]
    if max(rendered) >= vocab_size:
        raise ValueError("Qwen3.8 chat template emitted an ID outside model embeddings")
    summary.update(
        {
            "runtime_config_loaded": True,
            "runtime_tokenizer_loaded": True,
            "runtime_tokenizer_tokens": len(tokenizer),
            "medium_template_tokens": len(rendered),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-runtime-load", action="store_true")
    args = parser.parse_args()

    summary = validate_static_compatibility(args.reference_model, args.candidate_model)
    if not args.skip_runtime_load:
        validate_runtime(args.reference_model, args.candidate_model, summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
