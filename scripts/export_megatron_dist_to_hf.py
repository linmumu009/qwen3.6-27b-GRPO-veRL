#!/usr/bin/env python3
"""Export a complete veRL Megatron distributed actor checkpoint to HF.

The export is deliberately performed as a single CPU/Gloo rank. Megatron
distributed checkpoints are topology independent, so this reconstructs TP=1,
PP=1, CP=1 before asking the model-specific Megatron Bridge to emit HF keys.
That avoids the PP=2 online-export failure where only the first pipeline stage
was written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any
import uuid


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def resolve_model_checkpoint(actor_checkpoint: Path) -> Path:
    manifest_path = actor_checkpoint / "ckpt_contents.json"
    manifest = _read_json(manifest_path)
    model_entry = ((manifest.get("contents") or {}).get("model") or {})
    model_format = str(model_entry.get("format") or "")
    if model_format != "megatron_dist_checkpoint":
        raise ValueError(f"expected megatron_dist_checkpoint, got {model_format!r}")
    relative_path = Path(str(model_entry.get("path") or ""))
    if not relative_path.parts or relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe or empty model checkpoint path: {relative_path}")
    model_checkpoint = actor_checkpoint / relative_path
    if not (model_checkpoint / "metadata.json").is_file() and not (model_checkpoint / ".metadata").is_file():
        raise FileNotFoundError(f"Megatron checkpoint metadata missing: {model_checkpoint}")
    if not any(model_checkpoint.glob("*.distcp")):
        raise FileNotFoundError(f"Megatron checkpoint shards missing: {model_checkpoint}")
    return model_checkpoint


def validate_export_paths(actor_checkpoint: Path, base_model: Path, output_dir: Path) -> Path:
    actor_checkpoint = actor_checkpoint.resolve()
    base_model = base_model.resolve()
    output_dir = output_dir.resolve()
    if not actor_checkpoint.is_dir():
        raise FileNotFoundError(f"actor checkpoint missing: {actor_checkpoint}")
    if not (base_model / "config.json").is_file():
        raise FileNotFoundError(f"base config missing: {base_model / 'config.json'}")
    if not (base_model / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"base weight index missing: {base_model}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if output_dir == actor_checkpoint or actor_checkpoint in output_dir.parents:
        raise ValueError("output must not be inside the resume-capable actor checkpoint")
    return resolve_model_checkpoint(actor_checkpoint)


def _safetensor_inventory(model_dir: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    from safetensors import safe_open

    index = _read_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"weight_map missing from {model_dir}")
    inventory: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard_name in sorted({str(value) for value in weight_map.values()}):
        shard_path = model_dir / shard_name
        if not shard_path.is_file() or shard_path.stat().st_size == 0:
            raise FileNotFoundError(f"referenced shard missing or empty: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_slice(key)
                inventory[key] = (tuple(tensor.get_shape()), str(tensor.get_dtype()))
    return inventory


def verify_exact_hf_export(base_model: Path, output_dir: Path) -> dict[str, Any]:
    base_inventory = _safetensor_inventory(base_model)
    output_inventory = _safetensor_inventory(output_dir)
    missing = sorted(set(base_inventory) - set(output_inventory))
    extra = sorted(set(output_inventory) - set(base_inventory))
    shape_mismatched = sorted(
        key
        for key in set(base_inventory) & set(output_inventory)
        if base_inventory[key][0] != output_inventory[key][0]
    )
    dtype_different = sorted(
        key
        for key in set(base_inventory) & set(output_inventory)
        if base_inventory[key][1] != output_inventory[key][1]
    )
    output_index = _read_json(output_dir / "model.safetensors.index.json")
    weight_map = output_index["weight_map"]
    referenced_shards = sorted({str(value) for value in weight_map.values()})
    layer_ids = sorted(
        {
            int(parts[3])
            for key in output_inventory
            for parts in [key.split(".")]
            if len(parts) > 4 and parts[:3] == ["model", "language_model", "layers"] and parts[3].isdigit()
        }
    )
    errors: list[str] = []
    if missing:
        errors.append(f"missing {len(missing)} tensors")
    if extra:
        errors.append(f"unexpected {len(extra)} tensors")
    if shape_mismatched:
        errors.append(f"shape mismatch for {len(shape_mismatched)} tensors")
    if layer_ids != list(range(64)):
        errors.append(f"language layer coverage is not 0..63: {layer_ids[:4]}...{layer_ids[-4:]}")
    required_fragments = ("A_log", "conv1d", "in_proj_qkv", "linear_attn.norm")
    missing_families = [frag for frag in required_fragments if not any(frag in key for key in output_inventory)]
    if missing_families:
        errors.append(f"Qwen3.6 GDN tensor families missing: {missing_families}")
    return {
        "valid": not errors,
        "errors": errors,
        "base_tensor_count": len(base_inventory),
        "output_tensor_count": len(output_inventory),
        "referenced_shard_count": len(referenced_shards),
        "total_safetensor_bytes": sum((output_dir / shard).stat().st_size for shard in referenced_shards),
        "language_layer_ids": layer_ids,
        "missing_tensor_count": len(missing),
        "extra_tensor_count": len(extra),
        "shape_mismatch_count": len(shape_mismatched),
        "dtype_difference_count": len(dtype_different),
        "missing_tensor_examples": missing[:20],
        "extra_tensor_examples": extra[:20],
        "shape_mismatch_examples": shape_mismatched[:20],
        "dtype_difference_examples": dtype_different[:20],
        "dtype_difference_details": [
            {
                "key": key,
                "base": {"shape": base_inventory[key][0], "dtype": base_inventory[key][1]},
                "output": {"shape": output_inventory[key][0], "dtype": output_inventory[key][1]},
            }
            for key in dtype_different[:20]
        ],
    }


def _base_fallback_keys(base_model: Path, model_checkpoint: Path) -> list[str]:
    metadata_text = (model_checkpoint / "metadata.json").read_text(encoding="utf-8", errors="ignore")
    if "language_model.mtp" in metadata_text:
        raise RuntimeError("checkpoint contains MTP weights; base-model MTP fallback would be unsafe")
    weight_map = _read_json(base_model / "model.safetensors.index.json").get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("base model weight_map is missing")
    keys = sorted(key for key in weight_map if key.startswith("mtp."))
    if len(keys) != 15:
        raise RuntimeError(f"expected 15 frozen base MTP tensors, got {len(keys)}")
    return keys


def _iter_base_tensors(base_model: Path, keys: list[str]):
    from safetensors import safe_open

    weight_map = _read_json(base_model / "model.safetensors.index.json")["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(str(weight_map[key]), []).append(key)
    for shard_name, shard_keys in sorted(by_shard.items()):
        with safe_open(base_model / shard_name, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                yield key, handle.get_tensor(key)


def _save_hf_with_frozen_base_fallback(
    bridge: Any,
    model: list[Any],
    base_model: Path,
    model_checkpoint: Path,
    staging: Path,
) -> list[str]:
    fallback_keys = _base_fallback_keys(base_model, model_checkpoint)
    additional_files = getattr(bridge._model_bridge, "ADDITIONAL_FILE_PATTERNS", None)
    bridge.hf_pretrained.save_artifacts(
        staging,
        original_source_path=base_model,
        additional_files=additional_files,
    )

    def combined_generator():
        yield from bridge.export_hf_weights(model, cpu=True, show_progress=True)
        yield from _iter_base_tensors(base_model, fallback_keys)

    source = bridge.hf_pretrained.state.source
    if not hasattr(source, "save_generator"):
        raise TypeError("HuggingFace state source does not support streaming safetensors save")
    source.save_generator(
        combined_generator(),
        staging,
        strict=True,
        distributed_save=False,
        save_every_n_ranks=1,
    )
    return fallback_keys


def export_checkpoint(actor_checkpoint: Path, base_model: Path, output_dir: Path) -> dict[str, Any]:
    model_checkpoint = validate_export_paths(actor_checkpoint, base_model, output_dir)
    staging = output_dir.with_name(f".{output_dir.name}.incomplete-{uuid.uuid4().hex[:10]}")
    staging.mkdir(parents=True, exist_ok=False)

    import torch
    import mindspeed.megatron_adaptor  # noqa: F401 - installs Ascend Megatron patches
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.model_load_save import temporary_distributed_context
    from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing

    try:
        bridge = AutoBridge.from_hf_pretrained(str(base_model), trust_remote_code=True)
        with temporary_distributed_context(backend="gloo"):
            provider = bridge.to_megatron_provider(load_weights=False)
            provider.perform_initialization = False
            provider.bf16 = True
            provider.fp16 = False
            provider.params_dtype = torch.bfloat16
            provider.tensor_model_parallel_size = 1
            provider.pipeline_model_parallel_size = 1
            provider.context_parallel_size = 1
            provider.sequence_parallel = False
            provider.virtual_pipeline_model_parallel_size = None
            provider.finalize()
            model = provider.provide_distributed_model(
                wrap_with_ddp=False,
                use_cpu_initialization=True,
                mixed_precision_wrapper=None,
            )
            if len(model) != 1:
                raise RuntimeError(f"expected one TP1/PP1 model chunk, got {len(model)}")
            skeleton = {"model": model[0].sharded_state_dict()}
            loaded = load_dist_checkpointing(skeleton, str(model_checkpoint))
            unexpected_roots = set(loaded) - {"model", "content_metadata"}
            if "model" not in loaded or unexpected_roots:
                raise RuntimeError(f"unexpected checkpoint roots: {sorted(loaded)}")
            incompatible = model[0].load_state_dict(loaded["model"], strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"strict model load failed: {incompatible}")
            fallback_keys = _save_hf_with_frozen_base_fallback(
                bridge, model, base_model, model_checkpoint, staging
            )

        verification = verify_exact_hf_export(base_model, staging)
        if not verification["valid"]:
            raise RuntimeError(f"HF verification failed: {verification['errors']}")
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "actor_checkpoint": str(actor_checkpoint.resolve()),
            "megatron_model_checkpoint": str(model_checkpoint.resolve()),
            "base_model": str(base_model.resolve()),
            "conversion": "Megatron Bridge Qwen3.6 TP1/PP1/CP1 CPU strict export",
            "frozen_base_fallback_keys": fallback_keys,
            "resume_checkpoint_modified": False,
            "verification": verification,
        }
        (staging / "llin_export_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        details = traceback.format_exc()
        failure_path = staging / "EXPORT_FAILED.txt"
        try:
            failure_path.write_text(
                "Export did not pass all gates. This directory is incomplete and must not be evaluated.\n\n"
                + details,
                encoding="utf-8",
            )
        except OSError:
            pass
        print(details, file=sys.stderr, flush=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        result = verify_exact_hf_export(args.base_model.resolve(), args.output_dir.resolve())
    else:
        result = export_checkpoint(args.actor_checkpoint, args.base_model, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("valid", result.get("verification", {}).get("valid", False)) else 1


if __name__ == "__main__":
    sys.exit(main())
