#!/usr/bin/env python3
"""Fail closed when a veRL actor checkpoint is structurally incomplete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _nonempty_files(directory: Path, pattern: str) -> list[Path]:
    return sorted(path for path in directory.glob(pattern) if path.is_file() and path.stat().st_size > 0)


def verify_hf_checkpoint(base_model_dir: Path, model_dir: Path) -> dict[str, Any]:
    base_index_path = base_model_dir / "model.safetensors.index.json"
    output_index_path = model_dir / "model.safetensors.index.json"
    errors: list[str] = []
    if not base_index_path.is_file():
        errors.append(f"base index missing: {base_index_path}")
    if not output_index_path.is_file():
        errors.append(f"checkpoint index missing: {output_index_path}")
    if errors:
        return {"format": "huggingface", "valid": False, "errors": errors}

    base_map = _read_json(base_index_path).get("weight_map")
    output_map = _read_json(output_index_path).get("weight_map")
    if not isinstance(base_map, dict) or not isinstance(output_map, dict):
        return {
            "format": "huggingface",
            "valid": False,
            "errors": ["base or checkpoint index has no object-valued weight_map"],
        }

    base_keys = set(base_map)
    output_keys = set(output_map)
    referenced_shards = sorted({str(value) for value in output_map.values()})
    missing_shards = [name for name in referenced_shards if not (model_dir / name).is_file()]
    empty_shards = [
        name
        for name in referenced_shards
        if (model_dir / name).is_file() and (model_dir / name).stat().st_size == 0
    ]
    missing_tensors = sorted(base_keys - output_keys)
    extra_tensors = sorted(output_keys - base_keys)
    errors.extend(f"referenced shard missing: {name}" for name in missing_shards)
    errors.extend(f"referenced shard empty: {name}" for name in empty_shards)
    if missing_tensors:
        errors.append(f"checkpoint is missing {len(missing_tensors)} base-model tensors")
    if extra_tensors:
        errors.append(f"checkpoint contains {len(extra_tensors)} unexpected tensors")
    return {
        "format": "huggingface",
        "valid": not errors,
        "errors": errors,
        "base_tensor_count": len(base_keys),
        "checkpoint_tensor_count": len(output_keys),
        "referenced_shard_count": len(referenced_shards),
        "missing_shards": missing_shards,
        "empty_shards": empty_shards,
        "missing_tensor_count": len(missing_tensors),
        "extra_tensor_count": len(extra_tensors),
        "missing_tensor_examples": missing_tensors[:20],
        "extra_tensor_examples": extra_tensors[:20],
    }


def verify_megatron_dist_checkpoint(model_dir: Path) -> dict[str, Any]:
    metadata_candidates = [model_dir / ".metadata", model_dir / "metadata.json"]
    metadata_files = [path for path in metadata_candidates if path.is_file() and path.stat().st_size > 0]
    shard_files = _nonempty_files(model_dir, "*.distcp")
    errors: list[str] = []
    if not metadata_files:
        errors.append("Megatron dist checkpoint metadata is missing or empty")
    if not shard_files:
        errors.append("Megatron dist checkpoint has no non-empty .distcp shards")
    return {
        "format": "megatron_dist_checkpoint",
        "valid": not errors,
        "errors": errors,
        "metadata_files": [path.name for path in metadata_files],
        "shard_count": len(shard_files),
        "total_shard_bytes": sum(path.stat().st_size for path in shard_files),
    }


def verify_checkpoint(checkpoint_dir: Path, base_model_dir: Path | None = None) -> dict[str, Any]:
    manifest_path = checkpoint_dir / "actor" / "ckpt_contents.json"
    if not manifest_path.is_file():
        return {
            "valid": False,
            "checkpoint_dir": str(checkpoint_dir),
            "errors": [f"checkpoint manifest missing: {manifest_path}"],
        }
    manifest = _read_json(manifest_path)
    model_entry = ((manifest.get("contents") or {}).get("model") or {})
    model_format = str(model_entry.get("format") or "")
    relative_path = str(model_entry.get("path") or "")
    model_dir = checkpoint_dir / "actor" / relative_path
    if model_format == "huggingface":
        if base_model_dir is None:
            result = {
                "format": model_format,
                "valid": False,
                "errors": ["--base-model-dir is required for HuggingFace checkpoint validation"],
            }
        else:
            result = verify_hf_checkpoint(base_model_dir, model_dir)
    elif model_format == "megatron_dist_checkpoint":
        result = verify_megatron_dist_checkpoint(model_dir)
    else:
        result = {
            "format": model_format or None,
            "valid": False,
            "errors": [f"unsupported or missing checkpoint model format: {model_format!r}"],
        }
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "global_step": manifest.get("global_step"),
        "manifest_model_path": relative_path,
        **result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_checkpoint(args.checkpoint_dir, args.base_model_dir)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
