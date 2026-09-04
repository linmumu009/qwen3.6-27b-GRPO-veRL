#!/usr/bin/env python3
"""Measure tensor- and layer-level changes between two sharded HF checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_group(name: str) -> tuple[str, str]:
    match = LAYER_RE.match(name)
    if match:
        remainder = match.group(2)
        return f"language_layer_{int(match.group(1)):02d}", remainder.split(".", 1)[0]
    if name.startswith("model.visual."):
        return "visual_encoder", name.split(".", 3)[2]
    if name.startswith("mtp."):
        return "mtp", name.split(".", 2)[1]
    if "embed_tokens" in name:
        return "language_embedding", "embedding"
    if name.startswith("lm_head."):
        return "lm_head", "head"
    if name.startswith("model.language_model.norm."):
        return "language_final_norm", "norm"
    return "other", name.split(".", 1)[0]


def summarize_accumulator(value: dict[str, float | int]) -> dict[str, Any]:
    base_sq = float(value["base_sq"])
    candidate_sq = float(value["candidate_sq"])
    delta_sq = float(value["delta_sq"])
    dot = float(value["dot"])
    denominator = math.sqrt(base_sq * candidate_sq)
    return {
        "tensors": int(value["tensors"]),
        "elements": int(value["elements"]),
        "changed_elements": int(value["changed_elements"]),
        "changed_fraction": round(int(value["changed_elements"]) / int(value["elements"]), 9),
        "base_l2": math.sqrt(base_sq),
        "delta_l2": math.sqrt(delta_sq),
        "relative_l2": math.sqrt(delta_sq / base_sq) if base_sq else None,
        "cosine_similarity": dot / denominator if denominator else None,
        "max_abs_delta": float(value["max_abs_delta"]),
    }


def add_stats(accumulator: dict[str, float | int], stats: dict[str, Any]) -> None:
    accumulator["tensors"] += 1
    accumulator["elements"] += stats["elements"]
    accumulator["changed_elements"] += stats["changed_elements"]
    accumulator["base_sq"] += stats["base_sq"]
    accumulator["candidate_sq"] += stats["candidate_sq"]
    accumulator["delta_sq"] += stats["delta_sq"]
    accumulator["dot"] += stats["dot"]
    accumulator["max_abs_delta"] = max(accumulator["max_abs_delta"], stats["max_abs_delta"])


def new_accumulator() -> dict[str, float | int]:
    return {
        "tensors": 0,
        "elements": 0,
        "changed_elements": 0,
        "base_sq": 0.0,
        "candidate_sq": 0.0,
        "delta_sq": 0.0,
        "dot": 0.0,
        "max_abs_delta": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--chunk-elements", type=int, default=8_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    baseline_index_path = args.baseline / "model.safetensors.index.json"
    candidate_index_path = args.candidate / "model.safetensors.index.json"
    baseline_index = json.loads(baseline_index_path.read_text(encoding="utf-8"))
    candidate_index = json.loads(candidate_index_path.read_text(encoding="utf-8"))
    baseline_map = baseline_index["weight_map"]
    candidate_map = candidate_index["weight_map"]
    if set(baseline_map) != set(candidate_map):
        raise ValueError("checkpoint tensor key sets differ")

    shard_pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name in sorted(baseline_map):
        shard_pairs[(baseline_map[name], candidate_map[name])].append(name)

    started = time.perf_counter()
    overall = new_accumulator()
    groups: dict[str, dict[str, float | int]] = defaultdict(new_accumulator)
    modules: dict[str, dict[str, float | int]] = defaultdict(new_accumulator)
    tensor_rows: list[dict[str, Any]] = []
    dtype_pairs: dict[str, int] = defaultdict(int)
    shape_mismatch_count = 0
    with torch.inference_mode():
        for (baseline_shard, candidate_shard), names in sorted(shard_pairs.items()):
            with safe_open(args.baseline / baseline_shard, framework="pt", device="cpu") as left_handle:
                with safe_open(args.candidate / candidate_shard, framework="pt", device="cpu") as right_handle:
                    for name in names:
                        left = left_handle.get_tensor(name)
                        right = right_handle.get_tensor(name)
                        if tuple(left.shape) != tuple(right.shape):
                            shape_mismatch_count += 1
                            continue
                        dtype_pairs[f"{left.dtype}->{right.dtype}"] += 1
                        left_flat = left.reshape(-1)
                        right_flat = right.reshape(-1)
                        stats = {
                            "elements": left.numel(),
                            "changed_elements": 0,
                            "base_sq": 0.0,
                            "candidate_sq": 0.0,
                            "delta_sq": 0.0,
                            "dot": 0.0,
                            "max_abs_delta": 0.0,
                        }
                        for start in range(0, left.numel(), args.chunk_elements):
                            stop = min(start + args.chunk_elements, left.numel())
                            left_chunk_raw = left_flat[start:stop]
                            right_chunk_raw = right_flat[start:stop]
                            stats["changed_elements"] += int(torch.count_nonzero(left_chunk_raw != right_chunk_raw))
                            left_chunk = left_chunk_raw.float()
                            right_chunk = right_chunk_raw.float()
                            delta = right_chunk - left_chunk
                            stats["base_sq"] += float(torch.sum(left_chunk * left_chunk, dtype=torch.float64))
                            stats["candidate_sq"] += float(torch.sum(right_chunk * right_chunk, dtype=torch.float64))
                            stats["delta_sq"] += float(torch.sum(delta * delta, dtype=torch.float64))
                            stats["dot"] += float(torch.sum(left_chunk * right_chunk, dtype=torch.float64))
                            stats["max_abs_delta"] = max(stats["max_abs_delta"], float(torch.max(torch.abs(delta))))
                        group, module = tensor_group(name)
                        add_stats(overall, stats)
                        add_stats(groups[group], stats)
                        add_stats(modules[f"{group}/{module}"], stats)
                        tensor_rows.append(
                            {
                                "name": name,
                                "group": group,
                                "module": module,
                                "elements": stats["elements"],
                                "changed_fraction": stats["changed_elements"] / stats["elements"],
                                "delta_l2": math.sqrt(stats["delta_sq"]),
                                "relative_l2": math.sqrt(stats["delta_sq"] / stats["base_sq"])
                                if stats["base_sq"]
                                else None,
                                "max_abs_delta": stats["max_abs_delta"],
                            }
                        )
                        del left, right, left_flat, right_flat

    result = {
        "schema_version": 1,
        "diagnostic_type": "hf_checkpoint_weight_delta",
        "baseline_model": args.baseline_label,
        "candidate_model": args.candidate_label,
        "baseline_index_sha256": sha256_file(baseline_index_path),
        "candidate_index_sha256": sha256_file(candidate_index_path),
        "tensor_keys": len(baseline_map),
        "shard_pairs": len(shard_pairs),
        "shape_mismatches": shape_mismatch_count,
        "dtype_pairs": dict(sorted(dtype_pairs.items())),
        "overall": summarize_accumulator(overall),
        "by_group": {name: summarize_accumulator(value) for name, value in sorted(groups.items())},
        "by_group_module": {name: summarize_accumulator(value) for name, value in sorted(modules.items())},
        "top_tensors_by_relative_l2": sorted(
            (row for row in tensor_rows if row["relative_l2"] is not None),
            key=lambda row: row["relative_l2"], reverse=True,
        )[:30],
        "top_tensors_by_delta_l2": sorted(tensor_rows, key=lambda row: row["delta_l2"], reverse=True)[:30],
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "raw_weights_included": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.startswith("top_")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
