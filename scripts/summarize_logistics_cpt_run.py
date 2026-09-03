#!/usr/bin/env python3
"""Validate and summarize a logistics CPT veRL run without source text."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from scripts.verify_checkpoint_integrity import verify_checkpoint


STEP_RE = re.compile(r"^step:(?P<step>\d+) - (?P<metrics>.+)$", re.MULTILINE)
FATAL_PATTERNS = (
    "Error executing job with overrides",
    "ChildFailedError",
    "OutOfMemoryError",
)


def parse_metrics(text: str) -> dict[int, dict[str, float]]:
    parsed: dict[int, dict[str, float]] = {}
    for match in STEP_RE.finditer(text):
        step = int(match.group("step"))
        metrics: dict[str, float] = {}
        for segment in match.group("metrics").split(" - "):
            key, raw_value = segment.split(":", 1)
            metrics[key] = float(raw_value)
        if step in parsed and parsed[step] != metrics:
            raise ValueError(f"conflicting duplicate metrics for step {step}")
        parsed[step] = metrics
    return parsed


def summarize(run_dir: Path, expected_steps: int, expected_tokens: int) -> dict[str, object]:
    stdout_paths = sorted(run_dir.glob("torchrun_logs/*/attempt_0/*/stdout.log"))
    stderr_paths = sorted(run_dir.glob("torchrun_logs/*/attempt_0/*/stderr.log"))
    if not stdout_paths:
        raise ValueError("no torchrun stdout logs found")

    all_stdout = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in stdout_paths)
    all_stderr = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in stderr_paths)
    combined = all_stdout + "\n" + all_stderr
    fatal_matches = [pattern for pattern in FATAL_PATTERNS if pattern in combined]
    fatal_rank_tracebacks = len(re.findall(r"\[rank\d+\]: Traceback", combined))
    if fatal_matches or fatal_rank_tracebacks:
        raise ValueError(f"fatal log signatures found: {fatal_matches}, rank_tracebacks={fatal_rank_tracebacks}")

    metrics = parse_metrics(all_stdout)
    expected_sequence = list(range(1, expected_steps + 1))
    if sorted(metrics) != expected_sequence:
        raise ValueError(f"step sequence mismatch: expected={expected_sequence} actual={sorted(metrics)}")

    required = {
        "perf/max_memory_allocated_gb",
        "perf/max_memory_reserved_gb",
        "perf/cpu_memory_used_gb",
        "train/loss",
        "train/grad_norm",
        "train/lr",
        "train/global_tokens",
        "train/total_tokens(B)",
    }
    for step, values in metrics.items():
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"step {step} missing metrics: {missing}")
        if not all(math.isfinite(values[key]) for key in required):
            raise ValueError(f"step {step} contains non-finite metrics")

    total_tokens = int(sum(metrics[step]["train/global_tokens"] for step in expected_sequence))
    if total_tokens != expected_tokens:
        raise ValueError(f"token total mismatch: expected={expected_tokens} actual={total_tokens}")

    checkpoint_dir = run_dir / "checkpoints" / f"global_step_{expected_steps}"
    checkpoint = verify_checkpoint(checkpoint_dir)
    if not checkpoint["valid"]:
        raise ValueError(f"checkpoint integrity failed: {checkpoint['errors']}")
    if checkpoint.get("save_contents") != ["model", "extra"]:
        raise ValueError(f"unexpected save_contents: {checkpoint.get('save_contents')}")
    if checkpoint.get("optimizer_declared"):
        raise ValueError("optimizer must not be present in the pilot checkpoint")

    losses = [metrics[step]["train/loss"] for step in expected_sequence]
    grad_norms = [metrics[step]["train/grad_norm"] for step in expected_sequence]
    result = {
        "schema_version": 1,
        "run_name": run_dir.name,
        "status": "complete",
        "exit_code": 0,
        "steps": expected_steps,
        "sequence_tokens_with_eos": total_tokens,
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "minimum": min(losses),
            "maximum": max(losses),
            "mean": sum(losses) / len(losses),
        },
        "grad_norm_before_clip": {
            "last": grad_norms[-1],
            "maximum": max(grad_norms),
        },
        "clip_grad": 1.0,
        "learning_rate": {
            "first": metrics[1]["train/lr"],
            "last": metrics[expected_steps]["train/lr"],
        },
        "peak_memory_gb": {
            "npu_allocated": max(values["perf/max_memory_allocated_gb"] for values in metrics.values()),
            "npu_reserved": max(values["perf/max_memory_reserved_gb"] for values in metrics.values()),
            "cpu_used": max(values["perf/cpu_memory_used_gb"] for values in metrics.values()),
        },
        "checkpoint": {
            "global_step": checkpoint["global_step"],
            "layout": checkpoint["checkpoint_layout"],
            "format": checkpoint["format"],
            "shard_count": checkpoint["shard_count"],
            "total_shard_bytes": checkpoint["total_shard_bytes"],
            "save_contents": checkpoint["save_contents"],
            "optimizer_declared": checkpoint["optimizer_declared"],
            "promotion_allowed": False,
        },
        "nonfatal_compatibility_messages": {
            "mstx_range_end": combined.count("mstx.range_end() missing 1 required positional argument"),
            "mixed_fused_layer_norm_safe_import": combined.count(
                "apex.normalization.fused_layer_norm' has no attribute 'MixedFusedLayerNorm'"
            ),
        },
        "fatal_log_signatures": 0,
        "source_content_included": False,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--expected-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(args.run_dir.resolve(), args.expected_steps, args.expected_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
