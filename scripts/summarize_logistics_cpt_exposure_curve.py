#!/usr/bin/env python3
"""Validate and summarize a fixed CPT exposure curve."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from scripts.summarize_logistics_cpt_run import FATAL_PATTERNS, parse_metrics
from scripts.verify_checkpoint_integrity import verify_checkpoint


def summarize_curve(
    run_dir: Path,
    *,
    steps_per_exposure: int,
    total_exposures: int,
    sequence_tokens_per_exposure: int,
    checkpoint_exposures: tuple[int, ...],
    experiment: str = "single_book_cpt_exposure_curve_2x_4x",
) -> dict[str, object]:
    if steps_per_exposure < 1 or total_exposures < 1 or sequence_tokens_per_exposure < 1:
        raise ValueError("step, exposure, and token counts must be positive")
    if not checkpoint_exposures or any(value < 1 or value > total_exposures for value in checkpoint_exposures):
        raise ValueError("checkpoint exposures must fall inside the exposure curve")

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
        raise ValueError(
            f"fatal log signatures found: {fatal_matches}, rank_tracebacks={fatal_rank_tracebacks}"
        )

    total_steps = steps_per_exposure * total_exposures
    metrics = parse_metrics(all_stdout)
    if sorted(metrics) != list(range(1, total_steps + 1)):
        raise ValueError("training log does not contain one unique metric row for every expected step")
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
        missing = required - set(values)
        if missing or not all(math.isfinite(values[key]) for key in required):
            raise ValueError(f"step {step} has missing or non-finite metrics: {sorted(missing)}")

    exposure_rows: list[dict[str, object]] = []
    for exposure in range(1, total_exposures + 1):
        first_step = (exposure - 1) * steps_per_exposure + 1
        last_step = exposure * steps_per_exposure
        window = [metrics[step] for step in range(first_step, last_step + 1)]
        tokens = int(sum(row["train/global_tokens"] for row in window))
        if tokens != sequence_tokens_per_exposure:
            raise ValueError(
                f"exposure {exposure} token mismatch: expected={sequence_tokens_per_exposure} actual={tokens}"
            )
        losses = [row["train/loss"] for row in window]
        exposure_rows.append(
            {
                "exposure": exposure,
                "first_step": first_step,
                "last_step": last_step,
                "sequence_tokens_with_eos": tokens,
                "loss": {
                    "first": losses[0],
                    "last": losses[-1],
                    "minimum": min(losses),
                    "maximum": max(losses),
                    "mean": statistics.fmean(losses),
                },
                "learning_rate": {"first": window[0]["train/lr"], "last": window[-1]["train/lr"]},
                "maximum_grad_norm_before_clip": max(row["train/grad_norm"] for row in window),
            }
        )

    checkpoints: list[dict[str, object]] = []
    for exposure in checkpoint_exposures:
        step = exposure * steps_per_exposure
        result = verify_checkpoint(run_dir / "checkpoints" / f"global_step_{step}")
        if not result["valid"]:
            raise ValueError(f"checkpoint at exposure {exposure} failed integrity: {result['errors']}")
        if result.get("global_step") != step or result.get("save_contents") != ["model", "extra"]:
            raise ValueError(f"checkpoint at exposure {exposure} has an unexpected manifest")
        if result.get("optimizer_declared"):
            raise ValueError(f"checkpoint at exposure {exposure} must not include optimizer state")
        checkpoints.append(
            {
                "exposure": exposure,
                "global_step": step,
                "format": result["format"],
                "shard_count": result["shard_count"],
                "total_shard_bytes": result["total_shard_bytes"],
                "save_contents": result["save_contents"],
                "optimizer_declared": result["optimizer_declared"],
                "valid": True,
            }
        )

    return {
        "schema_version": 1,
        "run_name": run_dir.name,
        "experiment": experiment,
        "status": "complete",
        "source_content_included": False,
        "promotion_allowed": False,
        "total_exposures": total_exposures,
        "steps_per_exposure": steps_per_exposure,
        "total_steps": total_steps,
        "sequence_tokens_with_eos_per_exposure": sequence_tokens_per_exposure,
        "total_sequence_tokens_with_eos": sequence_tokens_per_exposure * total_exposures,
        "exposures": exposure_rows,
        "checkpoints": checkpoints,
        "peak_memory_gb": {
            "npu_allocated": max(row["perf/max_memory_allocated_gb"] for row in metrics.values()),
            "npu_reserved": max(row["perf/max_memory_reserved_gb"] for row in metrics.values()),
            "cpu_used": max(row["perf/cpu_memory_used_gb"] for row in metrics.values()),
        },
        "fatal_log_signatures": 0,
        "nonfatal_compatibility_messages": {
            "mixed_fused_layer_norm_safe_import": combined.count(
                "apex.normalization.fused_layer_norm' has no attribute 'MixedFusedLayerNorm'"
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps-per-exposure", type=int, default=29)
    parser.add_argument("--total-exposures", type=int, default=4)
    parser.add_argument("--sequence-tokens-per-exposure", type=int, default=336702)
    parser.add_argument("--checkpoint-exposure", type=int, action="append", default=[])
    parser.add_argument("--experiment", default="single_book_cpt_exposure_curve_2x_4x")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = tuple(args.checkpoint_exposure or [2, 4])
    result = summarize_curve(
        args.run_dir.resolve(),
        steps_per_exposure=args.steps_per_exposure,
        total_exposures=args.total_exposures,
        sequence_tokens_per_exposure=args.sequence_tokens_per_exposure,
        checkpoint_exposures=checkpoints,
        experiment=args.experiment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
