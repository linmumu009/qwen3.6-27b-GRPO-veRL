#!/usr/bin/env python3
"""Summarize 20-step GRPO timings, NPU utilization, cache hits and long tails."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import fmean, median


METRIC_RE = re.compile(
    r"(?:^| - )([A-Za-z0-9_./]+):(?:np\.(?:float64|int32|int64)\()?([-+0-9.eE]+)\)?"
)
CACHE_RE = re.compile(r"Prefix cache hit rate:\s*([0-9.]+)%", re.IGNORECASE)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def parse_step_metrics(text: str) -> list[dict[str, float]]:
    steps: list[dict[str, float]] = []
    for line in text.splitlines():
        if "training/global_step:" not in line:
            continue
        metrics = {key: float(value) for key, value in METRIC_RE.findall(line)}
        if "training/global_step" in metrics:
            steps.append(metrics)
    return steps


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "mean": fmean(values) if values else None,
        "median": median(values) if values else None,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def read_npu_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_final_cache_counters(path: Path | None) -> dict[str, float | int | None]:
    """Aggregate the final per-engine vLLM prefix-cache counters."""
    if path is None or not path.exists():
        return {
            "engines": 0,
            "hits": None,
            "queries": None,
            "hit_rate_pct": None,
        }
    final_metrics: dict[str, float] = {}
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                metrics = json.loads(line).get("metrics", {})
            except (AttributeError, json.JSONDecodeError):
                continue
            if metrics:
                final_metrics = metrics
    hit_items = {
        key: value
        for key, value in final_metrics.items()
        if key.startswith("vllm:prefix_cache_hits_total{")
        or key.startswith("vllm:prefix_cache_hits{")
    }
    query_items = {
        key: value
        for key, value in final_metrics.items()
        if key.startswith("vllm:prefix_cache_queries_total{")
        or key.startswith("vllm:prefix_cache_queries{")
    }
    hits = sum(hit_items.values())
    queries = sum(query_items.values())
    engines = {
        match.group(1)
        for key in (*hit_items, *query_items)
        if (match := re.search(r'engine="([^"]+)"', key))
    }
    return {
        "engines": len(engines),
        "hits": hits if hit_items else None,
        "queries": queries if query_items else None,
        "hit_rate_pct": 100.0 * hits / queries if query_items and queries > 0 else None,
    }


def stable_window(m05_rows: list[dict[str, str]]) -> tuple[datetime, datetime] | None:
    stable = [
        datetime.fromisoformat(row["timestamp"])
        for row in m05_rows
        if 2 <= int(row["completed_step"]) < 19
    ]
    return (min(stable), max(stable)) if stable else None


def summarize_npu(
    rows: list[dict[str, str]],
    window: tuple[datetime, datetime] | None,
) -> dict[str, object]:
    if window is not None:
        start, end = window
        rows = [
            row
            for row in rows
            if start <= datetime.fromisoformat(row["timestamp"]) <= end
        ]
    aicore = [float(row["aicore_pct"]) for row in rows]
    npu_util = [float(row["npu_util_pct"]) for row in rows]
    return {
        "records": len(rows),
        "aicore_pct": summarize_values(aicore),
        "npu_util_pct": summarize_values(npu_util),
        "active_record_ratio": (
            sum(value > 0 for value in aicore) / len(aicore) if aicore else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--m05-csv", type=Path, required=True)
    parser.add_argument("--m06-csv", type=Path, required=True)
    parser.add_argument("--cache-jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_text = args.driver_log.read_text(encoding="utf-8", errors="ignore")
    steps = parse_step_metrics(log_text)
    steady_steps = [
        step for step in steps if 2 <= int(step["training/global_step"]) <= 19
    ]
    tail_ratios = [
        step["timing_s/agent_loop/slowest/generate_sequences"]
        / step["timing_s/agent_loop/generate_sequences/mean"]
        for step in steady_steps
    ]
    gen_shares = [
        step["timing_s/gen"] / step["timing_s/step"]
        for step in steady_steps
        if step.get("timing_s/step", 0) > 0
    ]
    m05_rows = read_npu_rows(args.m05_csv)
    m06_rows = read_npu_rows(args.m06_csv)
    window = stable_window(m05_rows)
    cache_rates = [float(value) for value in CACHE_RE.findall(log_text)]
    tail_p90 = percentile(tail_ratios, 0.90)
    gen_share_mean = fmean(gen_shares) if gen_shares else None
    recommend_fully_async = bool(
        tail_p90 is not None
        and gen_share_mean is not None
        and tail_p90 >= 1.75
        and gen_share_mean >= 0.55
    )
    summary = {
        "completed_steps": len(steps),
        "steady_steps": len(steady_steps),
        "steady_timing_s": {
            key: summarize_values([step[key] for step in steady_steps if key in step])
            for key in (
                "timing_s/step",
                "timing_s/gen",
                "timing_s/update_actor",
                "timing_s/sync_rollout_weights",
            )
        },
        "steady_reward_mean": summarize_values(
            [step["critic/score/mean"] for step in steady_steps]
        ),
        "long_tail": {
            "slowest_to_mean_generate_ratio": summarize_values(tail_ratios),
            "generation_share_of_step": summarize_values(gen_shares),
            "recommend_bounded_fully_async": recommend_fully_async,
            "decision_rule": "p90 tail ratio >= 1.75 and mean generation share >= 0.55",
        },
        "prefix_cache": {
            "final_counters": read_final_cache_counters(args.cache_jsonl),
            "driver_reported_hit_rate_pct": summarize_values(cache_rates),
        },
        "npu_steady_window": (
            [window[0].isoformat(), window[1].isoformat()] if window else None
        ),
        "npu": {
            "m05_trainer": summarize_npu(m05_rows, window),
            "m06_rollout": summarize_npu(m06_rows, window),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
