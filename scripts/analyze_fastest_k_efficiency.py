#!/usr/bin/env python3
"""Summarize Fastest-K fully-async timing and rollout-quality evidence.

The analyzer is read-only. It consumes the durable runtime markers emitted by
the LLIN observability patches and optional rollout JSONL files, then prints a
machine-readable JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_FLOAT = r"[-+0-9.eE]+"
_PREWARM_RE = re.compile(
    rf"\[LLIN_PREWARM\] groups=(\d+) queued_tokens=(\d+) wait_s=({_FLOAT})"
)
_STAGE_RE = re.compile(
    rf"\[LLIN_TRAIN_STAGE\] step=(\d+) "
    rf"queue_wait_s=({_FLOAT}) deserialize_s=({_FLOAT}) assemble_s=({_FLOAT}) "
    rf"reward_s=({_FLOAT}) old_log_prob_s=({_FLOAT}) ref_log_prob_s=({_FLOAT}) "
    rf"adv_s=({_FLOAT}) update_actor_s=({_FLOAT}) step_s=({_FLOAT})"
)
_FASTEST_RE = re.compile(
    rf"\[LLIN_FASTEST_K\] candidates=(\d+) selected=(\d+) discarded=(\d+) "
    rf"completed_discarded=(\d+) physical_aborts=(\d+) quorum_s=({_FLOAT})"
)
_PARAM_SYNC_RE = re.compile(rf"timing_s/timing_s/param_sync:(?:np\.float64\()?({_FLOAT})")
_STALE_DROP_RE = re.compile(rf"fully_async/count/dropped_stale_samples:({_FLOAT})")


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(clean),
        "min": min(clean) if clean else None,
        "mean": mean(clean) if clean else None,
        "median": median(clean) if clean else None,
        "p90": quantile(clean, 0.90),
        "p95": quantile(clean, 0.95),
        "max": max(clean) if clean else None,
    }


def parse_driver_text(text: str) -> dict[str, Any]:
    text = _ANSI_RE.sub("", text)
    prewarm_match = _PREWARM_RE.search(text)
    stages = []
    for match in _STAGE_RE.finditer(text):
        values = [float(value) for value in match.groups()[1:]]
        stages.append(
            dict(
                zip(
                    (
                        "step",
                        "queue_wait_s",
                        "deserialize_s",
                        "assemble_s",
                        "reward_s",
                        "old_log_prob_s",
                        "ref_log_prob_s",
                        "adv_s",
                        "update_actor_s",
                        "step_s",
                    ),
                    [int(match.group(1)), *values],
                    strict=True,
                )
            )
        )

    fastest = [
        {
            "candidates": int(match.group(1)),
            "selected": int(match.group(2)),
            "discarded": int(match.group(3)),
            "completed_discarded": int(match.group(4)),
            "physical_aborts": int(match.group(5)),
            "quorum_s": float(match.group(6)),
        }
        for match in _FASTEST_RE.finditer(text)
    ]
    sync_values = [float(value) for value in _PARAM_SYNC_RE.findall(text)]
    stale_values = [float(value) for value in _STALE_DROP_RE.findall(text)]
    steady = [item for item in stages if item["step"] >= 2]

    return {
        "prewarm": (
            {
                "groups": int(prewarm_match.group(1)),
                "queued_tokens": int(prewarm_match.group(2)),
                "wait_s": float(prewarm_match.group(3)),
            }
            if prewarm_match
            else None
        ),
        "steps": stages,
        "step_count": len(stages),
        "steady_state": {
            "definition": "steps >= 2; step 1 excluded as cold actor compilation",
            "steps": len(steady),
            "queue_wait_s": describe(item["queue_wait_s"] for item in steady),
            "actor_update_s": describe(item["update_actor_s"] for item in steady),
            "step_s": describe(item["step_s"] for item in steady),
            "trainer_idle_ratio": (
                sum(item["queue_wait_s"] for item in steady)
                / sum(item["step_s"] for item in steady)
                if steady and sum(item["step_s"] for item in steady)
                else None
            ),
        },
        "fastest_k": {
            "groups_observed": len(fastest),
            "quorum_s": describe(item["quorum_s"] for item in fastest),
            "candidates": sorted({item["candidates"] for item in fastest}),
            "selected": sorted({item["selected"] for item in fastest}),
            "discarded_total": sum(item["discarded"] for item in fastest),
            "completed_discarded_total": sum(item["completed_discarded"] for item in fastest),
            "physical_aborts_total": sum(item["physical_aborts"] for item in fastest),
        },
        "param_sync_s": describe(sync_values),
        "dropped_stale_samples_final": stale_values[-1] if stale_values else None,
    }


def iter_rollout_rows(directory: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def summarize_rollouts(directory: Path) -> dict[str, Any]:
    rows = list(iter_rollout_rows(directory))
    strict_scores = []
    for row in rows:
        tool_used = bool(float(row.get("tool_used", 0.0)))
        required_table_used = bool(float(row.get("required_table_used", 0.0)))
        final_answer_correct = bool(float(row.get("final_answer_correct", 0.0)))
        if tool_used and required_table_used and final_answer_correct:
            strict_scores.append(1.0)
        elif tool_used and required_table_used:
            strict_scores.append(0.2)
        elif tool_used:
            strict_scores.append(0.05)
        else:
            strict_scores.append(0.0)
    return {
        "rows": len(rows),
        "files": len(list(directory.glob("*.jsonl"))),
        "score": describe(float(row.get("score", 0.0)) for row in rows),
        "historical_answer_correct": sum(float(row.get("answer_correct", 0.0)) for row in rows),
        "evidence_contains_expected": sum(
            float(row.get("evidence_contains_expected", 0.0)) for row in rows
        ),
        "strict_final_answer_correct": sum(
            float(row.get("final_answer_correct", 0.0)) for row in rows
        ),
        "strict_full_reward_count": sum(score == 1.0 for score in strict_scores),
        "strict_reward_replay": describe(strict_scores),
        "required_table_used": sum(float(row.get("required_table_used", 0.0)) for row in rows),
        "tool_used": sum(float(row.get("tool_used", 0.0)) for row in rows),
        "output_chars": describe(len(str(row.get("output") or "")) for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "driver": parse_driver_text(args.driver_log.read_text(encoding="utf-8", errors="replace")),
        "rollouts": summarize_rollouts(args.rollout_dir) if args.rollout_dir else None,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
