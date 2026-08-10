#!/usr/bin/env python3
"""Summarize the three frozen Step-120 oracle-ladder boss evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.compare_boss_exact_evaluations import index_unique, read_jsonl, summarize


def analyze(paths: dict[str, Path]) -> dict[str, object]:
    indexed = {arm: index_unique(read_jsonl(path), arm) for arm, path in paths.items()}
    task_sets = {arm: set(rows) for arm, rows in indexed.items()}
    if len({frozenset(values) for values in task_sets.values()}) != 1:
        raise ValueError("oracle arms do not contain identical task ids")
    summaries = {arm: summarize(list(rows.values())) for arm, rows in indexed.items()}
    control_correct = int(summaries["control"]["correct_numeric_count"])
    contract_correct = int(summaries["contract"]["correct_numeric_count"])
    oracle_correct = int(summaries["oracle"]["correct_numeric_count"])
    contract_gain = contract_correct - control_correct
    oracle_gain = oracle_correct - contract_correct
    if contract_gain >= 2 and contract_gain >= oracle_gain:
        primary = "task_understanding_or_planning"
    elif oracle_gain >= 2:
        primary = "evidence_acquisition"
    elif oracle_correct < len(next(iter(task_sets.values()))):
        primary = "final_synthesis_or_multiple_bottlenecks"
    else:
        primary = "no_single_dominant_bottleneck"
    return {
        "contract": "step120-oracle-ladder-v1",
        "task_count": len(next(iter(task_sets.values()))),
        "task_ids_identical": True,
        "arms": summaries,
        "numeric_correct_deltas": {
            "contract_minus_control": contract_gain,
            "oracle_minus_contract": oracle_gain,
            "oracle_minus_control": oracle_correct - control_correct,
        },
        "primary_bottleneck": primary,
        "diagnostic_only": True,
        "next_stage": "banded_reward_offline_replay",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("control", "contract", "oracle"):
        parser.add_argument(f"--{arm}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze({arm: getattr(args, arm) for arm in ("control", "contract", "oracle")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
