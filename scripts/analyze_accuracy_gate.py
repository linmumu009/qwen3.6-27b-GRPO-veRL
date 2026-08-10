#!/usr/bin/env python3
"""Gate a 2x8 canary with boss-exact validation and within-prompt signal."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.compare_boss_exact_evaluations import read_jsonl, summarize


def rollout_signal(directory: Path, expected_group_size: int) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(directory.glob("*.jsonl"), key=lambda item: int(item.stem)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            digest = hashlib.sha256(str(row.get("input") or "").encode("utf-8")).hexdigest()
            groups[(str(path), digest)].append(row)
    valid = [group for group in groups.values() if len(group) == expected_group_size]
    mixed = [
        group
        for group in valid
        if len({float(row.get("final_answer_correct") or 0) for row in group}) > 1
    ]
    all_wrong = [
        group
        for group in valid
        if not any(float(row.get("final_answer_correct") or 0) for row in group)
    ]
    return {
        "group_count": len(groups),
        "valid_group_count": len(valid),
        "valid_group_rate": len(valid) / len(groups) if groups else 0.0,
        "mixed_correct_group_count": len(mixed),
        "mixed_correct_group_rate": len(mixed) / len(valid) if valid else 0.0,
        "all_wrong_group_count": len(all_wrong),
    }


def analyze(
    rollout_dir: Path,
    boss_reward: Path,
    expected_group_size: int = 8,
) -> dict[str, Any]:
    signal = rollout_signal(rollout_dir, expected_group_size)
    boss = summarize(read_jsonl(boss_reward))
    checks = {
        "boss_val_has_20_tasks": boss["n"] == 20,
        "all_training_groups_have_8_responses": signal["valid_group_rate"] >= 0.99,
        "mixed_correct_rate_at_least_30pct": signal["mixed_correct_group_rate"] >= 0.30,
        "boss_numeric_correct_exceeds_step120": boss["correct_numeric_count"] >= 3,
        "completion_not_below_step120": boss["complete_count"] >= 16,
        "process_score_not_materially_regressed": (
            boss["process_score_mean"] is not None and boss["process_score_mean"] >= 0.80
        ),
    }
    return {
        "contract": "banded-v1-2groups-8responses-accuracy-gate",
        "rollout_signal": signal,
        "boss_exact": boss,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--boss-reward", type=Path, required=True)
    parser.add_argument("--expected-group-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    result = analyze(args.rollout_dir, args.boss_reward, args.expected_group_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.report_only and not result["gate_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
