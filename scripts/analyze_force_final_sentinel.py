#!/usr/bin/env python3
"""Compare the force-final sentinel against the same Step 120 greedy baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from scripts.prepare_force_final_sentinel import DEFAULT_TASK_IDS


KNOWN_INCOMPLETE = frozenset(DEFAULT_TASK_IDS[:4])
COMPLETION_GUARDRAILS = frozenset(DEFAULT_TASK_IDS[4:])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in result:
            raise ValueError(f"{label}: missing or duplicate task_id={task_id!r}")
        result[task_id] = row
    return result


def compact(row: dict[str, Any]) -> dict[str, Any]:
    reward = row.get("reward") or {}
    evidence = row.get("evidence") or {}
    return {
        "reward_total": float(reward.get("reward_total") or 0.0),
        "complete": int(reward.get("result_complete") or 0),
        "has_answer": int(reward.get("result_has_answer") or 0),
        "verdict": str(evidence.get("verdict") or ""),
        "verdict_fine": str(evidence.get("verdict_fine") or ""),
        "n_turns": int(reward.get("efficiency_n_turns") or 0),
        "n_sql": int(reward.get("efficiency_n_sql") or 0),
        "n_cmds": int(reward.get("efficiency_n_cmds") or 0),
    }


def analyze(
    baseline_rows: list[dict[str, Any]],
    sentinel_rows: list[dict[str, Any]],
    adapter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline = index(baseline_rows, "baseline")
    sentinel = index(sentinel_rows, "sentinel")
    expected = set(DEFAULT_TASK_IDS)
    if set(sentinel) != expected:
        raise ValueError(f"sentinel task mismatch: expected={sorted(expected)} got={sorted(sentinel)}")
    if not expected.issubset(baseline):
        raise ValueError("baseline does not contain all sentinel tasks")

    paired = []
    for task_id in DEFAULT_TASK_IDS:
        before = compact(baseline[task_id])
        after = compact(sentinel[task_id])
        paired.append(
            {
                "task_id": task_id,
                "role": "known_incomplete" if task_id in KNOWN_INCOMPLETE else "completion_guardrail",
                "baseline": before,
                "force_final": after,
                "reward_delta": after["reward_total"] - before["reward_total"],
                "completion_rescued": bool(not before["complete"] and after["complete"]),
                "guardrail_preserved": bool(
                    task_id not in COMPLETION_GUARDRAILS
                    or (after["complete"] >= before["complete"] and after["reward_total"] >= before["reward_total"])
                ),
            }
        )

    rescued = sum(row["completion_rescued"] for row in paired if row["task_id"] in KNOWN_INCOMPLETE)
    guardrails_preserved = all(
        row["guardrail_preserved"] for row in paired if row["task_id"] in COMPLETION_GUARDRAILS
    )
    # The gate is fail-closed: a missing adapter summary cannot prove that all
    # six trajectories ended with a terminal assistant answer.
    adapter_all_terminal = bool(adapter and adapter.get("all_terminal"))
    passed = rescued >= 2 and guardrails_preserved and adapter_all_terminal
    return {
        "analysis": "step120_force_final_48k_sentinel6",
        "gate": {
            "passed": passed,
            "required_rescues": 2,
            "rescued_incomplete_tasks": rescued,
            "guardrails_preserved": guardrails_preserved,
            "adapter_all_terminal": adapter_all_terminal,
            "decision": "proceed_to_5_step_canary" if passed else "refine_force_final_policy_before_training",
        },
        "means": {
            "baseline_reward": fmean(row["baseline"]["reward_total"] for row in paired),
            "force_final_reward": fmean(row["force_final"]["reward_total"] for row in paired),
            "reward_delta": fmean(row["reward_delta"] for row in paired),
            "baseline_turns": fmean(row["baseline"]["n_turns"] for row in paired),
            "force_final_turns": fmean(row["force_final"]["n_turns"] for row in paired),
        },
        "paired": paired,
        "adapter": adapter or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sentinel", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = json.loads(args.adapter.read_text(encoding="utf-8"))
    result = analyze(read_jsonl(args.baseline), read_jsonl(args.sentinel), adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
