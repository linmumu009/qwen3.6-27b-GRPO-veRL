#!/usr/bin/env python3
"""Compare two boss-original reward_judge outputs at task grain."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in result:
            raise ValueError(f"{label}: missing or duplicate task_id={task_id!r}")
        result[task_id] = row
    return result


def _avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def _process_score(reward: dict[str, Any]) -> float | None:
    weighted = [
        (reward.get("process_tables_hit"), 0.3),
        (reward.get("process_fields_used"), 0.3),
        (reward.get("process_docs_hit"), 0.2),
        (reward.get("process_task_fit"), 0.2),
    ]
    active = [(float(value), weight) for value, weight in weighted if value is not None]
    if not active:
        return None
    return sum(value * weight for value, weight in active) / sum(weight for _, weight in active)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [row.get("reward") or {} for row in rows]
    evidence = [row.get("evidence") or {} for row in rows]
    totals = [float(item.get("reward_total") or 0.0) for item in rewards]
    result_scores = [
        0.5 * float(item.get("result_has_answer") or 0.0)
        + 0.5 * float(item.get("result_correct_numeric") or 0.0)
        for item in rewards
    ]
    process_scores = [score for item in rewards if (score := _process_score(item)) is not None]

    def reward_mean(key: str) -> float | None:
        return _avg([float(item[key]) for item in rewards if item.get(key) is not None])

    def evidence_mean(key: str) -> float | None:
        return _avg([float(item[key]) for item in evidence if item.get(key) is not None])

    return {
        "n": len(rows),
        "reward_total_mean": _avg(totals),
        "reward_total_sum": sum(totals),
        "result_score_mean": _avg(result_scores),
        "process_score_mean": _avg(process_scores),
        "complete_count": sum(int(item.get("result_complete") or 0) for item in rewards),
        "has_answer_count": sum(int(item.get("result_has_answer") or 0) for item in rewards),
        "correct_numeric_count": sum(
            int(item.get("result_correct_numeric") or 0) for item in rewards
        ),
        "tables_hit_count": sum(int(item.get("process_tables_hit") or 0) for item in rewards),
        "fields_used_mean": reward_mean("process_fields_used"),
        "task_fit_mean": reward_mean("process_task_fit"),
        "n_turns_mean": reward_mean("efficiency_n_turns"),
        "n_sql_mean": reward_mean("efficiency_n_sql"),
        "n_cmds_mean": reward_mean("efficiency_n_cmds"),
        "dup_cmd_mean": reward_mean("efficiency_dup_cmd"),
        "answer_len_mean": evidence_mean("answer_len"),
        "verdict_counts": dict(sorted(Counter(item.get("verdict") for item in evidence).items())),
        "verdict_fine_counts": dict(
            sorted(Counter(item.get("verdict_fine") for item in evidence).items())
        ),
    }


def compare_prompts(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = index_unique(read_jsonl(left_path), "left trajectories")
    right = index_unique(read_jsonl(right_path), "right trajectories")
    common = sorted(set(left) & set(right))
    mismatches = [
        task_id
        for task_id in common
        if (left[task_id].get("messages") or [])[:2]
        != (right[task_id].get("messages") or [])[:2]
    ]
    return {
        "task_ids_identical": set(left) == set(right),
        "common_task_count": len(common),
        "identical_prompt_count": len(common) - len(mismatches),
        "mismatched_prompt_task_ids": mismatches,
    }


def compare(
    left_path: Path,
    right_path: Path,
    left_label: str,
    right_label: str,
    left_trajectories: Path | None = None,
    right_trajectories: Path | None = None,
) -> dict[str, Any]:
    left = index_unique(read_jsonl(left_path), left_label)
    right = index_unique(read_jsonl(right_path), right_label)
    if set(left) != set(right):
        raise ValueError("reward outputs do not contain identical task ids")

    pairs: list[dict[str, Any]] = []
    wins = losses = ties = 0
    for task_id in sorted(left):
        left_row, right_row = left[task_id], right[task_id]
        left_reward = float((left_row.get("reward") or {}).get("reward_total") or 0.0)
        right_reward = float((right_row.get("reward") or {}).get("reward_total") or 0.0)
        delta = right_reward - left_reward
        wins += delta > 1e-12
        losses += delta < -1e-12
        ties += abs(delta) <= 1e-12
        pairs.append(
            {
                "task_id": task_id,
                left_label: left_reward,
                right_label: right_reward,
                "delta": delta,
                f"{left_label}_verdict": (left_row.get("evidence") or {}).get("verdict"),
                f"{right_label}_verdict": (right_row.get("evidence") or {}).get("verdict"),
            }
        )

    left_summary = summarize(list(left.values()))
    right_summary = summarize(list(right.values()))
    numeric_deltas = {
        key: {
            left_label: left_summary[key],
            right_label: right_summary[key],
            "delta": right_summary[key] - left_summary[key],
        }
        for key in left_summary
        if isinstance(left_summary[key], (int, float))
        and isinstance(right_summary[key], (int, float))
    }
    result: dict[str, Any] = {
        "comparison": f"{left_label}_vs_{right_label}_same_tasks_boss_original_reward_judge",
        "task_ids_identical": True,
        left_label: left_summary,
        right_label: right_summary,
        "numeric_deltas": numeric_deltas,
        "paired_reward": {"wins": wins, "losses": losses, "ties": ties},
        "paired_results": pairs,
        "artifact_sha256": {
            f"{left_label}_reward": sha256(left_path),
            f"{right_label}_reward": sha256(right_path),
        },
    }
    if left_trajectories and right_trajectories:
        result["prompt_identity"] = compare_prompts(left_trajectories, right_trajectories)
        result["artifact_sha256"].update(
            {
                f"{left_label}_trajectories": sha256(left_trajectories),
                f"{right_label}_trajectories": sha256(right_trajectories),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--left-trajectories", type=Path)
    parser.add_argument("--right-trajectories", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.left,
        args.right,
        args.left_label,
        args.right_label,
        args.left_trajectories,
        args.right_trajectories,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
