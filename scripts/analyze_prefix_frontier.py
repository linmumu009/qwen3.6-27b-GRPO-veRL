#!/usr/bin/env python3
"""Aggregate local prefix rollouts and choose the next deterministic frontier states."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_results(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda value: str(value)):
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                state_id = str(row.get("prefix_state_id") or "")
                if not state_id:
                    raise ValueError(f"validation row lacks prefix_state_id: {path}:{line_index + 1}")
                row["_result_identity"] = f"{_sha(path)}:{line_index}"
                rows.append(row)
    return rows


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pq.write_table(pa.Table.from_pylist(rows), path)
    path.chmod(0o600)


def _metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _state_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["_result_identity"])
    scorable = [row for row in ordered if _metric(row, "train_mask") == 1.0]
    fixed = scorable[:8]
    return {
        "observed": len(ordered),
        "unknown": sum(_metric(row, "train_mask") != 1.0 for row in ordered),
        "scorable": len(scorable),
        "fixed8_complete": len(fixed) == 8,
        "fixed8_correct": int(sum(_metric(row, "final_answer_correct") == 1.0 for row in fixed)),
        "fixed8_pass": int(sum(_metric(row, "success") == 1.0 for row in fixed)),
        "coarse4_correct": int(
            sum(_metric(row, "final_answer_correct") == 1.0 for row in scorable[:4])
        ),
        "coarse4_complete": len(scorable) >= 4,
        "reward_mean_fixed8": (
            sum(_metric(row, "tiered_reward", _metric(row, "reward")) for row in fixed) / len(fixed)
            if fixed
            else 0.0
        ),
    }


def _boundary_violations(rows: list[dict[str, Any]]) -> Counter[str]:
    violations: Counter[str] = Counter()
    for row in rows:
        if _metric(row, "train_mask") != 1.0:
            continue
        reward = _metric(row, "tiered_reward", _metric(row, "reward"))
        final = _metric(row, "final_answer_correct") == 1.0
        success = _metric(row, "success") == 1.0
        if not final and reward > 0.2 + 1e-9:
            violations["wrong_reward_above_point2"] += 1
        if success and reward < 0.8 - 1e-9:
            violations["grounded_correct_reward_below_point8"] += 1
        if _metric(row, "guess_correct_blocked") == 1.0 and abs(reward) > 1e-9:
            violations["guess_reward_nonzero"] += 1
        if (
            _metric(row, "unsafe") == 1.0 or _metric(row, "budget_exceeded") == 1.0
        ) and abs(reward) > 1e-9:
            violations["unsafe_or_budget_reward_nonzero"] += 1
        if _metric(row, "generated_suffix_only_mask_verified") != 1.0:
            violations["suffix_mask_not_verified"] += 1
    return violations


def analyze(
    ladders: Path,
    validation_files: list[Path],
    output_root: Path,
    max_observed_per_state: int = 16,
) -> dict[str, Any]:
    ladder_rows = pq.read_table(ladders).to_pylist()
    by_state = {str(row["extra_info"]["prefix_state_id"]): row for row in ladder_rows}
    if len(by_state) != len(ladder_rows):
        raise ValueError("representative ladder has duplicate prefix states")
    results = _read_results(validation_files)
    unknown_states = sorted({str(row["prefix_state_id"]) for row in results} - set(by_state))
    if unknown_states:
        raise ValueError("validation results contain states outside the frozen representative ladders")
    violations = _boundary_violations(results)
    if violations:
        raise ValueError(f"frontier reward/mask boundary violation: {dict(violations)}")

    results_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        results_by_state[str(row["prefix_state_id"])].append(row)
    summaries = {state: _state_summary(rows) for state, rows in results_by_state.items()}
    task_states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ladder_rows:
        if row["extra_info"]["curriculum_split"] == "train":
            task_states[str(row["extra_info"]["task_id"])].append(row)

    accepted: list[dict[str, Any]] = []
    backups: list[dict[str, Any]] = []
    next_rows: list[dict[str, Any]] = []
    task_outcomes: Counter[str] = Counter()
    for task_id, states in sorted(task_states.items()):
        states.sort(
            key=lambda row: (
                int(row["extra_info"]["remaining_assistant_decisions"]),
                str(row["extra_info"]["curriculum_stage"]),
            )
        )
        state_ids = [str(row["extra_info"]["prefix_state_id"]) for row in states]
        completed_mixed = [
            row
            for row in states
            if summaries.get(str(row["extra_info"]["prefix_state_id"]), {}).get("fixed8_complete")
            and 2
            <= summaries[str(row["extra_info"]["prefix_state_id"])] ["fixed8_correct"]
            <= 6
        ]
        if completed_mixed:
            choice = max(
                completed_mixed,
                key=lambda row: int(row["extra_info"]["remaining_assistant_decisions"]),
            )
            accepted.append(choice)
            task_outcomes["accepted_2_to_6_of_8"] += 1
            continue

        completed_boundary = [
            row
            for row in states
            if summaries.get(str(row["extra_info"]["prefix_state_id"]), {}).get("fixed8_complete")
            and summaries[str(row["extra_info"]["prefix_state_id"])] ["fixed8_correct"] in {1, 7}
        ]
        backups.extend(completed_boundary)

        pending = None
        for row in states:
            state_id = str(row["extra_info"]["prefix_state_id"])
            summary = summaries.get(state_id)
            if not summary:
                continue
            if summary["scorable"] < 4 and summary["observed"] < max_observed_per_state:
                pending = row
                task_outcomes["resample_unknown_to_four_scorable"] += 1
                break
            if summary["coarse4_complete"] and 1 <= summary["coarse4_correct"] <= 3 and not summary["fixed8_complete"]:
                pending = row
                task_outcomes["fill_coarse_mixed_to_eight"] += 1
                break
        if pending is not None:
            next_rows.append(pending)
            continue

        observed_uniform: dict[int, str] = {}
        for index, state_id in enumerate(state_ids):
            summary = summaries.get(state_id)
            if not summary or not summary["coarse4_complete"]:
                continue
            if summary["coarse4_correct"] == 4:
                observed_uniform[index] = "correct"
            elif summary["coarse4_correct"] == 0:
                observed_uniform[index] = "wrong"
        brackets = [
            (left, right)
            for left, left_value in observed_uniform.items()
            for right, right_value in observed_uniform.items()
            if left < right and left_value == "correct" and right_value == "wrong"
        ]
        if brackets:
            left, right = min(brackets, key=lambda pair: (pair[1] - pair[0], pair))
            candidates = [index for index in range(left + 1, right) if index not in observed_uniform]
            if candidates:
                middle = min(candidates, key=lambda index: (abs(index - (left + right) / 2), index))
                next_rows.append(states[middle])
                task_outcomes["binary_search_next_depth"] += 1
                continue
        task_outcomes["no_mixed_frontier_on_representative"] += 1

    accepted_depths = {
        int(row["extra_info"]["remaining_assistant_decisions"]) for row in accepted
    }
    gate = len(accepted) >= 5 and len(accepted_depths) >= 2 and not violations
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_parquet(output_root / "next_round.runtime.sensitive.parquet", next_rows)
    _write_parquet(output_root / "accepted_frontier.runtime.sensitive.parquet", accepted)
    _write_parquet(output_root / "boundary_backup.runtime.sensitive.parquet", backups)
    safe = {
        "schema_version": "prefix-state-frontier-search-v1",
        "input_validation_files": len(validation_files),
        "result_rows": len(results),
        "result_identity_unique": len({row["_result_identity"] for row in results}) == len(results),
        "states_observed": len(summaries),
        "state_scorable_count_histogram": dict(
            sorted(Counter(value["scorable"] for value in summaries.values()).items())
        ),
        "state_unknown_count_histogram": dict(
            sorted(Counter(value["unknown"] for value in summaries.values()).items())
        ),
        "accepted_2_to_6_count": len(accepted),
        "accepted_remaining_depths": sorted(accepted_depths),
        "boundary_backup_1_or_7_count": len(backups),
        "next_round_state_count": len(next_rows),
        "task_outcome_counts": dict(sorted(task_outcomes.items())),
        "reward_boundary_violation_counts": dict(violations),
        "api_requests": 0,
        "optimizer_steps": 0,
        "actor_parameter_updates": 0,
        "checkpoints": 0,
        "frontier_gate_passed": gate,
        "frontier_gate_contract": ">=5 train tasks at 2..6/8 and >=2 remaining depths",
        "output_hashes": {
            name: _sha(output_root / name)
            for name in (
                "next_round.runtime.sensitive.parquet",
                "accepted_frontier.runtime.sensitive.parquet",
                "boundary_backup.runtime.sensitive.parquet",
            )
        },
    }
    safe_path = output_root / "frontier_gate.safe.json"
    safe_path.write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    safe_path.chmod(0o600)
    print(json.dumps(safe, sort_keys=True))
    return safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladders", type=Path, required=True)
    parser.add_argument("--validation", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-observed-per-state", type=int, default=16)
    args = parser.parse_args()
    analyze(args.ladders, args.validation, args.output_root, args.max_observed_per_state)


if __name__ == "__main__":
    main()
