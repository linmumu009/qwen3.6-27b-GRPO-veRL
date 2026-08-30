#!/usr/bin/env python3
"""Build a fail-closed 20-group prefix canary schedule from frontier results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


MAX_NOMINAL_GROUPS = 20


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(frontier: Path, ladders: Path, output: Path, summary: Path) -> dict[str, Any]:
    selected = pq.read_table(frontier).to_pylist()
    ladder_rows = pq.read_table(ladders).to_pylist()
    if len(selected) < 5:
        raise ValueError("canary requires at least five accepted frontier tasks")
    task_selected: dict[str, dict[str, Any]] = {}
    for row in selected:
        extra = row["extra_info"]
        task = str(extra["task_id"])
        if task in task_selected:
            raise ValueError("accepted frontier has duplicate tasks")
        task_selected[task] = row
    depths = {
        int(row["extra_info"]["remaining_assistant_decisions"]) for row in selected
    }
    if len(depths) < 2:
        raise ValueError("canary frontier must cover at least two depths")

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ladder_rows:
        task = str(row["extra_info"]["task_id"])
        if task in task_selected:
            by_task[task].append(row)
    candidates: dict[str, list[dict[str, Any]]] = {}
    for task, chosen in task_selected.items():
        rows = sorted(
            by_task[task],
            key=lambda row: (
                int(row["extra_info"]["remaining_assistant_decisions"]),
                str(row["extra_info"]["prefix_state_id"]),
            ),
        )
        selected_id = str(chosen["extra_info"]["prefix_state_id"])
        index = next(
            (i for i, row in enumerate(rows) if str(row["extra_info"]["prefix_state_id"]) == selected_id),
            None,
        )
        if index is None:
            raise ValueError("accepted frontier is outside representative ladder")
        ordered = [rows[index]]
        # Nearest earlier/later states are deterministic fallbacks when the
        # current-policy group becomes uniform after a prior update.
        for offset in range(1, len(rows)):
            for candidate_index in (index + offset, index - offset):
                if 0 <= candidate_index < len(rows) and rows[candidate_index] not in ordered:
                    ordered.append(rows[candidate_index])
        candidates[task] = ordered

    tasks = sorted(task_selected)
    schedule: list[dict[str, Any]] = []
    cursor = defaultdict(int)
    while len(schedule) < MAX_NOMINAL_GROUPS:
        task = tasks[len(schedule) % len(tasks)]
        options = candidates[task]
        source = options[cursor[task] % len(options)]
        cursor[task] += 1
        row = copy.deepcopy(source)
        row["extra_info"].update(
            {
                "prefix_canary_authorized": True,
                "prefix_canary_nominal_group_index": len(schedule),
                "prefix_canary_actual_optimizer_target": 5,
                "prefix_canary_max_nominal_groups": MAX_NOMINAL_GROUPS,
                "formal_full_training_allowed": False,
            }
        )
        schedule.append(row)
    if any(
        schedule[index]["extra_info"]["prefix_state_id"]
        == schedule[index + 1]["extra_info"]["prefix_state_id"]
        for index in range(len(schedule) - 1)
    ):
        raise ValueError("consecutive nominal groups unexpectedly reuse one prefix identity")

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pq.write_table(pa.Table.from_pylist(schedule), output)
    output.chmod(0o600)
    result = {
        "schema_version": "prefix-state-canary-schedule-v1",
        "accepted_frontier_sha256": _sha(frontier),
        "representative_ladders_sha256": _sha(ladders),
        "frontier_task_count": len(task_selected),
        "frontier_remaining_depths": sorted(depths),
        "nominal_group_count": len(schedule),
        "responses_per_group": 8,
        "groups_per_nominal_batch": 1,
        "target_actual_optimizer_steps": 5,
        "same_prefix_grouping_only": True,
        "fallback_movement": "deterministic_nearest_earlier_then_later_ready_prefix",
        "full_training_allowed": False,
        "schedule_sha256": _sha(output),
    }
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary.chmod(0o600)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--ladders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.frontier, args.ladders, args.output, args.safe_summary), sort_keys=True))


if __name__ == "__main__":
    main()
