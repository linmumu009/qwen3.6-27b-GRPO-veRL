#!/usr/bin/env python3
"""Safely replay frozen tiered-canary observability without inventing evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from llin_verl.grounded_trajectory_reward import JudgeState, _final_decision
from llin_verl.minimal_grounded_reward import _task_infrastructure


def _read_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _dataset(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in pq.read_table(path).to_pylist():
        extra = row.get("extra_info") or {}
        identity = str(extra.get("instruction_sha256") or "")
        truth = ((row.get("reward_model") or {}).get("ground_truth") or {})
        if identity and isinstance(truth, dict):
            result[identity] = truth
    return result


def replay(
    rollout_dir: Path,
    dataset: Path,
    database_root: Path,
) -> dict[str, Any]:
    rows = _read_rows(rollout_dir)
    truths = _dataset(dataset)
    before_states: Counter[str] = Counter()
    before_reasons: Counter[str] = Counter()
    after_states: Counter[str] = Counter()
    after_reasons: Counter[str] = Counter()
    migrations: Counter[str] = Counter()
    task_components: Counter[str] = Counter()
    final_components: Counter[str] = Counter()
    database_migrations: Counter[str] = Counter()
    historical_observability: Counter[str] = Counter()
    joins_missing = 0
    valid_task_hashes = 0
    valid_trajectory_hashes = 0

    for row in rows:
        old_state = str(row.get("judge_state") or "UNKNOWN")
        old_reason = str(row.get("judge_reason") or "missing")
        before_states[old_state] += 1
        before_reasons[old_reason] += 1
        task_identity = str(row.get("task_identity_sha256") or "")
        trajectory_identity = str(row.get("trajectory_identity_sha256") or "")
        valid_task_hashes += len(task_identity) == 64 and all(
            char in "0123456789abcdef" for char in task_identity.casefold()
        )
        valid_trajectory_hashes += len(trajectory_identity) == 64 and all(
            char in "0123456789abcdef" for char in trajectory_identity.casefold()
        )
        truth = truths.get(task_identity)
        if truth is None:
            joins_missing += 1
            new_state, new_reason = "UNKNOWN", "offline_task_join_missing"
        else:
            task, _ = _task_infrastructure(
                truth,
                {"pi_reward_database_root": str(database_root)},
            )
            final, _ = _final_decision(str(row.get("output") or ""), truth)
            task_components[f"{task.state.value}:{task.reason}"] += 1
            final_components[f"{final.state.value}:{final.reason}"] += 1
            database_migrations[
                f"stored_{bool(row.get('database_available'))}->replay_{task.state is JudgeState.PASS}"
            ] += 1

            unsafe = bool(row.get("unsafe"))
            budget = bool(row.get("budget_exceeded"))
            q = int(float(row.get("query_attempt_count") or 0))
            events = int(float(row.get("tool_event_count") or 0))
            if unsafe:
                new_state, new_reason = "FAIL", "unsafe"
            elif budget:
                new_state, new_reason = "FAIL", "budget_exceeded"
            elif task.state is JudgeState.UNKNOWN:
                new_state, new_reason = "UNKNOWN", task.reason
            elif q > 0 or events > 0:
                # The old private dump contains aggregate q/t fields but not
                # native tool events or their exact response-token counts.
                # Replaying it as PASS would fabricate evidence.
                historical_observability["tool_trajectory_token_history_unobservable"] += 1
                new_state, new_reason = "UNKNOWN", "historical_tool_response_cost_unobservable"
            elif old_state == "FAIL" and old_reason == "no_relevant_readonly_attempt":
                historical_observability["observable_no_attempt_fail"] += 1
                new_state, new_reason = "FAIL", "no_relevant_readonly_attempt"
            else:
                historical_observability["other_history_incomplete"] += 1
                new_state, new_reason = "UNKNOWN", "historical_runtime_evidence_incomplete"

        after_states[new_state] += 1
        after_reasons[new_reason] += 1
        migrations[f"{old_state}:{old_reason}->{new_state}:{new_reason}"] += 1

    return {
        "schema_version": "qwen38-tiered-observability-offline-replay-v1",
        "rows": len(rows),
        "dataset_tasks": len(truths),
        "task_join_missing": joins_missing,
        "before": {
            "judge_state_counts": dict(sorted(before_states.items())),
            "judge_reason_counts": dict(sorted(before_reasons.items())),
        },
        "after": {
            "judge_state_counts": dict(sorted(after_states.items())),
            "judge_reason_counts": dict(sorted(after_reasons.items())),
        },
        "reason_migration_matrix": dict(sorted(migrations.items())),
        "database_migration_counts": dict(sorted(database_migrations.items())),
        "task_component_counts": dict(sorted(task_components.items())),
        "final_component_counts": dict(sorted(final_components.items())),
        "historical_observability": dict(sorted(historical_observability.items())),
        "identity": {
            "valid_task_sha256": valid_task_hashes,
            "valid_trajectory_sha256": valid_trajectory_hashes,
            "native_request_and_event_fields_persisted_in_old_dump": False,
        },
        "old_tool_token_policy": "missing_history_remains_UNKNOWN_not_zero",
        "pass_promotion_from_missing_old_tool_evidence": 0,
        "sensitive_fields_emitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.rollout_dir, args.dataset, args.database_root)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps({"rows": report["rows"], "before": report["before"], "after": report["after"]}, sort_keys=True))


if __name__ == "__main__":
    main()
