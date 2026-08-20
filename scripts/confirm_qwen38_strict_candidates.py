#!/usr/bin/env python3
"""Confirm Qwen3.8 strict candidates with two fresh trajectories.

The initial adaptive 2+2+2 screen admits a provisional candidate after one
strict correct and one completed strict wrong answer.  This gate combines the
stored adaptive counts with two new strict-table outcomes and only keeps tasks
with at least two strict correct and two completed strict wrong trajectories.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CONTRACT = "llin-qwen38-robust-strict-candidate-confirmation-v1"
REWARD_CONTRACT = "banded-v2-strict-table-v1"


def _identity(row: dict[str, Any]) -> str:
    return str((row.get("extra_info") or {}).get("instruction_sha256") or "")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_private(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist([], schema=schema)
    pq.write_table(table, temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str((row.get("extra_info") or {}).get(field, "unknown")) for row in rows).items()
        )
    )


def confirm(
    candidate_path: Path,
    confirmation_path: Path,
    robust_path: Path,
    rejected_path: Path,
    safe_path: Path,
    *,
    expected_candidates: int,
    host_label: str,
) -> dict[str, Any]:
    candidate_table = pq.read_table(candidate_path)
    candidates = candidate_table.to_pylist()
    if len(candidates) != expected_candidates:
        raise ValueError(
            f"expected {expected_candidates} provisional candidates, got {len(candidates)}"
        )
    by_identity = {_identity(row): row for row in candidates}
    if "" in by_identity or len(by_identity) != len(candidates):
        raise ValueError("candidate identities are missing or duplicated")

    outcomes = _read_jsonl(confirmation_path)
    outcome_by_identity = {str(row.get("instruction_sha256") or ""): row for row in outcomes}
    if "" in outcome_by_identity or len(outcome_by_identity) != len(outcomes):
        raise ValueError("confirmation identities are missing or duplicated")
    if set(outcome_by_identity) != set(by_identity):
        raise ValueError("confirmation outcomes do not exactly cover candidates")

    robust: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    total_confirmation_trajectories = 0
    total_confirmation_completed = 0
    total_confirmation_correct = 0
    total_confirmation_timeouts = 0
    total_confirmation_runtime_errors = 0
    for identity, source in by_identity.items():
        extra = dict(source.get("extra_info") or {})
        if str(extra.get("adaptive_wave_contract") or "") != "llin-adaptive-dwh-2plus2plus2-v1":
            raise ValueError("candidate adaptive-wave contract mismatch")
        prior_samples = int(extra.get("adaptive_samples_observed", -1))
        if prior_samples not in (2, 4, 6):
            raise ValueError("candidate does not carry a valid adaptive sample count")
        prior_correct = int(extra.get("adaptive_correct_count", -1))
        prior_completed = int(extra.get("adaptive_completed_count", -1))
        prior_timeouts = int(extra.get("adaptive_timeout_count", 0))
        prior_runtime_errors = int(extra.get("adaptive_runtime_error_count", 0))
        if prior_correct < 1 or prior_completed - prior_correct < 1:
            raise ValueError("provisional candidate is not strict mixed")

        outcome = outcome_by_identity[identity]
        if str(outcome.get("outcome_contract") or "") != REWARD_CONTRACT:
            raise ValueError("confirmation outcome contract is not banded-v2 strict")
        new_correct = int(outcome.get("correct_count", 0))
        new_completed = int(outcome.get("completed_count", 0))
        new_timeouts = int(outcome.get("trajectory_timeout_count", 0))
        new_runtime_errors = int(outcome.get("runtime_error_count", 0))
        if not 0 <= new_correct <= new_completed <= 2:
            raise ValueError("confirmation outcome counts are invalid")

        total_correct = prior_correct + new_correct
        total_completed = prior_completed + new_completed
        total_wrong = total_completed - total_correct
        total_runtime_errors = prior_runtime_errors + new_runtime_errors
        accepted = total_correct >= 2 and total_wrong >= 2 and total_runtime_errors == 0
        reasons: list[str] = []
        if total_correct < 2:
            reasons.append("fewer_than_two_strict_correct")
        if total_wrong < 2:
            reasons.append("fewer_than_two_completed_strict_wrong")
        if total_runtime_errors:
            reasons.append("runtime_error")
        rejection_reasons.update(reasons)

        row = deepcopy(source)
        final_extra = dict(row.get("extra_info") or {})
        final_extra.update(
            {
                "strict_reward_contract": REWARD_CONTRACT,
                "robust_confirmation_contract": CONTRACT,
                "robust_confirmation_samples": 2,
                "robust_total_samples_observed": prior_samples + 2,
                "robust_total_correct_count": total_correct,
                "robust_total_completed_count": total_completed,
                "robust_total_wrong_count": total_wrong,
                "robust_total_timeout_count": prior_timeouts + new_timeouts,
                "robust_total_runtime_error_count": total_runtime_errors,
                "robust_candidate_accepted": accepted,
                "training_allowed": False,
                "promotion_allowed": False,
            }
        )
        row["extra_info"] = final_extra
        (robust if accepted else rejected).append(row)

        total_confirmation_trajectories += 2
        total_confirmation_completed += new_completed
        total_confirmation_correct += new_correct
        total_confirmation_timeouts += new_timeouts
        total_confirmation_runtime_errors += new_runtime_errors

    _write_private(robust_path, robust, candidate_table.schema)
    _write_private(rejected_path, rejected, candidate_table.schema)
    summary = {
        "contract": CONTRACT,
        "reward_contract": REWARD_CONTRACT,
        "host_label": host_label,
        "provisional_candidates": len(candidates),
        "confirmation_trajectories": total_confirmation_trajectories,
        "confirmation_completed_trajectories": total_confirmation_completed,
        "confirmation_strict_correct_trajectories": total_confirmation_correct,
        "confirmation_timeout_trajectories": total_confirmation_timeouts,
        "confirmation_runtime_error_trajectories": total_confirmation_runtime_errors,
        "robust_candidates": len(robust),
        "rejected_candidates": len(rejected),
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "robust_by_source_version": _counts(robust, "source_version"),
        "robust_by_difficulty": _counts(robust, "difficulty_level"),
        "candidate_dataset_sha256": _sha256(candidate_path),
        "robust_dataset_sha256": _sha256(robust_path),
        "rejected_dataset_sha256": _sha256(rejected_path),
        "minimum_completed_strict_correct": 2,
        "minimum_completed_strict_wrong": 2,
        "timeouts_count_as_completed_wrong": False,
        "runtime_errors_allowed": 0,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_hashes_final_answers_tool_outputs_or_server_paths": False,
    }
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--confirmation-outcomes", type=Path, required=True)
    parser.add_argument("--output-robust", type=Path, required=True)
    parser.add_argument("--output-rejected", type=Path, required=True)
    parser.add_argument("--output-safe-json", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, required=True)
    parser.add_argument("--host-label", required=True)
    args = parser.parse_args()
    result = confirm(
        args.candidates,
        args.confirmation_outcomes,
        args.output_robust,
        args.output_rejected,
        args.output_safe_json,
        expected_candidates=args.expected_candidates,
        host_label=args.host_label,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
