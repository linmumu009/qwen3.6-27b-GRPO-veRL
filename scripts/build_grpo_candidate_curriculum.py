#!/usr/bin/env python3
"""Build a deterministic five-exposure GRPO curriculum from train128.

The canonical 128-row split remains unchanged.  This script writes a private
640-row schedule ordered by the number of Step120 samples needed to first
observe both a correct and a completed-incorrect trajectory: 2, then 4, then
6.  Older fixed-eight-sample candidate sources are retained as a final legacy
bucket rather than being mislabeled as six-sample candidates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.freeze_grpo_candidate_pool import (
    difficulty,
    explicitly_enabled,
    file_sha256,
    instruction_identity,
)


CONTRACT = "llin-grpo-candidate-curriculum-2-4-6-legacy8-v1"
BUCKET_ORDER = (2, 4, 6, 8)
DIRECT_TWO_SAMPLE_SOURCES = {"v20_adaptive_h05", "v20_adaptive_h06"}


def _stable_key(row: dict[str, Any], seed: str) -> str:
    return hashlib.sha256(
        f"{seed}:{instruction_identity(row)}".encode("utf-8")
    ).hexdigest()


def curriculum_bucket(row: dict[str, Any]) -> int:
    """Return the truthful Step120 mixedness-observation bucket for a row."""

    extra = row.get("extra_info") or {}
    explicit = extra.get("adaptive_mixed_after_samples")
    if explicit not in (None, ""):
        try:
            value = int(explicit)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid adaptive_mixed_after_samples: {explicit}") from error
        if value not in (2, 4, 6):
            raise ValueError(f"unsupported adaptive mixedness bucket: {value}")
        return value

    source = str(extra.get("candidate_pool_source") or "")
    if source in DIRECT_TWO_SAMPLE_SOURCES:
        screen_samples = int(extra.get("adaptive_screen_samples") or 0)
        topup_samples = int(extra.get("adaptive_topup_samples") or 0)
        correct_count = int(extra.get("adaptive_screen_correct_count") or 0)
        if screen_samples == 2 and topup_samples == 0 and correct_count == 1:
            return 2

    # plan-first, semantic281, and old fixed-top-up candidates were established
    # with eight observations or a non-comparable fixed-eight protocol.  Keep
    # them, but do not fabricate a 2/4/6 early-stop label.
    return 8


def _stratified_order(
    rows: list[dict[str, Any]], *, seed: str
) -> list[dict[str, Any]]:
    """Round-robin difficulty/source strata with seeded order inside each."""

    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        extra = row.get("extra_info") or {}
        source = str(extra.get("candidate_pool_source") or "unknown")
        strata[(difficulty(row), source)].append(row)
    for key in strata:
        strata[key].sort(key=lambda row: _stable_key(row, f"{seed}:{key}"))

    ordered: list[dict[str, Any]] = []
    keys = sorted(strata, key=lambda key: (_stable_text_key(key[0]), key[1]))
    offset = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % max(1, len(keys))
    keys = keys[offset:] + keys[:offset]
    while any(strata[key] for key in keys):
        for key in keys:
            if strata[key]:
                ordered.append(strata[key].pop(0))
    return ordered


def _stable_text_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _copy_for_schedule(
    row: dict[str, Any], *, bucket: int, exposure: int, position: int
) -> dict[str, Any]:
    copied = dict(row)
    extra = dict(copied.get("extra_info") or {})
    extra.update(
        {
            "candidate_curriculum_contract": CONTRACT,
            "candidate_curriculum_bucket": bucket,
            "candidate_curriculum_exposure": exposure,
            "candidate_curriculum_position": position,
        }
    )
    copied["extra_info"] = extra
    return copied


def _validate_canonical(rows: list[dict[str, Any]], expected_tasks: int) -> None:
    if len(rows) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} canonical tasks, found {len(rows)}")
    identities = [instruction_identity(row) for row in rows]
    if len(set(identities)) != expected_tasks:
        raise ValueError("canonical training identities are not unique")
    for row in rows:
        if not explicitly_enabled(row, "training_allowed"):
            raise ValueError("canonical training row is not training-enabled")
        if explicitly_enabled(row, "promotion_allowed"):
            raise ValueError("canonical training row unexpectedly enables promotion")


def _validate_schedule_rows(
    canonical: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    *,
    exposures: int,
) -> dict[str, int]:
    canonical_ids = {instruction_identity(row) for row in canonical}
    expected_rows = len(canonical) * exposures
    if len(schedule) != expected_rows:
        raise ValueError(f"expected {expected_rows} schedule rows, found {len(schedule)}")
    counts = Counter(instruction_identity(row) for row in schedule)
    if set(counts) != canonical_ids or set(counts.values()) != {exposures}:
        raise ValueError("schedule does not expose every canonical task exactly as requested")
    buckets = []
    positions = []
    for row in schedule:
        extra = row.get("extra_info") or {}
        if extra.get("candidate_curriculum_contract") != CONTRACT:
            raise ValueError("schedule row has the wrong curriculum contract")
        buckets.append(int(extra.get("candidate_curriculum_bucket")))
        positions.append(int(extra.get("candidate_curriculum_position")))
    if buckets != sorted(buckets, key=BUCKET_ORDER.index):
        raise ValueError("schedule is not ordered 2 -> 4 -> 6 -> legacy8")
    if positions != list(range(1, expected_rows + 1)):
        raise ValueError("schedule positions are not contiguous")
    return dict(sorted(Counter(buckets).items()))


def _write_private_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_curriculum(
    canonical_path: Path,
    *,
    output_path: Path,
    safe_summary_path: Path,
    expected_canonical_sha256: str,
    expected_tasks: int = 128,
    exposures: int = 5,
    groups_per_step: int = 2,
    seed: str = "20260817-curriculum-v1",
) -> dict[str, Any]:
    canonical_path = canonical_path.resolve(strict=True)
    if file_sha256(canonical_path) != expected_canonical_sha256:
        raise ValueError("canonical train sha256 mismatch")
    canonical = pq.read_table(canonical_path).to_pylist()
    _validate_canonical(canonical, expected_tasks)
    if exposures <= 0 or groups_per_step <= 0:
        raise ValueError("exposures and groups_per_step must be positive")

    by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical:
        by_bucket[curriculum_bucket(row)].append(row)
    if set(by_bucket) - set(BUCKET_ORDER):
        raise ValueError("curriculum contains an unsupported bucket")

    schedule: list[dict[str, Any]] = []
    for bucket in BUCKET_ORDER:
        rows = by_bucket.get(bucket, [])
        for exposure in range(1, exposures + 1):
            ordered = _stratified_order(
                [dict(row) for row in rows],
                seed=f"{seed}:bucket={bucket}:exposure={exposure}",
            )
            for row in ordered:
                schedule.append(
                    _copy_for_schedule(
                        row,
                        bucket=bucket,
                        exposure=exposure,
                        position=len(schedule) + 1,
                    )
                )

    scheduled_bucket_counts = _validate_schedule_rows(
        canonical, schedule, exposures=exposures
    )
    _write_private_parquet(output_path, schedule)
    persisted = pq.read_table(output_path).to_pylist()
    _validate_schedule_rows(canonical, persisted, exposures=exposures)

    task_bucket_counts = dict(
        sorted(Counter(curriculum_bucket(row) for row in canonical).items())
    )
    cumulative_groups = 0
    phases = []
    for bucket in BUCKET_ORDER:
        groups = task_bucket_counts.get(bucket, 0) * exposures
        start_group = cumulative_groups + 1 if groups else cumulative_groups
        cumulative_groups += groups
        phases.append(
            {
                "bucket": bucket,
                "tasks": task_bucket_counts.get(bucket, 0),
                "groups": groups,
                "start_group": start_group,
                "end_group": cumulative_groups,
            }
        )
    mixed_boundary_updates = sum(
        1
        for phase in phases[:-1]
        if phase["groups"] and phase["end_group"] % groups_per_step
    )
    summary = {
        "contract": CONTRACT,
        "canonical_tasks": expected_tasks,
        "exposures_per_task": exposures,
        "schedule_rows": len(schedule),
        "groups_per_optimizer_step": groups_per_step,
        "optimizer_steps": len(schedule) // groups_per_step,
        "bucket_order": list(BUCKET_ORDER),
        "bucket_semantics": {
            "2": "mixed correct/completed-incorrect observed by sample 2",
            "4": "mixed correct/completed-incorrect first observed by sample 4",
            "6": "mixed correct/completed-incorrect first observed by sample 6",
            "8": "legacy fixed-eight or non-comparable candidate protocol",
        },
        "task_bucket_counts": task_bucket_counts,
        "scheduled_bucket_counts": scheduled_bucket_counts,
        "phases": phases,
        "mixed_boundary_optimizer_steps": mixed_boundary_updates,
        "canonical_train_sha256": expected_canonical_sha256,
        "curriculum_sha256": file_sha256(output_path),
        "shuffle": False,
        "within_bucket_order": "difficulty_and_source_round_robin_with_seeded_hash",
        "sensitive_artifact_permissions": "0600",
        "contains_prompts_gold_sql_task_ids_environment_ids_or_server_paths": False,
    }
    _write_json(safe_summary_path, summary)
    return summary


def validate_curriculum(
    canonical_path: Path,
    curriculum_path: Path,
    safe_summary_path: Path,
    *,
    expected_tasks: int = 128,
    exposures: int = 5,
) -> dict[str, Any]:
    canonical = pq.read_table(canonical_path.resolve(strict=True)).to_pylist()
    schedule = pq.read_table(curriculum_path.resolve(strict=True)).to_pylist()
    _validate_canonical(canonical, expected_tasks)
    bucket_counts = _validate_schedule_rows(canonical, schedule, exposures=exposures)
    summary = json.loads(safe_summary_path.resolve(strict=True).read_text(encoding="utf-8"))
    if summary.get("contract") != CONTRACT:
        raise ValueError("curriculum summary contract mismatch")
    if summary.get("canonical_train_sha256") != file_sha256(canonical_path):
        raise ValueError("curriculum summary canonical hash mismatch")
    if summary.get("curriculum_sha256") != file_sha256(curriculum_path):
        raise ValueError("curriculum summary schedule hash mismatch")
    return {
        "status": "passed",
        "contract": CONTRACT,
        "canonical_tasks": len(canonical),
        "schedule_rows": len(schedule),
        "scheduled_bucket_counts": bucket_counts,
        "curriculum_sha256": file_sha256(curriculum_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--canonical-train", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--safe-summary", type=Path, required=True)
    build.add_argument("--expected-canonical-sha256", required=True)
    build.add_argument("--expected-tasks", type=int, default=128)
    build.add_argument("--exposures", type=int, default=5)
    build.add_argument("--groups-per-step", type=int, default=2)
    build.add_argument("--seed", default="20260817-curriculum-v1")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--canonical-train", type=Path, required=True)
    validate.add_argument("--curriculum", type=Path, required=True)
    validate.add_argument("--safe-summary", type=Path, required=True)
    validate.add_argument("--expected-tasks", type=int, default=128)
    validate.add_argument("--exposures", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        result = build_curriculum(
            args.canonical_train,
            output_path=args.output,
            safe_summary_path=args.safe_summary,
            expected_canonical_sha256=args.expected_canonical_sha256,
            expected_tasks=args.expected_tasks,
            exposures=args.exposures,
            groups_per_step=args.groups_per_step,
            seed=args.seed,
        )
    else:
        result = validate_curriculum(
            args.canonical_train,
            args.curriculum,
            args.safe_summary,
            expected_tasks=args.expected_tasks,
            exposures=args.exposures,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
