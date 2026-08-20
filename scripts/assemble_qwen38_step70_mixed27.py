#!/usr/bin/env python3
"""Freeze 27 Step70 strict-mixed tasks and a disjoint six-task sealed set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


CONTRACT = "llin-qwen38-step70-mixed27-four-exposure-v1"
REWARD_CONTRACT = "banded-v2-strict-table-v1"
EXPECTED_ORIGINAL_TASKS = 15
EXPECTED_HELDOUT_TASKS = 18
EXPECTED_TRAIN_TASKS = 27
EXPECTED_SEALED_TASKS = 6
EXPOSURES_PER_TASK = 4
RESPONSES_PER_GROUP = 8
GROUPS_PER_STEP = 2
SEALED_VERSION_TARGETS = {"v15": 2, "v20": 3, "v21": 1}
EXPECTED_ORIGINAL_VERSION_COUNTS = {"v15": 4, "v20": 9, "v21": 2}
EXPECTED_HELDOUT_VERSION_COUNTS = {"v15": 5, "v20": 11, "v21": 2}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(row: dict[str, Any]) -> tuple[str, str]:
    extra = row.get("extra_info") or {}
    instruction = str(extra.get("instruction_sha256") or "")
    gold = str(extra.get("gold_sha256") or "")
    if not instruction or not gold:
        raise ValueError("candidate identity is incomplete")
    return instruction, gold


def source_version(row: dict[str, Any]) -> str:
    raw = str((row.get("extra_info") or {}).get("source_version") or "")
    for version in SEALED_VERSION_TARGETS:
        if raw == version or raw.endswith(f"_{version}"):
            return version
    raise ValueError(f"unsupported source version: {raw}")


def difficulty(row: dict[str, Any]) -> str:
    raw = (row.get("extra_info") or {}).get("difficulty_level")
    try:
        level = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid difficulty level: {raw}") from error
    if level not in range(1, 6):
        raise ValueError(f"difficulty level is outside 1..5: {level}")
    return str(level)


def stable_key(row: dict[str, Any], *, seed: str) -> str:
    instruction, gold = identity(row)
    return hashlib.sha256(f"{seed}:{instruction}:{gold}".encode()).hexdigest()


def stratified_order(rows: Sequence[dict[str, Any]], *, seed: str) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        extra = row.get("extra_info") or {}
        key = (source_version(row), difficulty(row), str(extra.get("qwen38_source_host") or ""))
        if not key[2]:
            raise ValueError("candidate source host is missing")
        strata[key].append(row)
    for key, values in strata.items():
        values.sort(key=lambda row: stable_key(row, seed=f"{seed}:{key}"))
    ordered: list[dict[str, Any]] = []
    keys = sorted(strata)
    while any(strata.values()):
        for key in keys:
            if strata[key]:
                ordered.append(strata[key].pop(0))
    return ordered


def validate_candidate(row: dict[str, Any], *, origin: str) -> None:
    identity(row)
    source_version(row)
    difficulty(row)
    extra = row.get("extra_info") or {}
    truth = (row.get("reward_model") or {}).get("ground_truth") or {}
    checks = {
        "training_disabled": extra.get("training_allowed") is False,
        "promotion_disabled": extra.get("promotion_allowed") is False,
        "table_answer_type": truth.get("answer_type") == "table",
        "expected_value_json": bool(truth.get("expected_value_json")),
        "verification_sql": bool(truth.get("verification_sql")),
    }
    if origin == "original70":
        checks["strict_replay_qualified"] = extra.get("strict_reward_contract") == REWARD_CONTRACT
    missing = sorted(name for name, passed in checks.items() if not passed)
    if missing:
        raise ValueError(f"{origin} candidate failed contract: {', '.join(missing)}")


def read_sources(
    sources: Sequence[tuple[str, Path]], *, origin: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for host, source in sources:
        if host not in {"m00", "m05", "m06"}:
            raise ValueError(f"unsupported source host: {host}")
        resolved = source.resolve(strict=True)
        if resolved in seen_paths:
            raise ValueError(f"duplicate source path: {resolved}")
        seen_paths.add(resolved)
        for row in pq.read_table(resolved).to_pylist():
            copied = deepcopy(row)
            copied.setdefault("extra_info", {})["qwen38_source_host"] = host
            validate_candidate(copied, origin=origin)
            rows.append(copied)
    ids = [identity(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{origin} sources contain duplicate candidate identities")
    return rows


def write_private(path: Path, rows: list[dict[str, Any]], *, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    table = pa.Table.from_pylist(rows, schema=schema) if rows else pa.Table.from_pylist([], schema=schema)
    pq.write_table(table, temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def authorize(row: dict[str, Any], *, role: str, position: int) -> dict[str, Any]:
    copied = deepcopy(row)
    copied.setdefault("extra_info", {}).update(
        {
            "training_allowed": role == "train",
            "promotion_allowed": False,
            "owner_authorized_training": role == "train",
            "qwen38_training_contract": CONTRACT,
            "strict_reward_contract": REWARD_CONTRACT,
            "qwen38_source_policy_step": 70,
            "qwen38_selection_role": role,
            "qwen38_canonical_position": position,
        }
    )
    return copied


def assemble(
    original_sources: Sequence[tuple[str, Path]],
    heldout_sources: Sequence[tuple[str, Path]],
    *,
    canonical_path: Path,
    schedule_path: Path,
    sealed_path: Path,
    safe_summary_path: Path,
    seed: str = "20260820-qwen38-step70-mixed27-v1",
) -> dict[str, Any]:
    original = read_sources(original_sources, origin="original70")
    heldout = read_sources(heldout_sources, origin="heldout1430")
    if len(original) != EXPECTED_ORIGINAL_TASKS or len(heldout) != EXPECTED_HELDOUT_TASKS:
        raise ValueError(f"source counts mismatch: original={len(original)}, heldout={len(heldout)}")
    if {identity(row) for row in original} & {identity(row) for row in heldout}:
        raise ValueError("original70 and heldout candidate identities overlap")
    original_versions = Counter(source_version(row) for row in original)
    heldout_versions = Counter(source_version(row) for row in heldout)
    if dict(original_versions) != EXPECTED_ORIGINAL_VERSION_COUNTS:
        raise ValueError(f"original version counts mismatch: {dict(original_versions)}")
    if dict(heldout_versions) != EXPECTED_HELDOUT_VERSION_COUNTS:
        raise ValueError(f"heldout version counts mismatch: {dict(heldout_versions)}")

    sealed_raw: list[dict[str, Any]] = []
    train_heldout: list[dict[str, Any]] = []
    for version, target in SEALED_VERSION_TARGETS.items():
        ordered = stratified_order(
            [row for row in heldout if source_version(row) == version],
            seed=f"{seed}:sealed:{version}",
        )
        sealed_raw.extend(ordered[:target])
        train_heldout.extend(ordered[target:])
    train_raw = [*original, *train_heldout]
    if len(train_raw) != EXPECTED_TRAIN_TASKS or len(sealed_raw) != EXPECTED_SEALED_TASKS:
        raise ValueError("train/sealed split shape mismatch")
    if {identity(row) for row in train_raw} & {identity(row) for row in sealed_raw}:
        raise ValueError("train and sealed identities overlap")

    canonical = [
        authorize(row, role="train", position=position)
        for position, row in enumerate(stratified_order(train_raw, seed=f"{seed}:train"), 1)
    ]
    sealed = [
        authorize(row, role="sealed_eval", position=position)
        for position, row in enumerate(stratified_order(sealed_raw, seed=f"{seed}:sealed-final"), 1)
    ]
    schedule: list[dict[str, Any]] = []
    for exposure in range(1, EXPOSURES_PER_TASK + 1):
        for row in stratified_order(canonical, seed=f"{seed}:exposure:{exposure}"):
            copied = deepcopy(row)
            copied["extra_info"].update(
                {
                    "qwen38_exposure": exposure,
                    "qwen38_schedule_position": len(schedule) + 1,
                }
            )
            schedule.append(copied)
    counts = Counter(identity(row) for row in schedule)
    if len(schedule) != 108 or set(counts.values()) != {EXPOSURES_PER_TASK}:
        raise ValueError("four-exposure schedule shape mismatch")

    schema = pa.Table.from_pylist(canonical).schema
    write_private(canonical_path, canonical, schema=schema)
    write_private(schedule_path, schedule, schema=pa.Table.from_pylist(schedule).schema)
    write_private(sealed_path, sealed, schema=schema)
    train_versions = Counter(source_version(row) for row in canonical)
    sealed_versions = Counter(source_version(row) for row in sealed)
    train_difficulties = Counter(difficulty(row) for row in canonical)
    sealed_difficulties = Counter(difficulty(row) for row in sealed)
    summary = {
        "contract": CONTRACT,
        "reward_contract": REWARD_CONTRACT,
        "source_policy_step": 70,
        "source_candidates": 33,
        "original70_strict_mixed_tasks": EXPECTED_ORIGINAL_TASKS,
        "heldout_strict_mixed_tasks": EXPECTED_HELDOUT_TASKS,
        "canonical_tasks": EXPECTED_TRAIN_TASKS,
        "sealed_tasks": EXPECTED_SEALED_TASKS,
        "exposures_per_task": EXPOSURES_PER_TASK,
        "schedule_groups": len(schedule),
        "responses_per_group": RESPONSES_PER_GROUP,
        "groups_per_optimizer_step": GROUPS_PER_STEP,
        "optimizer_steps": len(schedule) // GROUPS_PER_STEP,
        "new_rollout_trajectories": len(schedule) * RESPONSES_PER_GROUP,
        "train_source_version_counts": dict(sorted(train_versions.items())),
        "sealed_source_version_counts": dict(sorted(sealed_versions.items())),
        "train_difficulty_counts": dict(sorted(train_difficulties.items())),
        "sealed_difficulty_counts": dict(sorted(sealed_difficulties.items())),
        "canonical_sha256": file_sha256(canonical_path),
        "schedule_sha256": file_sha256(schedule_path),
        "sealed_sha256": file_sha256(sealed_path),
        "training_allowed": True,
        "sealed_training_allowed": False,
        "promotion_allowed": False,
        "sensitive_artifact_permissions": "0600",
        "contains_prompts_gold_sql_task_ids_environment_ids_or_server_paths": False,
    }
    write_json(safe_summary_path, summary)
    return summary


def validate(canonical: Path, schedule: Path, sealed: Path, summary_path: Path) -> dict[str, Any]:
    canonical_rows = pq.read_table(canonical.resolve(strict=True)).to_pylist()
    schedule_rows = pq.read_table(schedule.resolve(strict=True)).to_pylist()
    sealed_rows = pq.read_table(sealed.resolve(strict=True)).to_pylist()
    summary = json.loads(summary_path.resolve(strict=True).read_text(encoding="utf-8"))
    train_ids = {identity(row) for row in canonical_rows}
    sealed_ids = {identity(row) for row in sealed_rows}
    schedule_counts = Counter(identity(row) for row in schedule_rows)
    if summary.get("contract") != CONTRACT or summary.get("reward_contract") != REWARD_CONTRACT:
        raise ValueError("summary contract mismatch")
    if len(canonical_rows) != 27 or len(train_ids) != 27:
        raise ValueError("canonical pool is not 27 unique tasks")
    if len(sealed_rows) != 6 or len(sealed_ids) != 6 or train_ids & sealed_ids:
        raise ValueError("sealed pool shape or disjointness failed")
    if len(schedule_rows) != 108 or set(schedule_counts) != train_ids or set(schedule_counts.values()) != {4}:
        raise ValueError("schedule is not an exact four-exposure expansion")
    if any((row.get("extra_info") or {}).get("training_allowed") is not True for row in schedule_rows):
        raise ValueError("schedule contains a training-disabled row")
    if any((row.get("extra_info") or {}).get("training_allowed") is not False for row in sealed_rows):
        raise ValueError("sealed pool contains a training-enabled row")
    expected_hashes = {
        "canonical_sha256": file_sha256(canonical),
        "schedule_sha256": file_sha256(schedule),
        "sealed_sha256": file_sha256(sealed),
    }
    if any(summary.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("training artifact SHA256 mismatch")
    return {
        "status": "passed",
        "contract": CONTRACT,
        "canonical_tasks": 27,
        "sealed_tasks": 6,
        "schedule_groups": 108,
        "optimizer_steps": 54,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--original", action="append", nargs=2, metavar=("HOST", "PARQUET"), required=True)
    build.add_argument("--heldout", action="append", nargs=2, metavar=("HOST", "PARQUET"), required=True)
    for target in (build, sub.add_parser("validate")):
        target.add_argument("--canonical", type=Path, required=True)
        target.add_argument("--schedule", type=Path, required=True)
        target.add_argument("--sealed", type=Path, required=True)
        target.add_argument("--safe-summary", type=Path, required=True)
    build.add_argument("--seed", default="20260820-qwen38-step70-mixed27-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        result = assemble(
            [(host, Path(path)) for host, path in args.original],
            [(host, Path(path)) for host, path in args.heldout],
            canonical_path=args.canonical,
            schedule_path=args.schedule,
            sealed_path=args.sealed,
            safe_summary_path=args.safe_summary,
            seed=args.seed,
        )
    else:
        result = validate(args.canonical, args.schedule, args.sealed, args.safe_summary)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
