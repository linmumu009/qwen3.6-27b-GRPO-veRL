#!/usr/bin/env python3
"""Assemble the owner-authorized Qwen3.8 70-task, two-exposure GRPO pool."""

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


CONTRACT = "llin-qwen38-grpo-train70-two-exposure-v1"
REWARD_CONTRACT = "banded-v2-strict-table-v1"
EXPECTED_SOURCE_COUNTS = {"m00": 21, "m05": 20, "m06": 29}
EXPECTED_VERSION_COUNTS = {"v15": 12, "v20": 39, "v21": 19}
EXPECTED_DIFFICULTY_COUNTS = {"1": 2, "2": 25, "3": 17, "4": 16, "5": 10}


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


def _source_version(row: dict[str, Any]) -> str:
    raw = str((row.get("extra_info") or {}).get("source_version") or "")
    for version in EXPECTED_VERSION_COUNTS:
        if raw == version or raw.endswith(f"_{version}"):
            return version
    raise ValueError(f"unsupported source version: {raw}")


def _difficulty(row: dict[str, Any]) -> str:
    raw = str((row.get("extra_info") or {}).get("difficulty_level") or "")
    try:
        return str(int(raw))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid difficulty level: {raw}") from error


def _stable_key(row: dict[str, Any], *, seed: str) -> str:
    instruction, gold = identity(row)
    return hashlib.sha256(f"{seed}:{instruction}:{gold}".encode()).hexdigest()


def _stratified_order(rows: list[dict[str, Any]], *, seed: str) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        extra = row.get("extra_info") or {}
        key = (_difficulty(row), _source_version(row), str(extra["qwen38_source_host"]))
        strata[key].append(row)
    for key, values in strata.items():
        values.sort(key=lambda row: _stable_key(row, seed=f"{seed}:{key}"))
    keys = sorted(strata)
    ordered: list[dict[str, Any]] = []
    while any(strata.values()):
        for key in keys:
            if strata[key]:
                ordered.append(strata[key].pop(0))
    return ordered


def _write_private(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_source(row: dict[str, Any]) -> None:
    extra = row.get("extra_info") or {}
    ground_truth = (row.get("reward_model") or {}).get("ground_truth") or {}
    required = {
        "explicit_semantic_reviewed": extra.get("explicit_semantic_reviewed") is True,
        # The legacy review applicator used the source Arrow schema, which
        # discarded this newly added nested field in already-produced files.
        # Accept absence, but never accept a contradictory decision.
        "semantic_review_decision": extra.get("semantic_review_decision")
        in (None, "approved_candidate"),
        "training_disabled": extra.get("training_allowed") is False,
        "promotion_disabled": extra.get("promotion_allowed") is False,
        "table_answer_type": ground_truth.get("answer_type") == "table",
        "expected_value_json": bool(ground_truth.get("expected_value_json")),
        "verification_sql": bool(ground_truth.get("verification_sql")),
    }
    missing = sorted(name for name, passed in required.items() if not passed)
    if missing:
        raise ValueError(
            "source candidate failed the approved, disabled table contract: "
            + ", ".join(missing)
        )


def assemble(
    sources: Sequence[tuple[str, Path]],
    *,
    canonical_path: Path,
    schedule_path: Path,
    safe_summary_path: Path,
    seed: str = "20260818-qwen38-train70-v1",
) -> dict[str, Any]:
    source_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    schemas: list[pa.Schema] = []
    for host, path in sources:
        if host not in EXPECTED_SOURCE_COUNTS:
            raise ValueError(f"unsupported source host: {host}")
        table = pq.read_table(path.resolve(strict=True))
        schemas.append(table.schema)
        rows = table.to_pylist()
        source_counts[host] += len(rows)
        for row in rows:
            _validate_source(row)
            copied = deepcopy(row)
            copied["extra_info"]["qwen38_source_host"] = host
            source_rows.append(copied)
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("source Parquet schemas differ")
    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise ValueError(f"source counts mismatch: {dict(source_counts)}")
    ids = [identity(row) for row in source_rows]
    if len(source_rows) != 70 or len(set(ids)) != 70:
        raise ValueError("the authorized pool must contain 70 unique candidate identities")
    version_counts = Counter(_source_version(row) for row in source_rows)
    difficulty_counts = Counter(_difficulty(row) for row in source_rows)
    if dict(version_counts) != EXPECTED_VERSION_COUNTS:
        raise ValueError(f"source-version counts mismatch: {dict(version_counts)}")
    if dict(difficulty_counts) != EXPECTED_DIFFICULTY_COUNTS:
        raise ValueError(f"difficulty counts mismatch: {dict(difficulty_counts)}")

    canonical: list[dict[str, Any]] = []
    for position, row in enumerate(_stratified_order(source_rows, seed=f"{seed}:canonical"), 1):
        copied = deepcopy(row)
        copied["extra_info"].update(
            {
                "training_allowed": True,
                "promotion_allowed": False,
                "owner_authorized_training": True,
                "qwen38_training_contract": CONTRACT,
                "strict_reward_contract": REWARD_CONTRACT,
                "qwen38_canonical_position": position,
            }
        )
        canonical.append(copied)

    schedule: list[dict[str, Any]] = []
    for exposure in (1, 2):
        for row in _stratified_order(deepcopy(canonical), seed=f"{seed}:exposure={exposure}"):
            row["extra_info"].update(
                {
                    "qwen38_exposure": exposure,
                    "qwen38_schedule_position": len(schedule) + 1,
                }
            )
            schedule.append(row)
    scheduled_counts = Counter(identity(row) for row in schedule)
    if len(schedule) != 140 or set(scheduled_counts.values()) != {2}:
        raise ValueError("schedule must expose each of the 70 tasks exactly twice")
    if [row["extra_info"]["qwen38_exposure"] for row in schedule] != [1] * 70 + [2] * 70:
        raise ValueError("schedule exposure order is not pass-major")

    _write_private(canonical_path, canonical)
    _write_private(schedule_path, schedule)
    persisted = pq.read_table(schedule_path).to_pylist()
    if Counter(identity(row) for row in persisted) != scheduled_counts:
        raise ValueError("persisted schedule validation failed")
    summary = {
        "contract": CONTRACT,
        "reward_contract": REWARD_CONTRACT,
        "canonical_tasks": 70,
        "exposures_per_task": 2,
        "schedule_groups": 140,
        "groups_per_optimizer_step": 2,
        "responses_per_group": 8,
        "optimizer_steps": 70,
        "source_counts": dict(sorted(source_counts.items())),
        "source_version_counts": dict(sorted(version_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "strict_baseline_variance_tasks": 20,
        "owner_authorized_all_70_despite_strict_baseline_variance": True,
        "legacy_review_schema_missing_decision_is_accepted": True,
        "canonical_sha256": file_sha256(canonical_path),
        "schedule_sha256": file_sha256(schedule_path),
        "shuffle": False,
        "schedule_order": "two_passes_difficulty_version_host_round_robin_with_seeded_hash",
        "training_allowed": True,
        "promotion_allowed": False,
        "sensitive_artifact_permissions": "0600",
        "contains_prompts_gold_sql_task_ids_environment_ids_or_server_paths": False,
    }
    _write_json(safe_summary_path, summary)
    return summary


def validate(canonical: Path, schedule: Path, summary_path: Path) -> dict[str, Any]:
    canonical_rows = pq.read_table(canonical.resolve(strict=True)).to_pylist()
    schedule_rows = pq.read_table(schedule.resolve(strict=True)).to_pylist()
    summary = json.loads(summary_path.resolve(strict=True).read_text(encoding="utf-8"))
    if summary.get("contract") != CONTRACT or summary.get("reward_contract") != REWARD_CONTRACT:
        raise ValueError("training summary contract mismatch")
    if len(canonical_rows) != 70 or len({identity(row) for row in canonical_rows}) != 70:
        raise ValueError("canonical pool is not 70 unique tasks")
    counts = Counter(identity(row) for row in schedule_rows)
    if len(schedule_rows) != 140 or set(counts) != {identity(row) for row in canonical_rows} or set(counts.values()) != {2}:
        raise ValueError("schedule is not an exact two-exposure expansion")
    if any((row.get("extra_info") or {}).get("training_allowed") is not True for row in schedule_rows):
        raise ValueError("schedule contains a training-disabled row")
    if any((row.get("extra_info") or {}).get("promotion_allowed") is not False for row in schedule_rows):
        raise ValueError("schedule unexpectedly enables promotion")
    if file_sha256(canonical) != summary.get("canonical_sha256") or file_sha256(schedule) != summary.get("schedule_sha256"):
        raise ValueError("training artifact SHA256 mismatch")
    return {"status": "passed", "contract": CONTRACT, "canonical_tasks": 70, "schedule_groups": 140, "optimizer_steps": 70}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source", action="append", nargs=2, metavar=("HOST", "PARQUET"), required=True)
    build.add_argument("--canonical", type=Path, required=True)
    build.add_argument("--schedule", type=Path, required=True)
    build.add_argument("--safe-summary", type=Path, required=True)
    build.add_argument("--seed", default="20260818-qwen38-train70-v1")
    check = sub.add_parser("validate")
    check.add_argument("--canonical", type=Path, required=True)
    check.add_argument("--schedule", type=Path, required=True)
    check.add_argument("--safe-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        result = assemble(
            [(host, Path(path)) for host, path in args.source],
            canonical_path=args.canonical,
            schedule_path=args.schedule,
            safe_summary_path=args.safe_summary,
            seed=args.seed,
        )
    else:
        result = validate(args.canonical, args.schedule, args.safe_summary)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
