#!/usr/bin/env python3
"""Create and validate a deterministic, difficulty-stratified GRPO split.

The source and outputs are sensitive Parquets.  Only the safe JSON summary may
be committed or logged.  The frozen source is never modified: explicit owner
authorization is recorded on the derived train rows, while the held-out rows
remain training-disabled and evaluation-only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.freeze_grpo_candidate_pool import (
    answer_type,
    difficulty,
    explicitly_enabled,
    file_sha256,
    instruction_identity,
    last_user_message,
    verifier_material,
)


CONTRACT = "llin-grpo-candidate-difficulty-split-v1"


def _difficulty_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _stable_key(row: dict[str, Any], seed: str) -> str:
    identity = instruction_identity(row)
    return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()


def _counts(rows: Iterable[dict[str, Any]], field) -> dict[str, int]:
    return dict(sorted(Counter(field(row) for row in rows).items()))


def allocate_train_counts(
    difficulty_counts: dict[str, int], train_rows: int
) -> dict[str, int]:
    """Allocate train rows with Hamilton largest remainders by difficulty."""

    total = sum(difficulty_counts.values())
    if total <= 0 or not 0 < train_rows < total:
        raise ValueError(f"invalid split size: train={train_rows}, total={total}")
    exact = {
        label: count * train_rows / total for label, count in difficulty_counts.items()
    }
    allocation = {label: int(value) for label, value in exact.items()}
    remaining = train_rows - sum(allocation.values())
    order = sorted(
        difficulty_counts,
        key=lambda label: (
            -(exact[label] - allocation[label]),
            _difficulty_sort_key(label),
        ),
    )
    for label in order[:remaining]:
        allocation[label] += 1
    if sum(allocation.values()) != train_rows:
        raise AssertionError("difficulty allocation does not sum to train size")
    if any(allocation[label] > difficulty_counts[label] for label in allocation):
        raise AssertionError("difficulty allocation exceeds source count")
    return dict(sorted(allocation.items(), key=lambda item: _difficulty_sort_key(item[0])))


def _copy_for_partition(
    row: dict[str, Any],
    *,
    partition: str,
    authorization_record: str,
) -> dict[str, Any]:
    copied = dict(row)
    extra = dict(copied.get("extra_info") or {})
    if partition == "train":
        extra.update(
            {
                "candidate_split": "train",
                "candidate_split_contract": CONTRACT,
                "owner_authorized_training": True,
                "owner_authorization_record": authorization_record,
                "source_training_allowed": False,
                "training_allowed": True,
                "promotion_allowed": False,
                "evaluation_only": False,
            }
        )
        training_allowed = True
    elif partition == "test":
        extra.update(
            {
                "candidate_split": "test",
                "candidate_split_contract": CONTRACT,
                "owner_authorized_training": False,
                "owner_authorization_record": authorization_record,
                "source_training_allowed": False,
                "training_allowed": False,
                "promotion_allowed": False,
                "evaluation_only": True,
            }
        )
        training_allowed = False
    else:
        raise ValueError(f"unsupported partition: {partition}")
    copied["extra_info"] = extra
    if "training_allowed" in copied:
        copied["training_allowed"] = training_allowed
    if "promotion_allowed" in copied:
        copied["promotion_allowed"] = False
    return copied


def _write_private_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _write_private_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _validate_source_rows(rows: list[dict[str, Any]], expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} source rows, found {len(rows)}")
    identities: set[str] = set()
    for row in rows:
        if not last_user_message(row) or verifier_material(row) is None:
            raise ValueError("source row is missing prompt or hidden verifier material")
        if explicitly_enabled(row, "training_allowed"):
            raise ValueError("frozen source unexpectedly enables training")
        if explicitly_enabled(row, "promotion_allowed"):
            raise ValueError("frozen source unexpectedly enables promotion")
        identity = instruction_identity(row)
        if identity in identities:
            raise ValueError("frozen source instruction identities are not unique")
        identities.add(identity)


def split_pool(
    source_path: Path,
    *,
    train_path: Path,
    test_path: Path,
    safe_summary_path: Path,
    private_environment_manifest_path: Path,
    expected_source_sha256: str,
    expected_rows: int,
    train_rows: int,
    seed: str,
    authorization_record: str,
    owner_authorized_training: bool,
) -> dict[str, Any]:
    if not owner_authorized_training:
        raise ValueError("explicit --owner-authorized-training is required")
    if not authorization_record.strip():
        raise ValueError("authorization record must be non-empty")
    source_path = source_path.resolve(strict=True)
    actual_source_sha256 = file_sha256(source_path)
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            f"source sha256 mismatch: {actual_source_sha256} != {expected_source_sha256}"
        )
    source_rows = pq.read_table(source_path).to_pylist()
    _validate_source_rows(source_rows, expected_rows)

    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_difficulty[difficulty(row)].append(row)
    source_difficulty_counts = {
        label: len(rows) for label, rows in by_difficulty.items()
    }
    train_targets = allocate_train_counts(source_difficulty_counts, train_rows)

    train_source_rows: list[dict[str, Any]] = []
    test_source_rows: list[dict[str, Any]] = []
    for label in sorted(by_difficulty, key=_difficulty_sort_key):
        ordered = sorted(by_difficulty[label], key=lambda row: _stable_key(row, seed))
        cutoff = train_targets[label]
        train_source_rows.extend(ordered[:cutoff])
        test_source_rows.extend(ordered[cutoff:])
    train_source_rows.sort(key=lambda row: _stable_key(row, f"{seed}:train-order"))
    test_source_rows.sort(key=lambda row: _stable_key(row, f"{seed}:test-order"))

    train = [
        _copy_for_partition(
            row, partition="train", authorization_record=authorization_record
        )
        for row in train_source_rows
    ]
    test = [
        _copy_for_partition(
            row, partition="test", authorization_record=authorization_record
        )
        for row in test_source_rows
    ]
    if len(train) != train_rows or len(test) != expected_rows - train_rows:
        raise AssertionError("partition row counts changed")
    train_identities = {instruction_identity(row) for row in train}
    test_identities = {instruction_identity(row) for row in test}
    if train_identities & test_identities:
        raise AssertionError("train and test identities overlap")
    if len(train_identities | test_identities) != expected_rows:
        raise AssertionError("train and test do not cover the source")

    _write_private_parquet(train_path, train)
    _write_private_parquet(test_path, test)
    environment_ids = sorted(
        {
            str((row.get("extra_info") or {}).get("environment_id") or "").strip()
            for row in source_rows
        }
    )
    if not environment_ids or any(not value for value in environment_ids):
        raise ValueError("every row must name a non-empty environment_id")
    _write_private_json(
        private_environment_manifest_path,
        {"contract": CONTRACT, "environment_ids": environment_ids},
    )

    persisted_train = pq.read_table(train_path).to_pylist()
    persisted_test = pq.read_table(test_path).to_pylist()
    _validate_partitions(persisted_train, persisted_test, expected_rows, train_rows)
    summary = {
        "contract": CONTRACT,
        "source_rows": expected_rows,
        "train_rows": len(train),
        "test_rows": len(test),
        "seed": seed,
        "source_sha256": actual_source_sha256,
        "train_sha256": file_sha256(train_path),
        "test_sha256": file_sha256(test_path),
        "unique_environment_count": len(environment_ids),
        "difficulty_counts": {
            "source": _counts(source_rows, difficulty),
            "train": _counts(train, difficulty),
            "test": _counts(test, difficulty),
        },
        "answer_type_counts": {
            "source": _counts(source_rows, answer_type),
            "train": _counts(train, answer_type),
            "test": _counts(test, answer_type),
        },
        "candidate_source_counts": {
            partition: _counts(
                rows,
                lambda row: str(
                    (row.get("extra_info") or {}).get("candidate_pool_source")
                    or "unknown"
                ),
            )
            for partition, rows in (("train", train), ("test", test))
        },
        "owner_authorized_training": True,
        "semantic_review_requirement_overridden_by_explicit_owner_request": True,
        "promotion_allowed": False,
        "test_evaluation_only": True,
        "sensitive_artifact_permissions": "0600",
        "contains_prompts_gold_sql_task_ids_environment_ids_or_server_paths": False,
    }
    safe_summary_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = safe_summary_path.with_name(
        safe_summary_path.name + f".tmp.{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(safe_summary_path)
    return summary


def _validate_partitions(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    expected_rows: int,
    expected_train_rows: int,
) -> None:
    if len(train) != expected_train_rows:
        raise ValueError(f"expected {expected_train_rows} train rows, found {len(train)}")
    if len(test) != expected_rows - expected_train_rows:
        raise ValueError(
            f"expected {expected_rows - expected_train_rows} test rows, found {len(test)}"
        )
    train_ids: set[str] = set()
    test_ids: set[str] = set()
    for partition, rows, identities in (
        ("train", train, train_ids),
        ("test", test, test_ids),
    ):
        for row in rows:
            identity = instruction_identity(row)
            if identity in identities:
                raise ValueError(f"{partition} identities are not unique")
            identities.add(identity)
            extra = row.get("extra_info") or {}
            if extra.get("candidate_split") != partition:
                raise ValueError(f"{partition} row has the wrong split marker")
            if explicitly_enabled(row, "promotion_allowed"):
                raise ValueError(f"{partition} row unexpectedly enables promotion")
            if partition == "train":
                if not explicitly_enabled(row, "training_allowed"):
                    raise ValueError("train row is not explicitly training-enabled")
                if extra.get("owner_authorized_training") is not True:
                    raise ValueError("train row lacks owner authorization")
            elif explicitly_enabled(row, "training_allowed"):
                raise ValueError("test row unexpectedly enables training")
            if partition == "test" and extra.get("evaluation_only") is not True:
                raise ValueError("test row is not marked evaluation-only")
    if train_ids & test_ids:
        raise ValueError("train and test identities overlap")
    if len(train_ids | test_ids) != expected_rows:
        raise ValueError("train and test do not cover the expected source rows")


def validate_split(
    *,
    train_path: Path,
    test_path: Path,
    safe_summary_path: Path,
    expected_rows: int,
    expected_train_rows: int,
    sandbox_root: Path | None = None,
) -> dict[str, Any]:
    train = pq.read_table(train_path.resolve(strict=True)).to_pylist()
    test = pq.read_table(test_path.resolve(strict=True)).to_pylist()
    _validate_partitions(train, test, expected_rows, expected_train_rows)
    summary = json.loads(safe_summary_path.resolve(strict=True).read_text(encoding="utf-8"))
    if summary.get("contract") != CONTRACT:
        raise ValueError("safe summary contract mismatch")
    if summary.get("train_sha256") != file_sha256(train_path):
        raise ValueError("train sha256 does not match safe summary")
    if summary.get("test_sha256") != file_sha256(test_path):
        raise ValueError("test sha256 does not match safe summary")
    environments = {
        str((row.get("extra_info") or {}).get("environment_id") or "").strip()
        for row in [*train, *test]
    }
    if any(not value for value in environments):
        raise ValueError("split contains an empty environment_id")
    missing_environment_count = 0
    if sandbox_root is not None:
        sandbox_root = sandbox_root.resolve(strict=True)
        missing_environment_count = sum(
            not (sandbox_root / environment_id).is_dir()
            for environment_id in environments
        )
        if missing_environment_count:
            raise FileNotFoundError(
                f"{missing_environment_count} split environments are missing"
            )
    return {
        "status": "passed",
        "contract": CONTRACT,
        "train_rows": len(train),
        "test_rows": len(test),
        "unique_environment_count": len(environments),
        "missing_environment_count": missing_environment_count,
        "train_sha256": file_sha256(train_path),
        "test_sha256": file_sha256(test_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    split = subparsers.add_parser("split")
    split.add_argument("--source", type=Path, required=True)
    split.add_argument("--train-output", type=Path, required=True)
    split.add_argument("--test-output", type=Path, required=True)
    split.add_argument("--safe-summary", type=Path, required=True)
    split.add_argument("--private-environment-manifest", type=Path, required=True)
    split.add_argument("--expected-source-sha256", required=True)
    split.add_argument("--expected-rows", type=int, default=161)
    split.add_argument("--train-rows", type=int, default=128)
    split.add_argument("--seed", default="20260817")
    split.add_argument("--authorization-record", required=True)
    split.add_argument("--owner-authorized-training", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--train", type=Path, required=True)
    validate.add_argument("--test", type=Path, required=True)
    validate.add_argument("--safe-summary", type=Path, required=True)
    validate.add_argument("--expected-rows", type=int, default=161)
    validate.add_argument("--expected-train-rows", type=int, default=128)
    validate.add_argument("--sandbox-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "split":
        result = split_pool(
            args.source,
            train_path=args.train_output,
            test_path=args.test_output,
            safe_summary_path=args.safe_summary,
            private_environment_manifest_path=args.private_environment_manifest,
            expected_source_sha256=args.expected_source_sha256,
            expected_rows=args.expected_rows,
            train_rows=args.train_rows,
            seed=args.seed,
            authorization_record=args.authorization_record,
            owner_authorized_training=args.owner_authorized_training,
        )
    else:
        result = validate_split(
            train_path=args.train,
            test_path=args.test,
            safe_summary_path=args.safe_summary,
            expected_rows=args.expected_rows,
            expected_train_rows=args.expected_train_rows,
            sandbox_root=args.sandbox_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
