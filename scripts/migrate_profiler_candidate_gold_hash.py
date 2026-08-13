#!/usr/bin/env python3
"""Repair legacy profiler candidate hashes without exposing sensitive rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.profile_boss_sandbox_catalog import canonical_hash


GOLD_KEYS = {"answer_type", "value", "verification_sql"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migrate(
    source: Path,
    output: Path,
    *,
    expected_rows: int,
    expected_repairs: int,
) -> dict:
    rows = []
    repairs = 0
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected object")
            instruction = str(row.get("instruction") or "")
            if canonical_hash(instruction) != row.get("instruction_sha256"):
                raise ValueError(f"line {line_number}: instruction hash mismatch")
            gold = row.get("gold")
            if not isinstance(gold, dict) or set(gold) != GOLD_KEYS:
                raise ValueError(f"line {line_number}: unexpected exported gold shape")
            repaired_hash = canonical_hash(gold)
            if row.get("gold_sha256") != repaired_hash:
                repairs += 1
                row["gold_sha256"] = repaired_hash
            rows.append(row)

    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(rows)}")
    if repairs != expected_repairs:
        raise ValueError(f"expected {expected_repairs} repairs, got {repairs}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(output)
    return {
        "contract": "boss-profiler-exported-gold-hash-migration-v1",
        "rows": len(rows),
        "repaired_rows": repairs,
        "source_sha256": file_sha256(source),
        "output_sha256": file_sha256(output),
        "contains_prompts_gold_sql_task_ids_or_server_paths": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-repairs", type=int, required=True)
    args = parser.parse_args()
    result = migrate(
        args.source,
        args.output,
        expected_rows=args.expected_rows,
        expected_repairs=args.expected_repairs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
