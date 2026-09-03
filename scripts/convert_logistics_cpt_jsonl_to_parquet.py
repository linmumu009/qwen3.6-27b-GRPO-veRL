#!/usr/bin/env python3
"""Convert the private CPT JSONL to a mode-0600 Parquet file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_COLUMNS = (
    "text",
    "part",
    "chapter",
    "record_id",
    "source_line_start",
    "source_line_end",
    "text_sha256",
    "token_count",
    "source_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(input_path: Path, output_path: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = sorted(set(REQUIRED_COLUMNS) - set(record))
            if missing:
                raise ValueError(f"line {line_number} missing columns: {missing}")
            if hashlib.sha256(record["text"].encode("utf-8")).hexdigest() != record["text_sha256"]:
                raise ValueError(f"line {line_number} text hash mismatch")
            records.append({column: record[column] for column in REQUIRED_COLUMNS})

    if not records:
        raise ValueError("input JSONL contains no records")

    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    table = pa.Table.from_pylist(records)
    pq.write_table(table, temporary_path, compression="zstd")
    os.chmod(temporary_path, 0o600)
    os.replace(temporary_path, output_path)
    os.chmod(output_path, 0o600)
    return {
        "rows": len(records),
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert(args.input.resolve(), args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
