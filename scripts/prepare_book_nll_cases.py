#!/usr/bin/env python3
"""Prepare private, fixed token windows for paired book NLL evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable


CONTRACT = "book-fixed-passage-nll-cases-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_windows(
    records: Iterable[list[int]], *, window_tokens: int, case_count: int, seed: int
) -> list[dict[str, object]]:
    if window_tokens < 2:
        raise ValueError("window_tokens must be at least 2")
    if case_count < 1:
        raise ValueError("case_count must be positive")
    candidates: list[tuple[int, int, list[int]]] = []
    for record_index, token_ids in enumerate(records):
        for start in range(0, len(token_ids) - window_tokens + 1, window_tokens):
            candidates.append((record_index, start, token_ids[start : start + window_tokens]))
    if len(candidates) < case_count:
        raise ValueError(f"only {len(candidates)} non-overlapping windows are available")
    selected = random.Random(seed).sample(candidates, case_count)
    rows: list[dict[str, object]] = []
    for record_index, start, token_ids in sorted(selected, key=lambda value: (value[0], value[1])):
        encoded = ",".join(str(value) for value in token_ids).encode("ascii")
        digest = hashlib.sha256(encoded).hexdigest()
        rows.append(
            {
                "case_id": f"r{record_index:04d}-o{start:06d}-{digest[:12]}",
                "token_ids_sha256": digest,
                "token_ids": token_ids,
            }
        )
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--window-tokens", type=int, default=512)
    parser.add_argument("--case-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    table = pq.read_table(args.parquet, columns=[args.text_key])
    texts = [str(value) for value in table.column(args.text_key).to_pylist()]
    records = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    rows = build_windows(
        records,
        window_tokens=args.window_tokens,
        case_count=args.case_count,
        seed=args.seed,
    )
    write_json(args.private_output, {"contract": CONTRACT, "cases": rows})
    try:
        os.chmod(args.private_output, 0o600)
    except OSError:
        pass
    safe = {
        "schema_version": 1,
        "contract": CONTRACT,
        "source_content_included": False,
        "token_ids_included": False,
        "source_parquet_sha256": sha256_file(args.parquet),
        "tokenizer_json_sha256": sha256_file(args.model / "tokenizer.json"),
        "source_records": len(records),
        "available_nonoverlapping_windows": sum(len(value) // args.window_tokens for value in records),
        "cases": len(rows),
        "window_tokens": args.window_tokens,
        "scored_tokens_per_case": args.window_tokens - 1,
        "seed": args.seed,
        "case_hashes": [str(row["token_ids_sha256"]) for row in rows],
    }
    write_json(args.safe_output, safe)
    print(json.dumps({key: value for key, value in safe.items() if key != "case_hashes"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
