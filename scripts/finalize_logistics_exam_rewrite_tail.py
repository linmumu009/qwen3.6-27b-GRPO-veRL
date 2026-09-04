#!/usr/bin/env python3
"""Complete a validated rewrite artifact with explicit identity fallbacks."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.rewrite_logistics_exam_stems import (
    REWRITE_VERSION,
    normalize_space,
    read_jsonl,
    sha256_file,
    text_hash,
    write_json,
    write_jsonl_private,
)


IDENTITY_MODEL = "identity-fallback-by-construction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    return parser.parse_args()


def validate_partial_row(row: dict[str, Any], source: dict[str, Any]) -> None:
    item_hash = str(source["item_hash"])
    original = normalize_space(source["question"])
    rewritten = normalize_space(row.get("rewritten_question") or "")
    if str(row.get("item_hash") or "") != item_hash:
        raise ValueError(f"rewrite item hash mismatch: {item_hash}")
    if str(row.get("original_question_sha256") or "") != text_hash(original):
        raise ValueError(f"rewrite source hash mismatch: {item_hash}")
    if str(row.get("rewritten_question_sha256") or "") != text_hash(rewritten):
        raise ValueError(f"rewrite text hash mismatch: {item_hash}")
    if row.get("semantic_validation_passed") is not True:
        raise ValueError(f"rewrite semantic validation did not pass: {item_hash}")
    if not rewritten or rewritten.casefold() == original.casefold():
        raise ValueError(f"validated paraphrase is empty or unchanged: {item_hash}")


def identity_row(source: dict[str, Any]) -> dict[str, Any]:
    item_hash = str(source["item_hash"])
    original = normalize_space(source["question"])
    digest = text_hash(original)
    return {
        "rewrite_version": REWRITE_VERSION,
        "generation_backend": "deterministic_identity_fallback",
        "generation_model": IDENTITY_MODEL,
        "item_hash": item_hash,
        "original_question_sha256": digest,
        "rewritten_question_sha256": digest,
        "rewritten_question": original,
        "deterministic_validation_passed": False,
        "semantic_validation_passed": True,
        "semantic_validation_method": "identity_by_construction",
        "attempts": 0,
        "identity_fallback": True,
    }


def main() -> int:
    args = parse_args()
    for output in (args.private_output, args.safe_output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")
    actual_source_sha256 = sha256_file(args.source)
    if args.expected_source_sha256 and actual_source_sha256 != args.expected_source_sha256:
        raise ValueError("source SHA-256 does not match the frozen benchmark identity")

    source_rows = read_jsonl(args.source)
    partial_rows = read_jsonl(args.partial)
    source_by_hash = {str(row["item_hash"]): row for row in source_rows}
    if len(source_by_hash) != len(source_rows):
        raise ValueError("source item hashes are not unique")
    partial_by_hash: dict[str, dict[str, Any]] = {}
    for row in partial_rows:
        item_hash = str(row.get("item_hash") or "")
        if not item_hash or item_hash in partial_by_hash or item_hash not in source_by_hash:
            raise ValueError("partial rewrites contain an unknown or duplicate item hash")
        validate_partial_row(row, source_by_hash[item_hash])
        partial_by_hash[item_hash] = row

    ordered: list[dict[str, Any]] = []
    for source in source_rows:
        item_hash = str(source["item_hash"])
        ordered.append(partial_by_hash.get(item_hash) or identity_row(source))
    write_jsonl_private(args.private_output, ordered)

    identity_count = len(source_rows) - len(partial_rows)
    model_counts = Counter(str(row["generation_model"]) for row in ordered)
    attempt_counts = Counter(int(row.get("attempts") or 0) for row in ordered)
    safe_summary = {
        "schema_version": 1,
        "rewrite_version": REWRITE_VERSION,
        "status": "complete_with_identity_fallback",
        "private_content_included": False,
        "source_items": len(source_rows),
        "validated_paraphrases": len(partial_rows),
        "identity_fallback_items": identity_count,
        "paraphrase_coverage": len(partial_rows) / len(source_rows),
        "semantic_validation_passed": len(ordered),
        "generation_model_counts": dict(sorted(model_counts.items())),
        "attempt_distribution": {str(key): value for key, value in sorted(attempt_counts.items())},
        "source_sha256": actual_source_sha256,
        "partial_input_sha256": sha256_file(args.partial),
        "private_output_sha256": sha256_file(args.private_output),
    }
    write_json(args.safe_output, safe_summary)
    os.chmod(args.safe_output, 0o644)
    print(json.dumps(safe_summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
