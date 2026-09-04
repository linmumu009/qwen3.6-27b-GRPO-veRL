#!/usr/bin/env python3
"""Write a content-free structural audit of private logistics benchmark cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


REFERENTIAL_ANSWER_PATTERNS = (
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\ball of the above\b",
        r"\bnone of the above\b",
        r"\bboth (?:a|b|c|d|\d)\b",
        r"\b(?:a|b|c|d) and (?:a|b|c|d)\b",
        r"\b(?:all|none) (?:are|is) correct\b",
        r"\bthe above\b",
        r"以上(?:都|均|皆|全)",
        r"以上(?:说法|选项|答案)",
        r"(?:都|均|皆|全)(?:正确|错误|不正确)",
        r"(?:前|上述)[一二两三四五六七八九十\d]+项",
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError("source contains no cases")
    return rows


def is_referential_answer(text: str) -> bool:
    return any(pattern.search(text.strip()) for pattern in REFERENTIAL_ANSWER_PATTERNS)


def audit(rows: list[dict[str, Any]], source_path: Path) -> dict[str, Any]:
    required = {"dataset", "question_type", "question", "options", "expected", "item_hash"}
    missing_required = 0
    invalid_expected = 0
    empty_questions = 0
    empty_options = 0
    referential_answer_items = 0
    label_prefixed_correct_answer_items = 0
    question_with_cjk_items = 0
    correct_answer_text_hashes: list[str] = []

    for row in rows:
        if not required <= set(row):
            missing_required += 1
            continue
        question = str(row["question"]).strip()
        if re.search(r"[\u3400-\u9fff]", question):
            question_with_cjk_items += 1
        options = row["options"]
        expected = row["expected"]
        if not question:
            empty_questions += 1
        if not isinstance(options, list) or not options or any(not str(value).strip() for value in options):
            empty_options += 1
            continue
        if (
            not isinstance(expected, list)
            or not expected
            or any(isinstance(value, bool) or not isinstance(value, int) for value in expected)
            or any(value < 0 or value >= len(options) for value in expected)
        ):
            invalid_expected += 1
            continue
        answer_texts = [str(options[index]).strip() for index in sorted(set(expected))]
        if any(is_referential_answer(value) for value in answer_texts):
            referential_answer_items += 1
        if any(re.match(r"^(?:\[[A-Z0-9]+\]|[A-Z0-9]+[.):])\s+", value) for value in answer_texts):
            label_prefixed_correct_answer_items += 1
        canonical = json.dumps(answer_texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        correct_answer_text_hashes.append(hashlib.sha256(canonical).hexdigest())

    item_hashes = [str(row.get("item_hash", "")) for row in rows]
    dataset_counts = Counter(str(row.get("dataset")) for row in rows)
    question_type_counts = Counter(str(row.get("question_type")) for row in rows)
    option_count_distribution = Counter(
        len(row.get("options", [])) for row in rows if isinstance(row.get("options"), list)
    )
    expected_count_distribution = Counter(
        len(row.get("expected", [])) for row in rows if isinstance(row.get("expected"), list)
    )

    return {
        "schema_version": 1,
        "private_content_included": False,
        "source_sha256": sha256_file(source_path),
        "rows": len(rows),
        "fields": sorted(set().union(*(row.keys() for row in rows))),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "question_type_counts": dict(sorted(question_type_counts.items())),
        "option_count_distribution": {
            str(key): value for key, value in sorted(option_count_distribution.items())
        },
        "expected_count_distribution": {
            str(key): value for key, value in sorted(expected_count_distribution.items())
        },
        "unique_item_hashes": len(set(item_hashes)),
        "unique_correct_answer_text_hashes": len(set(correct_answer_text_hashes)),
        "missing_required_rows": missing_required,
        "empty_question_rows": empty_questions,
        "empty_option_rows": empty_options,
        "invalid_expected_rows": invalid_expected,
        "referential_correct_answer_items": referential_answer_items,
        "label_prefixed_correct_answer_items": label_prefixed_correct_answer_items,
        "question_with_cjk_items": question_with_cjk_items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(load_rows(args.source), args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
