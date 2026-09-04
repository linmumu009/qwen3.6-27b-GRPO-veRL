#!/usr/bin/env python3
"""Summarize private rewrite diagnostics without emitting exam content."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    stage_counts = Counter(str(row.get("stage")) for row in rows)
    verifier_values: dict[str, Counter[str]] = {
        key: Counter() for key in ("equivalent", "same_correct_answer", "question_only")
    }
    issue_count_distribution: Counter[int] = Counter()
    for row in rows:
        payload = row.get("verifier_payload")
        if not isinstance(payload, dict):
            continue
        for key in verifier_values:
            value: Any = payload.get(key)
            verifier_values[key][f"{type(value).__name__}:{value}"] += 1
        issues = payload.get("issues")
        issue_count_distribution[len(issues) if isinstance(issues, list) else -1] += 1
    result = {
        "private_content_included": False,
        "diagnostic_rows": len(rows),
        "stage_counts": dict(sorted(stage_counts.items())),
        "verifier_value_counts": {
            key: dict(sorted(counts.items())) for key, counts in verifier_values.items()
        },
        "verifier_issue_count_distribution": {
            str(key): value for key, value in sorted(issue_count_distribution.items())
        },
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
