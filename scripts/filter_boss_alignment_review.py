#!/usr/bin/env python3
"""Apply evidence-backed task rejections without changing existing split assignments."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def parse_rejections(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--reject-task must use TASK_ID=REASON: {raw!r}")
        task_id, reason = (part.strip() for part in raw.split("=", 1))
        if not task_id or not reason:
            raise ValueError(f"--reject-task must use TASK_ID=REASON: {raw!r}")
        if task_id in result:
            raise ValueError(f"duplicate rejected task: {task_id}")
        result[task_id] = reason
    return result


def filter_review(
    rows: list[dict[str, Any]], rejections: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("alignment review row missing task_id")
        if task_id in by_task:
            raise ValueError(f"duplicate alignment review task_id: {task_id}")
        by_task[task_id] = row
    missing = sorted(set(rejections) - set(by_task))
    if missing:
        raise ValueError(f"rejected tasks not found in alignment review: {missing}")

    kept = [row for row in rows if row["task_id"] not in rejections]
    before = Counter(str(row.get("split") or "") for row in rows)
    after = Counter(str(row.get("split") or "") for row in kept)
    audit = {
        "contract": "boss-alignment-quality-rejection-v1",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "input_rows": len(rows),
        "output_rows": len(kept),
        "split_before": dict(sorted(before.items())),
        "split_after": dict(sorted(after.items())),
        "preserved_split_assignments": True,
        "rejections": [
            {
                "task_id": task_id,
                "reason": reason,
                "instruction_sha256": by_task[task_id].get("instruction_sha256"),
                "gold_sha256": by_task[task_id].get("gold_sha256"),
                "split": by_task[task_id].get("split"),
            }
            for task_id, reason in sorted(rejections.items())
        ],
    }
    return kept, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-review", type=Path, required=True)
    parser.add_argument("--output-review", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--reject-task", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rejections = parse_rejections(args.reject_task)
    if not rejections:
        raise ValueError("at least one --reject-task is required")
    kept, audit = filter_review(read_jsonl(args.input_review), rejections)
    args.output_review.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_review.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in kept),
        encoding="utf-8",
    )
    args.output_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
