#!/usr/bin/env python3
"""Select the fixed force-final sentinel tasks from the formal validation Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_TASK_IDS = (
    "task_000070",
    "task_000080",
    "task_000133",
    "task_000196",
    "task_000048",
    "task_000269",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_id(row: dict[str, Any]) -> str:
    return str(((row.get("reward_model") or {}).get("ground_truth") or {}).get("task_id") or "")


def select_rows(rows: list[dict[str, Any]], requested: tuple[str, ...]) -> list[dict[str, Any]]:
    indexed = {task_id(row): row for row in rows}
    missing = [value for value in requested if value not in indexed]
    if missing:
        raise ValueError(f"sentinel tasks missing from validation set: {missing}")
    return [indexed[value] for value in requested]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    args = parser.parse_args()

    from datasets import Dataset

    source = Dataset.from_parquet(str(args.input))
    requested = tuple(args.task_ids or DEFAULT_TASK_IDS)
    selected_rows = select_rows(source.to_list(), requested)
    selected = Dataset.from_list(selected_rows, features=source.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(str(args.output))

    roles = {
        "known_incomplete": list(requested[:4]),
        "completion_guardrail": list(requested[4:]),
    }
    answer_types = {
        value: str(((row.get("reward_model") or {}).get("ground_truth") or {}).get("answer_type") or "")
        for value, row in zip(requested, selected_rows, strict=True)
    }
    manifest = {
        "contract": "step120-force-final-sentinel6",
        "source": str(args.input),
        "source_sha256": file_sha256(args.input),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "rows": len(selected_rows),
        "task_ids": list(requested),
        "roles": roles,
        "answer_types": answer_types,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
