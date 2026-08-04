#!/usr/bin/env python3
"""Losslessly export boss sandbox task JSONLs into one identity manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_boss_aligned_dataset import canonical_hash, write_jsonl


TASK_FILES = (
    ("dwh", "dwh_tasks.jsonl", "natural_language_instruction"),
    ("kb", "kb_tasks.jsonl", "instruction"),
    ("hybrid", "hybrid_tasks.jsonl", "instruction"),
)


def export_tasks(sandbox_root: Path, version: str) -> list[dict]:
    base = sandbox_root / "sft" / version
    result = []
    seen = set()
    for task_type, filename, instruction_field in TASK_FILES:
        path = base / filename
        with path.open(encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                source = json.loads(line)
                task_id = str(source.get("task_id") or "").strip()
                if not task_id or task_id in seen:
                    raise ValueError(f"missing or duplicate task_id in {path}:{row_number}")
                seen.add(task_id)
                result.append(
                    {
                        **source,
                        "task_id": task_id,
                        "v": version,
                        "type": task_type,
                        "source_task_file": str(path),
                        "source_task_row": row_number,
                        "source_task_sha256": canonical_hash(source),
                        "source_instruction_field": instruction_field,
                    }
                )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox-root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = export_tasks(args.sandbox_root, args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
