#!/usr/bin/env python3
"""Validate an ``npu-smi info`` process table without exposing process details."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


PROCESS_ROW = re.compile(
    r"^\|\s*(?P<npu>\d+)\s+(?P<chip>\d+)\s+\|\s*(?P<pid>\d+)\s+\|\s*"
    r"(?P<name>[A-Za-z0-9_][^|]*)\|",
    re.MULTILINE,
)


def summarize_npu_process_table(text: str, *, host_label: str) -> dict[str, object]:
    """Return only table validity and process count; never return process identities."""

    if not text.strip():
        raise ValueError("npu-smi output is empty")
    if "Process id" not in text or "Process name" not in text:
        raise ValueError("npu-smi process table header is missing")
    rows = list(PROCESS_ROW.finditer(text))
    return {
        "schema_version": "qwen38-prefix-npu-process-gate-v1",
        "host": host_label,
        "table_valid": True,
        "process_count": len(rows),
        "idle": not rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = summarize_npu_process_table(sys.stdin.read(), host_label=args.host_label)
    except ValueError as exc:
        summary = {
            "schema_version": "qwen38-prefix-npu-process-gate-v1",
            "host": args.host_label,
            "table_valid": False,
            "process_count": None,
            "idle": False,
            "error": str(exc),
        }
        args.output.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
        args.output.chmod(0o600)
        return 2
    args.output.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
