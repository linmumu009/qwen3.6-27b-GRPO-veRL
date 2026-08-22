#!/usr/bin/env python3
"""Record a safe human attestation for a private shadow-audit packet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def attest(packet: Path, output: Path, *, reviewer: str, reviewed_at: str) -> dict:
    rows = read_jsonl(packet)
    if len(rows) != 16:
        raise ValueError("manual audit packet must contain exactly 16 samples")
    identities = [row["trajectory_identity_sha256"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("manual audit packet contains duplicate samples")
    checklist_failures = [
        identity
        for identity, row in zip(identities, rows, strict=True)
        if not all(bool(value) for value in row.get("audit_checklist", {}).values())
    ]
    strata = {
        f"{answer_type}|{correctness}": sum(
            row["answer_type"] == answer_type and int(row["correctness"]) == correctness
            for row in rows
        )
        for answer_type in ("numeric", "table")
        for correctness in (0, 1)
    }
    if any(count == 0 for count in strata.values()):
        raise ValueError("manual audit packet does not cover all answer/correctness strata")
    result = {
        "contract": "qwen38-trajectory-process-shadow-manual-audit-v1",
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "sample_count": len(rows),
        "sample_identity_sha256": identities,
        "stratum_counts": strata,
        "checklist_failure_count": len(checklist_failures),
        "checklist_failures": checklist_failures,
        "status": "pass" if not checklist_failures else "fail",
        "review_scope": (
            "redacted structural evidence: formula, tool-event presence, successful/matching SQL "
            "counts, required table/field coverage, hard gates, and process components"
        ),
        "sensitive_trajectory_content_exported": False,
        "formal_training_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at", required=True)
    args = parser.parse_args()
    result = attest(
        args.packet,
        args.output,
        reviewer=args.reviewer,
        reviewed_at=args.reviewed_at,
    )
    print(json.dumps({"status": result["status"], "sample_count": result["sample_count"]}))


if __name__ == "__main__":
    main()
