#!/usr/bin/env python3
"""Emit a safe structural audit of private P_verified samples.

This is intentionally not labelled a human semantic review.  It proves that
the sampled positives have successful answer-bearing SQL and an internally
consistent final parse without exporting private answers or SQL text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def attest(path: Path, output: Path, *, minimum_positives: int = 20) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    positives = [row for row in rows if float(row.get("process_verified", 0)) == 1.0]
    checks = []
    for row in positives:
        checklist = row.get("audit_checklist") or {}
        checks.append(
            bool(row.get("successful_sql_count", 0) > 0)
            and bool(row.get("answer_bearing_sql_count", 0) > 0)
            and float(row.get("last_answer_bearing_consistent", 0)) == 1.0
            and bool(checklist.get("verified_process_requires_answer_bearing_successful_sql"))
            and not bool(row.get("numeric_final_parse_ambiguous", 0))
        )
    result = {
        "contract": "verified-process-private-structural-audit-v1",
        "audit_kind": "automatic_structural_not_human_semantic",
        "private_sample_rows": len(rows),
        "verified_positive_samples": len(positives),
        "structural_passes": sum(checks),
        "structural_failures": len(checks) - sum(checks),
        "minimum_positive_samples": minimum_positives,
        "structural_precision_proxy": round(sum(checks) / len(checks), 8) if checks else None,
        "human_precision_established": False,
        "status": "pass" if len(positives) >= minimum_positives and all(checks) else "fail",
        "process_bonus_promotion_allowed": False,
        "private_content_exported": False,
    }
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-positives", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(attest(args.packet, args.output, minimum_positives=args.minimum_positives), sort_keys=True))


if __name__ == "__main__":
    main()
