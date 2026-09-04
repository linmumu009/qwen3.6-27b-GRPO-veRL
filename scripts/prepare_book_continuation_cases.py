#!/usr/bin/env python3
"""Regenerate fixed private continuation cases and verify their safe hashes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.audit_book_memorization import build_cases_from_text, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--prefix", type=int, action="append", required=True)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--target-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--reference-safe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    text = args.text.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"verified": True, "prefixes": {}}
    for prefix in sorted(set(args.prefix)):
        cases = build_cases_from_text(
            text,
            sample_count=args.sample_count,
            prefix_tokens=prefix,
            target_tokens=args.target_tokens,
            seed=args.seed,
        )
        reference_path = args.reference_safe_dir / f"step120_p{prefix}_n{args.sample_count}.safe.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        expected_hashes = [str(value) for value in reference["case_hashes"]]
        actual_hashes = [
            __import__("hashlib").sha256((case.prefix + "\0" + case.target).encode("utf-8")).hexdigest()
            for case in cases
        ]
        if actual_hashes != expected_hashes:
            raise ValueError(f"regenerated case hashes differ for prefix {prefix}")
        output = args.output_dir / f"p{prefix}_n{len(cases)}.jsonl"
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for case in cases:
                handle.write(json.dumps({
                    "case_id": case.case_id,
                    "prefix": case.prefix,
                    "target": case.target,
                }, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
        os.chmod(output, 0o600)
        summary["prefixes"][str(prefix)] = {
            "cases": len(cases),
            "case_hashes_match_reference": True,
        }
    write_json(args.output_dir.parent.parent.parent / "safe" / "continuation" / "case_rebuild.safe.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
