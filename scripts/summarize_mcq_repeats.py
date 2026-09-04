#!/usr/bin/env python3
"""Build a per-item majority-vote result from repeated private MCQ evaluations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.evaluate_logistics_knowledge import build_safe_result, load_private_rows, sha256_bytes, write_private_rows


def majority_rows(repeats: list[dict[str, dict[str, Any]]]) -> tuple[list[dict[str, Any]], int]:
    if len(repeats) < 3 or len(repeats) % 2 == 0:
        raise ValueError("an odd number of at least three repeats is required")
    hashes = set(repeats[0])
    if not hashes or any(set(rows) != hashes for rows in repeats[1:]):
        raise ValueError("repeat item hash sets differ or are empty")
    output: list[dict[str, Any]] = []
    no_majority = 0
    for item_hash in sorted(hashes):
        rows = [repeat[item_hash] for repeat in repeats]
        immutable = ("dataset", "category", "question_type", "question", "options", "expected")
        if any(any(row[key] != rows[0][key] for key in immutable) for row in rows[1:]):
            raise ValueError(f"repeat content differs for {item_hash}")
        votes = Counter(tuple(int(value) for value in row.get("parsed", [])) for row in rows if row.get("parse_ok"))
        winner: tuple[int, ...] = ()
        if votes:
            winner, count = votes.most_common(1)[0]
            if count <= len(repeats) // 2:
                winner = ()
        if not winner:
            no_majority += 1
        expected = tuple(int(value) for value in rows[0]["expected"])
        output.append(
            {
                **{key: rows[0][key] for key in (
                    "prompt_version", "chat_template_disable_thinking", "item_hash", "dataset", "source_id",
                    "category", "question_type", "question", "options", "expected",
                )},
                "prediction": json.dumps({"answers": list(winner)}, separators=(",", ":")) if winner else "",
                "reasoning": "",
                "parsed": list(winner),
                "parse_ok": bool(winner),
                "correct": bool(winner) and winner == expected,
                "elapsed_sec": 0.0,
                "usage": None,
                "error": None,
            }
        )
    return output, no_majority


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", action="append", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    args = parser.parse_args()
    repeat_maps = [load_private_rows(path) for path in args.repeat]
    rows, no_majority = majority_rows(repeat_maps)
    write_private_rows(args.private_output, rows)
    result = build_safe_result(
        rows,
        model=args.model_label,
        endpoint_label="m05-offline-vllm-three-repeat-item-majority",
        concurrency=64,
        input_hashes={"cases_jsonl": sha256_bytes(args.cases.read_bytes())},
        elapsed_sec=0.0,
        chat_template_disable_thinking=True,
    )
    result["aggregation"] = {
        "method": "per_item_strict_majority_of_parsed_answers",
        "repeats": len(args.repeat),
        "items_without_majority": no_majority,
        "accuracy_by_repeat": [
            round(sum(bool(row["correct"]) for row in repeat.values()) / len(repeat), 6) for repeat in repeat_maps
        ],
    }
    args.safe_output.parent.mkdir(parents=True, exist_ok=True)
    args.safe_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": args.model_label,
        "items": len(rows),
        "majority_accuracy": result["accuracy"],
        **result["aggregation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
