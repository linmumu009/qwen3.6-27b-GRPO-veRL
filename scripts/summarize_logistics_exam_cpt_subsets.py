#!/usr/bin/env python3
"""Summarize baseline/direct/rewritten majority results by rewrite coverage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from scripts.evaluate_logistics_knowledge import load_private_rows, sha256_bytes
from scripts.rewrite_logistics_exam_stems import read_jsonl


def mcnemar_exact(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(improved, regressed) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_summary(
    baseline: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired rows must be non-empty and equal length")
    improved = sum(not bool(left["correct"]) and bool(right["correct"]) for left, right in zip(baseline, candidate))
    regressed = sum(bool(left["correct"]) and not bool(right["correct"]) for left, right in zip(baseline, candidate))
    baseline_correct = sum(bool(row["correct"]) for row in baseline)
    candidate_correct = sum(bool(row["correct"]) for row in candidate)
    return {
        "items": len(baseline),
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "baseline_accuracy": round(baseline_correct / len(baseline), 6),
        "candidate_accuracy": round(candidate_correct / len(candidate), 6),
        "delta_accuracy_points": round((candidate_correct - baseline_correct) * 100 / len(baseline), 4),
        "improved_0_to_1": improved,
        "regressed_1_to_0": regressed,
        "net_correct": candidate_correct - baseline_correct,
        "mcnemar_exact_pvalue": round(mcnemar_exact(improved, regressed), 12),
    }


def subset_summary(
    names: Sequence[str],
    result_maps: Sequence[dict[str, dict[str, Any]]],
    item_hashes: Sequence[str],
) -> dict[str, Any]:
    ordered = sorted(item_hashes)
    rows_by_model = [[result[item_hash] for item_hash in ordered] for result in result_maps]
    output: dict[str, Any] = {
        "items": len(ordered),
        "models": {
            name: {
                "correct": sum(bool(row["correct"]) for row in rows),
                "accuracy": round(sum(bool(row["correct"]) for row in rows) / len(rows), 6),
            }
            for name, rows in zip(names, rows_by_model)
        },
        "pairwise": [],
    }
    for left_index in range(len(names)):
        for right_index in range(left_index + 1, len(names)):
            output["pairwise"].append(
                {
                    "baseline": names[left_index],
                    "candidate": names[right_index],
                    **paired_summary(rows_by_model[left_index], rows_by_model[right_index]),
                }
            )
    return output


def parse_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("result must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("result must be LABEL=PATH")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", type=parse_result, required=True)
    parser.add_argument("--rewrites", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.result) < 2:
        raise ValueError("at least two model results are required")
    names = [name for name, _path in args.result]
    if len(set(names)) != len(names):
        raise ValueError("result labels must be unique")
    paths = [path for _name, path in args.result]
    result_maps = [load_private_rows(path) for path in paths]
    hashes = set(result_maps[0])
    if not hashes or any(set(result) != hashes for result in result_maps[1:]):
        raise ValueError("result item hash sets differ or are empty")
    for item_hash in hashes:
        immutable = ("dataset", "category", "question_type", "expected")
        if any(
            any(result[item_hash][key] != result_maps[0][item_hash][key] for key in immutable)
            for result in result_maps[1:]
        ):
            raise ValueError(f"result metadata differs for {item_hash}")

    rewrite_rows = read_jsonl(args.rewrites)
    rewrite_map = {str(row["item_hash"]): row for row in rewrite_rows}
    if len(rewrite_map) != len(rewrite_rows) or set(rewrite_map) != hashes:
        raise ValueError("rewrite coverage does not exactly match result items")
    paraphrased = sorted(item_hash for item_hash, row in rewrite_map.items() if not row.get("identity_fallback"))
    identity = sorted(item_hash for item_hash, row in rewrite_map.items() if row.get("identity_fallback"))

    summary = {
        "schema_version": 1,
        "status": "complete",
        "private_content_included": False,
        "rewrite_coverage": {
            "all_items": len(hashes),
            "validated_paraphrases": len(paraphrased),
            "identity_fallback_items": len(identity),
            "paraphrase_coverage": len(paraphrased) / len(hashes),
        },
        "subsets": {
            "all": subset_summary(names, result_maps, sorted(hashes)),
            "validated_paraphrases": subset_summary(names, result_maps, paraphrased),
            "identity_fallback": subset_summary(names, result_maps, identity),
        },
        "input_sha256": {
            **{f"result:{name}": sha256_bytes(path.read_bytes()) for name, path in args.result},
            "rewrites": sha256_bytes(args.rewrites.read_bytes()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
