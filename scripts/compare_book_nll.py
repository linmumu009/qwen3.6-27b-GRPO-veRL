#!/usr/bin/env python3
"""Build a safe paired comparison for fixed book-passage NLL results."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from scripts.prepare_book_nll_cases import write_json
from scripts.run_vllm_prompt_nll import RESULT_CONTRACT


def load_rows(path: Path) -> tuple[str, dict[str, dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != RESULT_CONTRACT:
        raise ValueError("result contract mismatch")
    rows = {str(row["case_id"]): row for row in payload["rows"]}
    if len(rows) != len(payload["rows"]):
        raise ValueError("duplicate case id")
    return str(payload["model_label"]), rows


def bootstrap_ci(values: list[float], *, seed: int, samples: int = 10000) -> tuple[float, float]:
    rng = random.Random(seed)
    size = len(values)
    means = [statistics.fmean(values[rng.randrange(size)] for _ in range(size)) for _ in range(samples)]
    means.sort()
    return means[int(samples * 0.025)], means[int(samples * 0.975)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.02)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_label, baseline = load_rows(args.baseline)
    candidate_label, candidate = load_rows(args.candidate)
    if set(baseline) != set(candidate):
        raise ValueError("paired case ids differ")
    ordered = sorted(baseline)
    for case_id in ordered:
        if baseline[case_id]["token_ids_sha256"] != candidate[case_id]["token_ids_sha256"]:
            raise ValueError(f"case hash mismatch: {case_id}")
        if baseline[case_id]["scored_tokens"] != candidate[case_id]["scored_tokens"]:
            raise ValueError(f"token count mismatch: {case_id}")
    baseline_nll = statistics.fmean(float(baseline[key]["nll"]) for key in ordered)
    candidate_nll = statistics.fmean(float(candidate[key]["nll"]) for key in ordered)
    deltas = [float(candidate[key]["nll"]) - float(baseline[key]["nll"]) for key in ordered]
    ci_low, ci_high = bootstrap_ci(deltas, seed=args.bootstrap_seed)
    relative = (baseline_nll - candidate_nll) / baseline_nll
    gate = relative >= args.minimum_relative_improvement and ci_high < 0
    result = {
        "schema_version": 1,
        "comparison_type": "paired_fixed_book_passage_nll",
        "baseline_model": baseline_label,
        "candidate_model": candidate_label,
        "source_content_included": False,
        "cases": len(ordered),
        "scored_tokens": sum(int(baseline[key]["scored_tokens"]) for key in ordered),
        "baseline_mean_nll": round(baseline_nll, 9),
        "candidate_mean_nll": round(candidate_nll, 9),
        "mean_paired_delta_candidate_minus_baseline": round(statistics.fmean(deltas), 9),
        "relative_nll_improvement": round(relative, 9),
        "perplexity": {
            "baseline": round(math.exp(baseline_nll), 9),
            "candidate": round(math.exp(candidate_nll), 9),
        },
        "paired_cases": {
            "candidate_lower_nll": sum(value < 0 for value in deltas),
            "candidate_higher_nll": sum(value > 0 for value in deltas),
            "equal": sum(value == 0 for value in deltas),
        },
        "paired_bootstrap_95pct_ci_delta": [round(ci_low, 9), round(ci_high, 9)],
        "uptake_gate": {
            "minimum_relative_nll_improvement": args.minimum_relative_improvement,
            "requires_ci_below_zero": True,
            "passed": gate,
        },
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
