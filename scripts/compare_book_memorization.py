"""Build a content-free paired comparison for book continuation audits.

Inputs contain private source passages and model outputs.  The output contains
only hashes, aggregate metrics, and recurrence counts, so it is safe to review
or commit without disclosing the source text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.audit_book_memorization import PROMPT_VERSION, normalize_tokens, write_json
except ModuleNotFoundError:  # pragma: no cover - supports a standalone remote copy
    from audit_book_memorization import PROMPT_VERSION, normalize_tokens, write_json


THRESHOLDS = (1, 3, 5, 10, 20)


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row["case_id"])
        if case_id in rows:
            raise ValueError(f"duplicate case_id {case_id!r} in {path}:{line_number}")
        rows[case_id] = row
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def parse_named_paths(values: list[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"expected PREFIX=PATH, got {value!r}")
        prefix = int(label)
        if prefix in parsed:
            raise ValueError(f"duplicate prefix length {prefix}")
        parsed[prefix] = Path(raw_path)
    return parsed


def _occurrence_count(haystack: list[str], needle: list[str]) -> int:
    if not needle or len(needle) > len(haystack):
        return 0
    first = needle[0]
    return sum(
        haystack[index : index + len(needle)] == needle
        for index, token in enumerate(haystack[: len(haystack) - len(needle) + 1])
        if token == first
    )


def compare_pair(
    baseline_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
    *,
    source_tokens: list[str],
) -> dict[str, Any]:
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("baseline and candidate case_id sets differ")

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for case_id in sorted(baseline_rows):
        baseline = baseline_rows[case_id]
        candidate = candidate_rows[case_id]
        if baseline["source_hash"] != candidate["source_hash"]:
            raise ValueError(f"source hash mismatch for {case_id}")
        if baseline.get("error") or candidate.get("error"):
            raise ValueError(f"failed request present for {case_id}")
        if baseline.get("prompt_version") != PROMPT_VERSION or candidate.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"prompt protocol mismatch for {case_id}")
        if not baseline.get("chat_template_disable_thinking") or not candidate.get("chat_template_disable_thinking"):
            raise ValueError(f"thinking template not disabled for {case_id}")
        pairs.append((baseline, candidate))

    baseline_prefix = [int(left["exact_prefix_tokens"]) for left, _ in pairs]
    candidate_prefix = [int(right["exact_prefix_tokens"]) for _, right in pairs]
    baseline_f1 = [float(left["token_f1"]) for left, _ in pairs]
    candidate_f1 = [float(right["token_f1"]) for _, right in pairs]
    hashes = [left["source_hash"] for left, _ in pairs]

    high_match: list[dict[str, Any]] = []
    for left, right in pairs:
        exact_tokens = max(int(left["exact_prefix_tokens"]), int(right["exact_prefix_tokens"]))
        if exact_tokens < 10:
            continue
        sequence = normalize_tokens(str(left["target"]))[:exact_tokens]
        high_match.append(
            {
                "case_id": left["case_id"],
                "source_hash": left["source_hash"],
                "matched_sequence_sha256": hashlib.sha256("\0".join(sequence).encode("utf-8")).hexdigest(),
                "baseline_exact_prefix_tokens": int(left["exact_prefix_tokens"]),
                "candidate_exact_prefix_tokens": int(right["exact_prefix_tokens"]),
                "matched_sequence_occurrences_in_source": _occurrence_count(source_tokens, sequence),
            }
        )

    exact_delta = [right - left for left, right in zip(baseline_prefix, candidate_prefix)]
    return {
        "cases": len(pairs),
        "paired_case_hashes_sha256": hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest(),
        "empty_predictions": {
            "baseline": sum(bool(left.get("empty_prediction")) for left, _ in pairs),
            "candidate": sum(bool(right.get("empty_prediction")) for _, right in pairs),
        },
        "mean_exact_prefix_tokens": {
            "baseline": round(statistics.fmean(baseline_prefix), 6),
            "candidate": round(statistics.fmean(candidate_prefix), 6),
            "delta": round(statistics.fmean(exact_delta), 6),
        },
        "max_exact_prefix_tokens": {
            "baseline": max(baseline_prefix),
            "candidate": max(candidate_prefix),
        },
        "mean_token_f1": {
            "baseline": round(statistics.fmean(baseline_f1), 6),
            "candidate": round(statistics.fmean(candidate_f1), 6),
            "delta": round(statistics.fmean(candidate_f1) - statistics.fmean(baseline_f1), 6),
        },
        "paired_exact_prefix_change": {
            "candidate_higher": sum(delta > 0 for delta in exact_delta),
            "candidate_lower": sum(delta < 0 for delta in exact_delta),
            "same": sum(delta == 0 for delta in exact_delta),
        },
        "thresholds": {
            str(threshold): {
                "baseline_rate": round(
                    sum(value >= threshold for value in baseline_prefix) / len(pairs), 6
                ),
                "candidate_rate": round(
                    sum(value >= threshold for value in candidate_prefix) / len(pairs), 6
                ),
                "candidate_only": sum(left < threshold <= right for left, right in zip(baseline_prefix, candidate_prefix)),
                "baseline_only": sum(right < threshold <= left for left, right in zip(baseline_prefix, candidate_prefix)),
            }
            for threshold in THRESHOLDS
        },
        "identical_prediction_rate": round(
            sum(str(left.get("prediction", "")) == str(right.get("prediction", "")) for left, right in pairs)
            / len(pairs),
            6,
        ),
        "high_match_diagnostics": high_match,
    }


def build_comparison(
    baseline_paths: dict[int, Path],
    candidate_paths: dict[int, Path],
    *,
    source_path: Path,
    baseline_model: str,
    candidate_model: str,
) -> dict[str, Any]:
    if set(baseline_paths) != set(candidate_paths):
        raise ValueError("baseline and candidate prefix configurations differ")
    source_bytes = source_path.read_bytes()
    source_tokens = normalize_tokens(source_bytes.decode("utf-8"))
    results = {
        str(prefix): compare_pair(
            load_rows(baseline_paths[prefix]),
            load_rows(candidate_paths[prefix]),
            source_tokens=source_tokens,
        )
        for prefix in sorted(baseline_paths)
    }
    return {
        "schema_version": 1,
        "audit_type": "paired_verbatim_continuation_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "source_content_included": False,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_normalized_tokens": len(source_tokens),
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "prefix_configurations": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", required=True, metavar="PREFIX=PRIVATE_JSONL")
    parser.add_argument("--candidate", action="append", required=True, metavar="PREFIX=PRIVATE_JSONL")
    parser.add_argument("--source", type=Path, required=True, help="Private source text used only for hashes/counts")
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = build_comparison(
        parse_named_paths(args.baseline),
        parse_named_paths(args.candidate),
        source_path=args.source,
        baseline_model=args.baseline_model,
        candidate_model=args.candidate_model,
    )
    write_json(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
