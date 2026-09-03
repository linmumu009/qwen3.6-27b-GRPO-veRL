import json
from pathlib import Path

import pytest

from scripts.audit_book_memorization import PROMPT_VERSION
from scripts.compare_book_memorization import build_comparison, compare_pair, parse_named_paths


ROOT = Path(__file__).resolve().parents[1]


def _row(case_id: str, source_hash: str, prediction: str, target: str, exact: int, f1: float) -> dict:
    return {
        "case_id": case_id,
        "source_hash": source_hash,
        "prompt_version": PROMPT_VERSION,
        "chat_template_disable_thinking": True,
        "prediction": prediction,
        "target": target,
        "exact_prefix_tokens": exact,
        "token_f1": f1,
        "empty_prediction": not bool(prediction),
        "error": None,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compare_pair_is_paired_and_content_free() -> None:
    digest = "a" * 64
    target = " ".join(f"word{i}" for i in range(20))
    baseline = {"c1": _row("c1", digest, target, target, 10, 0.5)}
    candidate = {"c1": _row("c1", digest, target + " extra", target, 12, 0.6)}

    result = compare_pair(baseline, candidate, source_tokens=(target + " " + target).split())

    assert result["paired_exact_prefix_change"]["candidate_higher"] == 1
    assert result["thresholds"]["10"]["baseline_rate"] == 1.0
    assert result["high_match_diagnostics"][0]["matched_sequence_occurrences_in_source"] == 2
    assert "word0" not in str(result)


def test_compare_pair_rejects_case_hash_mismatch() -> None:
    baseline = {"c1": _row("c1", "a" * 64, "x", "x", 1, 1.0)}
    candidate = {"c1": _row("c1", "b" * 64, "x", "x", 1, 1.0)}
    with pytest.raises(ValueError, match="source hash mismatch"):
        compare_pair(baseline, candidate, source_tokens=["x"])


def test_build_comparison_hashes_source_and_supports_multiple_prefixes(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("alpha beta gamma delta", encoding="utf-8")
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    row = _row("c1", "c" * 64, "gamma", "gamma delta", 1, 0.5)
    _write_rows(baseline, [row])
    _write_rows(candidate, [row])

    result = build_comparison(
        {32: baseline},
        {32: candidate},
        source_path=source,
        baseline_model="base",
        candidate_model="candidate",
    )

    assert result["source_content_included"] is False
    assert result["prefix_configurations"]["32"]["cases"] == 1
    assert len(result["source_sha256"]) == 64


def test_parse_named_paths_rejects_duplicate_prefix() -> None:
    with pytest.raises(ValueError, match="duplicate prefix"):
        parse_named_paths(["32=a.jsonl", "32=b.jsonl"])


def test_committed_safe_comparison_contract() -> None:
    path = ROOT / "docs" / "logistics_book_memorization_base_vs_step120_20260903.safe.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["source_content_included"] is False
    assert report["source_sha256"] == "b2b4ed0156dea4076f86894478c8db2643ee05f25fb8af6bdf4622ee70b93bd1"
    assert set(report["prefix_configurations"]) == {"32", "64", "128"}

    for result in report["prefix_configurations"].values():
        assert result["cases"] == 200
        assert result["empty_predictions"] == {"baseline": 0, "candidate": 0}
        assert result["max_exact_prefix_tokens"]["baseline"] < 20
        assert result["max_exact_prefix_tokens"]["candidate"] < 20
        assert result["thresholds"]["10"]["candidate_only"] == 0
        assert result["thresholds"]["10"]["baseline_only"] == 0
        assert all(
            item["baseline_exact_prefix_tokens"] == item["candidate_exact_prefix_tokens"]
            for item in result["high_match_diagnostics"]
        )
