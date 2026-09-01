from scripts.audit_book_memorization import (
    aggregate,
    build_cases_from_text,
    longest_exact_prefix,
    token_f1,
)


def test_longest_exact_prefix_is_token_normalized() -> None:
    assert longest_exact_prefix("Order picking, packing and dispatch", "Order picking, packing differs") == 4


def test_token_f1_handles_partial_overlap() -> None:
    score = token_f1("alpha beta gamma", "alpha beta delta")
    assert round(score, 6) == 0.666667


def test_build_cases_is_deterministic_and_bounded() -> None:
    text = " ".join(f"token{index}" for index in range(200))
    first = build_cases_from_text(text, sample_count=8, prefix_tokens=12, target_tokens=6, seed=7)
    second = build_cases_from_text(text, sample_count=8, prefix_tokens=12, target_tokens=6, seed=7)
    assert first == second
    assert len(first) == 8
    assert all(len(case.case_id) > 8 for case in first)


def test_aggregate_excludes_raw_content() -> None:
    rows = [
        {
            "case_id": "c1",
            "source_hash": "a" * 64,
            "prediction": "private prediction",
            "target": "private target",
            "exact_prefix_tokens": 5,
            "target_tokens": 10,
            "token_f1": 0.5,
            "error": None,
        }
    ]
    summary = aggregate(rows, model="qwen3.6-27b", source_name="sample", concurrency=64)
    serialized = str(summary)
    assert "private prediction" not in serialized
    assert "private target" not in serialized
    assert summary["exact_prefix_rate"]["5"] == 1.0
    assert summary["concurrency"] == 64
