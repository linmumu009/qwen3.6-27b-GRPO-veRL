from types import SimpleNamespace

from scripts.audit_book_memorization import (
    ContinuationCase,
    _call_one,
    aggregate,
    build_cases_from_text,
    is_loopback_url,
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


def test_build_cases_are_stratified_and_non_overlapping() -> None:
    text = " ".join(f"token{index}" for index in range(500))
    cases = build_cases_from_text(text, sample_count=20, prefix_tokens=12, target_tokens=8, seed=11)
    spans = []
    for case in cases:
        first_token = case.prefix.split()[0]
        start = int(first_token.removeprefix("token"))
        spans.append((start, start + 20))
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


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


def test_call_one_disables_qwen_chat_template_thinking() -> None:
    captured = {}

    class Completions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="gamma delta", reasoning_content="")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    row = _call_one(
        client,
        "local-model",
        ContinuationCase("c1", "alpha beta", "gamma delta"),
        16,
        0,
        chat_template_disable_thinking=True,
    )

    assert captured["extra_body"] == {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert row["exact_prefix_tokens"] == 2
    assert row["empty_prediction"] is False
    assert row["reasoning_only"] is False


def test_aggregate_reports_empty_and_reasoning_only_without_raw_text() -> None:
    rows = [
        {
            "case_id": "c1",
            "source_hash": "b" * 64,
            "prediction": "",
            "reasoning": "private reasoning",
            "exact_prefix_tokens": 0,
            "token_f1": 0.0,
            "empty_prediction": True,
            "reasoning_only": True,
            "error": None,
        }
    ]
    summary = aggregate(
        rows,
        model="local-model",
        source_name="book",
        concurrency=64,
        prefix_tokens=64,
        target_tokens=32,
        seed=20260903,
        chat_template_disable_thinking=True,
    )

    assert summary["empty_predictions"] == 1
    assert summary["reasoning_only_responses"] == 1
    assert summary["prefix_tokens"] == 64
    assert "private reasoning" not in str(summary)


def test_loopback_url_allows_local_empty_api_key() -> None:
    assert is_loopback_url("http://127.0.0.1:8083/v1")
    assert is_loopback_url("http://localhost:8000/v1")
    assert not is_loopback_url("https://dashscope.aliyuncs.com/compatible-mode/v1")
