import pytest

from scripts.summarize_mcq_repeats import majority_rows


def _row(parsed: list[int]) -> dict:
    return {
        "prompt_version": "v",
        "chat_template_disable_thinking": True,
        "item_hash": "h",
        "dataset": "d",
        "source_id": "s",
        "category": "c",
        "question_type": "single_choice",
        "question": "q",
        "options": ["a", "b", "c"],
        "expected": [1],
        "parsed": parsed,
        "parse_ok": bool(parsed),
        "correct": parsed == [1],
    }


def test_majority_rows_uses_per_item_answer_vote() -> None:
    repeats = [{"h": _row([1])}, {"h": _row([0])}, {"h": _row([1])}]
    rows, no_majority = majority_rows(repeats)
    assert rows[0]["parsed"] == [1]
    assert rows[0]["correct"] is True
    assert no_majority == 0


def test_majority_rows_rejects_even_repeat_count() -> None:
    with pytest.raises(ValueError, match="odd number"):
        majority_rows([{"h": _row([1])}, {"h": _row([1])}])
