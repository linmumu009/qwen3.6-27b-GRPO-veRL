from dataclasses import asdict
import json

import pytest

from scripts.evaluate_logistics_knowledge import _make_item, parse_answers
from scripts.extract_logistics_improved_pairs import improved_pairs, verified_predictions


def item(number, *, dataset="LogistikaBench", expected=(1, 2)):
    return _make_item(dataset=dataset, source_id=str(number), category="unit",
        question_type="multiple_choice", question=f"Question {number}",
        options=["First option", "完整的第二项", "Third option"], expected=expected)


def prediction(case, answer):
    parsed, valid = parse_answers(answer, len(case.options))
    return {**asdict(case), "options": list(case.options), "expected": list(case.expected),
        "prediction": answer, "reasoning": "", "parsed": list(parsed),
        "parse_ok": valid, "correct": valid and parsed == case.expected, "error": ""}


def test_filter_uses_paired_ids_and_resolves_multiselect_full_text():
    cases = [item(1), item(2), item(3, dataset="SC-bench-knowledge"), item(4)]
    native_answers = ['{"answers":[0]}', '{"answers":[1,2]}', 'invalid', '{"answers":[0]}']
    best_answers = ['{"answers":[1,2]}', '{"answers":[1,2]}', '{"answers":[2,1]}', '{"answers":[0]}']
    native = {c.item_hash: prediction(c, a) for c, a in zip(cases, native_answers)}
    best = {c.item_hash: prediction(c, a) for c, a in zip(cases, best_answers)}
    # Input dictionary order must not change item alignment.
    pairs = improved_pairs(cases, dict(reversed(list(native.items()))), best)
    assert [p["pair_id"] for p in pairs] == ["LOG-0001", "SC-0001"]
    assert [p["source_id"] for p in pairs] == ["1", "3"]
    assert pairs[0]["native"]["answer_texts"] == ["First option"]
    assert pairs[0]["best"]["answer_texts"] == ["完整的第二项", "Third option"]
    assert pairs[0]["gold_texts"] == pairs[0]["best"]["answer_texts"]
    assert len(pairs[0]["options"]) == 3
    assert pairs[1]["native"]["prediction"] == "invalid"
    assert pairs[1]["native"]["answer_texts"] == []


@pytest.mark.parametrize("fault", ["duplicate", "missing", "question", "source_id", "parsed", "correct", "api_error"])
def test_rejects_misalignment_or_inconsistent_saved_scoring(tmp_path, fault):
    cases = [item(1), item(2)]
    rows = [prediction(c, '{"answers":[1,2]}') for c in cases]
    if fault == "duplicate":
        rows.append(rows[0])
    elif fault == "missing":
        rows.pop()
    elif fault in ("question", "source_id"):
        rows[0][fault] = "changed"
    elif fault == "parsed":
        rows[0]["parsed"] = [0]
    elif fault == "correct":
        rows[0]["correct"] = False
    else:
        rows[0]["error"] = "request failed"
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        verified_predictions(path, cases)


def test_reads_utf8_predictions_and_preserves_actual_response(tmp_path):
    case = item(1)
    row = prediction(case, '{"answers":[0]}')
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    assert verified_predictions(path, [case])[case.item_hash] == row
