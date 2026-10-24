import json
from pathlib import Path

import pytest
import yaml

from scripts.extract_opensource_improved_pairs import (
    BASE_TAG, TRAINED_TAG, compare_runs, confirm_aime_identity, read_jsonl,
)


def run(root, tag, correct, *, prompt="Question\u2028with embedded Unicode separator", finish="stop",
        error=None, gold="42", budget=8192):
    root.mkdir()
    pred = {"task_id": "case_0000", "benchmark": "GSM8K_test", "prompt": prompt,
            "response": "answer", "reasoning_content": "", "finish_reason": finish}
    if error:
        pred["error"] = error
    score = {"task_id": "case_0000", "golden_answer": gold, "correct": correct,
             "extracted_answer": "42" if correct else "41", "evidence": "saved score"}
    for name, row in [("predictions.jsonl", pred), ("scores.jsonl", score)]:
        (root / name).write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({"benchmark": "GSM8K_test", "total": 1,
        "correct": int(correct)}), encoding="utf-8")
    (root / "config_snapshot.yaml").write_text(yaml.safe_dump({
        "model": {"tag": tag, "path": "/model"},
        "benchmark": {"name": "GSM8K_test", "max_tokens": budget},
        "vllm": {"temperature": 0, "enable_thinking": False, "api_key": "do-not-export"},
    }), encoding="utf-8")
    return root


def compare(n, t):
    return compare_runs(n, t, "GSM8K_test", "nothink_8192")


def test_same_question_and_saved_answers_preserved_with_unicode_separator(tmp_path):
    n = run(tmp_path / "native", BASE_TAG, False)
    t = run(tmp_path / "trained", TRAINED_TAG, True)
    cases, audit = compare(n, t)
    assert len(cases) == 1
    assert "\u2028" in cases[0]["question"]
    assert cases[0]["native"]["scoring"]["extracted_answer"] == "41"
    assert cases[0]["trained"]["scoring"]["extracted_answer"] == "42"
    assert "do-not-export" not in json.dumps(audit)


@pytest.mark.parametrize("change,expected_status", [
    ({"prompt": "different question"}, "excluded_prompt_mismatch"),
    ({"tag": BASE_TAG}, "excluded_model_tag_conflict"),
    ({"finish": "length"}, "no_complete_improvement"),
    ({"finish": "error", "error": "request failed"}, "no_complete_improvement"),
])
def test_excludes_unusable_or_incomparable_candidates(tmp_path, change, expected_status):
    n = run(tmp_path / "native", BASE_TAG, False)
    kwargs = {"tag": TRAINED_TAG, **change}
    t = run(tmp_path / "trained", correct=True, **kwargs)
    cases, audit = compare(n, t)
    assert cases == []
    assert audit["status"] == expected_status


@pytest.mark.parametrize("change", [{"gold": "43"}, {"budget": 30720}])
def test_rejects_gold_or_budget_mismatch(tmp_path, change):
    n = run(tmp_path / "native", BASE_TAG, False)
    t = run(tmp_path / "trained", TRAINED_TAG, True, **change)
    with pytest.raises(ValueError):
        compare(n, t)


def test_rejects_duplicate_and_missing_score_ids(tmp_path):
    n = run(tmp_path / "native", BASE_TAG, False)
    t = run(tmp_path / "trained", TRAINED_TAG, True)
    path = n / "predictions.jsonl"
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_jsonl(path)
    (t / "scores.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        compare(t, t)


def test_user_confirmation_preserves_conflict_and_promotes_aime_only():
    payload = {"cases": [], "pending_aime": [{"dataset": "AIME24", "source_status": "pending"}],
        "manifest": {"audits": [{"dataset": "AIME24", "status": "pending_model_identity",
                                  "model_tag_conflict": True}],
                     "limitations": ["30K AIME model labels conflict", "Other limitation"]}}
    confirm_aime_identity(payload)
    assert len(payload["cases"]) == 1 and payload["pending_aime"] == []
    assert payload["manifest"]["audits"][0]["model_tag_conflict"] is True
    assert payload["manifest"]["audits"][0]["status"] == "included_user_confirmed_identity"
    assert payload["manifest"]["aime_identity_confirmation"]["reply"] == "确认是开源续训 Step120"
