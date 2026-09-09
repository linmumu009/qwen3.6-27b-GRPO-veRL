import pytest
import hashlib
import json
import sys

from scripts.summarize_mcq_repeats import majority_rows
from scripts.summarize_mcq_repeats import main


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


@pytest.mark.parametrize('mismatch', [False, True])
def test_repeat_request_metadata(tmp_path, monkeypatch, mismatch):
    cases = tmp_path / 'cases.jsonl'
    cases.write_text('{}\n', encoding='utf-8')
    args = ['summary', '--cases', str(cases), '--model-label', 'test',
            '--private-output', str(tmp_path / 'majority.jsonl'),
            '--safe-output', str(tmp_path / 'majority.safe.json')]
    for i in range(3):
        private = tmp_path / f'{i}.jsonl'
        private.write_text(json.dumps(_row([1])) + '\n', encoding='utf-8')
        safe = tmp_path / f'{i}.safe.json'
        safe.write_text(json.dumps({'items': 1,
            'input_sha256': {'cases_jsonl': hashlib.sha256(cases.read_bytes()).hexdigest()},
            'request': {'max_output_tokens': 128 if mismatch and i == 2 else 96}}), encoding='utf-8')
        args += ['--repeat', str(private), '--repeat-safe', str(safe)]
    monkeypatch.setattr(sys, 'argv', args)
    if mismatch:
        with pytest.raises(ValueError, match='contract mismatch'):
            main()
    else:
        assert main() == 0
        result = json.loads((tmp_path / 'majority.safe.json').read_text())
        assert result['request'] == {'max_output_tokens': 96}
        assert result['request_provenance'] == 'validated_per_repeat_safe_reports'
