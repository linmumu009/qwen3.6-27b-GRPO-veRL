import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('adjudicate', Path(__file__).parents[1] / 'scripts/adjudicate_book_sft_concerns.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_lossless_units():
    source = 'abc\n ' * 400
    assert ''.join(m.source_units(source).values()) == source


def test_fail_closed_decisions():
    d = {'issue_id': 0, 'category': 'material', 'reason': 'Scope differs', 'source_ids': ['S001']}
    assert m.validate({'decisions': [d]}, ['issue'], {'S001':'text'}) == 'valid_adjudication'
    assert m.validate({'decisions': [d,d]}, ['a','b'], {'S001':'text'}) == 'invalid_response'
    assert m.validate({'decisions': [d]}, ['issue'], {}) == 'invalid_evidence_id'
    d['source_ids'] = []
    assert m.validate({'decisions': [d]}, ['issue'], {}) == 'invalid_evidence_id'
    d['category'] = 'uncertain'
    assert m.validate({'decisions': [d]}, ['issue'], {}) == 'valid_adjudication'
