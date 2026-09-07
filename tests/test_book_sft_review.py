import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('review', Path(__file__).parents[1] / 'scripts/review_book_sft_pilot.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_quote_categories():
    assert m.quote_status({}, '') == 'missing_or_short'
    qa = {'support_quotes': ['Inventory is held for future use.']}
    assert m.quote_status(qa, 'Inventory is held for future use.') == 'exact'
    assert m.quote_status(qa, 'Inventory is held\nfor future use.') == 'whitespace_only'
    assert m.quote_status(qa, 'Inventory was held for future use.') == 'content_mismatch'


def test_fail_closed():
    assert m.decision({}, '') == 'invalid_review'
    v = dict.fromkeys(('all_claims_covered', 'standalone', 'unambiguous', 'scope_correct', 'type_match'), True)
    v.update(claims=[{'claim': 'A', 'supported': True, 'quote': 'A valid exact evidence quote.'}], issues=[])
    assert m.decision(v, 'A valid exact evidence quote.') == 'recheck_pass'
    assert m.decision(v, 'Different source') == 'evidence_unverified'
    v['claims'][0]['supported'] = False
    assert m.decision(v, '') == 'concern'


def test_sampling():
    rows = [{'id': str(i) + kind, 'kind': kind, 'status': status} for status in
            ('auto_pass', 'structural_reject', 'audit_reject') for kind in
            ('definition', 'comparison', 'conditions', 'application') for i in range(10)]
    result = m.select(rows)
    assert len(result) == 28
    assert result == m.select(list(reversed(rows)))
    assert sum(r['status'] == 'auto_pass' for r in result) == 20
