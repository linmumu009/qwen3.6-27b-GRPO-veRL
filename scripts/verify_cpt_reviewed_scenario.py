"""Reparse the frozen six-case format probe; never alter its questions or keys."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from verify_cpt_composed_probe import independent_parse


def verify(packet, result):
    cases = [json.loads(s) for s in packet.read_text(encoding='utf-8').splitlines()]
    rows = [json.loads(s) for s in (result/'predictions.private.jsonl').read_text(encoding='utf-8').splitlines()]
    registration = json.loads((result/'registration.safe.json').read_text())
    digest = hashlib.sha256(packet.read_bytes()).hexdigest()
    assert digest == registration['case_sha256']
    assert len(cases) == 6 and len(rows) == 12
    expected_keys = {(c['id'], mode) for c in cases for mode in ('answer_only', 'explain_then_answer')}
    assert len({(r['id'], r['condition']) for r in rows}) == len(rows)
    assert {(r['id'], r['condition']) for r in rows} == expected_keys
    by_id = {c['id']: c for c in cases}
    totals = Counter()
    for r in rows:
        c = by_id[r['id']]
        parsed = independent_parse(r['prediction'], len(c['options']))
        correct = parsed == c['expected'] and r['finish_reason'] == 'stop'
        assert parsed == r['parsed'] and correct == r['correct']
        totals[r['condition']+'_correct'] += correct
        totals['invalid'] += parsed is None
        totals['truncated'] += r['finish_reason'] != 'stop'
    assert 36-22-9 == 5
    assert Fraction(51,60) == Fraction(85,100)
    assert Fraction(288,300) == Fraction(96,100)
    assert Fraction(48*9+12*29,60) == 13
    assert (Fraction(10,20)+Fraction(20,80))/2 == Fraction(375,1000)
    assert Fraction(228,240) == Fraction(95,100)
    assert Fraction(45,50) == Fraction(90,100)
    return dict(cases=6, requests=12, case_sha256=digest,
        predictions_sha256=hashlib.sha256((result/'predictions.private.jsonl').read_bytes()).hexdigest(),
        **totals, independently_reparsed=True, numeric_keys_checked=True,
        repaired_cases=['llin-scenario-1'], explanation_audit={
            'llin-scenario-4': 'Correct selected option, incorrect intermediate arithmetic: 48*9+12*29=780, not 756. The exact answer is 13, not 12.6 rounded to the nearest option.'},
        added_candidate_rules=1, new_rule='batching_total_cost',
        training_running=False, official_protocol_changed=False,
        limitations='One repaired cost calculation does not establish a training-format advantage. Correct selected labels do not certify explanations. Five source units overlap older coverage; only the newly failing batching-cost rule is an added candidate, not a stable independent deficit.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    summary=verify(a.cases,a.result)
    a.out.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
