"""Freeze anonymous ratings before comparing exact-prompt recall to old probes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from summarize_cpt_mechanism import blind, save, sha


def summarize(review, ratings, previous, out):
    freeze = json.loads((review/'freeze.safe.json').read_text())
    assert sha(review/'review.private.json') == freeze['review_sha256']
    assert sha(review/'identity_key.private.json') == freeze['key_sha256']
    questions = {r['blind_id']: r for r in json.loads((review/'review.private.json').read_text())}
    judged = json.loads(ratings.read_text())
    assert len(judged) == len(questions) == 72
    assert {r['blind_id'] for r in judged} == set(questions)
    for r in judged:
        q = questions[r['blind_id']]
        assert len(r['facts']) == len(q['required']) and all(type(x) is bool for x in r['facts'])
        assert type(r['forbidden_inference']) is bool and type(r['uncertain']) is bool
        assert isinstance(r['reason'], str) and r['reason'].strip()
    stamp = review/'ratings_frozen.safe.json'
    if stamp.exists():
        assert json.loads(stamp.read_text())['ratings_sha256'] == sha(ratings)
    else:
        save(stamp, dict(items=72, ratings_sha256=sha(ratings)))
    mapping = json.loads((review/'identity_key.private.json').read_text())
    aggregate = defaultdict(lambda: dict(calls=0, strict_pass=0, facts_correct=0, facts_total=0, truncated=0))
    units = defaultdict(lambda: dict(calls=0, strict_pass=0, facts_correct=0, facts_total=0))
    for r in judged:
        m = mapping[r['blind_id']]
        q = questions[r['blind_id']]
        passed = all(r['facts']) and not r['uncertain'] and not r['forbidden_inference'] and q['finish_reason'] == 'stop'
        for row in (aggregate[m['model']], units[(m['model'], m['unit_id'])]):
            row['calls'] += 1
            row['strict_pass'] += int(passed)
            row['facts_correct'] += sum(r['facts'])
            row['facts_total'] += len(r['facts'])
        aggregate[m['model']]['truncated'] += q['finish_reason'] != 'stop'
    assert set(aggregate) == {'step120', 'p1', 'K', 'R'}
    assert all(v['calls'] == 18 for v in aggregate.values())
    old = json.loads(previous.read_text())
    save(out, dict(new_generation_calls=72, training_steps=0, formal_reruns=0,
         direct=dict(aggregate), units=[dict(model=m, unit_id=u, **v) for (m,u),v in sorted(units.items())],
         old_split_prompt_direct=old['direct'], old_prompt_stability=old['unit_stability'],
         ratings_sha256=sha(ratings), previous_results_sha256=sha(previous),
         promotion=False, independent_holdout=False,
         limitation='Original prompt tests recall, not independent transfer. Full prompt and split prompts differ in task scope; do not compare strict-pass percentages directly.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['blind', 'summarize'])
    for k in ('run', 'probes', 'out', 'review', 'ratings', 'previous'):
        p.add_argument('--'+k, type=Path)
    a = p.parse_args()
    if a.action == 'blind':
        blind(a.run, a.probes, a.out)
    else:
        summarize(a.review, a.ratings, a.previous, a.out)
