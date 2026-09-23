"""Offline audit of all changed items; never relabel or generate model outputs."""
import hashlib
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('CPT_resources/llin-neutral-change-audit-20260923-01')
PRIOR = Path('CPT_resources/llin-neutral-prefix-20260923-01')
CASES = Path('CPT_resources/llin-transfer-audit-20260915-01/frozen_cases.private.jsonl')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def norm(text):
    return ' '.join(text.casefold().split())


def equivalent_indices(options):
    groups = defaultdict(list)
    for i, text in enumerate(options):
        groups[norm(text)].append(i)
    return [indices for indices in groups.values() if len(indices) > 1]


def same_answer_text(options, left, right):
    if left is None or right is None:
        return False
    return sorted(norm(options[i]) for i in left) == sorted(norm(options[i]) for i in right)


def verify_reference(ref, cache):
    path = Path(ref['path'].replace('\\', '/'))
    if path not in cache:
        cache[path] = path.read_text(encoding='utf-8')
    content = cache[path]
    quote = ref.get('quote', ref.get('exact_quote'))
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in content.splitlines()]
        if 'line_1based' in ref:
            row = rows[ref['line_1based'] - 1]
            assert ref['key'] in (row.get('key'), row.get('id'), 'book:' + row.get('id', ''))
        else:
            matches = [r for r in rows if r.get('id') == ref['locator']['record_id']]
            assert len(matches) == 1
            row = matches[0]
            assert row['source_id'] == ref['source_id']
        text = row['text']
        assert hashlib.sha256(text.encode()).hexdigest() == ref['text_sha256']
        assert quote in text
    else:
        assert sha(path) == ref['file_sha256']
        if 'line_1based' in ref:
            text = '\n'.join(content.split('\n')[ref['line_1based']-1:ref['line_end_1based']])
            assert quote == text
            assert hashlib.sha256(content.encode()).hexdigest() == ref['text_sha256']
        else:
            loc = ref['locator']
            text = content[loc['character_start']:loc['character_end']]
            assert quote == text
            assert hashlib.sha256(text.encode()).hexdigest() == ref['text_sha256']
    if 'quote_sha256' in ref:
        assert hashlib.sha256(quote.encode()).hexdigest() == ref['quote_sha256']


def finalize():
    ledger = read(BASE / 'structural_ledger.private.json')
    safe = read(BASE / 'structural.safe.json')
    cache = {}
    combined = []
    for change in ('gain', 'loss'):
        path = BASE / f'{change}_review.private.json'
        review = read(path)['items']
        expected = {r['id']: r for r in ledger if r['change'] == change}
        assert len(review) == len(expected) and {r['id'] for r in review} == set(expected)
        for r in review:
            old = expected[r['id']]
            assert r['question'] == old['case']['question']
            assert r.get('full_option_count_reviewed', r.get('option_count_reviewed')) == 269
            assert r['source_status'] in 'ABCD'
            if r['source_status'] == 'A':
                assert r['source_refs']
            for ref in r['source_refs']:
                verify_reference(ref, cache)
            combined.append(dict(**old, current_review=r))
        safe[f'{change}_source_status'] = dict(Counter(r['source_status'] for r in review))
        safe['input_sha256'][str(path).replace('\\', '/')] = sha(path)
    for path in cache:
        safe['input_sha256'][str(path).replace('\\', '/')] = sha(path)
    save(BASE / 'reviewed_ledger.private.json', combined)
    safe.update(status='closed_offline_source_review_complete', reviewed_items=len(combined),
                source_reference_count=sum(len(r['current_review']['source_refs']) for r in combined),
                all_source_quotes_and_fingerprints_verified=True,
                reviewed_ledger_sha256=sha(BASE/'reviewed_ledger.private.json'),
                historical_source_cohorts_unchanged=True, training_ready=False,
                human_review_limitations='Source text extraction, no new page-image validation; outcomes visible; no independent blind claim')
    save(Path('docs/cpt_neutral_change_audit_20260923.safe.json'), safe)
    print(json.dumps({k:safe[k] for k in ['status','reviewed_items','gain_source_status','loss_source_status','source_reference_count']}, ensure_ascii=False))


def build():
    prior_safe = read(Path('docs/cpt_neutral_prefix_results_20260923.safe.json'))
    assert sha(PRIOR / 'results.private.json') == prior_safe['private_ledger_sha256']
    offline = read(PRIOR / 'offline.safe.json')
    fingerprints = {k.replace('\\', '/'): v for k, v in offline['input_sha256'].items()}
    assert sha(CASES) == fingerprints[str(CASES).replace('\\', '/')]
    cases = {c['item_hash']: c for c in map(json.loads, CASES.read_text(encoding='utf-8').splitlines())}
    results = read(PRIOR / 'results.private.json')
    assert len(results) == 305
    changed = [r for r in results if r['before']['correct'] != r['after']['correct']]
    assert len(changed) == 31
    pairs_path = Path('CPT_resources/llin-p1-error-map-20260923-01/paired_cases.private.json')
    pairs = read(pairs_path)
    linked = defaultdict(list)
    for pair in pairs:
        linked[pair['large']].append(pair)
    ledger = []
    for result in changed:
        c = cases[result['id']]
        options = c['options']
        peers = []
        for pair in linked[result['id']]:
            small = cases[pair['small']]
            # These are dataset-internal consistency clues, never source evidence.
            assert norm(c['question'].split('Which of the following best matches:', 1)[1]) == norm(small['question'].split('Which of the following best matches:', 1)[1])
            assert sorted(norm(options[i]) for i in c['expected']) == sorted(norm(small['options'][i]) for i in small['expected'])
            peers.append(dict(case=small, historical_p1_correct=pair['small_correct']))
        row = dict(id=result['id'], change='loss' if result['before']['correct'] else 'gain',
                   case=c, before=result['before'], after=result['after'], historical_source_status=result['source_status'],
                   casefold_duplicate_groups=equivalent_indices(options),
                   expected_has_casefold_duplicate=any(set(c['expected']) & set(g) for g in equivalent_indices(options)),
                   changed_index_same_normalized_text=same_answer_text(options, result['before']['answer'], result['after']['answer']),
                   natural_pairs=peers)
        ledger.append(row)
    save(BASE / 'structural_ledger.private.json', ledger)
    pool = cases[results[0]['id']]['options']
    assert all(cases[r['id']]['options'] == pool for r in results)
    # Post-hoc label category is descriptive only and cannot become an inference router.
    pros_cons = [r for r in results if any(norm(pool[i]) in ('advantage', 'disadvantage') for i in cases[r['id']]['expected'])]
    def counts(rows):
        return dict(n=len(rows), before=sum(r['before']['correct'] for r in rows),
                    after=sum(r['after']['correct'] for r in rows),
                    gains=sum(not r['before']['correct'] and r['after']['correct'] for r in rows),
                    losses=sum(r['before']['correct'] and not r['after']['correct'] for r in rows))
    safe = dict(status='offline_structural_audit_complete', new_model_calls=0, training_runs=0,
                official_scores_changed=False, changed_counts=dict(Counter(r['change'] for r in ledger)),
                full_pool_casefold_duplicate_groups=len(equivalent_indices(pool)),
                changed_expected_has_casefold_duplicate=dict(Counter(r['change'] for r in ledger if r['expected_has_casefold_duplicate'])),
                changed_index_same_normalized_text=dict(Counter(r['change'] for r in ledger if r['changed_index_same_normalized_text'])),
                changed_natural_pair_counts=dict(Counter(r['change'] for r in ledger if r['natural_pairs'])),
                posthoc_pros_cons_label_cohort=counts(pros_cons),
                input_sha256={str(p).replace('\\', '/'):sha(p) for p in [CASES, PRIOR/'results.private.json', pairs_path]},
                structural_ledger_sha256=sha(BASE/'structural_ledger.private.json'),
                limitations=['Outcome-selected descriptive audit, not held-out validation',
                             'Casefold duplicates do not constitute a new scoring rule',
                             'Natural pairs and labels do not independently validate source truth',
                             'Post-hoc label categories cannot be used as inference features'])
    save(BASE/'structural.safe.json', safe)
    print(json.dumps(safe, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    finalize() if args.finalize else build()
