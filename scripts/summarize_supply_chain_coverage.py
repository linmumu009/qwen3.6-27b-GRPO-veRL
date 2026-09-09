"""Join source-only labels with QA identities and safe benchmark strata."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--coverage', type=Path, required=True)
    a = p.parse_args()
    read = lambda p: [json.loads(x) for x in p.read_text().splitlines()]
    labels = read(a.coverage / 'classification.private.jsonl')
    summary = json.loads((a.coverage / 'summary.safe.json').read_text())
    assert summary['status'] == 'complete' and summary['failed_calls'] == 0
    qa = [x for r in labels if r['kind'] == 'qa' for x in r['labels']]
    source = [x for r in labels if r['kind'] == 'source' for x in r['labels']]
    train = read(a.root / 'runs/book-capability-sft-20260909-01/train.messages.private.jsonl')
    actual = {x['id'] for x in train}
    selected = [x for x in qa if x['id'] in actual]
    assert len(qa) == len({x['id'] for x in qa}) == 436
    assert len(selected) == len(actual) == 355 and len(source) == 115
    coverage = {}
    for name, rows in [('book_chunks', source), ('all_qa', qa), ('actual_train', selected)]:
        coverage[name] = {'items': len(rows), 'primary_domain': dict(Counter(x['primary_domain'] for x in rows)),
                          'capabilities_multilabel': dict(Counter(c for x in rows for c in x['capabilities'])),
                          'evidence': dict(Counter(x['usable_evidence'] for x in rows))}
    result = {'coverage': coverage, 'api_summary': summary, 'models': {}, 'private_content_included': False,
              'limitations': ['AI labels are provisional, not expert-certified knowledge coverage.',
                             'Topic distribution mismatch is not proof of a training cause.',
                             'Step120 historical baseline is not contemporaneous with aligned CPT/SFT.',
                             'All official benchmarks have been repeatedly observed; not pristine holdouts.']}
    run = a.root / 'runs/book-capability-sft-aligned-20260909-01'
    paths = {'step120_historical': a.root / 'runs/logistics-cpt-diagnostics-20260904/safe/public_eval/step120.majority.safe.json',
             'cpt_aligned': run / 'cpt.majority.safe.json', 'sft_aligned': run / 'sft.majority.safe.json'}
    hashes = set()
    for model, path in paths.items():
        r = json.loads(path.read_text()); rows = r['rows']
        assert len(rows) == len({x['item_hash'] for x in rows}) == 1672
        assert sum(x['correct'] for x in rows) == r['correct']
        hashes.add(r['input_sha256']['cases_jsonl'])
        groups = defaultdict(list)
        for x in rows:
            size = 'large_pool' if x['choice_count'] > 12 else 'ordinary'
            groups[x['dataset'], x['category'], x['question_type'], size].append(x)
        result['models'][model] = {'correct': r['correct'], 'items': len(rows),
            'input_report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'strata': [dict(dataset=k[0], category=k[1], question_type=k[2], choice_pool=k[3],
                           items=len(v), correct=sum(x['correct'] for x in v),
                           errors=sum(not x['correct'] for x in v)) for k, v in sorted(groups.items())]}
    assert len(hashes) == 1
    result['benchmark_case_sha256'] = hashes.pop()
    with (a.coverage / 'coverage_join.safe.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'coverage': coverage, 'models': {k: {'correct': v['correct'], 'errors': v['items'] - v['correct']}
                     for k, v in result['models'].items()}}, indent=2))


if __name__ == '__main__':
    main()
