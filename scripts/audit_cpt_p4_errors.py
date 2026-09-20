"""Reproduce the P1/P4 changed-item audit; emit no training examples or model calls.

Semantic annotations are a separate, inspectable Codex review, not an automatic
causal diagnosis. Full questions, options, answers and predictions remain private.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_cpt_transfer import CASE_SHA, paired, read_rows, require, sha, verify_predictions, verify_protocols
from prepare_cpt_p4_cumulative import OLD_MESSAGES_SHA, append_released
from run_logistics_strategy_diagnostic import messages_for
from verify_cpt_pilot_result import independently_parse


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def error_shape(gold, answer):
    if answer is None:
        return 'no_majority'
    missing, extra = set(gold) - set(answer), set(answer) - set(gold)
    return 'substitution' if missing and extra else 'omission' if missing else 'extra' if extra else 'correct'


def audit(resources, annotations, private_out):
    cases_path = resources / 'llin-transfer-audit-20260915-01/frozen_cases.private.jsonl'
    require(sha(cases_path) == CASE_SHA, 'formal snapshot mismatch')
    rows = read_rows(cases_path)
    cases = {r['item_hash']: r for r in rows}
    require(len(cases) == len(rows) == 1672, 'case count mismatch')
    dirs = {
        'step120': resources / 'llin-transfer-formal-20260917-01/step120_current',
        'p1': resources / 'llin-transfer-formal-20260917-01/p1',
        'p4': resources / 'llin-transfer-p4-run-20260918-01/formal5016',
    }
    protocols = {m: read(d / 'protocol.safe.json') for m, d in dirs.items()}
    verify_protocols(protocols, CASE_SHA, len(rows))
    raw = {m: read_rows(d / 'predictions.private.jsonl') for m, d in dirs.items()}
    scores = {m: verify_predictions(cases, raw[m], read(d / 'scores.safe.json')) for m, d in dirs.items()}
    # A second parser checks the stored parsed arrays independently.
    for records in raw.values():
        for record in records:
            parsed, valid = independently_parse(record['prediction'], len(cases[record['item_hash']]['options']))
            require(parsed == record['parsed'] and (valid and not record['truncated']) == record['valid'],
                    'independent parser mismatch')

    messages_path = resources / 'llin-transfer-p4-data-20260918-01/train.messages.private.jsonl'
    messages = read_rows(messages_path)
    require(sha(messages_path) == '1f67f334845f2ac84f9515a38612135b6238645f763c8b142a466303452884b7',
            'P4 training messages mismatch')
    original = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in messages[:315]).encode()
    require(hashlib.sha256(original).hexdigest() == OLD_MESSAGES_SHA, 'P1 original messages changed')
    packet = resources / 'llin-transfer-p4-package-20260918-01/packet'
    require(append_released(messages[:315], read_rows(packet / 'train.private.jsonl'),
                            read(packet / 'release.safe.json')) == messages, 'P4 append reconstruction mismatch')
    source_path = resources / 'llin-cpt-sft-search-20260912/source_generation_requests.private.jsonl'
    book_path = resources / 'llin-knowledge-expansion-20260911/book_pages.private.jsonl'
    sources = read_rows(source_path)
    books = {r['id']: r for r in read_rows(book_path)}
    review = read(annotations)
    require(review['source_catalog_sha256'] == sha(source_path), 'review source catalog mismatch')
    require(review['book_pages_sha256'] == sha(book_path), 'review book snapshot mismatch')
    changed = {k for k in cases if scores['p1'][k]['correct'] != scores['p4'][k]['correct']}
    ledger, private = [], []
    for annotation in review['items']:
        keys = [k for k in changed if k.startswith(annotation['hash_prefix'])]
        require(len(keys) == 1, 'missing or ambiguous review hash')
        key = keys[0]
        case = cases[key]
        require(all(type(i) is int and 0 <= i < len(sources) for i in annotation['source_rows']),
                'bad source catalog reference')
        source_ids = [sources[i]['id'] for i in annotation['source_rows']]
        book_refs = []
        for identity in annotation['book_ids']:
            require(identity in books, 'unresolved source page')
            page = books[identity]
            book_refs.append(dict(id=identity, source_file=page['source_file'], locator=page['locator']))
        train_rows = annotation['training_rows']
        require(all(type(i) is int and 0 <= i < len(messages) for i in train_rows), 'bad training reference')
        messages_in = messages_for(case, 'original')[0]
        require(all(isinstance(m['content'], str) for m in messages_in), 'unexpected multimodal input')
        entry = dict(item_hash=key, dataset=case['dataset'], category=case['category'],
                     options=len(case['options']), gold_cardinality=len(case['expected']),
                     transition='gain' if scores['p4'][key]['correct'] else 'loss',
                     p1_unstable=scores['p1'][key]['repeat_unstable'], p4_unstable=scores['p4'][key]['repeat_unstable'],
                     step120_correct=scores['step120'][key]['correct'],
                     wrong_answer_shape=error_shape(case['expected'], scores['p1' if scores['p4'][key]['correct'] else 'p4'][key]['answer']),
                     **{k: v for k, v in annotation.items() if k not in ('hash_prefix', 'source_rows', 'book_ids', 'training_rows')},
                     source_ids=source_ids, book_references=book_refs,
                     training_message_ids=[messages[i]['id'] for i in train_rows],
                     training_message_rows_zero_based=train_rows)
        ledger.append(entry)
        private.append(dict(review=entry, case=case,
                            scores={m: s[key] for m, s in scores.items()}, prompt=messages_in,
                            training_messages=[messages[i] for i in train_rows],
                            source_records=[sources[i] for i in annotation['source_rows']],
                            source_pages=[books[i] for i in annotation['book_ids']]))
    require(len(ledger) == len(changed) == len({r['item_hash'] for r in ledger}), 'review coverage mismatch')
    ledger.sort(key=lambda x: x['item_hash'])
    tables, remaining = [], []
    for dataset in sorted({r['dataset'] for r in rows}):
        keys = [k for k, c in cases.items() if c['dataset'] == dataset]
        for before in ('step120', 'p1'):
            tables.append(dict(dataset=dataset, before_model=before, **paired(keys, scores[before], scores['p4'])))
        remaining.append(dict(dataset=dataset, p4_errors=sum(not scores['p4'][k]['correct'] for k in keys),
                              wrong_in_both=sum(not scores['p4'][k]['correct'] and not scores['p1'][k]['correct'] for k in keys)))
    transitions = {}
    for transition in ('gain', 'loss'):
        subset = [r for r in ledger if r['transition'] == transition]
        transitions[transition] = dict(items=len(subset), stable_both=sum(not r['p1_unstable'] and not r['p4_unstable'] for r in subset),
            p1_unstable=sum(r['p1_unstable'] for r in subset), p4_unstable=sum(r['p4_unstable'] for r in subset),
            wrong_answer_shapes=dict(Counter(r['wrong_answer_shape'] for r in subset)),
            review_mechanisms=dict(Counter(r['mechanism'] for r in subset)),
            training_coverage=dict(Counter(r['training_coverage'] for r in subset)),
            source_status=dict(Counter(r['source_status'] for r in subset)))
    # Cross-check the released report without treating it as the source of scores.
    historical = read(resources / 'llin-transfer-p4-run-20260918-01/comparison.safe.json')
    require(historical['new_requests'] == 5284, 'unexpected historical result')
    for table in tables:
        previous = next(r for r in historical['formal'] if r['dataset'] == table['dataset'])
        comparison = previous['versus_' + table['before_model']]
        require(all(table[k] == comparison[k] for k in ('n', 'before', 'after', 'gains', 'losses')),
                'historical formal table mismatch')
    union = []
    for dataset, target in [('SC-bench-knowledge', 196), ('LogistikaBench', 1224)]:
        keys = [k for k, c in cases.items() if c['dataset'] == dataset]
        ceiling = sum(scores['p1'][k]['correct'] or scores['p4'][k]['correct'] for k in keys)
        union.append(dict(dataset=dataset, target=target, oracle_union_correct=ceiling,
                          still_missing=target-ceiling, deployable_model=False))
    long_losses = [r for r in ledger if r['options'] == 269 and r['transition'] == 'loss']
    private_out.mkdir(parents=True, exist_ok=True)
    write(private_out / 'changed_items.private.json', private)
    return dict(schema_version=1, date='2026-09-20', review_method='Codex semantic review of every changed item; not independent human adjudication',
        analysis='posthoc descriptive audit of already-used formal evaluations', comparisons=tables,
        transitions=transitions, remaining_errors=remaining, p1_p4_oracle_union=union,
        long_list_losses=dict(items=len(long_losses), mechanisms=dict(Counter(r['mechanism'] for r in long_losses))),
        raw_checks={m: dict(requests=len(raw[m]), invalid=sum(x['invalid'] for x in scores[m].values()),
                           truncated=sum(x['truncated'] for x in scores[m].values()),
                           unstable_items=sum(x['repeat_unstable'] for x in scores[m].values())) for m in scores},
        inputs=dict(cases_sha256=CASE_SHA, raw_sha256={m: sha(d / 'predictions.private.jsonl') for m, d in dirs.items()},
                    protocols_sha256={m: sha(d / 'protocol.safe.json') for m, d in dirs.items()},
                    training_messages_sha256=sha(messages_path), old315_messages_sha256=OLD_MESSAGES_SHA,
                    sources_sha256=sha(source_path), book_pages_sha256=sha(book_path), annotations_sha256=sha(annotations)),
        training=dict(p1_records=315, p4_records=363, preserved_p1_prefix=True, new_tasks=16, new_exposures=48,
                      coverage_scope='Only these SFT messages; not all historical CPT or pretraining knowledge.'),
        changed_items=ledger, new_inference_calls=0, new_training=False, official_scores_modified=False,
        limits=['Changed items are not a representative sample of all remaining errors.',
                'Same-seed deterministic inference repeats are not independent training seeds.',
                'Content, format, exposure count and training length are confounded; annotations do not establish training causality.',
                'Source references marked related or unresolved do not establish a unique official answer.',
                'Not-found coverage means not identified in this bounded SFT review, not proof of model ignorance.',
                'Official questions, options and labels must not be copied or paraphrased into future training data.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resources', type=Path, default=Path('CPT_resources'))
    parser.add_argument('--annotations', type=Path, default=Path('docs/cpt_p4_error_review_20260920.safe.json'))
    parser.add_argument('--private-out', type=Path, default=Path('CPT_resources/llin-p4-error-audit-20260920-01'))
    parser.add_argument('--out', type=Path, default=Path('docs/cpt_p4_error_audit_20260920.safe.json'))
    args = parser.parse_args()
    result = audit(args.resources, args.annotations, args.private_out)
    write(args.out, result)
    print(json.dumps({k: result[k] for k in ('comparisons', 'transitions', 'remaining_errors', 'raw_checks')}, indent=2))
