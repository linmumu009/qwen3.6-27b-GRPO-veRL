"""Reparse frozen evaluations and prepare diagnostic-only, stratified review packets.

No training data or model calls are produced. Raw cases/answers stay in the
explicit private output directory; the public summary contains aggregates only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_logistics_knowledge import parse_answers

CASE_SHA = 'b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99'
COUNTS = {'SC-bench-knowledge': 226, 'LogistikaBench': 1446}
MODELS = ('step120', 'cpt16', 's4', 's5')
SEED = 20260915


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protocols(protocols, source_sha, count):
    expected = dict(cases_sha256=source_sha, cases=count, repeats=3,
                    temperature=0, seed=1024, max_tokens=96,
                    max_model_len=8192, tp=8, max_num_seqs=32, prompt='original')
    reference = None
    for label, protocol in protocols.items():
        require(bool(protocol.get('model')), f'{label}: missing model identity')
        require(all(protocol.get(k) == v for k, v in expected.items()), f'{label}: frozen protocol mismatch')
        require(len(protocol.get('prompt_hashes', [])) == count, f'{label}: missing prompt fingerprints')
        current = {k: v for k, v in protocol.items() if k != 'model'}
        if reference is None:
            reference = current
        require(current == reference, f'{label}: prompt/protocol drift')


def verify_predictions(cases, rows, saved):
    require(len(rows) == len(cases) * 3, 'request count mismatch')
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = row['item_hash']
        require(key in cases, 'unknown case')
        pair = (key, row['repeat'])
        require(type(row['repeat']) is int and row['repeat'] in (0, 1, 2) and pair not in seen,
                'duplicate or invalid repeat')
        seen.add(pair)
        case = cases[key]
        parsed, ok = parse_answers(row['prediction'], len(case['options']))
        require(type(row['truncated']) is bool and type(row['valid']) is bool and
                type(row['correct']) is bool, 'non-boolean score flags')
        valid = ok and not row['truncated']
        correct = valid and list(parsed) == sorted(case['expected'])
        require(row['parsed'] == list(parsed) and row['valid'] == valid and row['correct'] == correct,
                'raw answer and saved score disagree')
        require(row['dataset'] == case['dataset'], 'dataset mismatch')
        require(type(row['output_tokens']) is int and 0 <= row['output_tokens'] <= 96,
                'output budget violation')
        grouped[key].append(row)
    require(set(grouped) == set(cases) == set(saved), 'case coverage mismatch')
    result = {}
    for key, group in grouped.items():
        require(len(group) == 3, 'missing repeats')
        votes = Counter(tuple(r['parsed']) if r['valid'] else None for r in group)
        answer, n = votes.most_common(1)[0]
        majority = answer if n >= 2 and answer is not None else None
        correct = majority is not None and list(majority) == sorted(cases[key]['expected'])
        require(saved[key]['correct'] == correct and saved[key]['dataset'] == cases[key]['dataset'],
                'saved majority disagrees')
        result[key] = dict(correct=correct, answer=list(majority) if majority is not None else None,
                           repeat_unstable=len(votes) > 1,
                           invalid=sum(not r['valid'] for r in group),
                           truncated=sum(r['truncated'] for r in group))
    return result


def paired(keys, before, after, uncertainty=False):
    n = len(keys)
    gains = sum(not before[k]['correct'] and after[k]['correct'] for k in keys)
    losses = sum(before[k]['correct'] and not after[k]['correct'] for k in keys)
    result = dict(n=n, before=sum(before[k]['correct'] for k in keys),
                  after=sum(after[k]['correct'] for k in keys), gains=gains, losses=losses,
                  delta_pp=100 * (gains-losses) / n if n else None)
    if uncertainty and n:
        discordant = gains + losses
        result['mcnemar_exact_two_sided_p_unadjusted'] = min(1., 2 * sum(
            math.comb(discordant, k) for k in range(min(gains, losses)+1)) / (2 ** discordant))
        rng = random.Random(SEED)
        # Resample paired per-item differences, never the three deterministic repeats.
        deltas = [1] * gains + [-1] * losses + [0] * (n-gains-losses)
        boot = sorted(100 * sum(rng.choices(deltas, k=n)) / n for _ in range(4000))
        result['paired_item_bootstrap_95pct_pp'] = [boot[99], boot[3899]]
    return result


def stratum(case):
    return (case['category'], case['question_type'],
            'long_options' if len(case['options']) > 20 else 'short_options',
            'multi_answer' if len(case['expected']) > 1 else 'single_answer')


def select_review(cases, baseline, quota=80):
    sc = sorted(k for k, c in cases.items() if c['dataset'] == 'SC-bench-knowledge' and not baseline[k]['correct'])
    groups = defaultdict(list)
    for key, case in cases.items():
        if case['dataset'] == 'LogistikaBench' and not baseline[key]['correct']:
            groups[stratum(case)].append(key)
    for group in groups.values():
        group.sort(key=lambda k: hashlib.sha256(f'{SEED}:{k}'.encode()).hexdigest())
    quota = min(quota, sum(map(len, groups.values())))
    require(len(groups) <= quota, 'review budget cannot cover every stratum')
    allocation = {k: 1 for k in groups}
    while sum(allocation.values()) < quota:
        eligible = [k for k in sorted(groups) if allocation[k] < len(groups[k])]
        # Proportional allocation with at least one per observed stratum.
        chosen = max(eligible, key=lambda k: len(groups[k]) / (allocation[k]+1))
        allocation[chosen] += 1
    selected = sc + [v for k in sorted(groups) for v in groups[k][:allocation[k]]]
    return selected, [dict(stratum=list(k), available=len(groups[k]), selected=allocation[k])
                      for k in sorted(groups)]


def summarize(cases, models):
    summary = dict(comparisons=[], by_stratum=[], overlaps=[], baseline_counts={}, model_audits={})
    baseline = models['step120']
    for label, predictions in models.items():
        summary['model_audits'][label] = dict(items=len(predictions), requests=3*len(predictions),
            invalid=sum(r['invalid'] for r in predictions.values()),
            truncated=sum(r['truncated'] for r in predictions.values()),
            repeat_unstable_items=sum(r['repeat_unstable'] for r in predictions.values()))
    for dataset in sorted({c['dataset'] for c in cases.values()}):
        keys = sorted(k for k, c in cases.items() if c['dataset'] == dataset)
        summary['baseline_counts'][dataset] = sum(baseline[k]['correct'] for k in keys)
        gain_sets = {m: {k for k in keys if not baseline[k]['correct'] and models[m][k]['correct']}
                     for m in MODELS[1:]}
        loss_sets = {m: {k for k in keys if baseline[k]['correct'] and not models[m][k]['correct']}
                     for m in MODELS[1:]}
        for label in MODELS[1:]:
            summary['comparisons'].append(dict(dataset=dataset, model=label,
                **paired(keys, baseline, models[label], uncertainty=True)))
            strata = sorted({stratum(cases[k]) for k in keys})
            for group in strata:
                subset = [k for k in keys if stratum(cases[k]) == group]
                summary['by_stratum'].append(dict(dataset=dataset, model=label, stratum=list(group),
                    **paired(subset, baseline, models[label])))
        for a, b in combinations(MODELS[1:], 2):
            summary['overlaps'].append(dict(dataset=dataset, models=[a, b],
                gain_intersection=len(gain_sets[a] & gain_sets[b]), gain_union=len(gain_sets[a] | gain_sets[b]),
                loss_intersection=len(loss_sets[a] & loss_sets[b]), loss_union=len(loss_sets[a] | loss_sets[b]),
                a_only_gains=len(gain_sets[a]-gain_sets[b]), b_only_gains=len(gain_sets[b]-gain_sets[a])))
        summary.setdefault('retrospective_repair_union', []).append(dict(dataset=dataset,
            repairs=len(set.union(*gain_sets.values())), common_repairs=len(set.intersection(*gain_sets.values())),
            common_regressions=len(set.intersection(*loss_sets.values())),
            interpretation='Uses gold to select different models per item; not an ensemble score or achievable candidate.'))
    return summary


def audit_answer_content(cases, models):
    """Flag textual collisions without changing any index-based score."""
    norm = lambda text: re.sub(r'\s+', ' ', text).strip().casefold()
    results = []
    for dataset in sorted({c['dataset'] for c in cases.values()}):
        keys = [k for k, c in cases.items() if c['dataset'] == dataset]
        collisions = []
        gold_collisions = []
        for k in keys:
            options = [norm(o) for o in cases[k]['options']]
            counts = Counter(options)
            if len(counts) != len(options):
                collisions.append(k)
            if any(counts[options[i]] > 1 for i in cases[k]['expected']):
                gold_collisions.append(k)
        by_model = {}
        for label, predictions in models.items():
            same_text = wrong_single_cardinality = 0
            for k in keys:
                pred, case = predictions[k], cases[k]
                if pred['correct'] or pred['answer'] is None:
                    continue
                same_text += ({norm(case['options'][i]) for i in pred['answer']} ==
                              {norm(case['options'][i]) for i in case['expected']})
                wrong_single_cardinality += (case['question_type'] in ('single_choice', 'true_or_false')
                                             and len(pred['answer']) != 1)
            by_model[label] = dict(index_wrong_but_same_normalized_option_text=same_text,
                                  wrong_answer_cardinality_on_explicit_single_choice=wrong_single_cardinality)
        results.append(dict(dataset=dataset, items_with_option_text_collisions=len(collisions),
            items_with_gold_option_text_collision=len(gold_collisions), models=by_model))
    return dict(normalization='Whitespace collapse and casefold only; no semantic equivalence judgment.',
                official_scores_changed=False, datasets=results)


def audit_source_probes(root):
    from scripts.run_vllm_logistics_mcq import load_items
    result = {}
    for arm in ('s4', 's5'):
        cases = {c.item_hash: c for c in load_items(root/arm/'tasks.private.jsonl')}
        require(len(cases) == 253, 'source probe count changed')
        models = {}
        for label in ('step120_tasks', arm+'_tasks'):
            rows = read_rows(root/arm/(label+'.private.jsonl'))
            require(len(rows) == len(cases) and len({r['item_hash'] for r in rows}) == len(rows),
                    'source probe duplicate or missing output')
            scores = {}
            for row in rows:
                case = cases[row['item_hash']]
                for key in ('dataset', 'source_id', 'category', 'question_type', 'question'):
                    require(row[key] == getattr(case, key), 'source probe input mismatch')
                require(row['options'] == list(case.options) and row['expected'] == list(case.expected),
                        'source probe options/labels mismatch')
                require(row['chat_template_disable_thinking'] is True and
                        row['prompt_version'] == 'logistics-mcq-zero-based-json-v2' and row['error'] is None,
                        'source probe protocol/error mismatch')
                parsed, ok = parse_answers(row['prediction'], len(case.options))
                correct = ok and parsed == case.expected
                require(list(parsed) == row['parsed'] and ok == row['parse_ok'] and correct == row['correct'],
                        'source probe raw parse mismatch')
                scores[row['item_hash']] = dict(correct=correct, answer=list(parsed))
            require(set(scores) == set(cases), 'source probe coverage changed')
            models[label] = scores
        before, after = models['step120_tasks'], models[arm+'_tasks']
        groups = sorted({(c.dataset, c.category) for c in cases.values()})
        result[arm] = [dict(dataset=g[0], category=g[1], **paired(
            [k for k, c in cases.items() if (c.dataset, c.category) == g], before, after)) for g in groups]
    return dict(raw_answers_reparsed=1012, groups=result,
        limitations=['Historical source runner did not save finish_reason; truncation cannot be independently excluded.',
                     'These source probes were used in development and are not independent generalization tests.'])


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='frozen_cases.private.jsonl and four model directories')
    p.add_argument('--private-output', type=Path, required=True)
    p.add_argument('--safe-output', type=Path, required=True)
    args = p.parse_args()
    source = args.input/'frozen_cases.private.jsonl'
    require(sha(source) == CASE_SHA, 'frozen source changed')
    rows = read_rows(source)
    cases = {r['item_hash']: r for r in rows}
    require(len(cases) == len(rows) == sum(COUNTS.values()), 'duplicate/missing frozen cases')
    require(dict(Counter(r['dataset'] for r in rows)) == COUNTS, 'dataset counts changed')
    protocols = {m: json.loads((args.input/m/'protocol.safe.json').read_text()) for m in MODELS}
    verify_protocols(protocols, CASE_SHA, len(cases))
    models = {m: verify_predictions(cases, read_rows(args.input/m/'predictions.private.jsonl'),
        json.loads((args.input/m/'scores.safe.json').read_text())) for m in MODELS}
    summary = summarize(cases, models)
    summary['answer_content_audit'] = audit_answer_content(cases, models)
    summary['source_probe_audit'] = audit_source_probes(args.input)
    require(summary['baseline_counts'] == {'SC-bench-knowledge': 189, 'LogistikaBench': 1180}, 'baseline changed')
    selected, allocation = select_review(cases, models['step120'])
    summary.update(date='2026-09-15', schema_version=1, source_sha256=CASE_SHA,
        file_sha256={m: {n: sha(args.input/m/n) for n in ['predictions.private.jsonl','protocol.safe.json','scores.safe.json']}
                     for m in MODELS}, protocol_equal=True, raw_outputs_reparsed=True,
        review_selection=dict(sc_errors=37, logistika_errors=len(selected)-37, seed=SEED, strata=allocation),
        evidence_review_complete=False, training_started=False,
        limitations=['Repeated fixed-seed evaluations are not independent training seeds.',
            'Bootstrap assumes exchangeable items; shared sources can violate this assumption.',
            'Intervals and unadjusted tests are descriptive after repeated benchmark selection.',
            'Review sampling is stratified; unweighted sample error rates do not estimate full-set rates.',
            'No causal knowledge/application/format diagnosis without source and answer inspection.',
            'Review packets are diagnostic-only; official questions and rewrites cannot enter training.'])
    require(not args.safe_output.exists(), 'refuse to overwrite summary')
    os.umask(0o077)
    args.private_output.mkdir(parents=True, exist_ok=False)
    def private_rows(name, rows):
        with (args.private_output/name).open('x', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
    private_rows('item_deltas.private.jsonl', [dict(item_hash=k, dataset=c['dataset'],
        category=c['category'], question_type=c['question_type'], models={m: models[m][k] for m in MODELS})
        for k, c in cases.items()])
    private_rows('review_blinded.private.jsonl', [dict(review_id=k, dataset=cases[k]['dataset'],
        category=cases[k]['category'], question_type=cases[k]['question_type'],
        question=cases[k]['question'], options=cases[k]['options'], review_status='pending',
        source_excerpt=None, source_locator=None, sufficiency=None, independently_derived_answer=None)
        for k in selected])
    private_rows('review_key.private.jsonl', [dict(item_hash=k, expected=cases[k]['expected'],
        source_id=cases[k]['source_id'], models={m: models[m][k] for m in MODELS}) for k in selected])
    dump(args.safe_output, summary)
    print(json.dumps(dict(comparisons=summary['comparisons'], review_items=len(selected),
                         retrospective_repair_union=summary['retrospective_repair_union'])))


if __name__ == '__main__':
    main()
