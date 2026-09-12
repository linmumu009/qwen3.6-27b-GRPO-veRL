"""Freeze a CPT/SFT hypothesis search and source-only generation requests.

This prepares experiments, not models. It never reads benchmark questions or
answers and never promotes generated questions to reviewed training data.
"""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


BASELINE = {'SC-bench-knowledge': (189, 226), 'LogistikaBench': (1180, 1446)}


def target_correct(correct, total, percentage_points=3):
    target = Fraction(correct) + Fraction(str(percentage_points)) * total / 100
    return (target.numerator + target.denominator - 1) // target.denominator


def assess_target(table):
    """Unrounded, per-dataset criterion; a combined score cannot substitute."""
    rows = {r['dataset']: r for r in table}
    if len(rows) != len(table):
        raise ValueError('Duplicate dataset')
    result = {}
    for dataset, (before, total) in BASELINE.items():
        row = rows[dataset]
        if row['n'] != total or row['before'] != before:
            raise ValueError('Baseline or denominator changed')
        after = row['after']
        if type(after) is not int or not 0 <= after <= total:
            raise ValueError('Invalid correct count')
        target = target_correct(before, total)
        result[dataset] = dict(required_correct=target, actual_correct=after,
                               passed=after >= target,
                               gain_percentage_points=100*(after-before)/total)
    return dict(datasets=result, numerical_target_met=all(r['passed'] for r in result.values()),
                independent_confirmation_still_required=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(core, output):
    units = [json.loads(x) for x in (core/'reviewed_units.private.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    sources = {r['key']:r for r in map(json.loads,(core/'sources.private.jsonl').read_text(encoding='utf-8-sig').splitlines())}
    if len(units) != 205 or len({u['id'] for u in units}) != 205:
        raise ValueError('Unexpected reviewed knowledge release')
    # Hold together units sharing an evidence record; later semantic review may
    # merge more groups, never split this grouping to inflate development size.
    parent = {u['id']:u['id'] for u in units}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    owner = {}
    for u in units:
        if u['review_status'] != 'reviewed_selected_assertions':
            raise ValueError('Unreviewed source unit')
        for e in u['evidence']:
            source = sources[e['key']]
            if hashlib.sha256(source['text'].encode()).hexdigest() != e['sha256']:
                raise ValueError('Source fingerprint mismatch')
            if e['key'] in owner:
                a,b = sorted((find(u['id']),find(owner[e['key']])))
                parent[b] = a
            else:
                owner[e['key']] = u['id']
    requests = []
    for u in units:
        group = find(u['id'])
        split = 'dev' if int(hashlib.sha256(group.encode()).hexdigest(),16)%5 == 0 else 'train'
        requests.append(dict(id=u['id'], evidence_group=group, proposed_split=split,
            title=u['title'], scope=u['scope'], source_text=u['body'],
            evidence=u['evidence'], topics=u['topics'],
            required_tasks=['definition_discrimination','boundary_or_counterexample','application_with_new_entities'],
            instruction='Use only this source and its scope. Create self-contained closed-book tasks with a unique answer, a minimal rationale, and exact supporting source spans. Use independently constructed entities and quantities. Do not assume missing company rules. Do not mention benchmark questions. Return unsupported task types as unavailable. Do not put the source passage or its answer in the question.',
            review_status='generation_request_only', training_ready=False))
    output.mkdir(parents=True,exist_ok=False)
    (output/'source_generation_requests.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in requests),encoding='utf-8',newline='\n')
    plan = dict(version='llin-cpt-sft-search-v1',date='2026-09-12',status='prepared_not_started',
        target={k:dict(baseline=c,total=n,required_correct=target_correct(c,n)) for k,(c,n) in BASELINE.items()},
        baseline_protocol='Original frozen prompts, thinking off, temperature 0, seed 1024, max output 96, context 8192, TP8, 3 repeats; never alter labels.',
        primary_benchmarks_are_adaptive_selection_sets=True,
        confirmation='Fresh process rerun after candidate freeze plus separately reported general/Agent retention; three deterministic repeats are not independent training seeds.',
        experiments=[
            dict(id='S1',hypothesis='Direct Step120 SFT with the old verified tasks can learn without the CPT starting-point assumption.',
                 method='SFT',initialization='original_step120_model_only',data='existing_reviewed_591_tasks',
                 train_parquet_sha256='2f10b42c9a8ad0bde49b2c1887a6216727352c4871d1afdc5e84145e70dd9051',
                 learning_rate=2e-7,batch_size=3,epochs=1,steps=197),
            dict(id='S2',hypothesis='The same tasks are under-trained at 2e-7; changing only LR improves independent task development and benchmark scores.',
                 method='SFT',initialization='original_step120_model_only',data='same_as_S1',
                 learning_rate=1e-6,batch_size=3,epochs=1,steps=197),
            dict(id='S3',hypothesis='Task coverage and conditions, rather than more repetitions of easy questions, limit transfer.',
                 method='SFT',status='requires_generated_task_review_and_baseline_probes',
                 data='source_only_tasks_from_205_reviewed_units_plus_retention_examples'),
            dict(id='C1',hypothesis='Selected knowledge exposure is too diluted in full-book CPT.',method='CPT',
                 status='freeze_exposure_and_background_token_budget_after_source_probes',
                 keep_all_source_families_available=True)
        ],
        execution_gates=['SSH and storage/NPU availability','pin model and actual tokenizer',
                         'actual dataset loss-mask and complete 591-record token budget',
                         'source support and semantic grouping for new tasks',
                         'evaluate train/dev before updates','resource lock and immutable run directory'],
        decision_rules=['Compare S1/S2 with same initialization/data/order/steps and only different LR.',
                        'If reference likelihood improves but objective train/dev accuracy does not, do not simply increase epochs.',
                        'If train accuracy improves but dev does not, prioritize data coverage/generalization and retention.',
                        'If train and dev improve, freeze the next exposure test before viewing its results.',
                        'Retain all outcomes; no best-repeat reporting; do not train on official benchmark answers.'],
        source_units=len(units),source_evidence_groups=len({r['evidence_group'] for r in requests}),
        request_splits={s:sum(r['proposed_split']==s for r in requests) for s in ('train','dev')},
        generated_tasks=0,training_started=False,
        source_manifest_sha256=sha(core/'reviewed_units.private.jsonl'),
        requests_sha256=sha(output/'source_generation_requests.private.jsonl'))
    (output/'plan.safe.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    return plan


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--core',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.core,args.output),ensure_ascii=False,indent=2))
