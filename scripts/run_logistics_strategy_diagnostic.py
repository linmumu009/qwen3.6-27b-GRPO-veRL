"""Frozen, local-only logistics mechanism diagnostic; never produces training data."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

from scripts.evaluate_logistics_knowledge import EvalItem, build_messages, parse_answers

SEED = 20260908
CONDITIONS = ('original', 'permuted', 'deliberate')
CASE_SHA = 'b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def stratum(row):
    return ('sc_knowledge' if row['dataset'] == 'SC-bench-knowledge' else
            'logistika_269' if len(row['options']) == 269 else 'logistika_other')


def permutation(key, count):
    indices = list(range(count))
    random.Random(f'{SEED}:{key}').shuffle(indices)
    return indices


def position_sensitive(row):
    # Conservative flag, not a validated semantic test. All cases remain scored.
    text = row['question'] + '\n' + '\n'.join(row['options'])
    return bool(re.search(r'\b(above|below|preceding|following options|both [A-Z]|option\s+[A-Z0-9]|statements?\s+[A-Z0-9])\b', text, re.I))


def item_of(row):
    return EvalItem(row['item_hash'], row['dataset'], str(row['source_id']),
                    row['category'], row['question_type'], row['question'],
                    tuple(row['options']), tuple(row['expected']))


def messages_for(row, condition):
    item = item_of(row)
    order = list(range(len(item.options)))
    if condition == 'permuted':
        order = permutation(item.item_hash, len(item.options))
        item = replace(item, options=tuple(item.options[i] for i in order))
    messages = build_messages(item)
    if condition == 'deliberate':
        messages[0]['content'] = (
            'Solve the closed-book logistics question using only the listed zero-based option indices. '
            'First check whether the question asks for correct or incorrect statements, '
            'whether it asks for one or multiple answers, and any limits or conditions. '
            'Evaluate the candidate statements against these requirements; check for missing '
            'correct choices and extra incorrect choices. For a long candidate list, identify '
            'the relevant concept and verify the exact matching entry and its index. '
            'You may give concise reasoning before the answer, but do not emit JSON until '
            'the final answer. End with exactly one JSON object of the form {"answers":[0]}.')
    return messages, order


def choose(rows, baseline):
    selected, allocation = [], []
    for group, quota in [('logistika_269',24), ('logistika_other',12), ('sc_knowledge',12)]:
        for correct in (False, True):
            pool = [r for r in rows if stratum(r) == group and bool(baseline[r['item_hash']]['correct']) == correct]
            pool.sort(key=lambda r: hashlib.sha256(f'{SEED}:{r["item_hash"]}'.encode()).hexdigest())
            take = pool[:quota]
            selected.extend(take)
            allocation.append(dict(stratum=group, historical_correct=correct, available=len(pool), selected=len(take), requested=quota))
    return selected, allocation


def aggregate_history(rows, models):
    groups = {}
    for row in rows:
        key = (row['dataset'], row['category'], len(row['options']), 'multi' if len(row['expected']) > 1 else 'single')
        group = groups.setdefault(key, dict(dataset=key[0], category=key[1], choice_count=key[2], answer_cardinality=key[3], items=0, step120_correct=0, cpt_correct=0, improved=0, regressed=0, step120_parse_failures=0, cpt_parse_failures=0))
        group['items'] += 1
        a,b = [models[m][row['item_hash']] for m in ('step120','cpt')]
        for name, pred in [('step120',a),('cpt',b)]:
            group[name+'_correct'] += int(pred['correct'])
            group[name+'_parse_failures'] += int(not pred['parse_ok'])
        group['improved'] += int(not a['correct'] and b['correct'])
        group['regressed'] += int(a['correct'] and not b['correct'])
    return list(groups.values())


def audit_structure(root, out):
    source=root/'runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl'
    assert digest(source)==CASE_SHA
    rows=[json.loads(s) for s in source.read_text().splitlines()]
    norm=lambda s:re.sub(r'\s+',' ',s).strip().casefold()
    banks=Counter(); dupes=Counter(); same=defaultdict(list)
    for row in rows:
        opts=[norm(v) for v in row['options']]; counts=Counter(opts)
        if len(opts)==269:banks[hashlib.sha256(json.dumps(opts).encode()).hexdigest()]+=1
        dupes['items_with_exact_normalized_duplicate_options']+=int(any(v>1 for v in counts.values()))
        dupes['items_with_gold_option_text_duplicated']+=int(any(counts[opts[i]]>1 for i in row['expected']))
        key=hashlib.sha256(json.dumps([norm(row['question']),opts]).encode()).hexdigest()
        same[key].append(tuple(row['expected']))
    report=dict(items=len(rows),large_pool_unique_ordered_banks=len(banks),large_pool_bank_sizes=sorted(banks.values()),
        **dict(dupes),duplicate_question_and_options_groups=sum(len(v)>1 for v in same.values()),
        items_in_duplicate_question_and_options_groups=sum(len(v) for v in same.values() if len(v)>1),
        same_question_options_conflicting_gold_groups=sum(len(set(v))>1 for v in same.values()),
        normalization='casefold whitespace only; not semantic adjudication',private_content_included=False)
    paths={'step120':root/'runs/logistics-cpt-diagnostics-20260904/private/public_eval/step120.majority.jsonl',
        'cpt':root/'runs/logistics-cpt-exposure-curve-diagnostics-20260904/private/public_eval/cpt_4x.majority.jsonl'}
    outcomes={}
    for model,p in paths.items():
        count=Counter()
        for row in [json.loads(s) for s in p.read_text().splitlines()]:
            opts=[norm(v) for v in row['options']];counts=Counter(opts)
            if not any(counts[opts[i]]>1 for i in row['expected']):continue
            count['gold_duplicate_items']+=1;count['strict_correct']+=int(row['correct'])
            if row['parse_ok'] and not row['correct'] and sorted(opts[i] for i in row['parsed'])==sorted(opts[i] for i in row['expected']):
                count['strict_wrong_identical_normalized_text']+=1
        outcomes[model]=dict(count)
    report['historical_duplicate_target_outcomes']=outcomes
    write(out/'structure.safe.json',report)


def prepare(root, out):
    cases = root/'runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl'
    paths = {
        'step120': root/'runs/logistics-cpt-diagnostics-20260904/safe/public_eval/step120.majority.safe.json',
        'cpt': root/'runs/logistics-cpt-exposure-curve-diagnostics-20260904/safe/public_eval/cpt_4x.majority.safe.json'}
    assert digest(cases) == CASE_SHA, 'Case snapshot mismatch'
    rows = [json.loads(s) for s in cases.read_text().splitlines() if s.strip()]
    models = {m: {r['item_hash']:r for r in read(p)['rows']} for m,p in paths.items()}
    ids = {r['item_hash'] for r in rows}
    assert len(rows) == len(ids) == 1672
    for m in models:
        assert set(models[m]) == ids
        for row in rows:
            previous=models[m][row['item_hash']]
            assert int(previous['choice_count']) == len(row['options'])
            assert int(previous['expected_count']) == len(row['expected'])
            assert previous['dataset'] == row['dataset'] and previous['category'] == row['category']
    assert sum(r['correct'] for r in models['cpt'].values()) == 1371
    assert sum(r['correct'] for r in models['step120'].values()) == 1369
    selected, allocation = choose(rows, models['cpt'])
    assert len({r['item_hash'] for r in selected}) == len(selected) <= 96
    out.mkdir(parents=True, exist_ok=False)
    # Strip all historical generated text and reasoning from the private casepack.
    keys = ('item_hash','dataset','source_id','category','question_type','question','options','expected')
    private = [{**{k:r[k] for k in keys}, 'stratum':stratum(r),
                'historical_correct':bool(models['cpt'][r['item_hash']]['correct']),
                'position_sensitive_flag':position_sensitive(r)} for r in selected]
    write(out/'cases.private.json',private)
    summary = dict(seed=SEED, cases_sha256=CASE_SHA, input_hashes={m:digest(p) for m,p in paths.items()},
        selected_sha256=digest(out/'cases.private.json'), allocation=allocation,
        historical_cross=aggregate_history(rows,models), selected_items=len(private),
        selected_position_sensitive=sum(r['position_sensitive_flag'] for r in private),
        conditions=list(CONDITIONS), repeats=3, max_requests=len(private)*18,
        max_output_tokens={'original':96,'permuted':96,'deliberate':2048},
        training=False, private_content_included=False,
        limits=['Stratified case-control cohort; not population accuracy.',
                'Permutation may change index-relative option semantics; flagged cases are not clean causal evidence.',
                'Deliberate changes both prompt and token budget; not a pure reasoning intervention.'])
    write(out/'manifest.safe.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='historical_cross'}), flush=True)


def run(root, out, model):
    manifest = read(out/'manifest.safe.json')
    assert digest(out/'cases.private.json') == manifest['selected_sha256']
    cases = read(out/'cases.private.json')
    target = out/model
    target.mkdir(exist_ok=False)
    paths = {'step120':root/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource',
             'cpt':root/'runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/hf_export_step_116'}
    from vllm import LLM, SamplingParams
    (out/'status.txt').write_text(f'loading_{model}\n')
    llm = LLM(model=str(paths[model]), tensor_parallel_size=8, dtype='bfloat16', trust_remote_code=True,
              max_model_len=8192,max_num_seqs=64,gpu_memory_utilization=.8,seed=1024,enforce_eager=True)
    tok = llm.get_tokenizer()
    jobs = {}
    for condition in CONDITIONS:
        tasks=[]
        for row in cases:
            messages, order = messages_for(row,condition)
            prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(prompt,add_special_tokens=False)
            assert len(ids)+manifest['max_output_tokens'][condition] <= 8192, 'Prompt overflow'
            tasks.append((row,order,ids))
        jobs[condition]=tasks
    timings=[]
    with (target/'predictions.private.jsonl').open('x') as handle:
        for repeat in range(3):
            for condition in (CONDITIONS if repeat%2==0 else tuple(reversed(CONDITIONS))):
                tasks=jobs[condition]
                (out/'status.txt').write_text(f'{model}_{condition}_repeat{repeat+1}\n')
                start=time.monotonic()
                outputs=llm.generate([{'prompt_token_ids':t[2]} for t in tasks],SamplingParams(temperature=0,max_tokens=manifest['max_output_tokens'][condition],seed=1024))
                assert len(outputs)==len(tasks)
                for (row,order,ids), output in zip(tasks,outputs):
                    answer=output.outputs[0]
                    parsed,valid=parse_answers(answer.text,len(order))
                    mapped=sorted(order[i] for i in parsed) if valid else []
                    valid=valid and answer.finish_reason=='stop'
                    record=dict(item_hash=row['item_hash'],model=model,condition=condition,repeat=repeat,
                        mapped=mapped,parse_ok=valid,correct=valid and mapped==sorted(row['expected']),
                        text=answer.text,finish_reason=answer.finish_reason,input_tokens=len(ids),output_tokens=len(answer.token_ids))
                    handle.write(json.dumps(record,ensure_ascii=False)+'\n')
                handle.flush()
                elapsed=time.monotonic()-start
                timings.append(dict(condition=condition,repeat=repeat,seconds=elapsed))
                print(json.dumps(dict(model=model,condition=condition,repeat=repeat+1,items=len(tasks),seconds=elapsed)),flush=True)
    write(target/'run.safe.json',dict(model_path=str(paths[model]),config_sha256=digest(paths[model]/'config.json'),timings=timings,training=False))


def majority(preds):
    assert len(preds)==3 and len({p['repeat'] for p in preds})==3
    counts=Counter(tuple(p['mapped']) for p in preds if p['parse_ok'])
    answer,n=counts.most_common(1)[0] if counts else ((),0)
    return dict(answer=list(answer) if n>=2 else None,correct=sum(p['correct'] for p in preds)>=2,
                stable_wrong=all(p['parse_ok'] and not p['correct'] for p in preds) and n==3,
                repeat_unstable=len({tuple(p['mapped']) if p['parse_ok'] else None for p in preds})>1,
                parse_failures=sum(not p['parse_ok'] for p in preds),truncated=sum(p['finish_reason']=='length' for p in preds))


def summarize(out):
    cases=read(out/'cases.private.json'); results={}; costs={}
    for model in ('step120','cpt'):
        records=[json.loads(s) for s in (out/model/'predictions.private.jsonl').read_text().splitlines()]
        assert len(records)==len(cases)*9
        grouped=defaultdict(list)
        for r in records: grouped[(r['item_hash'],r['condition'])].append(r)
        results[model]={r['item_hash']:{c:majority(grouped[(r['item_hash'],c)]) for c in CONDITIONS} for r in cases}
        costs[model]={c:dict(input_tokens=sum(r['input_tokens'] for r in records if r['condition']==c),output_tokens=sum(r['output_tokens'] for r in records if r['condition']==c)) for c in CONDITIONS}
    table=[]
    for model in results:
        for group in ('logistika_269','logistika_other','sc_knowledge'):
            for history in (False,True):
                subset=[r for r in cases if r['stratum']==group and r['historical_correct']==history]
                for c in CONDITIONS:
                    curr=[results[model][r['item_hash']][c] for r in subset]
                    base=[results[model][r['item_hash']]['original'] for r in subset]
                    table.append(dict(model=model,stratum=group,historical_correct=history,condition=c,items=len(subset),
                        correct=sum(r['correct'] for r in curr),gains=sum(not b['correct'] and v['correct'] for b,v in zip(base,curr)),
                        losses=sum(b['correct'] and not v['correct'] for b,v in zip(base,curr)),
                        answer_changed=sum(b['answer']!=v['answer'] for b,v in zip(base,curr)),
                        parse_failures=sum(r['parse_failures'] for r in curr),truncated=sum(r['truncated'] for r in curr),
                        repeat_unstable=sum(r['repeat_unstable'] for r in curr)))
    stable=[r for r in cases if all(results['cpt'][r['item_hash']][c]['stable_wrong'] for c in CONDITIONS)]
    safe_rows=[dict(item_hash=r['item_hash'],stratum=r['stratum'],historical_correct=r['historical_correct'],
                   position_sensitive_flag=r['position_sensitive_flag'],results={m:{c:{k:v for k,v in results[m][r['item_hash']][c].items() if k!='answer'} for c in CONDITIONS} for m in results}) for r in cases]
    for row in safe_rows:
        for m in results:
            baseline=results[m][row['item_hash']]['original']
            for c in CONDITIONS:
                current=results[m][row['item_hash']][c]
                row['results'][m][c]['valid_majority_pair']=baseline['answer'] is not None and current['answer'] is not None
                row['results'][m][c]['answer_changed']=baseline['answer']!=current['answer']
    write(out/'result.safe.json',dict(items=len(cases),requests=len(cases)*18,table=table,costs=costs,
        stable_wrong_cpt_all_conditions=len(stable),rows=safe_rows,private_content_included=False,training=False))
    # This candidate pack stays private; reading text requires the current bounded permission.
    stable.sort(key=lambda r:hashlib.sha256(f'review:{SEED}:{r["item_hash"]}'.encode()).hexdigest())
    write(out/'review_candidates.private.json',stable[:12])
    (out/'status.txt').write_text('phase1_complete\n')
    print(json.dumps(dict(items=len(cases),stable_wrong_cpt=len(stable))),flush=True)


def prepare_review(root, out):
    selected=read(out/'review_candidates.private.json')
    assert len(selected)<=12
    old_path=root/'runs/step120-qa-diagnostic-20260907/cases.private.json'
    old={r['item_hash']:r for r in read(old_path)}
    banks={};review=[]
    for r in selected:
        bank=hashlib.sha256(json.dumps(r['options'],ensure_ascii=False).encode()).hexdigest()
        banks[bank]=r['options']
        prior=old.get(r['item_hash'])
        if prior:
            assert prior['question']==r['question'] and prior['options']==r['options']
        review.append(dict(item_hash=r['item_hash'],stratum=r['stratum'],question=r['question'],
            options_bank=bank,evidence=prior['evidence'] if prior else [],
            evidence_status='inherited_gold_blind_retrieval_not_verified' if prior else 'unavailable'))
    payload=dict(cases=review,option_banks=banks,source_cases_sha256=digest(old_path),
        new_gold_exposed=False,training_allowed=False)
    write(out/'review_blind.private.json',payload)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','summarize','structure','prepare-review'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=('step120','cpt'))
    a=p.parse_args();os.umask(0o077)
    if a.mode=='prepare':prepare(a.root,a.out)
    elif a.mode=='run':run(a.root,a.out,a.model)
    elif a.mode=='structure':audit_structure(a.root,a.out)
    elif a.mode=='prepare-review':prepare_review(a.root,a.out)
    else:summarize(a.out)


if __name__=='__main__':main()
