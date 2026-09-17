"""Read-only P1/P2 replay coverage audit and frozen training-fit probe builder.

No training data are produced. Coverage annotations are reviewed task operations,
not a keyword-derived causal claim. Existing retention answers only enter analysis.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

REVIEWED_SHA='6ac62685258fc72815095ea13b2afbc8d96a58faf6464f0f8e955306af3b7a89'
ARITHMETIC='llin-core-9adae425cbbabbd2bf0c-application_with_new_entities'
COUNTING='llin-core-b5b62400c11c1cace1b4-application_with_new_entities'

def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rank(r):return hashlib.sha256(('p1-replay-'+r['id']).encode()).hexdigest()

def cases_from_reviewed(rows):
    train=[r for r in rows if r['split']=='train']
    assert len(train)==135 and len({r['id'] for r in train})==135
    return [dict(dataset='replay_training_fit',source_id='p1-replay-'+r['id'],
        category=r['task']['form'],question_type='multiple_choice',
        question=r['task']['question'],options=r['task']['options'],expected=r['task']['correct_indices']) for r in train]

def build(resources,out):
    old_path=resources/'llin-source-reviewed-20260914-01/reviewed_tasks.private.jsonl'
    assert sha(old_path)==REVIEWED_SHA
    old=read(old_path);train=[r for r in old if r['split']=='train'];extra=sorted(train,key=rank)[:60]
    packet=resources/'llin-transfer-p1-20260916-03'
    target=read(packet/'train.private.jsonl');retention=read(packet/'retention.private.jsonl')
    target_sources={s for r in target for s in r['source_ids']}
    retention_sources={s for r in retention for s in r['source_ids']}
    assert not {r['unit_id'] for r in train}&retention_sources
    def coverage(rows):
        return dict(exposures=len(rows),unique_tasks=len({r['id'] for r in rows}),
            declared_forms=dict(Counter(r['task']['form'] for r in rows)),
            numerical_calculation=sum(r['id']==ARITHMETIC for r in rows),
            categorical_vehicle_count=sum(r['id']==COUNTING for r in rows),
            target_source_overlap=sum(r['unit_id'] in target_sources for r in rows),
            retention_source_overlap=sum(r['unit_id'] in retention_sources for r in rows))
    paths=dict(base=resources/'llin-transfer-p2-result-20260917-01/baseline',
        base_repeat=resources/'llin-transfer-p2-result-20260917-01/baseline_repeat',
        p1=packet/'complete_result/post',p1_repeat=resources/'llin-transfer-p1-stability-20260916-02/post_repeat',
        p2=resources/'llin-transfer-p2-result-20260917-01/post')
    predictions={k:{r['source_id']:r for r in read(p/'predictions.private.jsonl')} for k,p in paths.items()}
    family={};changes=[]
    for unit in sorted({r['unit'] for r in retention}):
        ids=[r['id'] for r in retention if r['unit']==unit]
        family[unit]=dict(n=len(ids),correct={k:sum(v[i]['correct'] for i in ids) for k,v in predictions.items()})
    for r in retention:
        flags={k:v[r['id']]['correct'] for k,v in predictions.items()}
        if len(set(flags.values()))>1:changes.append(dict(id=r['id'],family=r['unit'],correct=flags))
    out.mkdir(exist_ok=False,parents=True)
    cases=cases_from_reviewed(old)
    (out/'cases.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases),encoding='utf-8')
    result=dict(reviewed_sha256=sha(old_path),cases_sha256=sha(out/'cases.private.jsonl'),
        replay_p1=coverage(train),replay_p2=coverage(train+extra),p2_added=coverage(extra),
        arithmetic_annotation=dict(calculation_task=ARITHMETIC,counting_task=COUNTING,
            scope='Agent operation review of all 135 questions; numeric thresholds and quoted quantities are not arithmetic exercises.'),
        retention=dict(n=80,numeric_tasks=sum(isinstance(r['premises']['expected'],int) for r in retention),families=family,changed_items=changes),
        input_sha256={k:sha(p/'predictions.private.jsonl') for k,p in paths.items()},
        interpretation='Coverage mismatch is a descriptive finding, not proof of gradient conflict or causal forgetting. Zero held-out source overlap is intentional, not a defect to repair by leaking these sources.',
        training=False,formal_evaluation=False,requests_planned=405,
        prospective_probe='All 135 original replay training tasks once per Step120/P1/P2; same prompts and labels as actual training. Report all, do not use fit as retention or promotion.')
    save(out/'coverage.safe.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resources',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.resources,a.out),indent=2))
