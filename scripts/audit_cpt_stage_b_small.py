"""Blind export, frozen-review comparison, and bounded local overlap checks."""
import argparse
import json
import re
from pathlib import Path
from cpt_stage_b import read,save,sha,digest,object_of
from cpt_stage_b_citations import source_blocks,validate_option_evidence


def raw_tasks(base):
    requests=read(base/'author_requests.private.json')['requests']
    raw=[json.loads(x) for x in (base/'author/predictions.private.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(raw)==len(requests)==8
    parsed=[]
    for req,row in zip(requests,raw):
        assert req['id']==row['id'] and row['messages_sha256']==digest(req['messages'])
        try:
            task=object_of(row['text'])
            if not isinstance(task,dict):raise ValueError('not object')
        except (ValueError,TypeError):task={'parse_error':True}
        parsed.append((req,row,task))
    return parsed


def blind(base):
    rows=[]
    for req,raw,task in raw_tasks(base):
        rows.append(dict(id=req['id'],group=req['group'],form=req['form'],question=task.get('question'),options=task.get('options'),
                         complete=raw['finish_reason']=='stop' and 'parse_error' not in task,task_sha256=digest(task)))
    save(base/'blind_tasks.private.json',rows)
    return rows


def compare(base,out):
    judgments=read(base/'blind_judgments.private.json')
    blind_rows=read(base/'blind_tasks.private.json')
    assert judgments['blind_tasks_sha256']==sha(base/'blind_tasks.private.json')
    by_id={r['id']:r for r in judgments['records']}
    groups={g['id']:g for g in read(base/'groups.private.json')['groups']}
    result=[]
    for req,raw,task in raw_tasks(base):
        review=by_id[req['id']];errors=[];g=groups[req['group']]
        key=None
        evidence=task.get('option_evidence')
        if isinstance(evidence,list) and all(isinstance(r,dict) and type(r.get('option_index')) is int and r.get('relation') in ('entailed','contradicted','insufficient') for r in evidence):
            key=sorted(r['option_index'] for r in evidence if r['relation']=='entailed')
        try:
            assert set(task)=={'question','options','option_evidence','reasoning_structure'},'malformed or truncated task schema'
            assert raw['finish_reason']=='stop' and raw['output_tokens']<=2048,'incomplete generation'
            assert isinstance(task['question'],str) and 'Select all correct statements.' in task['question'],'registered explicit instruction missing'
            assert len(task['question'].split())<=150,'question word limit'
            opts=task['options'];assert isinstance(opts,list) and len(opts)==g['option_count'],'option count'
            assert all(isinstance(o,str) and o.strip() and len(o.split())<=40 for o in opts),'option word limit/type'
            assert len(set(o.casefold().strip() for o in opts))==len(opts),'duplicate options'
            validate_option_evidence(task['option_evidence'],len(opts),source_blocks(g))
            assert all(r['relation']!='insufficient' for r in task['option_evidence'])
            key=sorted(r['option_index'] for r in task['option_evidence'] if r['relation']=='entailed')
            assert 0<len(key)<len(opts)
        except (AssertionError,ValueError,TypeError,KeyError) as e:
            errors.append(type(e).__name__+': '+str(e))
        agree=key is not None and review['correct_indices'] is not None and key==review['correct_indices']
        result.append(dict(id=req['id'],group=req['group'],form=req['form'],structural_errors=errors,author_key=key,
            blind_key=review['correct_indices'],key_agreement=agree,blind_quality_pass=review['quality_pass'],
            quality_pass=not errors and agree and review['quality_pass'],review_note=review['reason'],task_sha256=digest(task)))
    save(out,dict(author_calls=8,additional_review_model_calls=0,blind_review_by_different_model=True,
        original_author_evidence_hidden_until_judgments_frozen=True,judgments_sha256=sha(base/'blind_judgments.private.json'),
        group_quality_pass={g:all(r['quality_pass'] for r in result if r['group']==g) for g in groups},records=result,
        overlap_still_required=True,training_allowed=False))


def normalize(text):
    return ' '.join(re.sub(r'\d+(?:\.\d+)?','#',text.casefold()).split())


def overlap(base,root,out):
    from verify_cpt_premise_tasks import EXCLUSION_FILES
    paths=[root/'llin-transfer-audit-20260915-01/frozen_cases.private.jsonl',root/'llin-transfer-p4-data-20260918-01/train.messages.private.jsonl']+[root/n for n in EXCLUSION_FILES]
    corpus=[];inputs=[];missing=[]
    def collect(v,path):
        if isinstance(v,dict):
            if isinstance(v.get('question'),str):corpus.append((path,v['question']))
            if isinstance(v.get('messages'),list):
                for m in v['messages']:
                    if m.get('role')=='user':corpus.append((path,m['content'].split('\n\nOptions:')[0].removeprefix('Question:\n')))
            for x in v.values():collect(x,path)
        elif isinstance(v,list):
            for x in v:collect(x,path)
    for path in paths:
        if not path.exists():missing.append(str(path));continue
        rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        collect(rows,str(path));inputs.append(dict(path=str(path.relative_to(root)),sha256=sha(path)))
    old=root/'llin-stage-b-20260920-01/run04/calls.private.jsonl'
    for row in map(json.loads,old.read_text(encoding='utf-8').splitlines()):
        if row['stage']=='author':collect(object_of(row['text']),str(old))
    inputs.append(dict(path=str(old.relative_to(root)),sha256=sha(old)))
    tasks=read(base/'blind_tasks.private.json');output=[]
    def shingles(text):
        w=re.findall(r'[a-z]+|#',normalize(text));return {' '.join(w[i:i+5]) for i in range(len(w)-4)}
    signatures=[(p,q,normalize(q),shingles(q)) for p,q in corpus]
    for t in tasks:
        q=t['question'];hits=[]
        if not isinstance(q,str):
            output.append(dict(id=t['id'],unavailable=True,hits=[]));continue
        s=shingles(q)
        for p,oldq,n,oldset in signatures:
            exact=normalize(q)==n;ratio=len(s&oldset)/max(1,min(len(s),len(oldset)))
            if exact or ratio>=.3:hits.append(dict(source=p,question_sha256=digest(oldq),exact_number_normalized=exact,shared_5gram_fraction_of_shorter=ratio))
        output.append(dict(id=t['id'],hits=sorted(hits,key=lambda h:-h['shared_5gram_fraction_of_shorter'])[:12]))
    save(out,dict(corpus_records=len(corpus),unique_normalized_questions=len({n for _,_,n,_ in signatures}),inputs=inputs,missing=missing,
        records=output,threshold=.3,limitation='Lexical question-only screen; semantic novelty and option overlap require manual review. Corpus limited to enumerated historical artifacts.',training_allowed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['blind','compare','overlap']);p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('CPT_resources'));p.add_argument('--out',type=Path);a=p.parse_args()
    if a.mode=='blind':print(json.dumps(blind(a.base),ensure_ascii=False,indent=2))
    elif a.mode=='compare':compare(a.base,a.out)
    else:overlap(a.base,a.root,a.out)
