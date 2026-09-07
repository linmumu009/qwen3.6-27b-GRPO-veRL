"""Freeze historical errors plus matched controls; retrieve independent handbook excerpts."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re

from scripts.prepare_logistics_mcq_fit import SOURCE_SHA256
from scripts.run_vllm_logistics_mcq import load_items
from scripts.evaluate_logistics_knowledge import load_private_rows, PROMPT_VERSION

BASELINE_SHA256='76d1cf72e4f013d144a402efa179b3a21d7af77b6527bb8971a0609557aa66b0'
STOP=set('the a an of to in is are and or for with which what how that this by on as from be at it not'.split())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def words(text):
    return [w for w in re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]',text.lower()) if w not in STOP]


def choose_cohort(items,baseline,controls=60):
    if set(baseline)!={r.item_hash for r in items}:
        raise ValueError('baseline/source coverage mismatch')
    for item in items:
        row=baseline[item.item_hash]
        if (row['prompt_version']!=PROMPT_VERSION or not row['chat_template_disable_thinking']
            or row['question']!=item.question or tuple(row['options'])!=item.options
            or tuple(row['expected'])!=item.expected):
            raise ValueError('baseline/source content mismatch')
        if bool(row['correct'])!=(bool(row['parse_ok']) and tuple(row['parsed'])==item.expected):
            raise ValueError('baseline correctness mismatch')
    errors=sorted((r for r in items if not baseline[r.item_hash]['correct']),
        key=lambda r:(r.dataset,r.category,r.question_type,r.item_hash))
    pool={r.item_hash:r for r in items if baseline[r.item_hash]['correct']}
    if not errors or len(pool)<controls or controls<1:
        raise ValueError('insufficient errors/controls')
    selected=[]
    for i in range(controls):
        anchor=errors[min(len(errors)-1,int((i+0.5)*len(errors)/controls))]
        def distance(r):
            return (r.dataset!=anchor.dataset,r.question_type!=anchor.question_type,
                r.category!=anchor.category,abs(len(r.options)-len(anchor.options)),
                abs(len(r.question)-len(anchor.question)),r.item_hash)
        pick=min(pool.values(),key=distance)
        selected.append((pick,anchor.item_hash,distance(pick)[:3]==(False,False,False)))
        del pool[pick.item_hash]
    return errors,selected


def retrieve(question,options,windows,limit=3):
    # Gold labels are deliberately absent from retrieval inputs.
    stem_terms=set(words(question))
    query=stem_terms | set(words(' '.join(options)))
    counts=[Counter(words(w['text'])) for w in windows]
    df=Counter(term for row in counts for term in row)
    average=sum(sum(row.values()) for row in counts)/max(len(counts),1)
    scores=[]
    for i,row in enumerate(counts):
        length=sum(row.values())
        score=sum((3 if t in stem_terms else 1)*math.log(1+(len(counts)-df[t]+0.5)/(df[t]+0.5))*row[t]*2.5/
            (row[t]+1.5*(0.25+0.75*length/max(average,1))) for t in query & row.keys())
        scores.append((score,i))
    # Avoid heavily overlapping windows from the same source block.
    selected=[]
    for score,index in sorted(scores,key=lambda v:(-v[0],v[1])):
        w=windows[index]
        if score<=0:
            break
        if any(w['record']==s['record'] and abs(w['start_token']-s['start_token'])<384 for s in selected):
            continue
        selected.append({**w,'retrieval_score':score})
        if len(selected)==limit:
            break
    return selected


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('refusing overwrite')
    source=args.root/'runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl'
    baseline=source.with_name('step120.majority.jsonl')
    book=args.root/'runs/logistics-cpt-20260903/private/handbook8e_cpt_4096.parquet'
    if digest(source)!=SOURCE_SHA256 or digest(baseline)!=BASELINE_SHA256:
        raise ValueError('frozen source/baseline hash mismatch')
    items=load_items(source)
    errors,controls=choose_cohort(items,load_private_rows(baseline))
    if len(items)!=1672 or len(errors)!=303:
        raise ValueError('expected original 303 errors in 1672 cases')
    from transformers import AutoTokenizer
    import pandas as pd
    model=args.root/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource'
    tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=True)
    windows=[]
    for record,text in enumerate(pd.read_parquet(book)['text']):
        ids=tokenizer.encode(text,add_special_tokens=False)
        for start in range(0,len(ids),288):
            chunk=ids[start:start+384]
            if len(chunk)<80:
                continue
            passage=tokenizer.decode(chunk,skip_special_tokens=True)
            windows.append({'record':record,'start_token':start,'token_count':len(chunk),
                'text':passage,'text_sha256':hashlib.sha256(passage.encode()).hexdigest()})
    os.umask(0o077)
    args.output.mkdir(parents=True)
    (args.output/'status.txt').write_text('retrieving_independent_book_excerpts\n')
    control_map={r.item_hash:(anchor,exact) for r,anchor,exact in controls}
    chosen=sorted(errors+[r for r,_,_ in controls],key=lambda r:r.item_hash)
    cases=[]
    for item in chosen:
        group='original_error' if item.item_hash not in control_map else 'original_correct_control'
        cases.append({**item.__dict__,'cohort':group,
            'matched_error_hash':control_map.get(item.item_hash,(None,False))[0],
            'evidence':retrieve(item.question,item.options,windows)})
    write_json(args.output/'cases.private.json',cases)
    manifest={'private_content_included':False,'training_allowed':False,
        'source_sha256':SOURCE_SHA256,'historical_majority_sha256':BASELINE_SHA256,
        'book_parquet_sha256':digest(book),'book_source':'user-authorized Handbook of Logistics and Distribution Management, 8th edition',
        'cases_sha256':digest(args.output/'cases.private.json'),
        'errors':len(errors),'controls':len(controls),'items':len(cases),
        'exact_dataset_type_category_controls':sum(exact for _,_,exact in controls),
        'cohort_by_dataset':dict(Counter(r['cohort']+' / '+r['dataset'] for r in cases)),
        'retrieval_queries_use_gold':False,'book_windows':len(windows),
        'zero_retrieval_cases':sum(not r['evidence'] for r in cases),
        'coverage_note':'lexical retrieval is not verified sufficiency; a single handbook may omit topics or jurisdictions',
        'historical_error_selection_not_recomputed':True,
        'control_sampling':'60 systematic error anchors stratified by dataset/category/type; nearest unused original-correct match',
        'model':'Step120, not any exam-CPT/SFT checkpoint'}
    write_json(args.output/'cohort.safe.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
