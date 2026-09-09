"""Frozen six-replacement probe; source-only cases, no training or pool promotion."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
from probe_book_task_aligned_value import MODEL_PATHS, messages, read, write, digest, BOOK_SHA, source_units
from build_targeted_book_groups import call_api, unpack, ids_valid
from grade_book_task_aligned_value import SYSTEM, classify_open

SHA='9b400cea4b20ed7639cc45c72379f885d00d286486476a2fc207851a224522b7'

def conditions(q):
    if q['mode'] not in ('closed','evidence'):raise ValueError('Unknown mode')
    return ('closed','evidence') if q['mode']=='closed' else ('evidence',)

def prepare(root,out):
    path=root/'runs/book-task-repair-20260908-02/reviewed_replacements.private.jsonl'
    assert digest(path)==SHA
    cases=[json.loads(x) for x in path.read_text().splitlines()]
    assert len(cases)==6 and len({q['id'] for q in cases})==6
    assert Counter(q['mode'] for q in cases)=={'closed':4,'evidence':2}
    source=root/'runs/logistics-cpt-20260903/private/handbook8e_cpt_4096.jsonl'
    assert digest(source)==BOOK_SHA
    sources={r['record_id']:r['text'] for r in map(json.loads,source.read_text().splitlines())}
    for q in cases:
        text=sources[q['source_id']]; assert hashlib.sha256(text.encode()).hexdigest()==q['source_hash']
        units=source_units(text); assert q['reference_units']=={i:units[i] for i in q['source_ids']}
    out.mkdir(parents=True,exist_ok=False);write(out/'cases.private.json',cases)
    write(out/'protocol.safe.json',dict(items=6,closed_items=4,evidence_only_items=2,source_candidates_sha256=SHA,cases_sha256=digest(out/'cases.private.json'),models=MODEL_PATHS,generations_per_model=10,generations_total=20,max_output_tokens=512,temperature=0,seed=1024,thinking=False,repeats=1,judge_repeats=2,max_api_calls=12,training=False,old_scores_inherited=False,script_sha256=digest(Path(__file__))))

def run(root,out,model):
    meta=read(out/'protocol.safe.json');assert digest(out/'cases.private.json')==meta['cases_sha256']
    cases=read(out/'cases.private.json');dest=out/model;dest.mkdir(exist_ok=False)
    from vllm import LLM,SamplingParams
    (out/'status.txt').write_text('loading_'+model)
    path=root/MODEL_PATHS[model]
    llm=LLM(model=str(path),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,max_num_seqs=64,gpu_memory_utilization=.8,seed=1024,enforce_eager=True)
    tok=llm.get_tokenizer();jobs=[]
    for q in cases:
        for c in conditions(q):
            reference='\n'.join(k+': '+v for k,v in q['reference_units'].items())
            msgs,_=messages(q,c,reference)
            ids=tok.encode(tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
            assert len(ids)+512<=8192;jobs.append((q,c,ids))
    outputs=llm.generate([{'prompt_token_ids':ids} for _,_,ids in jobs],SamplingParams(temperature=0,max_tokens=512,seed=1024))
    assert len(outputs)==len(jobs)==meta['generations_per_model']
    records=[]
    for (q,c,ids),output in zip(jobs,outputs):
        answer=output.outputs[0]
        records.append(dict(id=q['id'],condition=c,model=model,text=answer.text,finish_reason=answer.finish_reason,output_tokens=len(answer.token_ids),prompt_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    with (dest/'answers.private.jsonl').open('x') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    write(dest/'summary.safe.json',dict(model=model,model_path=str(path),model_config_sha256=digest(path/'config.json'),responses=len(records),finish_reasons=dict(Counter(r['finish_reason'] for r in records)),answers_sha256=digest(dest/'answers.private.jsonl'),cases_sha256=meta['cases_sha256'],training=False))
    (out/'status.txt').write_text(model+'_inference_complete')

def grade(out,config_path):
    meta=read(out/'protocol.safe.json');assert digest(out/'cases.private.json')==meta['cases_sha256']
    cases=read(out/'cases.private.json');responses={}
    for m in MODEL_PATHS:
        path=out/m/'answers.private.jsonl';sm=read(out/m/'summary.safe.json')
        assert digest(path)==sm['answers_sha256'] and sm['cases_sha256']==meta['cases_sha256']
        rows=[json.loads(x) for x in path.read_text().splitlines()];assert len(rows)==meta['generations_per_model']
        for r in rows:
            key=(r['id'],m+':'+r['condition']);assert key not in responses;responses[key]=r
    expected={(q['id'],m+':'+c) for q in cases for m in MODEL_PATHS for c in conditions(q)}
    assert set(responses)==expected
    for q in cases:
        for c in conditions(q):assert responses[q['id'],'step120:'+c]['prompt_ids_sha256']==responses[q['id'],'cpt:'+c]['prompt_ids_sha256']
    config=read(config_path);dest=out/'grading';dest.mkdir(exist_ok=False)
    def process(job):
        q,repeat=job;labels=[m+':'+c for m in MODEL_PATHS for c in conditions(q)]
        random.Random('repair-grade-'+q['id']).shuffle(labels)
        if repeat:labels.reverse()
        mapping=dict(zip('ABCD',labels));r=dict(id=q['id'],repeat=repeat,mapping=mapping)
        if any(responses[q['id'],label]['finish_reason']!='stop' for label in labels):r['status']='truncated';return r
        # Dynamic label count is explicit, never pad two-answer evidence tasks with duplicates.
        system=SYSTEM.replace('FOUR anonymous','the supplied anonymous').replace("{id:'A'|'B'|'C'|'D',score", "{id:string,score").replace('Every A/B/C/D must occur EXACTLY once.', 'Every ID in allowed_answer_ids must occur EXACTLY once; no other IDs.')
        system+='\n'+meta.get('judge_policy_addendum','')
        data=dict(question=q['question'],reference_answer=q['answer'],rubric=q['rubric'],numbered_source=q['reference_units'],allowed_source_ids=list(q['reference_units']),allowed_answer_ids=list(mapping),answers=[dict(id=k,text=responses[q['id'],v]['text']) for k,v in mapping.items()])
        try:
            r['api_called']=True;response=call_api(config,system,data,2200);r['response']=response;v=unpack(response);r['verdict']=v
            ans=v.get('answers') if isinstance(v,dict) else None
            valid=isinstance(ans,list) and len(ans)==len(mapping) and all(isinstance(x,dict) for x in ans) and sorted(x.get('id','') for x in ans)==sorted(mapping)
            valid=valid and all(type(x.get('score')) is int and x['score'] in (0,1,2) and ids_valid(x.get('source_ids'),q['reference_units']) and isinstance(x.get('reason'),str) and x['reason'].strip() for x in ans)
            if not valid or any(type(v.get(k)) is not bool for k in ('question_valid','rubric_valid')):r['status']='invalid_grade'
            elif not v['question_valid'] or not v['rubric_valid']:r['status']='quality_quarantine'
            else:r.update(status='scored',scores={mapping[x['id']]:x['score'] for x in ans})
        except Exception as e:r.update(status='api_or_parse_failure',error_type=type(e).__name__)
        return r
    with ThreadPoolExecutor(max_workers=64) as pool: reviews=list(pool.map(process,[(q,i) for q in cases for i in (0,1)]))
    with (dest/'reviews.private.jsonl').open('x') as f:
        for r in reviews:f.write(json.dumps(r)+'\n')
    results=[]
    for q in cases:
        pair=[r for r in reviews if r['id']==q['id']]
        agreed=all(r['status']=='scored' for r in pair) and pair[0]['scores']==pair[1]['scores']
        row=dict(id=q['id'],mode=q['mode'],status='agreed' if agreed else 'unresolved',review_statuses=[r['status'] for r in pair])
        if agreed:
            row['scores']=pair[0]['scores']
            if q['mode']=='closed':row['decisions']={m:classify_open(row['scores'][m+':closed'],row['scores'][m+':evidence']) for m in MODEL_PATHS}
        results.append(row)
    value=dict(items=len(cases),generations=meta['generations_total'],prompt_identity_verified=True,results=results,api_calls=sum(r.get('api_called',False) for r in reviews),usage={k:sum(r.get('response',{}).get('usage',{}).get(k,0) for r in reviews) for k in ('prompt_tokens','completion_tokens','total_tokens')},training=False,training_ready=False,full_groups_reviewed=False,old_pools_modified=False)
    write(out/'result.safe.json',value);(out/'status.txt').write_text('complete_no_training');print(json.dumps(value,indent=2))

if __name__=='__main__':
    os.umask(0o077);p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','grade']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=list(MODEL_PATHS));p.add_argument('--api-config',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.out)
    elif a.action=='run':run(a.root,a.out,a.model)
    else:grade(a.out,a.api_config)
