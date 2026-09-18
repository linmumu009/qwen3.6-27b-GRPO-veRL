"""Frozen four-order baseline and blind source objections, with no training."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
from run_cpt_remediation_review import review_prompt,review_decision,MODELS
from run_cpt_source_authoring_pilot import closed_prompt,answer
from run_cpt_transfer_screen import digest
from run_cpt_formal_transfer import model_identity,ROOT
from prepare_cpt_remediation_packet import sha

def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def save(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

def specs(rows):
    result=[]
    for row in rows:
        order=list(range(4));random.Random(int(hashlib.sha256(row['id'].encode()).hexdigest()[:16],16)).shuffle(order)
        for variant in range(4):
            permutation=order[variant:]+order[:variant]
            result.append(dict(id=row['id'],variant=variant,order=permutation,
                expected=sorted(i for i,j in enumerate(permutation) if j in row['correct_indices'])))
    return result

def summarize(rows,raw,reviews,label):
    plan=specs(rows);assert len(raw)==len(plan)==352
    by_id={r['id']:r for r in rows};counts=Counter();details=[];per_task=defaultdict(list)
    for s,r in zip(plan,raw):
        row=by_id[s['id']];assert all(s[k]==r[k] for k in ('id','variant','order'))
        assert r['prompt_text_sha256']==hashlib.sha256(closed_prompt(row,s).encode()).hexdigest()
        assert 0<r['output_tokens']<=96 and r['prompt_tokens']+96<=8192
        pred=answer(r['text']) if r['finish_reason']=='stop' else None;good=pred==s['expected']
        counts[row['split']+'_correct']+=good;counts['invalid']+=pred is None;counts['truncated']+=r['finish_reason']!='stop'
        original=sorted(s['order'][i] for i in pred) if pred is not None else None
        detail=dict(id=s['id'],variant=s['variant'],split=row['split'],unit=row['unit'],correct=good,original_order_prediction=original)
        details.append(detail);per_task[s['id']].append(detail)
    stability=[]
    for identity,items in per_task.items():
        assert len(items)==4
        stability.append(dict(id=identity,split=by_id[identity]['split'],unit=by_id[identity]['unit'],correct_orders=sum(x['correct'] for x in items),
            invariant_answer=len({json.dumps(x['original_order_prediction']) for x in items})==1))
    assert len(reviews)==(88 if label=='p1' else 0)
    decisions=[]
    for row,r in zip(rows,reviews):
        assert r['id']==row['id'] and r['prompt_text_sha256']==hashlib.sha256(review_prompt(row).encode()).hexdigest()
        assert 0<r['output_tokens']<=1536 and r['prompt_tokens']+1536<=8192
        decisions.append(review_decision(row,r))
    return dict(model_label=label,counts=dict(counts),per_call=details,per_task=stability,
        all_orders_correct={s:sum(x['correct_orders']==4 for x in stability if x['split']==s) for s in ('train','dev','retention')},
        invariant_answer_tasks=sum(x['invariant_answer'] for x in stability),review_decisions=decisions,review_accepted=sum(x['accepted'] for x in decisions),
        closed_calls=352,review_calls=len(reviews),training_allowed=False)

def run(packet,model_manifest,out,label):
    import fcntl
    manifest=json.loads((packet/'manifest.safe.json').read_text());rows=read(packet/'cases.private.jsonl')
    assert len(rows)==88 and all(r['training_allowed'] is False for r in rows)
    for name,h in manifest['files'].items():assert sha(packet/name)==h
    assert json.loads((packet/'audit.safe.json').read_text())['packet_sha256']==manifest['files']['cases.private.jsonl']
    identity=json.loads(model_manifest.read_text());model=Path(MODELS[label]);assert model_identity(model)==identity
    os.umask(0o077);out.mkdir(exist_ok=False)
    def status(stage,**kw):save(out/'status.safe.json',dict(status=stage,training_allowed=False,**kw))
    status('waiting_for_lock')
    try:
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            from vllm import LLM,SamplingParams
            tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
            plan=specs(rows);by_id={r['id']:r for r in rows}
            texts=[closed_prompt(by_id[s['id']],s) for s in plan]+([review_prompt(r) for r in rows] if label=='p1' else [])
            prompts=[]
            for i,t in enumerate(texts):
                ids=tok.encode(tok.apply_chat_template([dict(role='user',content=t)],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
                assert len(ids)+(96 if i<352 else 1536)<=8192;prompts.append(dict(prompt_token_ids=ids))
            save(out/'registration.safe.json',dict(model=str(model),model_manifest_sha256=sha(model_manifest),packet_sha256=manifest['files']['cases.private.jsonl'],
                seed=20922,temperature=0,closed_calls=352,review_calls=88 if label=='p1' else 0,option_orders=4,max_closed_tokens=96,max_review_tokens=1536,
                max_model_len=8192,tp=8,max_num_seqs=16,chunk=8,thinking=False,code_sha256=sha(Path(__file__)),training_allowed=False))
            status('loading_model')
            llm=LLM(model=str(model),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,max_num_seqs=16,gpu_memory_utilization=.8,seed=20922,enforce_eager=True)
            stages=[]
            for name,start,end,budget in [('closed.private.jsonl',0,352,96),('review.private.jsonl',352,len(texts),1536)]:
                raw=[]
                with (out/name).open('x',encoding='utf-8') as f:
                    for first in range(start,end,8):
                        ps=prompts[first:min(first+8,end)]
                        outputs=sorted(llm.generate(ps,SamplingParams(temperature=0,max_tokens=budget,seed=20922)),key=lambda x:int(x.request_id));assert len(outputs)==len(ps)
                        for i,o in enumerate(outputs,first):
                            ids=prompts[i]['prompt_token_ids'];assert list(o.prompt_token_ids)==ids;v=o.outputs[0]
                            meta={k:plan[i][k] for k in ('id','variant','order')} if i<352 else dict(id=rows[i-352]['id'])
                            r=dict(meta,text=v.text,finish_reason=v.finish_reason,output_tokens=len(v.token_ids),prompt_tokens=len(ids),prompt_text_sha256=hashlib.sha256(texts[i].encode()).hexdigest(),prompt_token_sha256=digest(ids))
                            raw.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n')
                        f.flush();status(name,completed=len(raw),total=end-start)
                stages.append(raw)
            assert model_identity(model)==identity
            result=summarize(rows,*stages,label);save(out/'summary.safe.json',result);status('completed_pending_operator_audit')
    except BaseException as error:status('failed',error=type(error).__name__+': '+str(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('packet','model-manifest','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--model-label',choices=list(MODELS),required=True);a=p.parse_args();run(a.packet,a.model_manifest,a.out,a.model_label)
