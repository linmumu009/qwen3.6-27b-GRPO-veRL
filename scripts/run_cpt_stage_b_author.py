"""Bounded source-only authoring. Stops for local review; never evaluates or trains."""
import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from pathlib import Path
from cpt_stage_b import sha,save,read,digest,author_prompt,reviewer_prompt,validate_task,review_ok,object_of,FORMS,device_idle,encode_prompt

ROOT=Path('/workspace/llin-verl-grpo')
MODELS={'step120_current':ROOT/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource',
        'p1':ROOT/'runs/llin-transfer-p1-run-20260916-02/training/llin-step120-p1-hf'}


def identity(model):
    index=read(model/'model.safetensors.index.json'); shards=sorted(set(index['weight_map'].values()))
    if len(index['weight_map'])!=1199 or len(shards)!=15: raise ValueError('model layout')
    out=dict(path=str(model),metadata_sha256={},shards_stat={},weight_content_hashed=False)
    for name in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'):
        out['metadata_sha256'][name]=sha(model/name)
    if out['metadata_sha256']['tokenizer.json']!='06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523': raise ValueError('tokenizer identity')
    for name in shards:
        if Path(name).name!=name: raise ValueError('shard path')
        s=(model/name).stat();out['shards_stat'][name]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    return out


def idle():
    result=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30,check=True)
    return device_idle(result.stdout)


def run(a):
    os.umask(0o077);a.out.mkdir(parents=True,exist_ok=False)
    def status(state,**kw):
        tmp=a.out/'status.tmp';save(tmp,dict(status=state,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),training_allowed=False,**kw));tmp.replace(a.out/'status.safe.json')
    try:
        manifest=read(a.packet/'manifest.safe.json');path=a.packet/'sources.private.json'
        if sha(path)!=manifest['source_packet_sha256']: raise ValueError('packet fingerprint')
        packet=read(path)
        if packet['training_allowed'] is not False or len(packet['groups'])!=8 or packet['forms']!=list(FORMS): raise ValueError('packet contract')
        allowed={'id','scope','operation','option_count','source_text','source_refs'}
        if any(set(g)!=allowed for g in packet['groups']): raise ValueError('author source isolation')
        identities={k:identity(v) for k,v in MODELS.items()};save(a.out/'models.safe.json',identities)
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(MODELS['step120_current'],trust_remote_code=True,local_files_only=True)
        def tokens(text):
            return encode_prompt(tok,text)
        requests=[dict(id=g['id']+'-'+f,group=g['id'],form=f,prompt=author_prompt(g,f)) for g in packet['groups'] for f in FORMS]
        for r in requests:r['tokens']=tokens(r['prompt'])
        save(a.out/'registration.safe.json',dict(source_packet_sha256=sha(path),author_calls_max=64,author_requests=32,author_temperature=.4,review_temperature=0,author_seed=20920,review_seed=20921,max_tokens=2048,max_model_len=8192,max_num_seqs=32,tp=8,wait_seconds=a.wait_seconds,models_sha256=sha(a.out/'models.safe.json'),code_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('cpt_stage_b.py'))},training_allowed=False))
        deadline=time.monotonic()+a.wait_seconds
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            while True:
                locked=False
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
                    ready,used=idle()
                    if ready: break
                except BlockingIOError: used=[]
                finally:
                    if locked and not locals().get('ready',False):fcntl.flock(lock,fcntl.LOCK_UN)
                if time.monotonic()>=deadline:status('expired_waiting_for_device');return
                status('waiting_for_device',memory_used_mib=used,deadline_remaining_seconds=round(deadline-time.monotonic()))
                time.sleep(min(60,max(0,deadline-time.monotonic())))
            if identities!={k:identity(v) for k,v in MODELS.items()}: raise ValueError('model changed while waiting')
            status('loading_author_model')
            from vllm import LLM,SamplingParams
            llm=LLM(model=str(MODELS['step120_current']),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,max_num_seqs=32,gpu_memory_utilization=.8,seed=20920,enforce_eager=True)
            calls=0
            with (a.out/'calls.private.jsonl').open('x',encoding='utf-8') as log:
                def query(text,stage,id):
                    nonlocal calls
                    ids=tokens(text)
                    if calls>=64:raise ValueError('author budget exhausted')
                    try:
                        outputs=llm.generate([dict(prompt_token_ids=ids)],SamplingParams(temperature=.4 if stage=='author' else 0,max_tokens=2048,seed=20920 if stage=='author' else 20921))
                    except Exception as e:raise RuntimeError('inference failed') from e
                    calls+=1
                    if len(outputs)!=1 or list(outputs[0].prompt_token_ids)!=ids:raise RuntimeError('prompt identity')
                    v=outputs[0].outputs[0]
                    row=dict(id=id,stage=stage,prompt_sha256=digest(text),prompt_token_sha256=digest(ids),text=v.text,finish_reason=v.finish_reason,output_tokens=len(v.token_ids))
                    log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush();status('authoring',calls=calls,maximum=64)
                    if v.finish_reason!='stop':raise ValueError('truncated response')
                    return object_of(v.text)
                candidates=[]
                for r in requests:
                    g=next(g for g in packet['groups'] if g['id']==r['group'])
                    row={k:r[k] for k in ('id','group','form')}
                    try:
                        task=validate_task(query(r['prompt'],'author',r['id']),g)
                        row['task']=task
                        review=query(reviewer_prompt(g,task,r['form']),'review',r['id'])
                        row.update(review=review,blind_review_pass=review_ok(review,task))
                    except (ValueError,TypeError,KeyError,AttributeError) as e:row.update(blind_review_pass=False,rejection=str(e))
                    candidates.append(row);save(a.out/'candidates.private.json',candidates)
            if identities!={k:identity(v) for k,v in MODELS.items()}:raise ValueError('model identity changed')
            save(a.out/'summary.safe.json',dict(calls=calls,candidates=len(candidates),blind_review_pass=sum(r['blind_review_pass'] for r in candidates),local_review_required=True,diagnostic_calls=0,training_allowed=False))
            status('completed_pending_local_quality_and_overlap_review',calls=calls)
    except BaseException as e:status('failed',error=type(e).__name__+': '+str(e));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--wait-seconds',type=int,default=21600);a=p.parse_args()
    if not 0<=a.wait_seconds<=21600:raise ValueError('bounded wait must be <=6 hours')
    run(a)
