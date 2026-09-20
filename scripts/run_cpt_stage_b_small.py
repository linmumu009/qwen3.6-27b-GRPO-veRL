"""One bounded author batch, or a manually released diagnostic batch."""
import argparse
import fcntl
import json
import time
from pathlib import Path
from cpt_stage_b import read,save,sha,digest,specs,messages,summarize
from run_cpt_stage_b_author import ROOT,MODELS,identity,idle


def run(a):
    if sha(a.input)!=a.input_sha256:raise ValueError('input fingerprint')
    packet=read(a.input)
    if packet['training_allowed'] is not False:raise ValueError('training prohibited')
    author=packet.get('mode')=='author'
    if author:
        if a.label!='step120_current' or len(packet['requests'])!=8:raise ValueError('author contract')
        plan=packet['requests'];texts=[r['messages'] for r in plan];budget=2048;seed=20922;temperature=.4
    else:
        if packet.get('quality_released') is not True or packet.get('review_sha256') is None or not 1<=len(packet['groups'])<=2:raise ValueError('quality gate')
        plan=specs(packet,a.label);texts=[messages(packet,r) for r in plan];budget=96;seed=1024;temperature=0
    a.out.mkdir(parents=True,exist_ok=False)
    def status(s,**kw):
        p=a.out/'status.tmp';save(p,dict(status=s,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),training_allowed=False,**kw));p.replace(a.out/'status.safe.json')
    status('preflight')
    try:
        before={k:identity(v) for k,v in MODELS.items()};save(a.out/'models.safe.json',before)
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(MODELS[a.label],trust_remote_code=True,local_files_only=True)
        prompts=[]
        for msg in texts:
            rendered=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(rendered,add_special_tokens=False)
            if not ids or any(type(i) is not int or i<0 for i in ids) or len(ids)+budget>8192:raise ValueError('token/window contract')
            prompts.append(dict(prompt_token_ids=ids))
        save(a.out/'registration.safe.json',dict(input_sha256=a.input_sha256,mode='author' if author else 'diagnostic',model_label=a.label,
            calls=len(plan),max_tokens=budget,temperature=temperature,seed=seed,max_model_len=8192,max_num_seqs=32,tp=8,thinking=False,
            prompt_tokens=[len(x['prompt_token_ids']) for x in prompts],code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['run_cpt_stage_b_small.py','run_cpt_stage_b_author.py','cpt_stage_b.py','evaluate_logistics_knowledge.py']},training_allowed=False))
        deadline=time.monotonic()+a.wait_seconds
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            while True:
                ready=False;locked=False;used=[]
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True;ready,used=idle()
                    if ready:break
                except BlockingIOError:pass
                finally:
                    if locked and not ready:fcntl.flock(lock,fcntl.LOCK_UN)
                if time.monotonic()>=deadline:status('expired_waiting_for_device');return
                status('waiting_for_device',memory_used_mib=used);time.sleep(min(60,deadline-time.monotonic()))
            if before!={k:identity(v) for k,v in MODELS.items()}:raise ValueError('model changed')
            status('loading_model')
            from vllm import LLM,SamplingParams
            llm=LLM(model=str(MODELS[a.label]),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,max_num_seqs=32,gpu_memory_utilization=.8,seed=seed,enforce_eager=True)
            raw=[]
            with (a.out/'predictions.private.jsonl').open('x',encoding='utf-8') as f:
                # Author batch uses eight independent contexts. Diagnostic batches
                # keep the original prompt order; exact repeats are later in plan.
                for first in range(0,len(plan),32):
                    batch=prompts[first:first+32];status('generating',completed=len(raw),total=len(plan))
                    outputs=sorted(llm.generate(batch,SamplingParams(temperature=temperature,max_tokens=budget,seed=seed)),key=lambda o:int(o.request_id))
                    if len(outputs)!=len(batch):raise ValueError('output count')
                    for i,o in enumerate(outputs,first):
                        if list(o.prompt_token_ids)!=prompts[i]['prompt_token_ids']:raise ValueError('prompt mismatch')
                        v=o.outputs[0];r={k:v for k,v in plan[i].items() if k!='messages'}
                        r.update(text=v.text,finish_reason=v.finish_reason,output_tokens=len(v.token_ids),messages_sha256=digest(texts[i]),prompt_token_sha256=digest(prompts[i]['prompt_token_ids']))
                        raw.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n')
                    f.flush()
            if before!={k:identity(v) for k,v in MODELS.items()}:raise ValueError('model changed')
            if not author:save(a.out/'summary.safe.json',summarize(packet,raw,a.label))
            status('completed_pending_local_review' if author else 'completed_pending_verification',completed=len(raw),total=len(plan))
    except BaseException as e:status('failed',error=type(e).__name__+': '+str(e));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('input','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--input-sha256',required=True);p.add_argument('--label',choices=list(MODELS),required=True);p.add_argument('--wait-seconds',type=int,default=0)
    a=p.parse_args()
    if not 0<=a.wait_seconds<=21600:raise ValueError('wait bound')
    run(a)
