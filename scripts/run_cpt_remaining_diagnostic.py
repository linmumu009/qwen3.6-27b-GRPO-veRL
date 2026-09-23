"""Journalled 80-call, one-group diagnostic; never train or retry reservations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/workspace/llin-verl-grpo')
OLD=ROOT/'runs/llin-mechanism-run-20260920-01'
CODE=ROOT/'runs/llin-mechanism-package-20260920-02/scripts'
OUT=ROOT/'runs/llin-remaining-diagnostic-20260923-01'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, data):
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temp.replace(p)


def worker(package,label):
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from run_cpt_formal_transfer import model_identity
    from cpt_stage_b import specs,messages,digest,summarize
    reg=read(package/'execution.safe.json')
    packet=read(package/'diagnostic.private.json')
    assert sha(package/'diagnostic.private.json')==reg['packet_sha256']
    identity=read(OLD/(('step120' if label=='step120_current' else label)+'_direct')/'identity.safe.json')
    assert model_identity(Path(identity['path']))==identity
    plan=specs(packet,label)
    assert len(plan)==(16 if label=='step120_current' else 64)
    folder=OUT/label;folder.mkdir(exist_ok=True)
    done=folder/'completed.safe.json'
    if done.exists():
        d=read(done)
        assert d['calls']==len(plan) and sha(folder/'predictions.private.json')==d['predictions_sha256']
        return
    tok=AutoTokenizer.from_pretrained(identity['path'],local_files_only=True)
    prompts=[]
    for s in plan:
        msg=messages(packet,s)
        ids=tok.encode(tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        assert ids and len(ids)+96<=8192
        prompts.append(dict(prompt_token_ids=ids))
    save(folder/'identity.safe.json',identity)
    prompt_hash=digest(prompts)
    frozen=folder/'prompts.safe.json'
    prompt_reg=dict(packet_sha256=reg['packet_sha256'],prompt_token_sha256=prompt_hash,calls=len(plan),max_tokens=96,temperature=0,seed=1024,thinking=False)
    if frozen.exists(): assert read(frozen)==prompt_reg
    else: save(frozen,prompt_reg)
    batches=[]
    for start in range(0,len(plan),16):
        stem=f'batch-{start:03d}'
        result,reserve,complete=[folder/(stem+s) for s in ('.private.json','.reserved.safe.json','.completed.safe.json')]
        if complete.exists():
            d=read(complete);assert d['calls']==len(plan[start:start+16]) and sha(result)==d['sha256']
        else: assert not reserve.exists() and not result.exists(), 'Unresolved reservation; inspect, do not resubmit'
        batches.append((start,result,reserve,complete))
    llm=None
    for start,result,reserve,complete in batches:
        if complete.exists(): continue
        if llm is None:
            llm=LLM(model=identity['path'],tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,
                max_num_seqs=32,gpu_memory_utilization=.8,seed=1024,enforce_eager=True,enable_prefix_caching=False)
        batch=prompts[start:start+16]
        save(reserve,dict(start=start,calls=len(batch),prompt_token_sha256=digest(batch),packet_sha256=reg['packet_sha256']))
        outputs=sorted(llm.generate(batch,SamplingParams(temperature=0,max_tokens=96,seed=1024)),key=lambda o:int(o.request_id))
        assert len(outputs)==len(batch)
        rows=[]
        for idx,o in enumerate(outputs,start):
            assert list(o.prompt_token_ids)==prompts[idx]['prompt_token_ids'] and len(o.outputs)==1
            v=o.outputs[0];assert len(v.token_ids)<=96
            rows.append(dict(plan[idx],text=v.text,finish_reason=v.finish_reason,output_tokens=len(v.token_ids),
                messages_sha256=digest(messages(packet,plan[idx])),prompt_token_sha256=digest(prompts[idx]['prompt_token_ids'])))
        save(result,rows);save(complete,dict(calls=len(rows),sha256=sha(result)))
    assert model_identity(Path(identity['path']))==identity
    rows=[r for _,result,_,_ in batches for r in read(result)]
    save(folder/'predictions.private.json',rows)
    save(folder/'summary.safe.json',summarize(packet,rows,label))
    save(done,dict(calls=len(rows),predictions_sha256=sha(folder/'predictions.private.json'),prompt_token_sha256=prompt_hash))


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True)
    p.add_argument('--worker',choices=['step120_current','p1']);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    sys.path.insert(0,str(CODE));os.umask(0o077)
    if a.worker:worker(a.package,a.worker);return
    import fcntl
    from cpt_stage_b import device_idle,specs
    from run_cpt_formal_transfer import model_identity
    from run_logistics_cpt_curve_8x import evaluation_env
    reg=read(a.package/'execution.safe.json');packet=read(a.package/'diagnostic.private.json')
    assert reg['max_calls']==80 and reg['training_allowed'] is False
    assert sha(Path(__file__))==reg['runner_sha256'] and sha(a.package/'diagnostic.private.json')==reg['packet_sha256']
    assert packet['quality_released'] is True and packet['training_allowed'] is False
    assert [g['id'] for g in packet['groups']]==['offered_transport_capacity']
    assert sum(len(specs(packet,label)) for label in ('step120_current','p1'))==80
    for name,h in read(OLD/'registration.safe.json')['code'].items():assert sha(CODE/name)==h, 'old code changed'
    for name,h in reg['protocol_code_sha256'].items():assert sha(CODE/name)==h, 'protocol mismatch'
    for label in ('step120','p1'):
        identity=read(OLD/(label+'_direct')/'identity.safe.json')
        assert model_identity(Path(identity['path']))==identity, 'model identity mismatch'
    OUT.mkdir(exist_ok=True)
    with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle,used=device_idle(subprocess.check_output(['npu-smi','info'],text=True))
        assert idle,'Devices busy; no process terminated'
        assert shutil.disk_usage(OUT).free>5_000_000_000
        frozen=OUT/'execution.safe.json'
        if frozen.exists():assert read(frozen)==reg
        else:save(frozen,reg)
        save(OUT/'preflight.safe.json',dict(checked_at=time.time(),free_bytes=shutil.disk_usage(OUT).free,memory_mb=used,models_match=True,code_match=True))
        if a.preflight_only:return
        status=OUT/'status.safe.json'
        try:
            for label in ('step120_current','p1'):
                save(status,dict(state='running',stage=label,training_runs=0))
                with (OUT/(label+f'.{time.time_ns()}.log')).open('x') as log:
                    subprocess.run([sys.executable,str(Path(__file__)),'--package',str(a.package),'--worker',label],
                        env=evaluation_env(dict(os.environ)),stdout=log,stderr=subprocess.STDOUT,check=True)
            assert sum(read(OUT/label/'completed.safe.json')['calls'] for label in ('step120_current','p1'))==80
            save(status,dict(state='complete_pending_local_verification',calls=80,training_runs=0,formal_reruns=0))
        except BaseException as e:
            save(status,dict(state='failed_preserved',error=str(e),recovery='Inspect reservations before resuming; never automatically resend.'));raise


if __name__=='__main__':main()
