"""P1-only 144-call diagnostic. Preserve completed batches; never retry reservations."""
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
OUT=ROOT/'runs/llin-selection-diagnostic-20260923-01'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def save(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');temp.replace(p)


def batches(plan,prompts,folder,packet_hash,make_llm,sampling):
    entries=[]
    for start in range(0,len(plan),16):
        paths=[folder/(f'batch-{start:03d}'+s) for s in ('.private.json','.reserved.safe.json','.completed.safe.json')]
        result,reserve,complete=paths
        if complete.exists():
            d=read(complete)
            assert d['calls']==len(plan[start:start+16]) and sha(result)==d['sha256']
            assert read(reserve)==dict(start=start,calls=d['calls'],prompt_token_sha256=digest(prompts[start:start+16]),packet_sha256=packet_hash)
        else:assert not reserve.exists() and not result.exists(),'Unresolved reservation; do not resubmit'
        entries.append((start,*paths))
    llm=None
    for start,result,reserve,complete in entries:
        if complete.exists():continue
        if llm is None:llm=make_llm()
        batch=prompts[start:start+16]
        with reserve.open('x') as f:
            json.dump(dict(start=start,calls=len(batch),prompt_token_sha256=digest(batch),packet_sha256=packet_hash),f)
            f.flush();os.fsync(f.fileno())
        outputs=sorted(llm.generate(batch,sampling),key=lambda x:int(x.request_id))
        assert len(outputs)==len(batch)
        rows=[]
        for i,o in enumerate(outputs,start):
            assert list(o.prompt_token_ids)==prompts[i]['prompt_token_ids'] and len(o.outputs)==1
            v=o.outputs[0];assert len(v.token_ids)<=96
            rows.append(dict(id=plan[i]['id'],text=v.text,finish_reason=v.finish_reason,output_tokens=len(v.token_ids),
                messages_sha256=digest(plan[i]['messages']),prompt_token_sha256=digest(prompts[i]['prompt_token_ids'])))
        save(result,rows);save(complete,dict(calls=len(rows),sha256=sha(result)))
    return [r for _,p,_,_ in entries for r in read(p)]


def prepared(package):
    from transformers import AutoTokenizer
    from run_cpt_formal_transfer import model_identity
    reg=read(package/'execution.safe.json');packet=read(package/'packet.private.json')
    assert reg['max_calls']==144 and reg['training_allowed'] is False and packet['training_allowed'] is False
    assert reg['quality_released'] is True and sha(package/'quality.safe.json')==reg['quality_sha256']
    assert read(package/'quality.safe.json')['passed'] is True
    assert sha(Path(__file__))==reg['runner_sha256'] and sha(package/'packet.private.json')==reg['packet_sha256']
    assert len(packet['requests'])==144 and len({r['id'] for r in packet['requests']})==144 and len(packet['reuse'])==16
    for name,h in read(OLD/'registration.safe.json')['code'].items():assert sha(CODE/name)==h
    for name,h in reg['protocol_code_sha256'].items():assert sha(CODE/name)==h
    identity=read(OLD/'p1_direct/identity.safe.json')
    assert identity['path']==reg['model_path'] and model_identity(Path(identity['path']))==identity
    tok=AutoTokenizer.from_pretrained(identity['path'],local_files_only=True)
    def ids(msg):
        result=tok.encode(tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        assert result and len(result)+96<=8192
        return result
    for row in packet['reuse']:
        assert hashlib.sha256(json.dumps(ids(row['messages'])).encode()).hexdigest()==row['historical_prompt_token_sha256'],'historical prompt changed'
    prompts=[dict(prompt_token_ids=ids(r['messages'])) for r in packet['requests']]
    return reg,packet,identity,prompts


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',required=True,type=Path)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--worker',action='store_true');modes.add_argument('--preflight-only',action='store_true')
    p.add_argument('--lock-fd',type=int);a=p.parse_args()
    sys.path.insert(0,str(CODE));os.umask(0o077)
    if a.worker:
        import fcntl
        assert a.lock_fd is not None,'Worker requires coordinator lock descriptor'
        held=os.fstat(a.lock_fd);expected=(ROOT/'runs/.logistics-exam-cpt.lock').stat()
        assert (held.st_dev,held.st_ino)==(expected.st_dev,expected.st_ino),'Wrong lock descriptor'
        fcntl.flock(a.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from vllm import LLM,SamplingParams
        from run_cpt_formal_transfer import model_identity
        reg,packet,identity,prompts=prepared(a.package)
        folder=OUT/'p1';folder.mkdir(exist_ok=True)
        freeze=dict(packet_sha256=reg['packet_sha256'],prompt_token_sha256=digest(prompts),calls=144)
        if (folder/'prompts.safe.json').exists():assert read(folder/'prompts.safe.json')==freeze
        else:save(folder/'prompts.safe.json',freeze)
        save(folder/'identity.safe.json',identity)
        rows=batches(packet['requests'],prompts,folder,reg['packet_sha256'],
            lambda:LLM(model=identity['path'],tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,
                max_model_len=8192,max_num_seqs=32,gpu_memory_utilization=.8,seed=1024,enforce_eager=True,enable_prefix_caching=False),
            SamplingParams(temperature=0,max_tokens=96,seed=1024))
        assert model_identity(Path(identity['path']))==identity
        save(folder/'predictions.private.json',rows)
        save(folder/'completed.safe.json',dict(calls=len(rows),predictions_sha256=sha(folder/'predictions.private.json')))
        return
    import fcntl
    from cpt_stage_b import device_idle
    from run_logistics_cpt_curve_8x import evaluation_env
    OUT.mkdir(exist_ok=True)
    with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reg,packet,identity,prompts=prepared(a.package)
        idle,used=device_idle(subprocess.check_output(['npu-smi','info'],text=True))
        assert idle,'Devices busy; no process terminated'
        assert shutil.disk_usage(OUT).free>5_000_000_000
        frozen=OUT/'execution.safe.json'
        if frozen.exists():assert read(frozen)==reg
        else:save(frozen,reg)
        save(OUT/'preflight.safe.json',dict(checked_at=time.time(),free_bytes=shutil.disk_usage(OUT).free,memory_mb=used,
            model_matches=True,code_matches=True,historical_token_prompts_match=16,max_prompt_tokens=max(len(p['prompt_token_ids']) for p in prompts)))
        if a.preflight_only:return
        status=OUT/'status.safe.json'
        try:
            save(status,dict(state='running',stage='p1',training_runs=0))
            with (OUT/f'p1.{time.time_ns()}.log').open('x') as log:
                subprocess.run([sys.executable,str(Path(__file__)),'--package',str(a.package),'--worker','--lock-fd',str(lock.fileno())],
                    pass_fds=(lock.fileno(),),env=evaluation_env(dict(os.environ)),stdout=log,stderr=subprocess.STDOUT,check=True)
            assert read(OUT/'p1/completed.safe.json')['calls']==144
            save(status,dict(state='complete_pending_local_verification',calls=144,training_runs=0))
        except BaseException as e:
            save(status,dict(state='failed_preserved',error=str(e),recovery='Inspect reservations before resuming.'));raise


if __name__=='__main__':main()
