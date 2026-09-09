"""Full frozen benchmark comparison; two matched shards, no training or API."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace/llin-verl-grpo')
OUT=ROOT/'runs/supply-chain-full-prompt-comparison-20260909-01'
CASES=ROOT/'runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl'
CASE_SHA='b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99'
MODEL=ROOT/'runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/hf_export_step_116'
CONFIGS=[('original96','original',96),('guided2048','deliberate',2048)]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,indent=2),encoding='utf-8')


def partition(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],len(r['options'])>=100)].append(r)
    shards=[[],[]];offset=0
    for key in sorted(groups):
        for i,r in enumerate(sorted(groups[key],key=lambda x:x['item_hash'])):shards[(i+offset)%2].append(r)
        offset=(offset+len(groups[key]))%2
    assert not ({r['item_hash'] for r in shards[0]} & {r['item_hash'] for r in shards[1]})
    assert sum(map(len,shards))==len(rows)
    return shards


def vote(rows):
    assert len(rows)==3 and {r['repeat'] for r in rows}=={0,1,2}
    answers=[tuple(r['parsed']) if r['valid'] else None for r in rows]
    answer,n=Counter(answers).most_common(1)[0]
    return dict(correct=answer is not None and n>=2 and sum(r['correct'] for r in rows)>=2,
                stable=len(set(answers))==1,no_majority=n<2)


def worker(shard):
    from run_logistics_strategy_diagnostic import messages_for,parse_answers
    assert digest(CASES)==CASE_SHA
    rows=[json.loads(x) for x in CASES.read_text().split('\n') if x.strip()]
    cases=partition(rows)[shard];target=OUT/f'shard{shard}';target.mkdir(exist_ok=False)
    def status(s):save(target/'progress.safe.json',dict(status=s,shard=shard,cases=len(cases),completed=completed))
    completed=0;status('preflight')
    try:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True);jobs={}
        for label,condition,budget in CONFIGS:
            prompts=[]
            for r in cases:
                messages,_=messages_for(r,condition)
                ids=tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
                assert len(ids)+budget<=8192
                prompts.append(dict(prompt_token_ids=ids))
            jobs[label]=prompts
        from vllm import LLM,SamplingParams
        status('loading_model')
        llm=LLM(model=str(MODEL),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,max_num_seqs=32,gpu_memory_utilization=.8,seed=1024,enforce_eager=True)
        timings=[]
        with (target/'predictions.private.jsonl').open('x') as f:
            for repeat in range(3):
                for start in range(0,len(cases),128):
                    batch=cases[start:start+128]
                    order=CONFIGS if (repeat+start//128+shard)%2==0 else list(reversed(CONFIGS))
                    for label,_,budget in order:
                        status(f'{label}_repeat{repeat+1}_batch{start//128+1}')
                        t=time.monotonic()
                        outputs=llm.generate(jobs[label][start:start+128],SamplingParams(temperature=0,max_tokens=budget,seed=1024))
                        outputs=sorted(outputs,key=lambda x:int(x.request_id));assert len(outputs)==len(batch)
                        elapsed=time.monotonic()-t
                        for r,o in zip(batch,outputs):
                            assert o.outputs
                            ans=o.outputs[0];parsed,parse_ok=parse_answers(ans.text,len(r['options']))
                            valid=parse_ok and ans.finish_reason=='stop'
                            rec=dict(item_hash=r['item_hash'],dataset=r['dataset'],condition=label,repeat=repeat,shard=shard,parsed=list(parsed),valid=valid,correct=valid and list(parsed)==sorted(r['expected']),truncated=ans.finish_reason=='length',output_tokens=len(ans.token_ids),prediction=ans.text)
                            f.write(json.dumps(rec)+'\n');completed+=1
                        f.flush();timings.append(dict(condition=label,repeat=repeat,batch=start//128,items=len(batch),seconds=elapsed))
                        save(target/'timings.safe.json',timings);status('batch_complete')
        assert completed==len(cases)*6;status('complete')
    except BaseException:
        status('failed');raise


def summarize():
    records=[];timings=[]
    for shard in range(2):
        p=OUT/f'shard{shard}'
        records.extend(json.loads(x) for x in (p/'predictions.private.jsonl').read_text().split('\n') if x.strip())
        timings.extend(json.loads((p/'timings.safe.json').read_text()))
    assert len(records)==10032
    groups=defaultdict(list)
    for r in records:groups[r['item_hash'],r['condition']].append(r)
    assert len(groups)==3344
    table=[]
    for dataset in ['all','SC-bench-knowledge','LogistikaBench']:
        subset=[r for r in records if dataset=='all' or r['dataset']==dataset]
        ids={r['item_hash'] for r in subset}
        a={i:vote(groups[i,'original96']) for i in ids};b={i:vote(groups[i,'guided2048']) for i in ids}
        row=dict(dataset=dataset,n=len(ids),before=sum(r['correct'] for r in a.values()),after=sum(r['correct'] for r in b.values()),gains=sum(not a[i]['correct'] and b[i]['correct'] for i in ids),losses=sum(a[i]['correct'] and not b[i]['correct'] for i in ids))
        stable=[i for i in ids if a[i]['stable'] and b[i]['stable']]
        row['both_stable']=dict(n=len(stable),gains=sum(not a[i]['correct'] and b[i]['correct'] for i in stable),losses=sum(a[i]['correct'] and not b[i]['correct'] for i in stable))
        for label,_,_ in CONFIGS:
            rr=[r for r in subset if r['condition']==label]
            row[label]=dict(repeat_correct=[sum(r['correct'] for r in rr if r['repeat']==rep) for rep in range(3)],truncated=sum(r['truncated'] for r in rr),invalid=sum(not r['valid'] for r in rr),output_tokens=sum(r['output_tokens'] for r in rr))
        table.append(row)
    save(OUT/'summary.safe.json',dict(requests=10032,cases_sha256=CASE_SHA,table=table,worker_batch_seconds={label:sum(t['seconds'] for t in timings if t['condition']==label) for label,_,_ in CONFIGS},timing_note='Sum of batch times across two workers, not end-to-end latency or per-item latency',training=False,api_calls=0,decision='No automatic promotion; paired uncertainty, cost and retention review still required'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,choices=[0,1]);a=p.parse_args();os.umask(0o077)
    if a.shard is not None:return worker(a.shard)
    import fcntl
    with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert digest(CASES)==CASE_SHA
        rows=[json.loads(x) for x in CASES.read_text().split('\n') if x.strip()]
        assert len(rows)==len({r['item_hash'] for r in rows})==1672
        OUT.mkdir(exist_ok=False)
        def status(s):(OUT/'status.txt').write_text(s+'\n')
        code=Path(__file__).parent
        save(OUT/'protocol.safe.json',dict(cases_sha256=CASE_SHA,cases=1672,conditions=CONFIGS,repeats=3,max_requests=10032,max_output_token_budget=1672*3*(96+2048),model=str(MODEL),shard_sizes=list(map(len,partition(rows))),total_concurrency=64,workers=2,per_worker_concurrency=32,tensor_parallel=8,temperature=0,seed=1024,max_model_len=8192,training=False,api_calls=0,script_hash=digest(Path(__file__)),helper_hash=digest(code/'run_logistics_strategy_diagnostic.py'),limits=['Full benchmarks previously inspected; no unseen-test claim','No prompt tuning or checkpoint selection','No auto-promotion; original Agent/general retention not inferred','Output truncation and invalid answer count wrong']))
        processes=[];handles=[];started=time.monotonic();status('running')
        try:
            for shard in range(2):
                env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=','.join(str(i) for i in range(shard*8,shard*8+8)))
                handle=(OUT/f'shard{shard}.log').open('x');handles.append(handle)
                processes.append(subprocess.Popen([sys.executable,str(Path(__file__)), '--shard',str(shard)],env=env,stdout=handle,stderr=subprocess.STDOUT))
            # Keep the resource lock until both children finish, even if one fails.
            codes=[proc.wait() for proc in processes]
            if any(codes):raise RuntimeError(f'Worker exit codes {codes}')
            summarize();save(OUT/'walltime.safe.json',dict(seconds=time.monotonic()-started));status('complete_review_pending')
        except BaseException:
            status('failed');raise
        finally:
            for proc in processes:
                if proc.poll() is None:proc.wait()
            for handle in handles:handle.close()


if __name__=='__main__':main()
