"""Bounded no-training 2x2 prompt/budget diagnostic on the existing 96 cases."""
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from run_logistics_strategy_diagnostic import messages_for,majority,parse_answers


def main():
    root=Path('/workspace/llin-verl-grpo');old=root/'runs/logistics-strategy-diagnostic-20260908'
    out=root/'runs/supply-chain-prompt-factorial-20260909-02'
    model=root/'runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/hf_export_step_116'
    os.umask(0o077)
    with (root/'runs/.logistics-exam-cpt.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads((old/'manifest.safe.json').read_text());src=old/'cases.private.json'
        assert hashlib.sha256(src.read_bytes()).hexdigest()==manifest['selected_sha256']
        cases=json.loads(src.read_text());assert len(cases)==96
        out.mkdir(exist_ok=False)
        def save(name,data):(out/name).write_text(json.dumps(data,indent=2))
        def status(s):(out/'status.txt').write_text(s+'\n')
        configs=[('original96','original',96),('guided96','deliberate',96),('original2048','original',2048),('guided2048','deliberate',2048)]
        save('protocol.safe.json',dict(cases_sha256=manifest['selected_sha256'],model=str(model),cases=96,conditions=configs,repeats=3,max_requests=1152,max_output_tokens_budget=1234944,temperature=0,seed=1024,max_model_len=8192,concurrency=64,training=False,api_calls=0,selection='Reuse all 96 previously frozen cases; no new outcome-based selection',limits=['Case-control cohort, not population accuracy','Prompt intervention bundles guidance and permission for explanation','No evidence added; cannot identify absence of stored knowledge','Truncation counts wrong; repeats not independent items']))
        try:
            from vllm import LLM,SamplingParams
            status('loading_cpt')
            llm=LLM(model=str(model),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,max_num_seqs=64,gpu_memory_utilization=.8,seed=1024,enforce_eager=True)
            tok=llm.get_tokenizer();jobs={}
            for label,condition,budget in configs:
                prompts=[]
                for r in cases:
                    messages,_=messages_for(r,condition)
                    ids=tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
                    assert len(ids)+budget<=8192
                    prompts.append(dict(prompt_token_ids=ids))
                jobs[label]=prompts
            records=[]
            with (out/'predictions.private.jsonl').open('x') as f:
                for repeat in range(3):
                    order=configs[repeat:]+configs[:repeat]
                    for label,_,budget in order:
                        status(f'{label}_repeat{repeat+1}')
                        outputs=llm.generate(jobs[label],SamplingParams(temperature=0,max_tokens=budget,seed=1024))
                        outputs=sorted(outputs,key=lambda x:int(x.request_id));assert len(outputs)==96
                        for r,o in zip(cases,outputs):
                            answer=o.outputs[0];parsed,valid=parse_answers(answer.text,len(r['options']))
                            valid=valid and answer.finish_reason=='stop'
                            rec=dict(item_hash=r['item_hash'],condition=label,repeat=repeat,mapped=list(parsed),parse_ok=valid,correct=valid and list(parsed)==sorted(r['expected']),finish_reason=answer.finish_reason,output_tokens=len(answer.token_ids),text=answer.text)
                            records.append(rec);f.write(json.dumps(rec)+'\n')
                        f.flush()
            grouped=defaultdict(list)
            for r in records:grouped[r['item_hash'],r['condition']].append(r)
            result=[]
            for group in ['all','logistika_269','logistika_other','sc_knowledge']:
                subset=[r for r in cases if group=='all' or r['stratum']==group]
                for label,_,budget in configs:
                    mm=[majority(grouped[r['item_hash'],label]) for r in subset]
                    bb=[majority(grouped[r['item_hash'],'original96']) for r in subset]
                    result.append(dict(group=group,condition=label,n=len(subset),correct=sum(x['correct'] for x in mm),gains=sum(not b['correct'] and x['correct'] for b,x in zip(bb,mm)),losses=sum(b['correct'] and not x['correct'] for b,x in zip(bb,mm)),truncated=sum(x['truncated'] for x in mm),output_tokens=sum(x['output_tokens'] for r in subset for x in grouped[r['item_hash'],label])))
            save('summary.safe.json',dict(requests=len(records),table=result,training=False));status('complete_analysis_pending')
        except BaseException:
            status('failed');raise


if __name__=='__main__':main()
