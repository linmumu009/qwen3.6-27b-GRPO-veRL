"""P4 inference with the frozen two-order remediation protocol; parent owns lock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from run_cpt_remediation_review import specs,closed_prompt,summarize
from run_cpt_transfer_screen import digest
from prepare_cpt_p4_cumulative import read,sha

def run(cases,model,baseline,out):
    assert os.environ.get('LLIN_COORDINATOR_LOCKED')=='1'
    assert sha(cases)=='9d956b539b6f40a07e71ad4b925c681e49534c73e89873c80e825d7895a07848'
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    rows=read(cases);plan=specs(rows);by_id={r['id']:r for r in rows}
    assert len(rows)==44 and len(plan)==88
    tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
    texts=[closed_prompt(by_id[s['id']],s) for s in plan]
    prompts=[dict(prompt_token_ids=tok.encode(tok.apply_chat_template([dict(role='user',content=t)],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)) for t in texts]
    prior=read(baseline);assert len(prior)==88
    for s,t,p,b in zip(plan,texts,prompts,prior):
        assert all(s[k]==b[k] for k in ('id','variant','order'))
        assert hashlib.sha256(t.encode()).hexdigest()==b['prompt_text_sha256']
        assert digest(p['prompt_token_ids'])==b['prompt_token_sha256'] and len(p['prompt_token_ids'])==b['prompt_tokens']
        assert len(p['prompt_token_ids'])+96<=8192
    out.mkdir(exist_ok=False)
    registration=dict(model=str(model),cases_sha256=sha(cases),baseline_sha256=sha(baseline),seed=20922,temperature=0,max_tokens=96,max_model_len=8192,tp=8,max_num_seqs=16,chunk_size=8,thinking=False,calls=88,baseline_prompt_tokens_verified=True)
    (out/'registration.safe.json').write_text(json.dumps(registration,indent=2)+'\n')
    llm=LLM(model=str(model),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,max_num_seqs=16,gpu_memory_utilization=.8,seed=20922,enforce_eager=True)
    raw=[]
    with (out/'closed.private.jsonl').open('x',encoding='utf-8') as f:
        for start in range(0,len(plan),8):
            outputs=sorted(llm.generate(prompts[start:start+8],SamplingParams(temperature=0,max_tokens=96,seed=20922)),key=lambda o:int(o.request_id))
            assert len(outputs)==len(prompts[start:start+8])
            for i,o in enumerate(outputs,start):
                ids=prompts[i]['prompt_token_ids'];assert list(o.prompt_token_ids)==ids
                v=o.outputs[0];r=dict({k:plan[i][k] for k in ('id','variant','order')},text=v.text,finish_reason=v.finish_reason,output_tokens=len(v.token_ids),prompt_tokens=len(ids),prompt_text_sha256=hashlib.sha256(texts[i].encode()).hexdigest(),prompt_token_sha256=digest(ids))
                raw.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n')
            f.flush();(out/'progress.safe.json').write_text(json.dumps(dict(completed=len(raw),total=88)))
    result=summarize(rows,raw,[],'p4');result['remote_token_reconstruction_verified']=True
    (out/'summary.safe.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('cases','model','baseline','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cases,a.model,a.baseline,a.out)
