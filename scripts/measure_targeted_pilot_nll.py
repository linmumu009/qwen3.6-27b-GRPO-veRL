"""Fixed exact training-prefix answer NLL; no generation scoring or training."""
import argparse,json,hashlib,math,os
from pathlib import Path
from probe_targeted_book_groups import prompt_messages
from run_vllm_prompt_nll import chosen_logprob

def main():
    p=argparse.ArgumentParser()
    for k in ('root','model','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--label',required=True);a=p.parse_args();os.umask(0o077)
    data=a.root/'runs/targeted-book-pilot-prepared-20260908'
    train=list(map(json.loads,(data/'train.private.jsonl').read_text().splitlines()))
    dev=[r for r in map(json.loads,(data/'evaluation/candidates.private.jsonl').read_text().splitlines()) if r['split']=='dev']
    rows=train+dev
    assert len(train)==80 and len(dev)==39 and len({r['id'] for r in rows})==119
    import pandas as pd
    meta=json.loads((data/'manifest.safe.json').read_text())
    assert hashlib.sha256((data/'train.parquet').read_bytes()).hexdigest()==meta['train_parquet_sha256']
    actual={r['id']:r for r in pd.read_parquet(data/'train.parquet').to_dict('records')}
    from vllm import LLM,SamplingParams
    llm=LLM(model=str(a.model),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,
        max_model_len=4096,max_num_seqs=32,gpu_memory_utilization=.8,seed=1024,enforce_eager=True)
    t=llm.get_tokenizer();prompts=[];starts=[]
    for r in rows:
        prefix=t.encode(t.apply_chat_template(prompt_messages(r,0),tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        ids=prefix+t.encode(r['answer'],add_special_tokens=False)+[t.eos_token_id]
        assert len(ids)<=4096 and len(ids)>len(prefix)+1
        if r['id'] in actual:assert ids==list(actual[r['id']]['input_ids']) and len(prefix)==actual[r['id']]['answer_start']
        prompts.append({'prompt_token_ids':ids});starts.append(len(prefix))
    a.output.mkdir(parents=True,exist_ok=False)
    outs=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1,detokenize=False))
    assert len(outs)==119
    scores=[]
    for r,prefix,start,out in zip(rows,prompts,starts,outs):
        ids=prefix['prompt_token_ids'];assert list(out.prompt_token_ids)==ids
        lp=out.prompt_logprobs;assert len(lp)==len(ids)
        v=[-chosen_logprob(lp[i],ids[i]) for i in range(start,len(ids))]
        assert all(math.isfinite(x) and x>=-1e-5 for x in v)
        scores.append({'id':r['id'],'cohort':'development' if r['split']=='dev' else 'maintenance' if r['decision']=='maintenance_pool' else 'opportunity',
            'content_tokens':len(v)-1,'content_nll_sum':sum(v[:-1]),'eos_nll':v[-1],
            'first_content_nll':v[0],'input_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    (a.output/'scores.private.json').write_text(json.dumps(scores))
    summary={}
    for cohort in ('opportunity','maintenance','development'):
        rr=[r for r in scores if r['cohort']==cohort];n=sum(r['content_tokens'] for r in rr);s=sum(r['content_nll_sum'] for r in rr)
        summary[cohort]={'items':len(rr),'content_tokens':n,'content_nll':s/n,
             'mean_eos_nll':sum(r['eos_nll'] for r in rr)/len(rr),
             'answer_and_eos_nll':(s+sum(r['eos_nll'] for r in rr))/(n+len(rr)),
             'mean_first_content_nll':sum(r['first_content_nll'] for r in rr)/len(rr)}
    safe={'label':a.label,'cohorts':summary,'input_sha256':hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
          'scores_sha256':hashlib.sha256((a.output/'scores.private.json').read_bytes()).hexdigest(),'training':False,
          'limitation':'Teacher-forced reference likelihood, not generated factual correctness; exact original prompt only.'}
    (a.output/'summary.safe.json').write_text(json.dumps(safe,indent=2));print(json.dumps(safe))

if __name__=='__main__':main()
