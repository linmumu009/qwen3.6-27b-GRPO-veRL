"""Source-only paired response-format diagnostic; no training or official QA."""
import argparse,fcntl,hashlib,json,os,re
from pathlib import Path

def parse(text):
    hits=re.findall(r'\{\s*"answers"\s*:\s*\[[^\]]*\]\s*\}',text)
    if len(hits)!=1:return None
    try:
        a=json.loads(hits[0])['answers']
        if not isinstance(a,list) or any(type(x)!=int or not 0<=x<4 for x in a) or len(a)!=len(set(a)):return None
        return sorted(a)
    except (ValueError,TypeError):return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    os.umask(0o077)
    assert hashlib.sha256(a.cases.read_bytes()).hexdigest()==a.sha
    cases=[json.loads(s) for s in a.cases.read_text().splitlines()]
    assert len(cases)==6 and all(c['training_allowed'] is False for c in cases)
    a.out.mkdir(exist_ok=False)
    def save(n,v):(a.out/n).write_text(json.dumps(v,indent=2)+'\n')
    save('status.safe.json',{'status':'waiting_for_lock','training_running':False})
    try:
        with open('/workspace/llin-verl-grpo/runs/.logistics-exam-cpt.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            from vllm import LLM,SamplingParams
            model='/workspace/llin-verl-grpo/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource'
            tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True)
            specs=[];prompts=[]
            for c in cases:
                for condition,instruction in [('answer_only','Return only a JSON object with key "answers" containing all correct zero-based option indices.'),('explain_then_answer','Explain the governing rule and calculation in at most 35 words, then return a JSON object with key "answers" containing all correct zero-based option indices.')]:
                    question=instruction+'\nQuestion:\n'+c['question']+'\nOptions:\n'+'\n'.join(f'[{i}] {v}' for i,v in enumerate(c['options']))
                    rendered=tok.apply_chat_template([{'role':'user','content':question}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    tokens=tok.encode(rendered,add_special_tokens=False)
                    assert len(tokens)+96<=8192
                    prompts.append({'prompt_token_ids':tokens});specs.append((c,condition))
            save('registration.safe.json',dict(cases=6,requests=12,case_sha256=a.sha,model=model,max_tokens=96,temperature=0,seed=1024,tp=8,diagnostic_only=True,official_protocol_changed=False))
            save('status.safe.json',{'status':'loading_model','training_running':False})
            llm=LLM(model=model,tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,max_num_seqs=32,gpu_memory_utilization=.8,seed=1024,enforce_eager=True)
            outputs=sorted(llm.generate(prompts,SamplingParams(temperature=0,max_tokens=96,seed=1024)),key=lambda x:int(x.request_id))
            assert len(outputs)==12
            results=[]
            for (c,condition),prompt,out in zip(specs,prompts,outputs):
                assert list(out.prompt_token_ids)==prompt['prompt_token_ids']
                answer=out.outputs[0];parsed=parse(answer.text)
                results.append(dict(id=c['id'],condition=condition,prediction=answer.text,parsed=parsed,finish_reason=answer.finish_reason,correct=parsed==c['expected'] and answer.finish_reason=='stop',prompt_sha256=hashlib.sha256(json.dumps(prompt['prompt_token_ids']).encode()).hexdigest()))
            (a.out/'predictions.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
            save('status.safe.json',{'status':'completed_pending_verification','training_running':False})
    except BaseException as e:
        save('status.safe.json',{'status':'failed','error':str(e),'training_running':False});raise

if __name__=='__main__':main()
