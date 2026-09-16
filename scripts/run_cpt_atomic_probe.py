"""Judge every claim separately, retaining the original scenario and all options.

Six 16-token claim budgets equal the 96-token joint output cap, but input work is
repeated six times. This is a diagnostic decomposition, not a compute-matched
formal evaluation or a new independent set of knowledge units.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import cpt_composed_probe as joint


def parse_bool(text):
    try: value=json.loads(text.strip())
    except ValueError:return None
    if not isinstance(value,dict) or set(value)!={'correct'} or type(value['correct']) is not bool:return None
    return value['correct']


def prompt(case,condition,target):
    order=joint.variants(case)[0]
    original=joint.prompt(case,order,condition)
    body=original.split('\n',1)[1].replace(' Select all correct statements.','')
    index=order.index(target)
    return ('Judge only the designated statement, using the scenario and applicable source rules. The other options are candidate claims, not established facts. Return only {"correct":true} or {"correct":false}.\n'+body+
        '\nDesignated statement: option ['+str(index)+']. Decide whether this statement is correct.')


def selected_cases(packet,audit):
    cases=[json.loads(s) for s in Path(packet).read_text(encoding='utf-8').splitlines()];joint.validate(cases)
    decisions=[json.loads(s) for s in Path(audit).read_text().splitlines()]
    assert len(decisions)==len(cases) and {c['id'] for c in cases}=={r['id'] for r in decisions}
    assert all(r['case_sha256']==joint.sha(packet) and r['verdict'] in ('accept','hold','reject') for r in decisions)
    ids={r['id'] for r in decisions if r['verdict']=='accept'}
    selected=[c for c in cases if c['id'] in ids]
    assert len(selected)==31 and all(len(c['options'])==6 for c in selected)
    return selected


def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--case-sha',required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--audit-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert joint.sha(a.cases)==a.case_sha and joint.sha(a.audit)==a.audit_sha
    cases=selected_cases(a.cases,a.audit);by_id={c['id']:c for c in cases}
    os.umask(0o077);a.out.mkdir(exist_ok=False)
    def save(name,value):(a.out/name).write_text(json.dumps(value,indent=2)+'\n')
    def status(name,**kwargs):save('status.safe.json',dict(status=name,training_running=False,**kwargs))
    status('waiting_for_lock')
    try:
        with open(joint.LOCK,'a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            from vllm import LLM,SamplingParams
            tok=AutoTokenizer.from_pretrained(joint.MODEL,trust_remote_code=True,local_files_only=True)
            specs=[];prompts=[]
            for c in cases:
                for condition in ('closed_book','source_evidence'):
                    for target in range(6):
                        text=prompt(c,condition,target)
                        ids=tok.encode(tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
                        assert len(ids)+16<=8192
                        specs.append(dict(id=c['id'],condition=condition,target=target,expected=target in c['expected'],prompt_text_sha256=hashlib.sha256(text.encode()).hexdigest(),prompt_token_sha256=joint.token_sha(ids),prompt_tokens=len(ids)))
                        prompts.append(dict(prompt_token_ids=ids))
            save('registration.safe.json',dict(cases=len(cases),requests=len(specs),case_sha256=a.case_sha,semantic_review_sha256=a.audit_sha,model=joint.MODEL,max_tokens=16,per_case_condition_total_output_cap=96,repeated_input_compute=6,max_model_len=8192,tp=8,max_num_seqs=16,thinking=False,temperature=0,seed=1024,paired_joint_variant=0,training_started=False,official_protocol_changed=False,code_sha256=joint.sha(__file__),joint_code_sha256=joint.sha(joint.__file__)))
            status('loading_model')
            llm=LLM(model=joint.MODEL,tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,max_num_seqs=16,gpu_memory_utilization=.8,seed=1024,enforce_eager=True)
            status('inferring',completed=0,total=len(specs));all_rows=[]
            with (a.out/'predictions.private.jsonl').open('x') as f:
                for start in range(0,len(specs),16):
                    outs=sorted(llm.generate(prompts[start:start+16],SamplingParams(temperature=0,max_tokens=16,seed=1024)),key=lambda x:int(x.request_id))
                    assert len(outs)==len(prompts[start:start+16])
                    for offset,out in enumerate(outs):
                        i=start+offset;s=specs[i];assert list(out.prompt_token_ids)==prompts[i]['prompt_token_ids']
                        ans=out.outputs[0];parsed=parse_bool(ans.text)
                        r=dict(s,text=ans.text,parsed=parsed,finish_reason=ans.finish_reason,output_tokens=len(ans.token_ids),correct=parsed is not None and parsed==s['expected'] and ans.finish_reason=='stop')
                        all_rows.append(r);f.write(json.dumps(r)+'\n')
                    f.flush();status('inferring',completed=len(all_rows),total=len(specs))
            save('summary.safe.json',dict(requests=len(all_rows),correct=sum(r['correct'] for r in all_rows),invalid=sum(r['parsed'] is None for r in all_rows),truncated=sum(r['finish_reason']!='stop' for r in all_rows),training_ready=False))
            status('completed_pending_verification',requests=len(all_rows))
    except BaseException as exc:status('failed',error=type(exc).__name__,detail=str(exc));raise


if __name__=='__main__':main()
