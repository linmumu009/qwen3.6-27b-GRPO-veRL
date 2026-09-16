"""Validate and run audited source-based scenarios in paired option orders.

The packet carries private source text and option-level rationale. Neither labels
nor rationales enter inference prompts. Evidence conditions supply source facts
and are diagnostic only. They are never scored as closed-book performance.
"""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import re

MODEL = '/workspace/llin-verl-grpo/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource'
LOCK = '/workspace/llin-verl-grpo/runs/.logistics-exam-cpt.lock'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def arithmetic(expression):
    """Exact rational arithmetic only; never execute arbitrary packet code."""
    if not isinstance(expression, str) or len(expression) > 200:
        raise ValueError('arithmetic expression too long')
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) < 10**10:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            n = walk(node.operand)
            return -n if isinstance(node.op, ast.USub) else n
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult): return a*b
            if isinstance(node.op, ast.Div): return a/b
            if isinstance(node.op, ast.Pow) and b.denominator == 1 and 0 <= b <= 8:
                return a**int(b)
        raise ValueError('unsupported arithmetic')
    return walk(ast.parse(expression, mode='eval').body)


def validate(cases):
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('empty packet or duplicate cases')
    for c in cases:
        options, answers = c['options'], c['expected']
        if not 4 <= len(options) <= 8 or len({s.strip().casefold() for s in options}) != len(options):
            raise ValueError('invalid or duplicate options')
        if not answers or len(answers) >= len(options) or len(set(answers)) != len(answers) or any(type(a) is not int or not 0 <= a < len(options) for a in answers):
            raise ValueError('invalid answer set')
        if len(c['option_audit']) != len(options) or any(not x.strip() for x in c['option_audit']):
            raise ValueError('missing option audit')
        if not c['sources'] or not c['question'].endswith('Select all correct statements.'):
            raise ValueError('missing source or response scope')
        if c['training_allowed'] is not False or c['purpose'] != 'screen_only':
            raise ValueError('screening-only packet required')
        for expression, expected in c['arithmetic_checks']:
            if arithmetic(expression) != arithmetic(expected):
                raise ValueError('numerical audit failed: '+c['id'])


def variants(case):
    seed = int(hashlib.sha256(case['id'].encode()).hexdigest()[:16],16)
    order = list(range(len(case['options'])))
    random.Random(seed).shuffle(order)
    # A cyclic shift has no fixed option position. Both orders are shuffled
    # relative to authoring order; no recurring true-answer position is exposed.
    return [order, order[3:]+order[:3]]


def prompt(case, order, condition):
    text = 'Return only a JSON object with key "answers" containing all correct zero-based option indices.\n'
    if condition == 'source_evidence':
        text += 'Use the following archived source facts in the stated historical or statistical scope.\n'
        for source in case['sources']:
            body=re.sub(r'(?m)^[A-Z]\.[IVX]+(?:/[IVX]+)?[-.]\d+(?:\.\d+)?[ \t]+','',source['source_text'])
            text += source['scope']+'\n'+body+'\n'
    elif condition != 'closed_book':
        raise ValueError('unknown condition')
    text += 'Question:\n'+case['question']+'\nOptions:\n'
    return text+'\n'.join(f'[{i}] {case["options"][j]}' for i,j in enumerate(order))


def specifications(cases):
    result=[]
    for c in cases:
        for variant, order in enumerate(variants(c)):
            for condition in ('closed_book','source_evidence'):
                result.append(dict(id=c['id'], variant=variant, order=order, condition=condition,
                    expected=sorted(i for i,j in enumerate(order) if j in c['expected'])))
    return result


def parse(text,count):
    decoder=json.JSONDecoder();matches=[]
    for i,char in enumerate(text):
        if char!='{': continue
        try: obj,_=decoder.raw_decode(text[i:])
        except ValueError: continue
        if isinstance(obj,dict) and set(obj)=={'answers'}: matches.append(obj['answers'])
    if len(matches)!=1: return None
    values=matches[0]
    if not isinstance(values,list) or not values or len(values)>=count or len(set(map(str,values)))!=len(values) or any(type(v) is not int or not 0<=v<count for v in values):
        return None
    return sorted(values)


def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    os.umask(0o077)
    if sha(a.cases)!=a.sha: raise ValueError('packet fingerprint mismatch')
    cases=[json.loads(s) for s in a.cases.read_text(encoding='utf-8').splitlines()];validate(cases)
    if len(cases)!=32: raise ValueError('registered batch requires 32 audited scenarios')
    a.out.mkdir(exist_ok=False)
    def save(name,value): (a.out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    def status(value,**kwargs): save('status.safe.json',dict(status=value,training_running=False,**kwargs))
    status('waiting_for_lock')
    try:
        with open(LOCK,'a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            from vllm import LLM,SamplingParams
            tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True)
            by_id={c['id']:c for c in cases};specs=specifications(cases);prompts=[];text_hashes=[]
            for spec in specs:
                text=prompt(by_id[spec['id']],spec['order'],spec['condition'])
                rendered=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tokenizer.encode(rendered,add_special_tokens=False)
                if len(ids)+96>8192:raise ValueError('context overflow')
                prompts.append(dict(prompt_token_ids=ids));text_hashes.append(hashlib.sha256(text.encode()).hexdigest())
            save('registration.safe.json',dict(case_sha256=a.sha,model=MODEL,cases=len(cases),requests=len(specs),
                max_tokens=96,temperature=0,seed=1024,thinking=False,tp=8,max_num_seqs=16,max_model_len=8192,
                option_orders=2,repeats_per_order_condition=1,official_protocol_changed=False,training_started=False,
                source_evidence_not_closed_book=True,source_cleanup='strip line-start chapter identifiers only',code_sha256=sha(__file__)))
            status('loading_model')
            llm=LLM(model=MODEL,tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,max_num_seqs=16,gpu_memory_utilization=.8,seed=1024,enforce_eager=True)
            status('inferring',completed=0,total=len(specs))
            records=[]
            with (a.out/'predictions.private.jsonl').open('x',encoding='utf-8') as f:
                for start in range(0,len(specs),16):
                    batch=prompts[start:start+16]
                    outs=sorted(llm.generate(batch,SamplingParams(temperature=0,max_tokens=96,seed=1024)),key=lambda r:int(r.request_id))
                    if len(outs)!=len(batch):raise ValueError('request count mismatch')
                    for offset,out in enumerate(outs):
                        i=start+offset;s=specs[i];c=by_id[s['id']];ids=prompts[i]['prompt_token_ids']
                        if list(out.prompt_token_ids)!=ids:raise ValueError('request alignment mismatch')
                        answer=out.outputs[0];parsed=parse(answer.text,len(c['options']))
                        rec=dict(s,text=answer.text,parsed=parsed,correct=parsed==s['expected'] and answer.finish_reason=='stop',
                            finish_reason=answer.finish_reason,output_tokens=len(answer.token_ids),prompt_tokens=len(ids),
                            prompt_token_sha256=token_sha(ids),prompt_text_sha256=text_hashes[i])
                        records.append(rec);f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                    f.flush();status('inferring',completed=len(records),total=len(specs))
            save('summary.safe.json',dict(requests=len(records),by_condition={k:dict(requests=sum(r['condition']==k for r in records),correct=sum(r['condition']==k and r['correct'] for r in records)) for k in ('closed_book','source_evidence')},invalid=sum(r['parsed'] is None for r in records),truncated=sum(r['finish_reason']!='stop' for r in records),training_ready=False))
            status('completed_pending_verification',requests=len(records))
    except BaseException as exc:
        status('failed',error=type(exc).__name__,detail=str(exc));raise


if __name__=='__main__':main()
