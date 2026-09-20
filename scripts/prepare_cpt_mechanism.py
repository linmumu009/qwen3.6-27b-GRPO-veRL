"""Source-only K/R builder: never accepts formal questions, options or labels.

R selection is deterministic greedy token matching, with stable ID tie breaks.
It is frozen before any new inference; budget failure is terminal, not relaxed.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from prepare_cpt_p4_cumulative import OLD_MESSAGES_SHA, OLD_PARQUET_SHA, TOKENIZER_SHA
from qwen36_mcq_answer_dataset import validate_record


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def budget(rows):
    return dict(records=len(rows),sequence_tokens=sum(len(r['input_ids']) for r in rows),supervised_tokens=sum(r['loss_tokens'] for r in rows))


def validate_budgets(k,r,g):
    assert 1<=g<=6
    for b in (k,r):
        assert b['records']==315+3*g
        assert b['sequence_tokens']<=120000 and b['supervised_tokens']<=30000
    # Symmetric and conservative: smaller total is the denominator.
    for key,limit in [('supervised_tokens',.05),('sequence_tokens',.10)]:
        assert abs(k[key]-r[key])/min(k[key],r[key])<=limit,(key,k[key],r[key])


def prepare(original,sources_path,model,out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    assert sha(original/'train.messages.private.jsonl')==OLD_MESSAGES_SHA
    assert sha(original/'train.parquet')==OLD_PARQUET_SHA
    assert sha(model/'tokenizer.json')==TOKENIZER_SHA
    sources=json.loads(sources_path.read_text(encoding='utf8'));g=len(sources)
    assert 1<=g<=6 and len({s['unit_id'] for s in sources})==g
    allowed={'unit_id','title','body','scope','source_key','source_sha256','body_sha256'}
    assert all(set(s)==allowed for s in sources),'source-only schema violation'
    messages=read(original/'train.messages.private.jsonl');rows=pq.read_table(original/'train.parquet').to_pylist()
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    def encode(x):
        m=x['messages'];prefix=tok.encode(tok.apply_chat_template(m[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        ids=prefix+tok.encode(m[-1]['content'],add_special_tokens=False)+[tok.eos_token_id]
        mask=validate_record(ids,len(prefix),tok.eos_token_id,4096)
        return dict(id=x['id'],input_ids=ids,answer_start=len(prefix),loss_tokens=sum(mask))
    assert [encode(x) for x in messages]==rows
    additions=[]
    for s in sources:
        assert hashlib.sha256(s['body'].encode()).hexdigest()==s['body_sha256']
        for i in range(3):
            additions.append(dict(id='K-'+s['unit_id']+f'-{i}',messages=[
                dict(role='system',content='Explain the concept and its applicable scope accurately.'),
                dict(role='user',content=f"In the Eurostat/UNECE/ITF Glossary for Transport Statistics, fifth edition (2019), explain {s['title']} and its applicable scope."),
                dict(role='assistant',content=s['body'])]))
    krows=[encode(x) for x in additions];kb=budget(rows+krows)
    # P1's newly constructed speed/BOM/severity/Incoterms families are disjoint
    # from glossary vessel/rail/vehicle definitions. Restrict R to these reviewed
    # records, excluding all replay and any target-term occurrence in messages.
    permitted=('p1-speed-train-','p1-bom-train-','p1-severity-train-','p1-incoterms-train-')
    candidates=[]
    for x,r in zip(messages,rows):
        if not x['id'].startswith(permitted):continue
        text=' '.join(m['content'] for m in x['messages']).lower()
        if any(s['title'].lower() in text or s['body'].lower() in text for s in sources):continue
        candidates.append((x,r))
    assert len(candidates)>=3*g,'insufficient disjoint original records'
    chosen=[];sl=ss=0
    target_l=sum(r['loss_tokens'] for r in krows);target_s=sum(len(r['input_ids']) for r in krows)
    for i in range(3*g):
        fraction=(i+1)/(3*g)
        x,r=min(candidates,key=lambda pair:(
            abs(sl+pair[1]['loss_tokens']-fraction*target_l)/kb['supervised_tokens']+
            abs(ss+len(pair[1]['input_ids'])-fraction*target_s)/kb['sequence_tokens'],pair[0]['id']))
        chosen.append((x,r));candidates=[pair for pair in candidates if pair[0]['id']!=x['id']]
        sl+=r['loss_tokens'];ss+=len(r['input_ids'])
    radd=[]
    for i,(x,r) in enumerate(chosen):
        new=deepcopy(x);new['id']='R-'+x['id']+f'-{i}';radd.append(new)
    rr=[encode(x) for x in radd];rb=budget(rows+rr)
    validate_budgets(kb,rb,g)
    out.mkdir(exist_ok=False)
    for arm,extra,extra_rows in [('K',additions,krows),('R',radd,rr)]:
        d=out/arm;d.mkdir()
        pq.write_table(pa.Table.from_pylist(rows+extra_rows),d/'train.parquet')
        (d/'train.messages.private.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in messages+extra),encoding='utf8')
        for name in ('dev.parquet','dev.messages.private.jsonl'):shutil.copyfile(original/name,d/name)
    result={'units':g,'records_per_arm':315+3*g,'steps_per_arm':105+g,'lr':5e-7,'batch':3,'epochs':1,'seed':1,
            'budgets':{'K':kb,'R':rb},'sources_sha256':sha(sources_path),'old_messages_sha256':OLD_MESSAGES_SHA,
            'original_315_messages_and_tokens_unchanged':True,'formal_inputs_read':False,'training_allowed':False,
            'pending':'installed loader, seed, model identity and runtime comparability verification',
            'R_selection':'greedy normalized cumulative token residual; original ID tie break; no post scores',
            'R_original_ids':[x['id'] for x,r in chosen],
            'files':{arm:{n:sha(out/arm/n) for n in ('train.parquet','train.messages.private.jsonl','dev.parquet')} for arm in ('K','R')}}
    save(out/'budget.safe.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('original','sources','model','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.original,a.sources,a.model,a.out),indent=2))
