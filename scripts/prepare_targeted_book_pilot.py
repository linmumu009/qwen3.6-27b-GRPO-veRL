"""Frozen short SFT pilot; private textbook pools only, never benchmark labels."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from probe_targeted_book_groups import prompt_messages
from qwen36_mcq_answer_dataset import validate_record

EXPECTED={'opportunity':67,'maintenance':16,'development':39,'quarantine':55}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_training(opportunity,maintenance):
    # Reserve one from each larger topic, preserving all six handling opportunities.
    reserve=[]
    for topic in ('general','warehousing','transport'):
        candidates=sorted((r for r in opportunity if r['topic']==topic),
                          key=lambda r:hashlib.sha256(('pilot-v1:'+r['id']).encode()).hexdigest())
        if not candidates:raise ValueError('Missing required topic')
        reserve.append(candidates[0])
    held={r['id'] for r in reserve}
    train=[r for r in opportunity if r['id'] not in held]+maintenance
    if len(train)!=80 or len(held)!=3:raise ValueError('Unexpected pilot size')
    return sorted(train,key=lambda r:r['id']),reserve


def main():
    p=argparse.ArgumentParser()
    for key in ('pools','model','book','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();os.umask(0o077)
    meta=json.loads((a.pools/'summary.safe.json').read_text())
    if meta['pools']!=EXPECTED:raise ValueError('Unexpected frozen pools')
    pools={};ids=set();group_splits={};chapters={'train':set(),'dev':set()}
    source={r['record_id']:r for r in map(json.loads,a.book.read_text().splitlines())}
    for name,count in EXPECTED.items():
        path=a.pools/f'{name}.private.jsonl'
        if sha(path)!=meta['hashes'][name]:raise ValueError('Pool hash mismatch')
        rows=list(map(json.loads,path.read_text().splitlines()))
        if len(rows)!=count:raise ValueError('Pool count mismatch')
        for r in rows:
            if r['id'] in ids:raise ValueError('Duplicate ID')
            ids.add(r['id'])
            group_splits.setdefault(r['concept_group'],set()).add(r['split'])
            if name=='quarantine':continue
            if r['split']!=('dev' if name=='development' else 'train'):raise ValueError('Wrong split')
            if not r.get('semantic_screen_passed'):raise ValueError('Missing semantic screen')
            if not r['quality'] or any(v is not True for v in r['quality'].values()):raise ValueError('Quality gate')
            s=source[r['source_id']]
            if hashlib.sha256(s['text'].encode()).hexdigest()!=r['source_hash']:raise ValueError('Source hash')
            if s['chapter']!=r['chapter']:raise ValueError('Source chapter')
            chapters[r['split']].add(r['chapter'])
        pools[name]=rows
    if any(len(v)!=1 for v in group_splits.values()) or chapters['train']&chapters['dev']:
        raise ValueError('Split leakage')
    train,reserve=select_training(pools['opportunity'],pools['maintenance'])
    from transformers import AutoTokenizer
    import pandas as pd
    t=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    records=[]
    for r in train:
        prompt=t.apply_chat_template(prompt_messages(r,0),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=t.encode(prompt,add_special_tokens=False)
        answer=t.encode(r['answer'],add_special_tokens=False)
        tokens=prefix+answer+[t.eos_token_id]
        mask=validate_record(tokens,len(prefix),t.eos_token_id,4096)
        records.append({'id':r['id'],'input_ids':tokens,'answer_start':len(prefix),'loss_tokens':sum(mask)})
    a.output.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(records).to_parquet(a.output/'train.parquet',index=False)
    for name,rows in [('train',train),('reserve',reserve)]:
        (a.output/f'{name}.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    evaluation=a.output/'evaluation';evaluation.mkdir()
    # Maintenance is explicitly in-training, not a held-out generalization test.
    eval_rows=pools['development']+pools['maintenance']
    f=evaluation/'candidates.private.jsonl';f.write_text(''.join(json.dumps(r)+'\n' for r in eval_rows))
    (evaluation/'summary.safe.json').write_text(json.dumps({'candidates_sha256':sha(f),'items':len(eval_rows)}))
    safe={'train_count':80,'opportunity_count':64,'maintenance_count':16,'reserve_count':3,
          'development_count':39,'evaluation_count':55,'batch_size':8,'epochs':1,'steps':10,
          'save_steps':[5,10],'learning_rate':5e-7,'dropped_tail':0,
          'loss_tokens':sum(r['loss_tokens'] for r in records),
          'sequence_tokens':sum(len(r['input_ids']) for r in records),
          'max_length':max(len(r['input_ids']) for r in records),
          'train_topics':dict(Counter(r['topic'] for r in train)),
          'pool_hashes':meta['hashes'],'train_parquet_sha256':sha(a.output/'train.parquet'),
          'source_model':str(a.model),'source_book_sha256':sha(a.book),
          'training_started':False,'supervision':'reference answer plus EOS only; original question only; closed-book',
          'benchmark_text_in_training':False,'development_chapter_disjoint':True,
          'limitations':['Small nonrandom textbook development set; CPT has seen the full book.',
             'One short pilot, not a causal comparison of selection strategies; 16 maintenance items are in-training.',
             'No claim of Agent/general regression safety without separate evaluation.']}
    (a.output/'manifest.safe.json').write_text(json.dumps(safe,indent=2));print(json.dumps(safe,indent=2))


if __name__=='__main__':main()
