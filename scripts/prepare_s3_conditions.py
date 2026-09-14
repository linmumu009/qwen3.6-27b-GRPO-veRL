"""Freeze source-condition SFT data; official benchmarks are never inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('reviewed','conditions','model','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from qwen36_mcq_answer_dataset import validate_record
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    assert digest(a.model/'tokenizer.json')=='06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'
    reviewed=[json.loads(x) for x in a.reviewed.read_text(encoding='utf-8').splitlines()]
    conditions=[json.loads(x) for x in a.conditions.read_text(encoding='utf-8').splitlines()]
    assert len(reviewed)==157 and len(conditions)==55
    train_units={r['unit_id'] for r in reviewed if r['split']=='train'}
    assert all(r['unit_id'] in train_units for r in conditions)
    assert Counter(r['split'] for r in conditions)=={'train':44,'scenario_check':11}
    assert not ({r['group'] for r in reviewed if r['split']=='train'} & {r['group'] for r in reviewed if r['split']=='dev'})
    train=[];dev=[];cases=[]
    def task_messages(task,explain):
        instruction=('Briefly explain the governing classification condition, then return a JSON object with key "answers" containing all correct zero-based option indices.' if explain else 'Return exactly one JSON object with key "answers" containing all correct zero-based option indices. Do not include explanations or other text.')
        prompt=instruction+'\n\nQuestion:\n'+task['question']+'\n\nOptions:\n'+'\n'.join(f'[{i}] {s}' for i,s in enumerate(task['options']))
        answer=json.dumps({'answers':task['correct_indices']},separators=(',',':'))
        if explain:answer=task['explanation']+'\n'+answer
        return [dict(role='user',content=prompt),dict(role='assistant',content=answer)]
    def record(task,id,explain):
        messages=task_messages(task,explain)
        prompt=tok.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=tok.encode(prompt,add_special_tokens=False)
        values=prefix+tok.encode(messages[-1]['content'],add_special_tokens=False)+[tok.eos_token_id]
        mask=validate_record(values,len(prefix),tok.eos_token_id,4096)
        return dict(id=id,input_ids=values,answer_start=len(prefix),loss_tokens=sum(mask)),dict(id=id,messages=messages)
    def case(task,id,split,category):
        return dict(dataset=split,source_id=id,category=category,question_type='multi_choice' if len(task['correct_indices'])>1 else 'single_choice',question=task['question'],options=task['options'],expected=task['correct_indices'])
    for r in reviewed:
        t=r['task']
        target=train if r['split']=='train' else dev
        target.append(record(t,r['id'],False))
        cases.append(case(t,r['id'],'source_conditions_'+r['split'],'reviewed_source'))
    for r in conditions:
        copies=6 if r['split']=='train' else 1
        for exposure in range(copies):
            (train if r['split']=='train' else dev).append(record(r,r['id']+f'-exposure{exposure}',r['split']=='train' and exposure%2==1))
        cases.append(case(r,r['id'],'condition_training' if r['split']=='train' else 'condition_scenario_check',r['topics'][0]))
    assert len(train)==399 and len(dev)==33 and len(cases)==212
    assert len({r[0]['id'] for r in train+dev})==432
    a.out.mkdir(exist_ok=False)
    manifest=dict(arm='S3',lr=5e-7,batch_size=3,steps=133,epochs=1,
        source_condition_templates=11,condition_training_variants=44,condition_exposures=6,
        foundation_train_once=135,unique_training_tasks=179,
        validation_note='22 source-group-held-out tasks plus 11 same-template unseen-parameter checks; latter are not independent generalization.',
        preservation_note='135 previously reviewed source tasks serve as foundation retention; no claim of general/Agent preservation.',
        reviewed_sha256=digest(a.reviewed),conditions_sha256=digest(a.conditions),
        tokenizer_sha256=digest(a.model/'tokenizer.json'),training_started=False)
    for split,records in [('train',train),('dev',dev)]:
        pq.write_table(pa.Table.from_pylist([r[0] for r in records]),a.out/(split+'.parquet'))
        (a.out/(split+'.messages.private.jsonl')).write_text(''.join(json.dumps(r[1],ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        manifest[split]=dict(count=len(records),sequence_tokens=sum(len(r[0]['input_ids']) for r in records),loss_tokens=sum(r[0]['loss_tokens'] for r in records),parquet_sha256=digest(a.out/(split+'.parquet')))
    (a.out/'cases.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases),encoding='utf-8')
    manifest['cases_sha256']=digest(a.out/'cases.private.jsonl')
    (a.out/'manifest.safe.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
