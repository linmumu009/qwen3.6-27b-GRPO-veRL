"""Build the frozen S5 source-only task exposure; does not start training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

TASK_SHA = '8d1f5e3ce9772fc2ddf99e16ee92630dcbf8aa6a1b58e558b5f397d33c616ea2'


def main():
    p = argparse.ArgumentParser()
    for key in ('tasks', 'reviewed', 's3', 'probe', 'model', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from audit_cpt_sft_search_data import audit_records, TOKENIZER_SHA, digest
    from qwen36_mcq_answer_dataset import validate_record
    assert digest(a.tasks) == TASK_SHA
    assert digest(a.model/'tokenizer.json') == TOKENIZER_SHA
    assert digest(a.s3/'dev.parquet') == '3a35e5307cd46027b5e91f598917c70c0bc470175ba1de4ec534120edddc41f0'
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
    read = lambda f: [json.loads(x) for x in f.read_text(encoding='utf-8').splitlines()]
    tasks, reviewed = read(a.tasks), read(a.reviewed)
    assert len(tasks) == 32 and len(reviewed) == 157
    assert all(t['group'] == 'llin-core-010eea3da282a23f6b6b' for t in tasks)
    assert not ({r['group'] for r in reviewed if r['split'] == 'train'} &
                {r['group'] for r in reviewed if r['split'] == 'dev'})
    train, messages = [], []
    def add(task, identity, explain):
        instruction = ('Briefly explain the governing rule and calculation, then return a JSON object with key "answers" containing all correct zero-based option indices.' if explain else
                       'Return exactly one JSON object with key "answers" containing all correct zero-based option indices. Do not include explanations or other text.')
        prompt = instruction+'\n\nQuestion:\n'+task['question']+'\n\nOptions:\n'+'\n'.join(f'[{i}] {v}' for i,v in enumerate(task['options']))
        answer = json.dumps({'answers':task['correct_indices']}, separators=(',', ':'))
        if explain:
            answer = task['explanation']+'\n'+answer
        msg = [dict(role='user', content=prompt), dict(role='assistant', content=answer)]
        prefix = tok.encode(tok.apply_chat_template(msg[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
        values = prefix+tok.encode(answer, add_special_tokens=False)+[tok.eos_token_id]
        mask = validate_record(values, len(prefix), tok.eos_token_id, 4096)
        train.append(dict(id=identity, input_ids=values, answer_start=len(prefix), loss_tokens=sum(mask)))
        messages.append(dict(id=identity, messages=msg))
    for r in reviewed:
        if r['split'] == 'train':
            add(r['task'], r['id'], False)
    assert len(train) == 135
    for r in tasks:
        for exposure in range(3):
            add(r['task'], r['id']+f'-exposure{exposure}', exposure != 0)
    assert len(train) == 231
    audit = audit_records(train, messages, tok, 3)
    cases = read(a.s3/'cases.private.jsonl') + read(a.probe)
    for r in tasks:
        t = r['task']
        cases.append(dict(dataset='s5_condition_training', source_id=r['id'], category=r['family'],
                          question_type='single_choice', question=t['question'], options=t['options'], expected=t['correct_indices']))
    assert len(cases) == 253 and len({r['source_id'] for r in cases}) == 253
    a.out.mkdir(exist_ok=False)
    pq.write_table(pa.Table.from_pylist(train), a.out/'train.parquet')
    (a.out/'train.messages.private.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in messages), encoding='utf-8')
    for name in ('dev.parquet', 'dev.messages.private.jsonl'):
        shutil.copyfile(a.s3/name, a.out/name)
    (a.out/'cases.private.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in cases), encoding='utf-8')
    manifest = dict(arm='S5', lr=5e-7, batch_size=3, steps=77, epochs=1, train=audit,
                    tasks_sha256=TASK_SHA, reviewed_sha256=digest(a.reviewed), cases_sha256=digest(a.out/'cases.private.jsonl'),
                    training_started=False, fresh_optimizer=True, independent_generalization_claim=False)
    for split in ('train', 'dev'):
        rows = pq.read_table(a.out/(split+'.parquet')).to_pylist()
        manifest[split] = dict(count=len(rows), sequence_tokens=sum(len(r['input_ids']) for r in rows),
                               loss_tokens=sum(r['loss_tokens'] for r in rows), parquet_sha256=digest(a.out/(split+'.parquet')))
    (a.out/'manifest.safe.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
