"""Freeze source-only task SFT and disjoint objective development cases."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

HOLD = {'C39-U06-Q1': 'Unspecified author-specific first-step framework in manual spot check'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_key(row):
    # Option content is part of the task: generic stems alone are not duplicates.
    return tuple(re.sub(r'\s+', ' ', x).strip().casefold()
                 for x in [row['question'], *row['options']])


def messages(row):
    if row['kind'] == 'short_answer':
        return [{'role': 'user', 'content': row['question']},
                {'role': 'assistant', 'content': row['answer']}]
    choices = '\n'.join(f'[{i}] {v}' for i, v in enumerate(row['options']))
    explained = int(hashlib.sha256(row['id'].encode()).hexdigest(), 16) % 2 == 0
    instruction = ('Briefly explain the selection, then give a JSON object with key "answers" containing all correct zero-based option indices.'
                   if explained else 'Return exactly one JSON object with key "answers" containing all correct zero-based option indices. Do not include explanations or other text.')
    answer = json.dumps({'answers': row['correct_indices']}, separators=(',', ':'))
    if explained:
        answer = row['explanation'] + '\n' + answer
    return [{'role': 'user', 'content': instruction + '\n\nQuestion:\n' + row['question'] + '\n\nOptions:\n' + choices},
            {'role': 'assistant', 'content': answer}]


def clean(rows, units):
    assert len(rows) == len({r['id'] for r in rows}), 'Duplicate task ID'
    known = {u['id']: u for u in units}
    assert len(known) == len(units)
    seen = {}; kept = []; rejected = []
    for r in sorted(rows, key=lambda x: (x['split'] != 'dev', x['id'])):
        assert r['split'] in ('train', 'dev')
        assert known[r['unit_id']]['chapter'] == r['chapter']
        assert r['source_unit_ids'] and r['unit_id'] in r['source_unit_ids']
        assert all(known[i]['split'] == r['split'] for i in r['source_unit_ids']), 'Cross-split source'
        if r['id'] in HOLD:
            rejected.append({'id': r['id'], 'reason': HOLD[r['id']]}); continue
        key = input_key(r)
        if key in seen:
            # Contradictory duplicate labels require investigation, never silent first-wins.
            old = seen[key]
            assert (old['answer'], old['correct_indices']) == (r['answer'], r['correct_indices']), 'Conflicting duplicate target'
            rejected.append({'id': r['id'], 'reason': 'Exact normalized full-input duplicate'}); continue
        seen[key] = r; kept.append(r)
    assert not ({r['chapter'] for r in kept if r['split'] == 'train'} & {r['chapter'] for r in kept if r['split'] == 'dev'})
    return kept, rejected


def main():
    p = argparse.ArgumentParser()
    for k in ('source', 'model', 'out'): p.add_argument('--'+k, type=Path, required=True)
    a = p.parse_args(); os.umask(0o077)
    import pandas as pd
    from transformers import AutoTokenizer
    from qwen36_mcq_answer_dataset import validate_record
    rows = [json.loads(x) for x in (a.source/'tasks.private.jsonl').read_text().splitlines()]
    units = json.loads((a.source/'retained_units.private.json').read_text())
    kept, rejected = clean(rows, units)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    a.out.mkdir(parents=True, exist_ok=False)
    def write_rows(name, values):
        (a.out/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values))
    info = {'input_count':len(rows), 'retained':len(kept), 'rejected':rejected,
            'source_sha256':digest(a.source/'tasks.private.jsonl'),
            'units_sha256':digest(a.source/'retained_units.private.json'),
            'script_sha256':digest(Path(__file__)), 'supervision':'assistant and EOS only',
            'training':False, 'development_limit':'Chapters held out of SFT, not of prior book CPT; objective subset excludes free answers.',
            'format_policy':'One record/task. Deterministic ID hash chooses direct JSON or API-authored brief explanation plus JSON; free-answer tasks unchanged.'}
    for split in ('train','dev'):
        subset = [r for r in kept if r['split']==split]; records=[]; formatted=[]; cases=[]
        for r in subset:
            m = messages(r)
            prompt = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prefix = tok.encode(prompt, add_special_tokens=False)
            ids = prefix + tok.encode(m[-1]['content'], add_special_tokens=False) + [tok.eos_token_id]
            mask = validate_record(ids, len(prefix), tok.eos_token_id, 4096)
            records.append(dict(id=r['id'],input_ids=ids,answer_start=len(prefix),loss_tokens=sum(mask)))
            formatted.append(dict(id=r['id'],messages=m))
            if r['kind'] != 'short_answer':
                cases.append(dict(dataset='book_task_development' if split=='dev' else 'book_task_learning_check',source_id=r['id'],category=r['domain'],question_type=r['kind'],question=r['question'],options=r['options'],expected=r['correct_indices']))
        pd.DataFrame(records).to_parquet(a.out/(split+'.parquet'),index=False)
        write_rows(split+'.messages.private.jsonl',formatted)
        write_rows(split+'.cases.private.jsonl',cases)
        write_rows(split+'.tasks.private.jsonl',subset)
        info[split] = dict(count=len(records),objective_cases=len(cases),kind_counts=dict(Counter(r['kind'] for r in subset)),domain_counts=dict(Counter(r['domain'] for r in subset)),concepts=len({r['unit_id'] for r in subset}),loss_tokens=sum(r['loss_tokens'] for r in records),sequence_tokens=sum(len(r['input_ids']) for r in records),max_length=max(len(r['input_ids']) for r in records),parquet_sha256=digest(a.out/(split+'.parquet')),cases_sha256=digest(a.out/(split+'.cases.private.jsonl')))
    info.update(batch_size=8,steps_one_epoch=info['train']['count']//8,dropped_tail_one_epoch=info['train']['count']%8)
    (a.out/'manifest.safe.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))


if __name__ == '__main__': main()
