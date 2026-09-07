"""Convert filtered book-only messages into exact closed-book answer-only SFT."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def encode_messages(messages, tokenizer):
    prompt=tokenizer.apply_chat_template(messages[:1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix=tokenizer.encode(prompt,add_special_tokens=False)
    answer=tokenizer.encode(messages[1]['content'],add_special_tokens=False)
    return list(prefix)+list(answer)+[tokenizer.eos_token_id], len(prefix)


def main():
    p=argparse.ArgumentParser()
    for k in ('data','model','output'): p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(); os.umask(0o077)
    import pandas as pd
    from transformers import AutoTokenizer
    from qwen36_mcq_answer_dataset import validate_record
    manifest=json.loads((a.data/'dataset.safe.json').read_text())
    if manifest.get('retained')!=2000 or not manifest.get('dataset_ready'): raise ValueError('Dataset not ready')
    tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    a.output.mkdir(parents=True,exist_ok=False)
    info={}; chapter_sets={}; all_ids=set()
    for split in ('train','dev'):
        path=a.data/f'{split}.messages.private.jsonl'; raw=path.read_bytes()
        records=[]; chapters=set()
        for r in map(json.loads,raw.decode().splitlines()):
            if r['id'] in all_ids: raise ValueError('Duplicate ID across splits')
            all_ids.add(r['id']); chapters.add(r['chapter'])
            messages=r['messages']
            if [x['role'] for x in messages]!=['user','assistant']: raise ValueError('Invalid messages')
            ids,start=encode_messages(messages,tokenizer)
            mask=validate_record(ids,start,tokenizer.eos_token_id,4096)
            records.append({'id':r['id'],'input_ids':ids,'answer_start':start,'loss_tokens':sum(mask)})
        if not records: raise ValueError('Empty split')
        pd.DataFrame(records).to_parquet(a.output/f'{split}.parquet',index=False)
        info[split]={'count':len(records),'source_jsonl_sha256':hashlib.sha256(raw).hexdigest(),
            'sequence_tokens':sum(len(r['input_ids']) for r in records),'loss_tokens':sum(r['loss_tokens'] for r in records),
            'max_length':max(len(r['input_ids']) for r in records)}
        chapter_sets[split]=chapters
    if chapter_sets['train']&chapter_sets['dev']: raise ValueError('Chapter split overlap')
    info.update(total_count=len(all_ids),batch_size=8,epochs=1,steps=info['train']['count']//8,
        dropped_tail_per_epoch=info['train']['count']%8,source_model=str(a.model),supervision='answer and EOS only')
    (a.output/'manifest.safe.json').write_text(json.dumps(info,indent=2))


if __name__=='__main__': main()
