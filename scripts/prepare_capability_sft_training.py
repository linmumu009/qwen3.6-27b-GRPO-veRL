"""Freeze answer-only tokens for low-exposure CPT-origin SFT pilot."""
import argparse
import json
import os
from pathlib import Path
from prepare_book_sft_training import encode_messages
from qwen36_mcq_answer_dataset import validate_record
from build_targeted_book_groups import digest

def main():
    p=argparse.ArgumentParser()
    for k in ('root','model','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();os.umask(0o077)
    import pandas as pd
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    train=a.root/'runs/book-capability-sft-20260909-01/train.messages.private.jsonl'
    dev=a.root/'runs/book-capability-batch-20260908-01/organized/dev_candidates.private.jsonl'
    assert digest(train)=='07f50bbb7ea767ca6a38205db6e004e13eee7559ded71b205cf6e78bee025af5'
    assert digest(dev)=='88a838674b9162b856a9cd02beb8a2fda6289e169d80835a8bcd83facb4e81c3'
    a.out.mkdir(parents=True,exist_ok=False);info={};seen=set();chapters={}
    for split,path in [('train',train),('dev',dev)]:
        rows=[json.loads(x) for x in path.read_text().splitlines()];records=[];chapters[split]=set()
        for q in rows:
            assert q['id'] not in seen and q['split']==split;seen.add(q['id']);chapters[split].add(q['chapter'])
            messages=[dict(role='user',content=q['question']),dict(role='assistant',content=q['answer'])]
            if split=='train':assert q['messages']==messages
            ids,start=encode_messages(messages,tok);mask=validate_record(ids,start,tok.eos_token_id,4096)
            records.append(dict(id=q['id'],input_ids=ids,answer_start=start,loss_tokens=sum(mask)))
        pd.DataFrame(records).to_parquet(a.out/(split+'.parquet'),index=False)
        info[split]=dict(count=len(records),loss_tokens=sum(r['loss_tokens'] for r in records),sequence_tokens=sum(len(r['input_ids']) for r in records),max_length=max(len(r['input_ids']) for r in records),source_sha256=digest(path),parquet_sha256=digest(a.out/(split+'.parquet')))
    assert not chapters['train']&chapters['dev'] and info['train']['count']==355 and info['dev']['count']==71
    info.update(steps=44,batch_size=8,epochs=1,dropped_tail=3,learning_rate=2e-7,supervision='answer and EOS only',dev_use='Loss diagnostic only; unchanged references may contain rubric/answer defects. Not checkpoint selection.',training=False)
    (a.out/'manifest.safe.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))

if __name__=='__main__':main()
