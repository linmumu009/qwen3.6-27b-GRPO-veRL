"""Tokenize the released P1 packet and verify the installed DP=1 trainer loader."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

from qwen36_mcq_answer_dataset import Qwen36MCQAnswerDataset,validate_record
from audit_cpt_sft_search_data import TOKENIZER_SHA
from evaluate_logistics_knowledge import build_messages,_make_item


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def prepare(packet,reviewed,model,legacy_dev,out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    release=json.loads((packet/'release.safe.json').read_text())
    assert release['training_allowed'] is True
    assert sha(packet/'tasks.private.jsonl')==release['packet_sha256']
    assert sha(reviewed)==release['reviewed_sha256']
    manifest=json.loads((packet/'manifest.safe.json').read_text())
    for name,digest in manifest['files'].items():assert sha(packet/name)==digest
    assert sha(model/'tokenizer.json')==TOKENIZER_SHA
    assert sha(legacy_dev/'dev.parquet')=='3a35e5307cd46027b5e91f598917c70c0bc470175ba1de4ec534120edddc41f0'
    tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
    examples=[]
    def add(t,identity,explain=False):
        item=_make_item(dataset='training',source_id=identity,category='source_only',question_type='multiple_choice',question=t['question'],options=t['options'],expected=t['correct_indices'])
        messages=build_messages(item)
        answer=json.dumps({'answers':t['correct_indices']},separators=(',',':'))
        if explain:
            messages[0]['content']='Briefly explain the governing rule and its application, then return one JSON object with key "answers" containing all correct zero-based option indices.'
            answer=t['explanation']+'\n'+answer
        messages.append(dict(role='assistant',content=answer))
        examples.append(dict(id=identity,messages=messages))
    old=[r for r in read(reviewed) if r['split']=='train'];assert len(old)==135
    for r in old:add(r['task'],'p1-replay-'+r['id'])
    new=read(packet/'train.private.jsonl');assert len(new)==60
    for t in new:
        for exposure in range(3):add(t,t['id']+f'-exposure{exposure}',exposure!=0)
    assert len(examples)==315
    random.Random(20260916).shuffle(examples)
    rows=[]
    for x in examples:
        msg=x['messages']
        prefix=tok.encode(tok.apply_chat_template(msg[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        answer=tok.encode(msg[-1]['content'],add_special_tokens=False)
        ids=prefix+answer+[tok.eos_token_id]
        mask=validate_record(ids,len(prefix),tok.eos_token_id,4096)
        rows.append(dict(id=x['id'],input_ids=ids,answer_start=len(prefix),loss_tokens=sum(mask)))
    out.mkdir(exist_ok=False)
    pq.write_table(pa.Table.from_pylist(rows),out/'train.parquet')
    (out/'train.messages.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in examples),encoding='utf-8')
    for name in ('dev.parquet','dev.messages.private.jsonl'):shutil.copyfile(legacy_dev/name,out/name)
    budgets={}
    for split in ('train','dev'):
        table=pq.read_table(out/(split+'.parquet')).to_pylist();msgs={r['id']:r['messages'] for r in read(out/(split+'.messages.private.jsonl'))}
        assert len(table)%3==0 and len(msgs)==len(table)
        for r in table:
            msg=msgs[r['id']]
            prefix=tok.encode(tok.apply_chat_template(msg[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
            actual=prefix+tok.encode(msg[-1]['content'],add_special_tokens=False)+[tok.eos_token_id]
            assert actual==r['input_ids'] and len(prefix)==r['answer_start']
            assert sum(validate_record(actual,len(prefix),tok.eos_token_id,4096))==r['loss_tokens']
        budgets[split]=dict(records=len(table),sequence_tokens=sum(len(r['input_ids']) for r in table),loss_tokens=sum(r['loss_tokens'] for r in table),sha256=sha(out/(split+'.parquet')),max_length=max(len(r['input_ids']) for r in table))
    # Exercise actual trainer sampler/collator, including loss-mask alignment.
    import torch_npu
    from omegaconf import OmegaConf
    from verl.trainer.sft_trainer import SFTTrainer
    config=OmegaConf.create(dict(data=dict(train_batch_size=3,num_workers=0,pad_mode='no_padding',truncation='error',max_length=4096)))
    datasets={s:Qwen36MCQAnswerDataset(str(out/(s+'.parquet')),tok,config.data) for s in ('train','dev')}
    holder=SimpleNamespace(config=config,train_dataset=datasets['train'],val_dataset=datasets['dev'],engine=SimpleNamespace(get_data_parallel_rank=lambda:0,get_data_parallel_size=lambda:1))
    SFTTrainer._build_dataloader(holder)
    for split in ('train','dev'):
        loader=getattr(holder,'train_dataloader' if split=='train' else 'val_dataloader')
        order=list(getattr(holder,'train_sampler' if split=='train' else 'val_sampler'))
        assert sorted(order)==list(range(len(datasets[split])))
        seen=tokens=loss=0
        for batch in loader:
            for ids,mask in zip(batch['input_ids'].unbind(),batch['loss_mask'].unbind()):
                expected=datasets[split][order[seen]]
                assert ids.equal(expected['input_ids']) and mask.equal(expected['loss_mask'])
                seen+=1;tokens+=ids.numel();loss+=int(mask.sum())
        assert (seen,tokens,loss)==tuple(budgets[split][k] for k in ('records','sequence_tokens','loss_tokens'))
        budgets[split]['sampler_order_sha256']=hashlib.sha256(json.dumps(order).encode()).hexdigest()
    result=dict(budgets=budgets,tokenizer_sha256=TOKENIZER_SHA,release_sha256=sha(packet/'release.safe.json'),
        installed_loader_passed=True,data_parallel_size=1,train_shuffle_seed=20260916,
        validation_role='33 legacy frozen development records monitor loss only; new 40 transfer and 80 retention tasks are inference-only.',
        steps=105,lr=5e-7,batch=3,epochs=1,save_contents=['model','extra'],optimizer_resume_available=False)
    (out/'data_audit.safe.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('packet','reviewed','model','legacy-dev','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.packet,a.reviewed,a.model,a.legacy_dev,a.out),indent=2))
