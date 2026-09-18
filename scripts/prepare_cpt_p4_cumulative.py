"""Append only the released sixteen tasks; preserve every P1 message/token row."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

TRAIN_SHA='b2f0b062afa4cd34798e085b4b98a159269265829d896e60733d26d2364e85dc'
OLD_MESSAGES_SHA='78a4e8a4f0acf95c7a2e69aa342855ac60a1334a3f57fdb4677609363f2a4304'
OLD_PARQUET_SHA='7779c2ba6137eb91aaf80e56b262b4a4ed2620001eafd9097d5ce46f1e63bdf0'
TOKENIZER_SHA='06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]

def append_released(messages,tasks,release):
    from evaluate_logistics_knowledge import build_messages,_make_item
    assert release['training_allowed'] is True
    assert len(messages)==315 and len({x['id'] for x in messages})==315
    assert len(tasks)==16 and len({x['id'] for x in tasks})==16
    assert {t['id'] for t in tasks}==set(release['approved_train_ids'])
    assert not set(release['approved_train_ids']) & set(release['evaluation_only_ids'])
    result=deepcopy(messages)
    for t in tasks:
        assert t['split']=='train' and len(t['options'])==len(t['option_reasons'])==len(t['option_truth'])==4
        assert all(type(v) is bool for v in t['option_truth'])
        assert t['correct_indices']==[i for i,v in enumerate(t['option_truth']) if v]
        assert t['correct_indices'] and all(isinstance(s,str) and s.strip() for s in t['option_reasons'])
        for exposure in range(3):
            identity='p4-'+t['id']+f'-exposure{exposure}'
            item=_make_item(dataset='training',source_id=identity,category=t['unit'],question_type='multiple_choice',question=t['question'],options=t['options'],expected=t['correct_indices'])
            msgs=build_messages(item)
            answer=json.dumps({'answers':t['correct_indices']},separators=(',',':'))
            if exposure:
                msgs[0]['content']='Briefly explain the governing rule and its application, then return one JSON object with key "answers" containing all correct zero-based option indices.'
                explanation='\n'.join(f'Option {i}: '+reason for i,reason in enumerate(t['option_reasons']))
                answer=explanation+'\n'+answer
            msgs.append(dict(role='assistant',content=answer));result.append(dict(id=identity,messages=msgs))
    assert len(result)==len({r['id'] for r in result})==363 and result[:315]==messages
    return result

def prepare(original,packet,model,out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from qwen36_mcq_answer_dataset import Qwen36MCQAnswerDataset,validate_record
    assert sha(original/'train.messages.private.jsonl')==OLD_MESSAGES_SHA
    assert sha(original/'train.parquet')==OLD_PARQUET_SHA
    assert sha(original/'dev.parquet')=='3a35e5307cd46027b5e91f598917c70c0bc470175ba1de4ec534120edddc41f0'
    release=json.loads((packet/'release.safe.json').read_text())
    assert sha(packet/'train.private.jsonl')==release['train_file_sha256']==TRAIN_SHA
    assert sha(model/'tokenizer.json')==TOKENIZER_SHA
    messages=append_released(read(original/'train.messages.private.jsonl'),read(packet/'train.private.jsonl'),release)
    tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
    rows=[]
    for x in messages:
        msg=x['messages'];prefix=tok.encode(tok.apply_chat_template(msg[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        ids=prefix+tok.encode(msg[-1]['content'],add_special_tokens=False)+[tok.eos_token_id]
        rows.append(dict(id=x['id'],input_ids=ids,answer_start=len(prefix),loss_tokens=sum(validate_record(ids,len(prefix),tok.eos_token_id,4096))))
    assert rows[:315]==pq.read_table(original/'train.parquet').to_pylist()
    out.mkdir(exist_ok=False);pq.write_table(pa.Table.from_pylist(rows),out/'train.parquet')
    (out/'train.messages.private.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in messages),encoding='utf-8')
    for name in ('dev.parquet','dev.messages.private.jsonl'):shutil.copyfile(original/name,out/name)
    import torch_npu
    from omegaconf import OmegaConf
    from verl.trainer.sft_trainer import SFTTrainer
    config=OmegaConf.create(dict(data=dict(train_batch_size=3,num_workers=0,pad_mode='no_padding',truncation='error',max_length=4096)))
    ds={s:Qwen36MCQAnswerDataset(str(out/(s+'.parquet')),tok,config.data) for s in ('train','dev')}
    holder=SimpleNamespace(config=config,train_dataset=ds['train'],val_dataset=ds['dev'],engine=SimpleNamespace(get_data_parallel_rank=lambda:0,get_data_parallel_size=lambda:1))
    SFTTrainer._build_dataloader(holder);budgets={}
    for split in ('train','dev'):
        table=pq.read_table(out/(split+'.parquet')).to_pylist()
        order=list(getattr(holder,'train_sampler' if split=='train' else 'val_sampler'))
        assert sorted(order)==list(range(len(table)))
        seen=tokens=loss=0
        for batch in getattr(holder,'train_dataloader' if split=='train' else 'val_dataloader'):
            for ids,mask in zip(batch['input_ids'].unbind(),batch['loss_mask'].unbind()):
                expected=ds[split][order[seen]];assert ids.equal(expected['input_ids']) and mask.equal(expected['loss_mask'])
                seen+=1;tokens+=ids.numel();loss+=int(mask.sum())
        assert seen==len(table) and tokens==sum(len(r['input_ids']) for r in table) and loss==sum(r['loss_tokens'] for r in table)
        budgets[split]=dict(records=seen,sequence_tokens=tokens,loss_tokens=loss,sha256=sha(out/(split+'.parquet')),sampler_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest())
    result=dict(budgets=budgets,installed_loader_passed=True,original_315_messages_and_tokens_unchanged=True,
        tokenizer_sha256=TOKENIZER_SHA,release_sha256=sha(packet/'release.safe.json'),train_file_sha256=TRAIN_SHA,
        train_messages_sha256=sha(out/'train.messages.private.jsonl'),steps=121,batch=3,lr=5e-7,epochs=1,
        new_tasks=16,new_exposures=48,new_answer_only=16,new_explanation_answer=32,
        option_order='Original author order, unchanged across exposures; same policy as P1.',
        explanation_source='Released author option_reasons, no model-review answers.',
        evaluation_inputs_read=False,training_started=False,exact_token_budget_matched=False,
        sequence_token_delta=budgets['train']['sequence_tokens']-87340,supervised_token_delta=budgets['train']['loss_tokens']-12202)
    (out/'data_audit.safe.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('original','packet','model','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.original,a.packet,a.model,a.out),indent=2))
