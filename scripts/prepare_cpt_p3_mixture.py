"""P3 replaces only 60 extra replay records with released arithmetic."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def rank(identity):return hashlib.sha256(identity.encode()).hexdigest()

def mixture(messages,tasks,mapping):
    from evaluate_logistics_knowledge import _make_item,build_messages
    assert len(messages)==315 and len(tasks)==60 and len(mapping)==60
    result=deepcopy(messages);replacements=[]
    for task,entry in zip(sorted(tasks,key=lambda x:(x['operation'],x['id'])),mapping):
        i=entry['position'];assert messages[i]['id']==entry['new_id']
        item=_make_item(dataset='training',source_id=task['id'],category='source_only',question_type='multiple_choice',question=task['question'],options=task['options'],expected=task['correct_indices'])
        msg=build_messages(item)+[dict(role='assistant',content=json.dumps({'answers':task['correct_indices']},separators=(',',':')))]
        result[i]=dict(id=task['id'],messages=msg)
        replacements.append(dict(position=i,removed_id=messages[i]['id'],new_id=task['id']))
    assert sum(a==b for a,b in zip(messages,result))==255 and len({x['id'] for x in result})==315
    assert sum('-exposure' in x['id'] for x in result)==120
    assert sum(x['id'].startswith('p1-replay-') for x in result)==135
    return result,replacements


def prepare(original,packet,model,out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from audit_cpt_sft_search_data import TOKENIZER_SHA
    from qwen36_mcq_answer_dataset import Qwen36MCQAnswerDataset,validate_record
    prior=json.loads((original/'data_audit.safe.json').read_text())
    assert sha(original/'train.parquet')==prior['budgets']['train']['sha256']=='f0f7e1a7460b3d4a151b9a198464ff98dcc284beee53c3b6f9e53141d361f54d'
    from audit_cpt_p3_arithmetic import audit
    assert audit(packet)['passed']
    manifest=json.loads((packet/'manifest.safe.json').read_text())
    assert sha(packet/'train.private.jsonl')==manifest['files']['train.private.jsonl']
    assert sha(model/'tokenizer.json')==TOKENIZER_SHA
    tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
    messages=read(original/'train.messages.private.jsonl');new,mapping=mixture(messages,read(packet/'train.private.jsonl'),read(original/'replacement_map.private.jsonl'))
    original_rows=pq.read_table(original/'train.parquet').to_pylist()
    assert [r['id'] for r in original_rows]==[r['id'] for r in messages]
    by_id={r['id']:r for r in original_rows}
    rows=[]
    for x in new:
        msg=x['messages'];prefix=tok.encode(tok.apply_chat_template(msg[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        ids=prefix+tok.encode(msg[-1]['content'],add_special_tokens=False)+[tok.eos_token_id]
        row=dict(id=x['id'],input_ids=ids,answer_start=len(prefix),loss_tokens=sum(validate_record(ids,len(prefix),tok.eos_token_id,4096)))
        if x['id'] in by_id:assert row==by_id[x['id']]
        rows.append(row)
    out.mkdir(exist_ok=False)
    pq.write_table(pa.Table.from_pylist(rows),out/'train.parquet')
    for name,data in [('train.messages.private.jsonl',new),('replacement_map.private.jsonl',mapping)]:
        (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf-8')
    for name in ('dev.parquet','dev.messages.private.jsonl'):shutil.copyfile(original/name,out/name)
    assert sha(out/'dev.parquet')==prior['budgets']['dev']['sha256']
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
        assert seen==len(table) and tokens==sum(len(x['input_ids']) for x in table) and loss==sum(x['loss_tokens'] for x in table)
        budgets[split]=dict(records=seen,sequence_tokens=tokens,loss_tokens=loss,sha256=sha(out/(split+'.parquet')),sampler_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest())
    assert budgets['train']['sampler_order_sha256']==prior['budgets']['train']['sampler_order_sha256']
    result=dict(budgets=budgets,installed_loader_passed=True,steps=105,batch=3,lr=5e-7,
        target_exposures=120,replay_exposures=195,base_replay_exposures=135,arithmetic_exposures=60,unchanged_positions=255,target_answer_exposures=40,target_explanation_exposures=80,
        sequence_token_delta=budgets['train']['sequence_tokens']-78602,supervised_token_delta=budgets['train']['loss_tokens']-8770,
        exact_token_budget_matched=False,original_train_messages_sha256=sha(original/'train.messages.private.jsonl'),
        train_messages_sha256=sha(out/'train.messages.private.jsonl'),map_sha256=sha(out/'replacement_map.private.jsonl'),
        arithmetic_check_read_for_release_audit=True,arithmetic_check_used_for_training=False,
        original_retention_or_formal_inputs_read=False,training_started=False)
    (out/'data_audit.safe.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('original','packet','model','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.original,a.packet,a.model,a.out),indent=2))
