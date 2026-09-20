"""Installed-loader and baseline identity gate; no generation or training."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from prepare_cpt_mechanism import sha, read, budget, validate_budgets, save
from run_cpt_formal_transfer import model_identity
from run_logistics_cpt_curve_8x import BASE, CASES, ROOT
from run_logistics_strategy_diagnostic import messages_for


def gate(prepared,out):
    import torch_npu
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.trainer.sft_trainer import SFTTrainer
    from train_cpt_mechanism_sft import install_seed_contract
    install_seed_contract(SFTTrainer)
    from qwen36_mcq_answer_dataset import Qwen36MCQAnswerDataset,validate_record
    history=ROOT/'runs/llin-transfer-formal-20260917-01'
    reg=json.loads((history/'registration.safe.json').read_text())
    models={'step120_current':BASE,'p1':ROOT/'runs/llin-transfer-p1-run-20260916-02/training/llin-step120-p1-hf'}
    for name,model in models.items():assert model_identity(model)==reg['models'][name],name+' baseline model changed'
    for name,version in reg['versions'].items():assert importlib.metadata.version(name)==version,(name,'runtime changed')
    for directory,ref in reg['source_refs'].items():
        actual=subprocess.check_output(['git','-C',directory,'rev-parse','HEAD'],text=True).strip()
        assert actual==ref,(directory,'runtime source changed')
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    hashes=[]
    import hashlib
    for row in read(CASES):
        ids=tok.encode(tok.apply_chat_template(messages_for(row,'original')[0],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        hashes.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    for name in models:
        assert hashes==json.loads((history/name/'protocol.safe.json').read_text())['prompt_hashes']
    specification=json.loads((prepared/'budget.safe.json').read_text());budgets={};orders={}
    config=OmegaConf.create(dict(data=dict(train_batch_size=3,num_workers=0,pad_mode='no_padding',truncation='error',max_length=4096),trainer=dict(seed=1),engine=dict(seed=1)))
    for arm in ('K','R'):
        d=prepared/arm
        for n,digest in specification['files'][arm].items():assert sha(d/n)==digest
        for split in ('train','dev'):
            rows=pq.read_table(d/(split+'.parquet')).to_pylist();msgs=read(d/(split+'.messages.private.jsonl'))
            assert len(rows)==len(msgs)
            for r,x in zip(rows,msgs):
                prefix=tok.encode(tok.apply_chat_template(x['messages'][:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
                ids=prefix+tok.encode(x['messages'][-1]['content'],add_special_tokens=False)+[tok.eos_token_id]
                assert ids==r['input_ids'] and len(prefix)==r['answer_start']
                assert sum(validate_record(ids,len(prefix),tok.eos_token_id,4096))==r['loss_tokens']
        ds={s:Qwen36MCQAnswerDataset(str(d/(s+'.parquet')),tok,config.data) for s in ('train','dev')}
        holder=SimpleNamespace(config=config,train_dataset=ds['train'],val_dataset=ds['dev'],engine=SimpleNamespace(get_data_parallel_rank=lambda:0,get_data_parallel_size=lambda:1))
        SFTTrainer._build_dataloader(holder)
        for split in ('train','val'):
            dataset=ds['train' if split=='train' else 'dev'];order=list(getattr(holder,split+'_sampler'))
            assert sorted(order)==list(range(len(dataset)))
            seen=tokens=loss=0
            for batch in getattr(holder,split+'_dataloader'):
                for ids,mask in zip(batch['input_ids'].unbind(),batch['loss_mask'].unbind()):
                    item=dataset[order[seen]];assert ids.equal(item['input_ids']) and mask.equal(item['loss_mask'])
                    seen+=1;tokens+=ids.numel();loss+=int(mask.sum())
            if split=='train':
                budgets[arm]=dict(records=seen,sequence_tokens=tokens,supervised_tokens=loss)
                orders[arm]=order
                assert budgets[arm]==specification['budgets'][arm]
    assert orders['K']==orders['R']
    validate_budgets(budgets['K'],budgets['R'],specification['units'])
    result=dict(installed_loader_passed=True,baseline_model_and_runtime_match=True,prompts_match=True,
                budgets=budgets,same_sampler_order=True,sampler_sha256=hashlib.sha256(json.dumps(orders['K']).encode()).hexdigest(),
                budget_sha256=sha(prepared/'budget.safe.json'),training_allowed=True,trainer_seed=1,engine_seed=1,sampler_seed=1)
    save(out,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(gate(a.prepared,a.out)))
