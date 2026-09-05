"""Real-tokenizer, full-corpus answer mask and eval-prefix gate (no private output)."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--model',required=True)
    args=p.parse_args()
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from scripts.qwen36_mcq_answer_dataset import Qwen36MCQAnswerDataset
    from scripts.prepare_logistics_mcq_fit import SOURCE_SHA256, encode_item
    from scripts.run_vllm_logistics_mcq import load_items
    manifest=json.loads((args.data/'manifest.safe.json').read_text())
    parquet=args.data/'train.parquet'
    if hashlib.sha256(parquet.read_bytes()).hexdigest()!=manifest['parquet_sha256']:
        raise ValueError('parquet hash mismatch')
    if hashlib.sha256((Path(args.model)/'tokenizer.json').read_bytes()).hexdigest()!=manifest['tokenizer_sha256']:
        raise ValueError('tokenizer mismatch')
    source=Path('/workspace/llin-verl-grpo/runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA256:
        raise ValueError('frozen source mismatch')
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    dataset=Qwen36MCQAnswerDataset(str(parquet),tokenizer,
        OmegaConf.create({'pad_mode':'no_padding','truncation':'error','max_length':4096}))
    items=load_items(source)
    if len(dataset)!=1672 or len(items)!=1672:
        raise ValueError('case coverage mismatch')
    total=supervised=0
    for i,item in enumerate(items):
        expected=encode_item(item,tokenizer)
        actual=dataset[i]
        assert dataset.rows[i]['item_hash']==item.item_hash
        assert actual['input_ids'].tolist()==expected['input_ids']
        start=expected['answer_start']
        assert actual['loss_mask'][:start].sum().item()==0
        assert actual['loss_mask'][start:].all().item()
        assert actual['loss_mask'].roll(-1)[-1].item()==0
        total+=len(actual['input_ids'])
        supervised+=actual['loss_mask'].sum().item()
    assert total==manifest['sequence_tokens_per_epoch']
    assert supervised==manifest['supervised_tokens_per_epoch']
    result={'passed':True,'items':1672,'private_content_included':False,
        'same_as_official_eval_prefix':True,'gold_roundtrip':True,'eos_supervised':True,
        'sequence_tokens_per_epoch':total,'supervised_tokens_per_epoch':int(supervised)}
    (args.data/'gate.safe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
