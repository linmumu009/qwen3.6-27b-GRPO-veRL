"""Prepare private answer-only SFT using exactly the existing MCQ evaluation prompt."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from scripts.evaluate_logistics_knowledge import PROMPT_VERSION, build_messages, parse_answers
from scripts.run_vllm_logistics_mcq import load_items

SOURCE_SHA256 = 'b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99'


def encode_item(item, tokenizer, max_length=4096):
    prompt = tokenizer.apply_chat_template(build_messages(item), tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    answer = json.dumps({'answers': list(item.expected)}, separators=(',', ':'))
    parsed, valid = parse_answers(answer, len(item.options))
    if not valid or parsed != item.expected:
        raise ValueError('gold answer does not roundtrip through the official parser')
    answer_ids = list(tokenizer.encode(answer, add_special_tokens=False))
    if tokenizer.eos_token_id is None or not prompt_ids or not answer_ids:
        raise ValueError('missing prompt, answer or EOS')
    ids = prompt_ids + answer_ids + [int(tokenizer.eos_token_id)]
    # The evaluation prefix is preserved as token IDs, not retokenized across its boundary.
    if len(prompt_ids) + 96 > max_length or len(ids) > max_length:
        raise ValueError('sample exceeds the unchanged evaluation context')
    return {'item_hash': item.item_hash, 'dataset': item.dataset,
            'input_ids': ids, 'answer_start': len(prompt_ids),
            'answer_tokens': len(answer_ids), 'token_count': len(ids)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise ValueError('refusing overwrite')
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('frozen source hash mismatch')
    from transformers import AutoTokenizer
    import pandas as pd
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    items = load_items(args.source)
    if len(items) != 1672:
        raise ValueError('expected all 1672 cases')
    rows = [encode_item(item, tokenizer) for item in items]
    os.umask(0o077)
    args.output_dir.mkdir(parents=True)
    parquet = args.output_dir/'train.parquet'
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    result = {'contract': 'logistics-mcq-same-item-answer-sft-v1',
        'private_content_included': False, 'independent_test': False,
        'source_sha256': SOURCE_SHA256, 'items': len(rows),
        'by_dataset': dict(Counter(r['dataset'] for r in rows)),
        'prompt_version': PROMPT_VERSION, 'enable_thinking': False,
        'evaluation_prompt_tokens_preserved': True, 'all_options_retained': True,
        'sequence_tokens_per_epoch': sum(r['token_count'] for r in rows),
        'supervised_tokens_per_epoch': sum(r['answer_tokens']+1 for r in rows),
        'max_sequence_tokens': max(r['token_count'] for r in rows),
        'parquet_sha256': hashlib.sha256(parquet.read_bytes()).hexdigest(),
        'tokenizer_sha256': hashlib.sha256((Path(args.model)/'tokenizer.json').read_bytes()).hexdigest(),
        'epochs': 2, 'batch_size': 4, 'steps_per_epoch': 418, 'total_steps': 836,
        'checkpoint_steps': [418,836], 'optimizer_checkpoint_saved': False}
    (args.output_dir/'manifest.safe.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
