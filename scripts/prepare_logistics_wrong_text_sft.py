"""Prepare complete epochs of native/Step120 errors with full answer text."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random

from scripts.evaluate_logistics_knowledge import build_messages, parse_answers
from scripts.prepare_logistics_mcq_fit import SOURCE_SHA256
from scripts.run_vllm_logistics_mcq import load_items


def read_predictions(path, items):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {row['item_hash']: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {item.item_hash for item in items}:
        raise ValueError('prediction coverage or uniqueness mismatch')
    for item in items:
        row = indexed[item.item_hash]
        if (row['dataset'] != item.dataset or row['question'] != item.question or
                row['options'] != list(item.options) or row['expected'] != list(item.expected)):
            raise ValueError('prediction/source metadata mismatch')
        parsed, valid = parse_answers(row['prediction'], len(item.options))
        if row.get('error') or bool(row['correct']) != (valid and parsed == item.expected):
            raise ValueError('failed request or inconsistent saved correctness')
    return indexed


def select_errors(items, baseline, native):
    return [item for item in items
            if not baseline[item.item_hash]['correct'] or not native[item.item_hash]['correct']]


def training_example(item, seed=20261023):
    order = list(range(len(item.options)))
    rng = random.Random(f'{seed}:{item.item_hash}')
    rng.shuffle(order)
    if len(order) > 1 and order == list(range(len(order))):
        order = order[1:] + order[:1]
    expected = tuple(i for i, old in enumerate(order) if old in item.expected)
    shuffled = replace(item, options=tuple(item.options[i] for i in order), expected=expected)
    messages = build_messages(shuffled)
    messages[0]['content'] = (
        'Answer the closed-book logistics question. First return one JSON object '
        'in the form {"answers":[0]} using the listed zero-based option indices. '
        'For multiple choices include every correct index. Then write "Correct option text(s):" '
        'and reproduce the full text of every correct option as a JSON array of strings. '
        'Do not invent an explanation or abbreviate the option texts.'
    )
    texts = [shuffled.options[i] for i in expected]
    answer = json.dumps({'answers': list(expected)}, separators=(',', ':'))
    answer += '\nCorrect option text(s):\n' + json.dumps(texts, ensure_ascii=False)
    parsed, valid = parse_answers(answer, len(order))
    if not valid or parsed != expected or Counter(texts) != Counter(item.options[i] for i in item.expected):
        raise ValueError('answer text or shuffled answer mapping changed')
    return messages, answer, order


def encode(item, tokenizer):
    messages, answer, order = training_example(item)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    prefix = list(tokenizer.encode(prompt, add_special_tokens=False))
    suffix = list(tokenizer.encode(answer, add_special_tokens=False))
    ids = prefix + suffix + [int(tokenizer.eos_token_id)]
    if not prefix or not suffix or len(ids) > 4096:
        raise ValueError('empty/truncated training example')
    return dict(item_hash=item.item_hash, dataset=item.dataset, input_ids=ids,
        answer_start=len(prefix), answer_tokens=len(suffix), token_count=len(ids), option_order=order)


def training_schedule(item_count, epochs):
    if item_count < 1 or epochs < 1:
        raise ValueError('positive item and epoch counts required')
    batch = max(size for size in range(1, 9) if item_count % size == 0)
    steps_per_epoch = item_count // batch
    steps = steps_per_epoch * epochs
    return dict(batch_size=batch, steps_per_epoch=steps_per_epoch, total_epochs=epochs,
        total_steps=steps, evaluation_steps=sorted(set(range(50, steps + 1, 50)) | {steps}))


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'baseline', 'native', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing overwrite')
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('frozen source fingerprint mismatch')
    items = load_items(args.source)
    if Counter(item.dataset for item in items) != {'LogistikaBench': 1446, 'SC-bench-knowledge': 226}:
        raise ValueError('full benchmark coverage mismatch')
    baseline = read_predictions(args.baseline, items)
    native = read_predictions(args.native, items)
    selected = select_errors(items, baseline, native)
    if not selected:
        raise ValueError('no errors to train')
    from transformers import AutoTokenizer
    import pandas as pd
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [encode(item, tokenizer) for item in selected]
    # This trainer drops partial batches. Select an exact divisor: no omitted or repeated items.
    schedule = training_schedule(len(rows), args.epochs)
    os.umask(0o077)
    args.output.mkdir(parents=True)
    parquet = args.output / 'train.parquet'
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    manifest = dict(contract='logistics-wrong-union-full-text-sft-v1', independent_test=False,
        private_content_included=False, items=len(rows), full_evaluation_items=len(items),
        by_dataset=dict(Counter(item.dataset for item in selected)),
        baseline_errors=sum(not row['correct'] for row in baseline.values()),
        native_errors=sum(not row['correct'] for row in native.values()),
        source_sha256=SOURCE_SHA256,
        selection_sha256={key: hashlib.sha256(path.read_bytes()).hexdigest()
                          for key, path in [('baseline', args.baseline), ('native', args.native)]},
        parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
        full_answer_text_supervised=True, shuffled_options=True, all_options_retained=True,
        **schedule,
        sequence_tokens_per_epoch=sum(row['token_count'] for row in rows),
        sequence_tokens=sum(row['token_count'] for row in rows) * args.epochs,
        supervised_tokens=sum(row['answer_tokens'] + 1 for row in rows) * args.epochs,
        max_sequence_tokens=max(row['token_count'] for row in rows))
    (args.output / 'manifest.safe.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
