"""CPU check of the actual tensor-function body with the pinned Rust tokenizer.

This deliberately does not claim a complete veRL/container integration check.
Unavailable veRL imports are avoided by extracting only the function definition.
normalize_token_ids is identity for the adapter's already flat integer list.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from build_cpt_reviewed_core import EOS, TOKENIZER_SHA, digest, write_json


def check(parquet, tokenizer_path, output):
    if digest(tokenizer_path.read_bytes()) != TOKENIZER_SHA:
        raise ValueError('Unexpected tokenizer')
    rust = Tokenizer.from_file(str(tokenizer_path))

    class Adapter:
        eos_token_id = EOS
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return {'input_ids':rust.encode(text, add_special_tokens=add_special_tokens).ids}

    source_path = Path(__file__).with_name('qwen36_causal_lm_dataset.py')
    tree = ast.parse(source_path.read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_causal_lm_tensors')
    namespace = dict(torch=torch, F=F, Any=object, normalize_token_ids=lambda ids:list(ids),
                     DatasetPadMode=SimpleNamespace(RIGHT='right'))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), 'exec'), namespace)
    build = namespace['build_causal_lm_tensors']
    rows = pq.read_table(parquet).to_pylist()
    tokens = 0
    for row in rows:
        item = build(row['text'], Adapter(), max_length=4096, truncation='error', pad_mode='no_padding')
        ids, mask = item['input_ids'], item['loss_mask']
        if len(ids) != row['token_count'] or int(ids[-1]) != EOS or int((ids == EOS).sum()) != 1:
            raise ValueError('Length/EOS mismatch')
        if int(mask[0]) != 0 or not bool((mask[1:] == 1).all()):
            raise ValueError('Loss mask mismatch')
        shifted = torch.roll(mask, shifts=-1, dims=0)
        if int(shifted[-1]) != 0 or int(shifted[-2]) != 1:
            raise ValueError('Wrapped target unmasked or EOS target masked')
        tokens += int(mask.sum())
    # Also exercise the real padding branch, which is not used in this release.
    padded = build('Transport', Adapter(), max_length=64, truncation='error', pad_mode='right')
    count = int(padded['attention_mask'].sum())
    if not bool((padded['loss_mask'][count:] == 0).all()):
        raise ValueError('Padding supervised')
    summary = dict(records=len(rows), loss_tokens=tokens, parquet_sha256=digest(parquet.read_bytes()),
                   tensor_function_file_sha256=digest(source_path.read_bytes()),
                   tokenizer_sha256=TOKENIZER_SHA, single_eos=True, eos_target_supervised=True,
                   wraparound_masked=True, full_verl_container_integration_checked=False,
                   scope='Actual tensor function body, real CPU torch, flat-list Rust-tokenizer adapter; no model or training started')
    write_json(output, summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('parquet', 'tokenizer', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(check(args.parquet, args.tokenizer, args.output))
