"""CPU audit of real CPT tensors and the installed TP4/CP2 sequence packer.

This checks boundaries/positions; it is not a numerical NPU attention-kernel test.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    import torch
    import pandas as pd
    from transformers import AutoTokenizer
    from scripts.qwen36_causal_lm_dataset import build_causal_lm_tensors
    from verl.models.mcore import util
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    rows = pd.read_parquet(a.data)
    selected = [text for text in rows['text'] if text.count('[Chapter ') >= 2][:2]
    assert len(selected) == 2
    lengths = []
    for text in selected:
        tensor = build_causal_lm_tensors(text, tokenizer, max_length=4096, truncation='error', pad_mode='no_padding')
        length = len(tensor['input_ids'])
        assert tensor['position_ids'].tolist() == list(range(length))
        assert tensor['input_ids'][-1].item() == tokenizer.eos_token_id
        assert tensor['loss_mask'].sum().item() == length - 1
        assert tensor['loss_mask'][1:].all().item()
        lengths.append(length)
    # Unique positive IDs allow every real token to be traced through both CP shards.
    pieces = [torch.arange(1, lengths[0]+1), torch.arange(10001, 10001+lengths[1])]
    nested = torch.nested.nested_tensor(pieces, layout=torch.jagged)
    expected = Counter(torch.cat(pieces).tolist())
    results = []
    has_layout = 'cp_layout' in inspect.signature(util.preprocess_thd_engine).parameters
    for layout in (('zigzag', 'contiguous') if has_layout else ('zigzag',)):
        actual = Counter()
        boundaries = None
        for rank in (0, 1):
            with patch.object(util.mpu, 'get_tensor_model_parallel_world_size', return_value=4), \
                 patch.object(util.mpu, 'get_context_parallel_world_size', return_value=2), \
                 patch.object(util.mpu, 'get_context_parallel_rank', return_value=rank):
                packed, params, _ = util.preprocess_thd_engine(nested, **({'cp_layout': layout} if has_layout else {}))
            boundary = params.cu_seqlens_q.tolist()
            assert len(boundary) == 3 and boundary[0] == 0
            assert all(b-a >= length for a, b, length in zip(boundary, boundary[1:], lengths))
            assert boundaries is None or boundaries == boundary
            boundaries = boundary
            actual.update(v for v in packed.flatten().tolist() if v)
        assert actual == expected, 'Lost, duplicated, or mixed real tokens across CP ranks'
        results.append(dict(layout=layout, sample_boundaries=boundaries, unique_token_coverage=True))
    source = Path(inspect.getfile(util))
    result = dict(status='packing_contract_pass', tp=4, cp=2, samples=2, lengths=lengths,
        layouts=results, chapter_marker_creates_new_sequence=False, positions_continuous_within_sample=True,
        cross_sample_boundaries_preserved=True, numerical_attention_kernel_test=False,
        note='Boundary contract supports causal within-sample visibility; full NPU forward check remains separate',
        installed_packer_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest())
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
