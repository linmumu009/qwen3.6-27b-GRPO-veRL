"""Numerical visibility test for the installed packed causal NPU attention kernel.

Pairs with the separate TP4/CP2 packing audit; this is not a full-model accuracy test.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    import torch
    import torch_npu
    torch.npu.set_device(0)
    q = torch.zeros((128, 2, 128), dtype=torch.bfloat16, device='npu')
    k = torch.zeros_like(q)
    mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device='npu'), diagonal=1)
    def attend(v):
        return torch_npu.npu_fusion_attention(q, k, v, 2, 'TND', pse=None,
            padding_mask=None, atten_mask=mask, scale=128**-.5,
            pre_tockens=2147483647, next_tockens=0, keep_prob=1.0,
            inner_precise=0, sparse_mode=3, actual_seq_qlen=(64, 128),
            actual_seq_kvlen=(64, 128))[0].float().cpu()
    baseline = attend(torch.zeros_like(q))
    prefix = torch.zeros_like(q)
    prefix[:32] = 1
    changed_prefix = attend(prefix)
    suffix = torch.zeros_like(q)
    suffix[32:64] = 1
    changed_suffix = attend(suffix)
    delta_later = (changed_prefix[32:64]-baseline[32:64]).abs()
    cross_sample = (changed_prefix[64:]-baseline[64:]).abs().max().item()
    future_leak = (changed_suffix[:32]-baseline[:32]).abs().max().item()
    expected = torch.tensor([32/(i+1) for i in range(32,64)]).reshape(-1,1,1)
    error = (changed_prefix[32:64]-expected).abs().max().item()
    assert delta_later.min().item() > .49, 'Earlier fragment is invisible to later fragment'
    assert cross_sample < 1e-6, 'Packed sample boundary leaked'
    assert future_leak < 1e-6, 'Future tokens leaked into earlier fragment'
    assert error < .005, 'Packed attention differs from expected causal mean'
    result = dict(status='npu_packed_causal_visibility_pass', kernel='torch_npu.npu_fusion_attention',
        dtype='bfloat16', layout='TND', sample_endpoints=[64,128], internal_fragment_boundary=32,
        later_fragment_min_effect=delta_later.min().item(), other_sample_max_effect=cross_sample,
        future_to_past_max_effect=future_leak, max_causal_mean_error=error,
        full_model_end_to_end_test=False, model_training=False,
        scope='Actual packed causal kernel; combine with installed trainer packing contract audit')
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
