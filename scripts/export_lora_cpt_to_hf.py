"""Reconstruct frozen Step120 plus a pinned-Bridge adapter on CPU and export HF."""
import argparse
import json
import os
from pathlib import Path
import uuid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--actor-checkpoint', type=Path, required=True)
    p.add_argument('--base-model', type=Path, required=True)
    p.add_argument('--base-dist', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    from export_megatron_dist_to_hf import (
        validate_export_paths, verify_exact_hf_export, _save_hf_with_frozen_base_fallback)
    adapter_path = validate_export_paths(a.actor_checkpoint, a.base_model, a.output_dir)
    staging = a.output_dir.with_name('.'+a.output_dir.name+'.incomplete-'+uuid.uuid4().hex[:8])
    staging.mkdir(parents=True, exist_ok=False)
    import torch
    import mindspeed.megatron_adaptor
    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.lora import LoRA, LoRAMerge
    from megatron.bridge.peft.lora_layers import LoRALinear
    from lora_cpt_ascend_compat import install
    install()
    from megatron.bridge.training.model_load_save import temporary_distributed_context
    from megatron.bridge.training.checkpointing import apply_peft_adapter_filter_to_state_dict
    from verl.utils.model import load_mcore_dist_weights
    from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing

    bridge = AutoBridge.from_hf_pretrained(str(a.base_model), trust_remote_code=True)
    with temporary_distributed_context(backend='gloo'):
        provider = bridge.to_megatron_provider(load_weights=False)
        provider.perform_initialization = False
        provider.bf16 = True; provider.fp16 = False; provider.params_dtype = torch.bfloat16
        provider.tensor_model_parallel_size = 1; provider.pipeline_model_parallel_size = 1
        provider.context_parallel_size = 1; provider.sequence_parallel = False
        provider.virtual_pipeline_model_parallel_size = None
        provider.finalize()
        models = provider.provide_distributed_model(wrap_with_ddp=False,
            use_cpu_initialization=True, mixed_precision_wrapper=None)
        assert len(models) == 1
        load_mcore_dist_weights(models, str(a.base_dist))
        peft = LoRA(dim=64, alpha=128, dropout=0.0, lora_dtype=torch.bfloat16,
                    target_modules=['language_model.*.'+n for n in ('linear_qkv','linear_proj','linear_fc1','linear_fc2')])
        peft(models, training=True); peft.set_params_to_save(models)
        skeleton = apply_peft_adapter_filter_to_state_dict(
            {'model': models[0].sharded_state_dict()}, peft)
        loaded = load_dist_checkpointing(skeleton, str(adapter_path))
        state = loaded['model']
        assert state and all('.adapter.' in k for k in state), 'Non-adapter state in checkpoint'
        expected = {n for n, _ in models[0].named_parameters() if '.adapter.' in n}
        assert set(state) == expected, 'Adapter key set incomplete'
        result = models[0].load_state_dict(state, strict=False)
        assert not result.unexpected_keys and not (set(result.missing_keys) & expected)
        assert all(torch.isfinite(t).all() for t in state.values())
        assert any(torch.count_nonzero(t) for k,t in state.items() if 'linear_out' in k)
        merger = LoRAMerge(); merged = []

        def merge_children(module, prefix=''):
            for name, child in list(module.named_children()):
                full = prefix+'.'+name
                if isinstance(child, LoRALinear):
                    assert hasattr(child.to_wrap, 'weight'), 'Unregistered grouped adapter'
                    w = child.to_wrap.weight
                    expected_weight = (w.float() +
                        child.adapter.linear_out.weight.float() @ child.adapter.linear_in.weight.float() * 2).to(w.dtype)
                    merger.transform(child)
                    assert torch.equal(child.to_wrap.weight, expected_weight), 'Merge math mismatch'
                    setattr(module, name, child.to_wrap)
                    merged.append(full)
                else:
                    merge_children(child, full)
        merge_children(models[0])
        assert merged and not any('.adapter.' in n for n,_ in models[0].named_parameters())
        fallback = _save_hf_with_frozen_base_fallback(
            bridge, models, a.base_model, a.base_dist, staging)
    verification = verify_exact_hf_export(a.base_model, staging)
    assert verification['valid'], verification
    (staging/'llin_export_manifest.json').write_text(json.dumps(dict(
        conversion='Pinned Bridge LoRA rank64 alpha128 merged on CPU',
        adapter_checkpoint=str(adapter_path), base_model=str(a.base_model),
        base_dist=str(a.base_dist), merged_modules=merged, merge_math_verified=True,
        frozen_base_fallback_keys=fallback, verification=verification), indent=2))
    os.replace(staging, a.output_dir)


if __name__ == '__main__':
    main()
