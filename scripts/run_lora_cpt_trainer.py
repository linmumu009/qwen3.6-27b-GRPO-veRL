"""Process-scoped compatibility and frozen-weight audit for pinned Bridge LoRA."""
import hashlib
import json
import os
from pathlib import Path


def main():
    import torch
    import mindspeed.megatron_adaptor  # Install the existing Ascend compatibility layer first.
    from megatron.bridge.peft import utils
    from megatron.bridge.peft.lora import LoRA
    from lora_cpt_ascend_compat import install
    install()
    from verl.workers.engine.megatron import transformer_impl as engine

    source = os.environ['LLIN_LORA_SOURCE']
    audit = Path(os.environ['LLIN_LORA_AUDIT'])
    original_load = engine.load_mcore_dist_weights
    loaded = []

    def create_peft(config, dtype=None):
        assert config['type'] == 'lora' and config['rank'] == 64
        return LoRA(dim=config['rank'], alpha=config['alpha'], dropout=config['dropout'],
                    target_modules=list(config['target_modules']), lora_dtype=dtype)

    def create_hook(peft, training=True):
        def hook(model):
            chunks = model if isinstance(model, list) else [model]
            original_load(chunks, source)
            loaded.extend(chunks)
            result = peft(model, training=training)
            peft.set_params_to_save(result)
            return result
        return hook

    def reject_adapter_resume(*args, **kwargs):
        raise RuntimeError('Adapter resume is not registered for this fresh LoRA experiment')

    def already_loaded(model, path, **kwargs):
        assert path == source and loaded, 'Base weights must load before adapter injection'

    # The installed engine targets newer helper names; use the pinned native LoRA.
    utils.create_peft = create_peft
    utils.create_peft_hook = create_hook
    utils.load_peft_adapter_checkpoint = reject_adapter_resume
    engine.load_mcore_dist_weights = already_loaded
    original_optimizer = engine.MegatronEngine._build_optimizer
    tracked = []

    def fingerprints(modules):
        base = hashlib.sha256(); adapters = hashlib.sha256()
        counts = {'frozen': 0, 'trainable': 0}
        for i, module in enumerate(modules):
            for name, param in module.named_parameters():
                is_adapter = '.adapter.' in name
                assert not is_adapter or 'language_model.' in name, 'Visual adapter is not registered'
                assert param.requires_grad == is_adapter, f'Unexpected trainability: {name}'
                raw = param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                target = adapters if is_adapter else base
                target.update(f'{i}:{name}'.encode()); target.update(raw)
                counts['trainable' if is_adapter else 'frozen'] += param.numel()
        assert counts['trainable'] > 0 and counts['frozen'] > counts['trainable']
        return dict(base_sha256=base.hexdigest(), adapter_sha256=adapters.hexdigest(), **counts)

    def optimizer(self):
        before = fingerprints(self.module)
        audit.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank()
        (audit/f'rank{rank}.before.safe.json').write_text(json.dumps(before))
        tracked.append((self, before, rank))
        return original_optimizer(self)

    engine.MegatronEngine._build_optimizer = optimizer
    from verl.trainer import sft_trainer
    sft_trainer.main()
    for instance, before, rank in tracked:
        after = fingerprints(instance.module)
        assert before['base_sha256'] == after['base_sha256'], 'Frozen weights changed'
        assert before['adapter_sha256'] != after['adapter_sha256'], 'Adapter did not learn'
        (audit/f'rank{rank}.after.safe.json').write_text(json.dumps(dict(
            **after, base_unchanged=True, adapter_changed=True)))


if __name__ == '__main__':
    main()
