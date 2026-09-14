"""Recognize MindSpeed parallel linear layers in the pinned Bridge PEFT adapter."""


def install():
    from megatron.bridge.peft import utils, lora
    from mindspeed.te.pytorch.module.linear import TEColumnParallelLinear, TERowParallelLinear
    from mindspeed.te.pytorch.module.layernorm_column_parallel_linear import TELayerNormColumnParallelLinear
    from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
    original = utils.get_adapter_attributes_from_linear

    def attributes(module, is_expert=False):
        if type(module) in (TEColumnParallelLinear, TERowParallelLinear, TELayerNormColumnParallelLinear):
            assert not is_expert and not module.is_expert, 'Only dense model registered'
            row = type(module) is TERowParallelLinear
            return utils.AdapterAttributes(
                input_is_parallel=row, in_features=module.input_size,
                out_features=module.output_size, disable_tensor_parallel_comm=False,
                disable_sequence_parallel_comm=not module.config.sequence_parallel,
                base_linear_is_parallel=True)
        return original(module, is_expert=is_expert)

    utils.get_adapter_attributes_from_linear = attributes
    lora.get_adapter_attributes_from_linear = attributes
    original_forward = AdapterWrapper.base_linear_forward

    def base_forward(self, x, *args, **kwargs):
        if type(self.to_wrap) is TELayerNormColumnParallelLinear:
            # MindSpeed does not expose TE's return_layernorm_output option.
            # Recompute the same deterministic normalization for the adapter branch.
            output, bias = self.to_wrap(x, *args, **kwargs)
            norm = self.to_wrap._layernorm if self.to_wrap.config.normalization == 'LayerNorm' else self.to_wrap._rmsnorm
            return output, bias, norm(x)
        return original_forward(self, x, *args, **kwargs)

    AdapterWrapper.base_linear_forward = base_forward
