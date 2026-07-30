"""Compatibility helpers for the Ascend-validated Megatron-Bridge snapshot.

Megatron-Bridge commit de93536e contains the Qwen3.5 conversion code validated
by Ascend, but it predates three training helpers imported unconditionally by
newer veRL revisions. Keep the compatibility surface here so the vendor
snapshot itself remains untouched and auditable.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from megatron.core import tensor_parallel


class LinearForLastLayer(nn.Linear):
    """Replicated final projection compatible with Megatron output calls."""

    def __init__(self, input_size: int, output_size: int, sequence_parallel: bool) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=False)
        self.sequence_parallel = sequence_parallel
        if sequence_parallel:
            setattr(self.weight, "sequence_parallel", True)

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        del weight, runtime_gather_output
        logits = super().forward(input_).float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits,
                tensor_parallel_output_grad=False,
            )
        return logits, None


def _as_model_list(model: Any) -> list[Any]:
    return model if isinstance(model, list) else [model]


def make_value_model(hidden_size: int, sequence_parallel: bool):
    """Return a hook that replaces the final PP-stage head with a value head."""

    def hook(model: Any) -> list[Any]:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping
        from megatron.core import parallel_state

        AutoMapping.register_module_type("LinearForLastLayer", "replicated")
        model_chunks = _as_model_list(model)
        for model_chunk in model_chunks:
            if parallel_state.is_pipeline_last_stage():
                model_chunk.output_layer = LinearForLastLayer(
                    input_size=hidden_size,
                    output_size=1,
                    sequence_parallel=sequence_parallel,
                )
        return model_chunks

    return hook


def freeze_moe_router(model: Any) -> list[Any]:
    """Freeze router and shared-expert gate parameters when present."""

    model_chunks = _as_model_list(model)
    for model_chunk in model_chunks:
        layers = getattr(getattr(model_chunk, "decoder", None), "layers", None)
        if layers is None:
            continue
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            router = getattr(mlp, "router", None)
            if router is not None:
                _freeze_if_present(router, "weight")
                _freeze_if_present(router, "bias")
            shared_experts = getattr(mlp, "shared_experts", None)
            if shared_experts is not None:
                _freeze_if_present(shared_experts, "gate_weight")
                _freeze_if_present(shared_experts, "gate_bias")
    return model_chunks


def _freeze_if_present(module: Any, name: str) -> None:
    parameter = getattr(module, name, None)
    if parameter is not None:
        parameter.requires_grad = False


def create_ddp_config(
    wrap_with_ddp: bool = True,
    use_distributed_optimizer: bool = True,
    use_megatron_fsdp: bool = False,
    overrides: dict[str, object] | None = None,
    finalize: bool = True,
) -> object | None:
    """Backport Megatron-Bridge's external DDP config factory."""

    if not wrap_with_ddp:
        return None
    from megatron.bridge.training.config import DistributedDataParallelConfig

    values: dict[str, object] = {
        "use_distributed_optimizer": use_distributed_optimizer,
    }
    if use_megatron_fsdp:
        values.update(
            {
                "use_distributed_optimizer": True,
                "check_for_nan_in_grad": True,
                "use_megatron_fsdp": True,
                "data_parallel_sharding_strategy": "optim_grads_params",
                "overlap_grad_reduce": True,
            }
        )
    values.update(overrides or {})
    config = DistributedDataParallelConfig(**values)
    if finalize:
        config.finalize()
    return config


def _install_model_provider_compatibility() -> None:
    """Backport the provider finalization helper expected by newer veRL."""

    from megatron.bridge.models.model_provider import ModelProviderMixin

    if hasattr(ModelProviderMixin, "apply_overrides_and_finalize"):
        return

    def apply_overrides_and_finalize(
        self: Any,
        dtype: torch.dtype | None = None,
        overrides: dict[str, object] | None = None,
    ) -> Any:
        if dtype is not None:
            self.params_dtype = dtype
            self.fp16 = dtype == torch.float16
            self.bf16 = dtype == torch.bfloat16
        for name, value in (overrides or {}).items():
            # MindSpeed adds NPU-only runtime fields (for example
            # ``use_flash_attn``) that are intentionally absent from the
            # upstream provider dataclass at de93536e.
            setattr(self, name, value)
        self.finalize()
        return self

    ModelProviderMixin.apply_overrides_and_finalize = apply_overrides_and_finalize


_install_model_provider_compatibility()
