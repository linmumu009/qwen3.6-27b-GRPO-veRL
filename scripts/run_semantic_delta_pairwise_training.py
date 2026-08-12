#!/usr/bin/env python3
"""Run one reference-free pairwise ranking step on semantic SQL edit tokens."""

from __future__ import annotations

from functools import partial
import os
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.sft_trainer import SFTTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from scripts.epoch_aware_sequential_sampler import EpochAwareSequentialSampler
from scripts.semantic_delta_pairwise_loss import pairwise_loss_from_flat_sequences


def semantic_delta_pairwise_loss(
    model_output=None,
    data=None,
    dp_group=None,
    config=None,
    *,
    pairwise_beta: float = 1.0,
):
    if model_output is None:
        raise ValueError("pairwise training requires model output")
    log_prob = model_output["log_probs"]
    delta_mask = data["semantic_delta_mask"]
    candidate_sign = data["candidate_sign"]
    pair_index = data["pair_index"]
    global_batch_size = int(
        tu.get_non_tensor_data(data=data, key="global_batch_size", default=0)
    )
    if global_batch_size % 2:
        raise ValueError("pairwise global batch size must be even")
    return pairwise_loss_from_flat_sequences(
        log_prob_values=log_prob.values(),
        delta_mask_values=delta_mask.values(),
        candidate_sign_values=candidate_sign.values(),
        pair_index_values=pair_index.values(),
        offsets=delta_mask.offsets(),
        beta=pairwise_beta,
        global_pair_count=global_batch_size // 2,
        dp_size=int(data["dp_size"]),
    )


class PairwiseSemanticDeltaTrainer(SFTTrainer):
    """Keep adjacent pairs deterministic and fail closed outside the frozen DP=1 topology."""

    def _build_dataloader(self):
        if self.engine.get_data_parallel_size() != 1:
            raise ValueError("semantic-delta pairwise canary requires data parallel size 1")
        self.global_batch_size = int(self.config.data.train_batch_size)
        if self.global_batch_size != 32:
            raise ValueError("semantic-delta pairwise canary requires one 32-row global batch")
        self.train_batch_size_per_dp = self.global_batch_size
        self.train_sampler = EpochAwareSequentialSampler(self.train_dataset)
        self.collate_fn = SFTTensorCollator(self.config.data.pad_mode)
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=get_device_name(),
        )
        self.val_sampler = None
        self.val_dataloader = None


def run(config: DictConfig) -> None:
    initialize_global_process_group()
    try:
        trainer = PairwiseSemanticDeltaTrainer(config=config)
        if trainer.engine_config.forward_only:
            raise ValueError("pairwise canary must enable one optimizer step")
        trainer.training_client.set_loss_fn(
            loss_fn=partial(
                semantic_delta_pairwise_loss, pairwise_beta=float(config.pairwise.beta)
            )
        )
        trainer.fit()
    finally:
        destroy_global_process_group()


@hydra.main(config_path="pkg://verl.trainer.config", config_name="sft_trainer_engine", version_base=None)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    OmegaConf.resolve(config)
    run(config)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    main()
