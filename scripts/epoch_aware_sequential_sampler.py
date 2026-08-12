#!/usr/bin/env python3
"""Deterministic sampler compatible with trainers that call ``set_epoch``."""

from __future__ import annotations

from torch.utils.data import SequentialSampler


class EpochAwareSequentialSampler(SequentialSampler):
    """Preserve fixed row order while satisfying the distributed sampler API."""

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
