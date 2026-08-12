#!/usr/bin/env python3
"""Pure tensor helper for reference-free semantic-delta pairwise ranking."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


def pairwise_loss_from_flat_sequences(
    *,
    log_prob_values: torch.Tensor,
    delta_mask_values: torch.Tensor,
    candidate_sign_values: torch.Tensor,
    pair_index_values: torch.Tensor,
    offsets: torch.Tensor,
    beta: float,
    global_pair_count: int,
    dp_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if beta <= 0 or global_pair_count <= 0 or dp_size <= 0:
        raise ValueError("pairwise beta, global pair count and dp size must be positive")
    boundaries = [int(value) for value in offsets.detach().cpu().tolist()]
    groups: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
    for start, end in zip(boundaries, boundaries[1:]):
        if not start < end:
            raise ValueError("pairwise sequence offsets are not strictly increasing")
        mask = torch.roll(delta_mask_values[start:end], shifts=-1, dims=0).to(
            log_prob_values.dtype
        )
        count = mask.sum()
        if count.item() <= 0:
            raise ValueError("pairwise semantic-delta mask is empty after shift")
        signs = torch.unique(candidate_sign_values[start:end]).detach().cpu().tolist()
        pair_ids = torch.unique(pair_index_values[start:end]).detach().cpu().tolist()
        if len(signs) != 1 or int(signs[0]) not in (-1, 1):
            raise ValueError("pairwise candidate sign is not constant or valid")
        if len(pair_ids) != 1:
            raise ValueError("pairwise sequence contains multiple pair indices")
        sign = int(signs[0])
        pair_id = int(pair_ids[0])
        if sign in groups[pair_id]:
            raise ValueError("pairwise microbatch contains a duplicate candidate sign")
        groups[pair_id][sign] = (log_prob_values[start:end] * mask).sum() / count

    if not groups or any(set(candidates) != {-1, 1} for candidates in groups.values()):
        raise ValueError("pairwise microbatch must contain complete chosen/rejected pairs")
    margins = torch.stack(
        [candidates[1] - candidates[-1] for _, candidates in sorted(groups.items())]
    )
    raw_loss = -F.logsigmoid(beta * margins).sum()
    loss = raw_loss * dp_size / global_pair_count
    return loss, {
        "pairwise/margin_sum": margins.sum().detach(),
        "pairwise/chosen_preferred_count": (margins > 0).sum().detach(),
        "pairwise/pair_count": torch.tensor(
            len(groups), device=margins.device, dtype=margins.dtype
        ),
        "pairwise/raw_logistic_loss_sum": raw_loss.detach(),
    }
