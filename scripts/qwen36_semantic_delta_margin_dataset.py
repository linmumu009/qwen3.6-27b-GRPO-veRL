#!/usr/bin/env python3
"""Paired chosen/rejected SQL dataset with semantic-delta token masks."""

from __future__ import annotations

import torch

from scripts.qwen36_teacher_forced_diagnostic_dataset import (
    Qwen36TeacherForcedDiagnosticDataset,
)
from scripts.teacher_forced_component_masks import semantic_delta_token_masks


class Qwen36SemanticDeltaMarginDataset(Qwen36TeacherForcedDiagnosticDataset):
    """Add a paired edit mask while preserving the exact state-conditioned template."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        super().__init__(parquet_files, tokenizer, config, processor, max_samples)
        self._pair_index: dict[tuple[str, str], int] = {}
        for position, (_, row) in enumerate(self.dataframe.iterrows()):
            key = (str(row["source_task_id"]), str(row["candidate_label"]))
            if key in self._pair_index:
                raise ValueError(f"duplicate semantic-delta candidate: {key!r}")
            self._pair_index[key] = position
        task_ids = {task_id for task_id, _ in self._pair_index}
        expected = {(task_id, label) for task_id in task_ids for label in ("chosen", "rejected")}
        if set(self._pair_index) != expected:
            raise ValueError("semantic-delta dataset does not contain complete chosen/rejected pairs")

    @staticmethod
    def _sql_ids(item: dict[str, torch.Tensor]) -> tuple[list[int], list[int]]:
        positions = [
            index for index, value in enumerate(item["sql_shell_mask"].tolist()) if value
        ]
        return positions, [int(item["input_ids"][index].item()) for index in positions]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        result = super().__getitem__(item)
        row = self.dataframe.iloc[item]
        task_id = str(row["source_task_id"])
        label = str(row["candidate_label"])
        partner_label = "rejected" if label == "chosen" else "chosen"
        partner = super().__getitem__(self._pair_index[(task_id, partner_label)])

        current_positions, current_ids = self._sql_ids(result)
        _, partner_ids = self._sql_ids(partner)
        if label == "chosen":
            current_delta, _ = semantic_delta_token_masks(current_ids, partner_ids)
        else:
            _, current_delta = semantic_delta_token_masks(partner_ids, current_ids)
        mask = [0] * int(result["input_ids"].numel())
        for position, selected in zip(current_positions, current_delta, strict=True):
            mask[position] = int(selected)
        if not any(mask):
            raise ValueError(f"{task_id}::{label}: empty semantic-delta mask")
        result["semantic_delta_mask"] = torch.tensor(mask, dtype=torch.long)
        sign = 1 if label == "chosen" else -1
        pair_index = int(row["pair_index"])
        result["candidate_sign"] = torch.full_like(result["input_ids"], sign)
        result["pair_index"] = torch.full_like(result["input_ids"], pair_index)
        return result
