#!/usr/bin/env python3
"""Qwen3.6 repair SFT dataset with explicit component loss weights."""

from __future__ import annotations

import torch

from scripts.qwen36_teacher_forced_diagnostic_dataset import (
    Qwen36TeacherForcedDiagnosticDataset,
)
from scripts.teacher_forced_component_masks import build_sql_weighted_loss_mask


class Qwen36SQLWeightedSFTDataset(Qwen36TeacherForcedDiagnosticDataset):
    """Upweight SQL payload while downweighting already-solved tool boilerplate."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        self.tool_structure_weight = float(config.get("tool_structure_weight", 0.25))
        self.sql_payload_weight = float(config.get("sql_payload_weight", 8.0))
        self.final_answer_weight = float(config.get("final_answer_weight", 1.0))
        super().__init__(parquet_files, tokenizer, config, processor, max_samples)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        result = super().__getitem__(item)
        weighted = build_sql_weighted_loss_mask(
            tool_structure_mask=result["tool_structure_mask"].tolist(),
            sql_shell_mask=result["sql_shell_mask"].tolist(),
            final_answer_mask=result["final_answer_mask"].tolist(),
            tool_structure_weight=self.tool_structure_weight,
            sql_payload_weight=self.sql_payload_weight,
            final_answer_weight=self.final_answer_weight,
        )
        result["loss_mask"] = torch.tensor(weighted, dtype=torch.float32)
        return result
