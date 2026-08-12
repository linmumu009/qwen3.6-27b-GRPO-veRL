#!/usr/bin/env python3
"""Chosen-only first-action SFT with frozen structure/SQL weights."""

from __future__ import annotations

import torch

from scripts.qwen36_first_action_diagnostic_dataset import (
    Qwen36FirstActionDiagnosticDataset,
)


class Qwen36FirstActionWeightedSFTDataset(Qwen36FirstActionDiagnosticDataset):
    """Weight tool boilerplate at 0.25 and decoded SQL tokens at 8 by default."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        self.tool_structure_weight = float(config.get("tool_structure_weight", 0.25))
        self.sql_payload_weight = float(config.get("sql_payload_weight", 8.0))
        if not 0 < self.tool_structure_weight <= 32:
            raise ValueError("tool structure weight must be in (0, 32]")
        if not 0 < self.sql_payload_weight <= 32:
            raise ValueError("SQL payload weight must be in (0, 32]")
        super().__init__(parquet_files, tokenizer, config, processor, max_samples)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        result = super().__getitem__(item)
        structure = result["tool_structure_mask"].tolist()
        sql = result["sql_shell_mask"].tolist()
        weighted: list[float] = []
        for structure_token, sql_token in zip(structure, sql, strict=True):
            if structure_token and sql_token:
                raise ValueError("first-action weighted component masks overlap")
            weighted.append(
                float(structure_token) * self.tool_structure_weight
                + float(sql_token) * self.sql_payload_weight
            )
        if not any(weighted):
            raise ValueError("first-action weighted loss mask is empty")
        result["loss_mask"] = torch.tensor(weighted, dtype=torch.float32)
        return result
