#!/usr/bin/env python3
"""Qwen3.6 state-recovery SFT with one emphasized semantic critical token."""

from __future__ import annotations

import torch

from scripts.qwen36_teacher_forced_diagnostic_dataset import (
    Qwen36TeacherForcedDiagnosticDataset,
)
from scripts.teacher_forced_component_masks import (
    build_sql_weighted_loss_mask,
    emphasize_critical_sql_token,
)


class Qwen36CriticalTokenSFTDataset(Qwen36TeacherForcedDiagnosticDataset):
    """Keep base component weights and override one first-nongreedy SQL token."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        self.tool_structure_weight = float(config.get("tool_structure_weight", 0.25))
        self.sql_payload_weight = float(config.get("sql_payload_weight", 8.0))
        self.final_answer_weight = float(config.get("final_answer_weight", 1.0))
        self.critical_token_weight = float(config.get("critical_token_weight", 32.0))
        super().__init__(parquet_files, tokenizer, config, processor, max_samples)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        result = super().__getitem__(item)
        row = self.dataframe.iloc[item].to_dict()
        base = build_sql_weighted_loss_mask(
            tool_structure_mask=result["tool_structure_mask"].tolist(),
            sql_shell_mask=result["sql_shell_mask"].tolist(),
            final_answer_mask=result["final_answer_mask"].tolist(),
            tool_structure_weight=self.tool_structure_weight,
            sql_payload_weight=self.sql_payload_weight,
            final_answer_weight=self.final_answer_weight,
        )
        weighted, critical_mask = emphasize_critical_sql_token(
            weighted_loss_mask=base,
            sql_shell_mask=result["sql_shell_mask"].tolist(),
            critical_sql_token_offset=int(row["critical_sql_token_offset"]),
            critical_weight=self.critical_token_weight,
        )
        critical_positions = [index for index, value in enumerate(critical_mask) if value]
        if len(critical_positions) != 1:
            raise ValueError("critical token mask must contain exactly one position")
        actual_target_id = int(result["input_ids"][critical_positions[0]].item())
        expected_target_id = int(row["critical_sql_target_id"])
        if actual_target_id != expected_target_id:
            raise ValueError(
                f"critical target token mismatch: expected {expected_target_id}, got {actual_target_id}"
            )
        result["loss_mask"] = torch.tensor(weighted, dtype=torch.float32)
        result["critical_sql_token_mask"] = torch.tensor(critical_mask, dtype=torch.long)
        return result
