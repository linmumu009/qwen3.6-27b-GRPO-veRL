#!/usr/bin/env python3
"""Qwen3.6 repair dataset with mutually exclusive diagnostic target masks."""

from __future__ import annotations

from typing import Any

import torch

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.tokenizer.tokenizer import normalize_token_ids

from scripts.qwen36_assistant_mask_sft_dataset import (
    ASSISTANT_PREFIX,
    TURN_SUFFIX,
    Qwen36AssistantMaskSFTDataset,
)
from scripts.teacher_forced_component_masks import (
    assistant_mask_from_ranges,
    assistant_turn_ranges,
    build_repair_component_masks,
    normalize_assistant_turn_indices,
)


class Qwen36TeacherForcedDiagnosticDataset(Qwen36AssistantMaskSFTDataset):
    """Add tool-structure, shell-SQL and final-answer masks to the exact SFT rows."""

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        if self.pad_mode != DatasetPadMode.NO_PADDING:
            raise ValueError("teacher-forced diagnostic only supports no_padding")

        result = super().__getitem__(item)
        row_dict = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row_dict)
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = (
            self.enable_thinking[item] if self.enable_thinking is not None else self.enable_thinking_default
        )
        apply_kwargs: dict[str, Any] = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_kwargs["enable_thinking"] = bool(enable_thinking)

        rendered_text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=False,
            **apply_kwargs,
        )
        encoded = self.tokenizer(
            rendered_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = normalize_token_ids(encoded["input_ids"])
        input_ids = result["input_ids"].tolist()
        if encoded_ids != input_ids:
            raise ValueError("rendered text token IDs differ from the training chat-template IDs")

        prefix_ids = normalize_token_ids(
            self.tokenizer(ASSISTANT_PREFIX, add_special_tokens=False)["input_ids"]
        )
        suffix_ids = normalize_token_ids(
            self.tokenizer(TURN_SUFFIX, add_special_tokens=False)["input_ids"]
        )
        assistant_messages = [message for message in messages if message.get("role") == "assistant"]
        ranges = assistant_turn_ranges(
            input_ids,
            prefix_ids,
            suffix_ids,
            expected_turns=len(assistant_messages),
        )
        supervised_indices = normalize_assistant_turn_indices(
            row_dict.get("supervised_assistant_turn_indices"),
            len(assistant_messages),
        )
        if len(supervised_indices) != 2:
            raise ValueError("repair diagnostic requires exactly two supervised assistant turns")
        tool_turn_index, final_turn_index = supervised_indices
        tool_calls = assistant_messages[tool_turn_index].get("tool_calls") or []
        if len(tool_calls) != 1:
            raise ValueError("repair diagnostic requires exactly one teacher tool call")
        if assistant_messages[final_turn_index].get("tool_calls"):
            raise ValueError("repair diagnostic final supervised turn cannot call a tool")
        command = tool_calls[0]["function"]["arguments"]["command"]

        masks = build_repair_component_masks(
            input_ids=input_ids,
            offsets=encoded["offset_mapping"],
            rendered_text=rendered_text,
            command=command,
            turn_ranges=ranges,
            tool_turn_index=tool_turn_index,
            final_answer_turn_index=final_turn_index,
        )
        assistant_union = [
            int(tool or final)
            for tool, final in zip(
                masks["tool_turn_mask"], masks["final_answer_mask"], strict=True
            )
        ]
        if assistant_union != result["loss_mask"].tolist():
            raise ValueError("diagnostic component masks do not reconstruct the SFT loss mask")

        for key, mask in masks.items():
            if sum(mask) <= 0:
                raise ValueError(f"sample {item} has an empty {key}")
            result[key] = torch.tensor(mask, dtype=torch.long)
        all_assistant = assistant_mask_from_ranges(
            len(input_ids), ranges, list(range(len(ranges)))
        )
        context_assistant = [
            int(all_token and not supervised_token)
            for all_token, supervised_token in zip(
                all_assistant, result["loss_mask"].tolist(), strict=True
            )
        ]
        result["context_assistant_mask"] = torch.tensor(context_assistant, dtype=torch.long)
        return result
