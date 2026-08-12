#!/usr/bin/env python3
"""Qwen3.6 chosen-only first-action dataset with component masks."""

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
    assistant_turn_ranges,
    build_first_action_component_masks,
    normalize_assistant_turn_indices,
)


class Qwen36FirstActionDiagnosticDataset(Qwen36AssistantMaskSFTDataset):
    """Require exactly one supervised assistant bash/SQLite tool action."""

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        if self.pad_mode != DatasetPadMode.NO_PADDING:
            raise ValueError("first-action diagnostic only supports no_padding")
        result = super().__getitem__(item)
        row_dict = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row_dict)
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = (
            self.enable_thinking[item]
            if self.enable_thinking is not None
            else self.enable_thinking_default
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
            raise ValueError(
                "first-action rendered token IDs differ from training chat template"
            )

        assistant_messages = [
            message for message in messages if message.get("role") == "assistant"
        ]
        if len(assistant_messages) != 1:
            raise ValueError("first-action diagnostic requires one assistant turn")
        supervised = normalize_assistant_turn_indices(
            row_dict.get("supervised_assistant_turn_indices"), 1
        )
        if supervised != [0]:
            raise ValueError("first-action diagnostic must supervise assistant turn 0")
        calls = assistant_messages[0].get("tool_calls") or []
        if len(calls) != 1 or (calls[0].get("function") or {}).get("name") != "bash":
            raise ValueError("first-action target requires exactly one bash call")
        arguments = (calls[0].get("function") or {}).get("arguments") or {}
        if not isinstance(arguments, dict) or not isinstance(
            arguments.get("command"), str
        ):
            raise ValueError("first-action bash arguments must contain a command mapping")

        prefix_ids = normalize_token_ids(
            self.tokenizer(ASSISTANT_PREFIX, add_special_tokens=False)["input_ids"]
        )
        suffix_ids = normalize_token_ids(
            self.tokenizer(TURN_SUFFIX, add_special_tokens=False)["input_ids"]
        )
        ranges = assistant_turn_ranges(
            input_ids, prefix_ids, suffix_ids, expected_turns=1
        )
        masks = build_first_action_component_masks(
            input_ids=input_ids,
            offsets=encoded["offset_mapping"],
            rendered_text=rendered_text,
            command=arguments["command"],
            turn_ranges=ranges,
        )
        if masks["tool_turn_mask"] != result["loss_mask"].tolist():
            raise ValueError("first-action components do not reconstruct loss mask")
        for key, mask in masks.items():
            result[key] = torch.tensor(mask, dtype=torch.long)
        return result
