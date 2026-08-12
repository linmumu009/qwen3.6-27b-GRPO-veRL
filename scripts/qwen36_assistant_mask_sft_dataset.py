#!/usr/bin/env python3
"""Exact whole-conversation Qwen3.6 SFT dataset for veRL's custom_cls hook."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.tokenizer.tokenizer import normalize_token_ids

from scripts.teacher_forced_component_masks import (
    assistant_mask_from_ranges,
    assistant_turn_ranges,
    normalize_assistant_turn_indices,
)


ASSISTANT_PREFIX = "<|im_start|>assistant\n"
TURN_SUFFIX = "<|im_end|>\n"


def _find_all(sequence: list[int], needle: list[int]) -> list[int]:
    if not needle:
        raise ValueError("token marker must not be empty")
    return [
        index
        for index in range(len(sequence) - len(needle) + 1)
        if sequence[index : index + len(needle)] == needle
    ]


def build_assistant_loss_mask(
    input_ids: list[int],
    tokenizer: Any,
    expected_assistant_turns: int,
    supervised_assistant_turn_indices: Any = None,
) -> list[int]:
    """Mask assistant bodies, tool calls and closing tokens in a rendered Qwen chat."""

    prefix_ids = normalize_token_ids(tokenizer(ASSISTANT_PREFIX, add_special_tokens=False)["input_ids"])
    suffix_ids = normalize_token_ids(tokenizer(TURN_SUFFIX, add_special_tokens=False)["input_ids"])
    ranges = assistant_turn_ranges(
        input_ids,
        prefix_ids,
        suffix_ids,
        expected_turns=expected_assistant_turns,
    )
    selected = normalize_assistant_turn_indices(
        supervised_assistant_turn_indices,
        expected_assistant_turns,
    )
    return assistant_mask_from_ranges(len(input_ids), ranges, selected)


class Qwen36AssistantMaskSFTDataset(MultiTurnSFTDataset):
    """Use Qwen3.6's exact full chat template and train only assistant tokens.

    veRL's stock ``MultiTurnSFTDataset`` renders every message in isolation. The
    Qwen3.6 tool-use template requires the system message, tool schemas and user
    query to be rendered together. This class uses veRL's documented custom
    dataset hook while keeping the official trainer, collator, loss and engine.
    """

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row_dict = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row_dict)
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = (
            self.enable_thinking[item] if self.enable_thinking is not None else self.enable_thinking_default
        )
        apply_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_kwargs["enable_thinking"] = bool(enable_thinking)

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            **apply_kwargs,
        )
        input_id_list = normalize_token_ids(tokenized)
        assistant_turns = sum(message.get("role") == "assistant" for message in messages)
        loss_mask_list = build_assistant_loss_mask(
            input_id_list,
            self.tokenizer,
            assistant_turns,
            row_dict.get("supervised_assistant_turn_indices"),
        )

        input_ids = torch.tensor(input_id_list, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask_list, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        sequence_length = input_ids.shape[0]
        if sequence_length > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"sequence_length={sequence_length} is larger than max_length={self.max_length}")
            selection = slice(-self.max_length, None) if self.truncation == "left" else slice(0, self.max_length)
            input_ids = input_ids[selection]
            loss_mask = loss_mask[selection]
            attention_mask = attention_mask[selection]
            position_ids = position_ids[selection]

        if int(loss_mask.sum().item()) <= 0:
            raise ValueError(f"sample {item} has no assistant loss tokens after truncation")

        if self.pad_mode == DatasetPadMode.RIGHT and input_ids.shape[0] < self.max_length:
            pad_length = self.max_length - input_ids.shape[0]
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = F.pad(input_ids, (0, pad_length), value=pad_token_id)
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
            loss_mask = F.pad(loss_mask, (0, pad_length), value=0)
            position_ids = F.pad(position_ids, (0, pad_length), value=0)

        result = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
        if self.pad_mode == DatasetPadMode.RIGHT:
            result["attention_mask"] = attention_mask
        return result
