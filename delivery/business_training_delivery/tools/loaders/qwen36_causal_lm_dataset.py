#!/usr/bin/env python3
"""Raw-text causal-LM dataset for veRL's SFT trainer custom class hook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch.utils.data import Dataset

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tokenizer.tokenizer import normalize_token_ids


def build_causal_lm_tensors(
    text: str,
    tokenizer: Any,
    *,
    max_length: int,
    truncation: str,
    pad_mode: str,
) -> dict[str, torch.Tensor]:
    """Tokenize raw text, append EOS, and supervise every next-token target."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("causal-LM sample text must be a non-empty string")

    input_id_list = normalize_token_ids(tokenizer(text, add_special_tokens=False)["input_ids"])
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    input_id_list.append(int(eos_token_id))

    input_ids = torch.tensor(input_id_list, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

    # veRL rolls this mask one place to the left before applying next-token loss.
    # Zeroing each sample's first token therefore masks the wrapped prediction at
    # its final position while supervising every real target, including EOS.
    loss_mask = torch.ones_like(input_ids)
    loss_mask[0] = 0

    sequence_length = input_ids.shape[0]
    if sequence_length > max_length:
        if truncation == "error":
            raise ValueError(f"sequence_length={sequence_length} is larger than max_length={max_length}")
        selection = slice(-max_length, None) if truncation == "left" else slice(0, max_length)
        input_ids = input_ids[selection]
        attention_mask = attention_mask[selection]
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)
        loss_mask = loss_mask[selection]
        loss_mask[0] = 0

    if pad_mode == DatasetPadMode.RIGHT and input_ids.shape[0] < max_length:
        pad_length = max_length - input_ids.shape[0]
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = F.pad(input_ids, (0, pad_length), value=pad_token_id)
        attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
        position_ids = F.pad(position_ids, (0, pad_length), value=0)
        loss_mask = F.pad(loss_mask, (0, pad_length), value=0)

    result = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
    }
    if pad_mode == DatasetPadMode.RIGHT:
        result["attention_mask"] = attention_mask
    return result


class Qwen36CausalLMDataset(Dataset):
    """Read private Parquet text blocks without applying a chat template."""

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: Any,
        config: Any,
        processor: Any = None,
        max_samples: int = -1,
    ):
        # veRL resolves an AutoProcessor from the Qwen model path even for a
        # text-only custom dataset. Raw CPT intentionally ignores it.
        del processor

        self.text_key = config.get("text_key", "text")
        self.pad_mode = config.get("pad_mode", "no_padding")
        self.max_length = int(config.get("max_length", 4096))
        self.truncation = config.get("truncation", "error")
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = config.get("seed")
        self.sort_by_token_count_desc = bool(config.get("sort_by_token_count_desc", False))
        self.tokenizer = tokenizer

        if self.pad_mode not in {DatasetPadMode.RIGHT, DatasetPadMode.NO_PADDING, "right", "no_padding"}:
            raise ValueError(f"unsupported pad_mode={self.pad_mode}")
        if self.truncation not in {"error", "left", "right"}:
            raise ValueError(f"unsupported truncation={self.truncation}")

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        local_files = [copy_local_path_from_hdfs(str(path), verbose=True) for path in parquet_files]
        for path in local_files:
            if not Path(path).is_file():
                raise FileNotFoundError(path)

        frames = [pd.read_parquet(path, dtype_backend="pyarrow") for path in local_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        if self.text_key not in self.dataframe.columns:
            raise ValueError(f"missing required Parquet column: {self.text_key}")
        if self.sort_by_token_count_desc:
            if "token_count" not in self.dataframe.columns:
                raise ValueError("sort_by_token_count_desc requires token_count column")
            self.dataframe = self.dataframe.sort_values(
                by="token_count", ascending=False, kind="stable", ignore_index=True
            )

        total = len(self.dataframe)
        if max_samples > 0 and max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(total, size=max_samples, replace=False)
            else:
                indices = np.arange(max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()].reset_index(drop=True)
        print(f"causal-LM dataset len: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        text = self.dataframe.iloc[item][self.text_key]
        return build_causal_lm_tensors(
            text,
            self.tokenizer,
            max_length=self.max_length,
            truncation=self.truncation,
            pad_mode=self.pad_mode,
        )
