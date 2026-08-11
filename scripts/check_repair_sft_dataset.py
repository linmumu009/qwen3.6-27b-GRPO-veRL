#!/usr/bin/env python3
"""Tokenize a repair SFT parquet with veRL and audit assistant-only loss masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from verl.utils.tokenizer import hf_tokenizer

from qwen36_assistant_mask_sft_dataset import Qwen36AssistantMaskSFTDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true", help="write the gate without printing it")
    args = parser.parse_args()

    tokenizer = hf_tokenizer(args.model_path, trust_remote_code=True)
    config = OmegaConf.create(
        {
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "enable_thinking_default": False,
            "pad_mode": "no_padding",
            "max_length": args.max_length,
            "truncation": "error",
            "ignore_input_ids_mismatch": False,
            "apply_chat_template_kwargs": {},
        }
    )
    dataset = Qwen36AssistantMaskSFTDataset(
        parquet_files=str(args.train_file),
        tokenizer=tokenizer,
        config=config,
    )

    sample_summaries = []
    for index in range(len(dataset)):
        item = dataset[index]
        total_tokens = int(item["input_ids"].numel())
        assistant_tokens = int(item["loss_mask"].sum().item())
        if assistant_tokens <= 0:
            raise ValueError(f"sample {index} has no assistant loss tokens")
        if assistant_tokens >= total_tokens:
            raise ValueError(f"sample {index} does not mask non-assistant context")
        sample_summaries.append(
            {
                "index": index,
                "total_tokens": total_tokens,
                "assistant_loss_tokens": assistant_tokens,
                "masked_context_tokens": total_tokens - assistant_tokens,
            }
        )

    summary = {
        "contract": "repair-sft-tokenization-gate-v1",
        "rows": len(dataset),
        "max_length": args.max_length,
        "all_rows_have_assistant_loss": True,
        "all_rows_mask_non_assistant_context": True,
        "samples": sample_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
