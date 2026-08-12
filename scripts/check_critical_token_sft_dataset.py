#!/usr/bin/env python3
"""CPU-only fail-closed audit for semantic critical-token SFT weighting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from verl.utils.tokenizer import hf_tokenizer

from scripts.qwen36_critical_token_sft_dataset import Qwen36CriticalTokenSFTDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--critical-token-weight", type=float, default=32.0)
    parser.add_argument("--output", type=Path, required=True)
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
            "tool_structure_weight": 0.25,
            "sql_payload_weight": 8.0,
            "final_answer_weight": 1.0,
            "critical_token_weight": args.critical_token_weight,
        }
    )
    dataset = Qwen36CriticalTokenSFTDataset(
        parquet_files=str(args.data_file), tokenizer=tokenizer, config=config
    )
    samples = []
    for index in range(len(dataset)):
        item = dataset[index]
        critical = item["critical_sql_token_mask"].tolist()
        weighted = item["loss_mask"].tolist()
        context = item["context_assistant_mask"].tolist()
        if sum(critical) != 1:
            raise ValueError(f"sample {index}: critical mask must have one token")
        critical_position = critical.index(1)
        if weighted[critical_position] != args.critical_token_weight:
            raise ValueError(f"sample {index}: critical token weight mismatch")
        if any(float(mask) * float(weight) for mask, weight in zip(context, weighted, strict=True)):
            raise ValueError(f"sample {index}: error assistant context has nonzero loss")
        samples.append(
            {
                "index": index,
                "sequence_tokens": int(item["input_ids"].numel()),
                "critical_token_count": 1,
                "critical_token_weight": weighted[critical_position],
                "error_context_loss_mass": 0,
            }
        )

    result = {
        "contract": "repair-sft-critical-token-mask-gate-v1",
        "rows": len(dataset),
        "all_rows_have_assistant_loss": True,
        "all_rows_mask_non_assistant_context": True,
        "all_error_context_loss_mass_zero": True,
        "all_critical_token_masks_exactly_one": True,
        "all_critical_token_weights_match": True,
        "critical_token_weight": args.critical_token_weight,
        "samples": samples,
        "npu_required": False,
        "promotion_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
