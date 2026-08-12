#!/usr/bin/env python3
"""CPU-only fail-closed audit for state-conditioned selective SFT loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from verl.utils.tokenizer import hf_tokenizer

from scripts.qwen36_sql_weighted_sft_dataset import Qwen36SQLWeightedSFTDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--tool-structure-weight", type=float, default=0.25)
    parser.add_argument("--sql-payload-weight", type=float, default=8.0)
    parser.add_argument("--final-answer-weight", type=float, default=1.0)
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
            "tool_structure_weight": args.tool_structure_weight,
            "sql_payload_weight": args.sql_payload_weight,
            "final_answer_weight": args.final_answer_weight,
        }
    )
    dataset = Qwen36SQLWeightedSFTDataset(
        parquet_files=str(args.data_file), tokenizer=tokenizer, config=config
    )
    samples = []
    for index in range(len(dataset)):
        item = dataset[index]
        context = item["context_assistant_mask"].tolist()
        weighted = item["loss_mask"].tolist()
        context_tokens = int(sum(context))
        context_loss_mass = sum(
            float(context_token) * float(loss_weight)
            for context_token, loss_weight in zip(context, weighted, strict=True)
        )
        if context_tokens <= 0:
            raise ValueError(f"sample {index}: missing error assistant context tokens")
        if context_loss_mass != 0:
            raise ValueError(f"sample {index}: error assistant context has nonzero loss")
        counts = {
            key: int(item[key].sum().item())
            for key in ("tool_structure_mask", "sql_shell_mask", "final_answer_mask")
        }
        if any(value <= 0 for value in counts.values()):
            raise ValueError(f"sample {index}: empty supervised component mask")
        samples.append(
            {
                "index": index,
                "error_context_assistant_tokens": context_tokens,
                "error_context_loss_mass": context_loss_mass,
                **counts,
                "sequence_tokens": int(item["input_ids"].numel()),
            }
        )

    result = {
        "contract": "repair-sft-state-conditioned-mask-gate-v1",
        "rows": len(dataset),
        "all_error_context_assistant_masks_nonempty": True,
        "all_error_context_loss_mass_zero": True,
        "all_supervised_component_masks_nonempty": True,
        "all_rows_have_assistant_loss": True,
        "all_rows_mask_non_assistant_context": True,
        "truncation": "error",
        "weights": {
            "tool_structure": args.tool_structure_weight,
            "sql_payload": args.sql_payload_weight,
            "final_answer": args.final_answer_weight,
        },
        "samples": samples,
        "npu_required": False,
        "promotion_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
