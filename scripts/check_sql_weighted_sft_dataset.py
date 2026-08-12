#!/usr/bin/env python3
"""CPU-only fail-closed audit for the SQL-weighted repair SFT loss mask."""

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
    parser.add_argument("--max-length", type=int, default=2048)
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
        counts = {
            key: int(item[key].sum().item())
            for key in ("tool_structure_mask", "sql_shell_mask", "final_answer_mask")
        }
        expected_weight = (
            counts["tool_structure_mask"] * args.tool_structure_weight
            + counts["sql_shell_mask"] * args.sql_payload_weight
            + counts["final_answer_mask"] * args.final_answer_weight
        )
        actual_weight = float(item["loss_mask"].sum().item())
        if abs(actual_weight - expected_weight) > 1e-4:
            raise ValueError(f"sample {index}: weighted loss mask sum mismatch")
        samples.append(
            {
                "index": index,
                **counts,
                "weighted_loss_mass": actual_weight,
                "sql_loss_mass_share": counts["sql_shell_mask"]
                * args.sql_payload_weight
                / actual_weight,
            }
        )

    result = {
        "contract": "repair-sft-sql-weighted-mask-gate-v1",
        "rows": len(dataset),
        "weights": {
            "tool_structure": args.tool_structure_weight,
            "sql_payload": args.sql_payload_weight,
            "final_answer": args.final_answer_weight,
        },
        "all_component_masks_nonempty": True,
        "all_weighted_masks_match_contract": True,
        "sql_loss_mass_share": sum(row["sql_loss_mass_share"] for row in samples)
        / len(samples),
        "samples": samples,
        "npu_required": False,
        "promotion_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
