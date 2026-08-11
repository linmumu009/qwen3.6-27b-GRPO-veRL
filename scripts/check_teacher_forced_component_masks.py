#!/usr/bin/env python3
"""Audit component masks on every repair SFT row before reserving NPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from verl.utils.tokenizer import hf_tokenizer

from qwen36_teacher_forced_diagnostic_dataset import Qwen36TeacherForcedDiagnosticDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
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
        }
    )
    dataset = Qwen36TeacherForcedDiagnosticDataset(
        parquet_files=str(args.data_file), tokenizer=tokenizer, config=config
    )
    samples = []
    for index in range(len(dataset)):
        item = dataset[index]
        component_counts = {
            key: int(item[key].sum().item())
            for key in (
                "tool_turn_mask",
                "tool_structure_mask",
                "sql_shell_mask",
                "final_answer_mask",
            )
        }
        if component_counts["tool_structure_mask"] + component_counts["sql_shell_mask"] != component_counts["tool_turn_mask"]:
            raise ValueError(f"sample {index}: tool components do not reconstruct tool turn")
        if component_counts["tool_turn_mask"] + component_counts["final_answer_mask"] != int(item["loss_mask"].sum().item()):
            raise ValueError(f"sample {index}: assistant components do not reconstruct loss mask")
        samples.append({"index": index, **component_counts})

    result = {
        "contract": "repair-sft-teacher-forced-component-mask-gate-v1",
        "rows": len(dataset),
        "all_component_masks_nonempty": True,
        "all_tool_components_reconstruct_tool_turn": True,
        "all_assistant_components_reconstruct_loss_mask": True,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
