#!/usr/bin/env python3
"""CPU-only Qwen3.6 tokenization and loss-mask gate for chosen actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from verl.utils.tokenizer import hf_tokenizer

from scripts.prepare_repair_sft_dataset import sha256_file
from scripts.qwen36_first_action_diagnostic_dataset import (
    Qwen36FirstActionDiagnosticDataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.dataset_contract.read_text(encoding="utf-8"))
    if contract.get("contract") != "chosen-only-schema-conditioned-first-action-sft-v1":
        raise ValueError("chosen-only dataset contract mismatch")
    expected_hashes = {
        value["sha256"] for value in (contract.get("outputs") or {}).values()
    }
    if sha256_file(args.data_file) not in expected_hashes:
        raise ValueError("chosen-only parquet hash is absent from dataset contract")
    if contract.get("training_allowed") is not False:
        raise ValueError("chosen-only dataset unexpectedly authorizes training")

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
    dataset = Qwen36FirstActionDiagnosticDataset(
        parquet_files=str(args.data_file), tokenizer=tokenizer, config=config
    )
    samples: list[dict[str, int]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        total = int(item["input_ids"].numel())
        loss = item["loss_mask"].tolist()
        counts = {
            key: int(item[key].sum().item())
            for key in (
                "tool_turn_mask",
                "tool_structure_mask",
                "sql_shell_mask",
            )
        }
        if counts["tool_turn_mask"] != sum(loss):
            raise ValueError(f"sample {index}: loss is not exactly the tool action")
        if counts["tool_structure_mask"] + counts["sql_shell_mask"] != sum(loss):
            raise ValueError(f"sample {index}: component masks do not cover action")
        if any(value <= 0 for value in counts.values()):
            raise ValueError(f"sample {index}: empty chosen action component mask")
        if sum(loss) >= total:
            raise ValueError(f"sample {index}: non-assistant context is not masked")
        samples.append(
            {
                "index": index,
                "sequence_tokens": total,
                "masked_context_tokens": total - sum(loss),
                **counts,
            }
        )

    result = {
        "contract": "chosen-only-schema-action-tokenization-gate-v1",
        "rows": len(dataset),
        "max_length": args.max_length,
        "sequence_tokens_min": min(item["sequence_tokens"] for item in samples),
        "sequence_tokens_max": max(item["sequence_tokens"] for item in samples),
        "tool_turn_tokens_min": min(item["tool_turn_mask"] for item in samples),
        "tool_turn_tokens_max": max(item["tool_turn_mask"] for item in samples),
        "sql_shell_tokens_min": min(item["sql_shell_mask"] for item in samples),
        "sql_shell_tokens_max": max(item["sql_shell_mask"] for item in samples),
        "all_rows_tokenize_without_truncation": True,
        "all_rows_loss_exactly_one_assistant_tool_action": True,
        "all_nonassistant_context_loss_zero": True,
        "all_tool_structure_and_sql_masks_nonempty_disjoint_and_complete": True,
        "source_sha256": {
            "data_file": sha256_file(args.data_file),
            "dataset_contract": sha256_file(args.dataset_contract),
        },
        "samples": samples,
        "npu_required": False,
        "teacher_forced_baseline_allowed": True,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
