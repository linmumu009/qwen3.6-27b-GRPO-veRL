#!/usr/bin/env python3
"""CPU-only token and pair audit for the semantic-delta likelihood gate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from omegaconf import OmegaConf
from verl.utils.tokenizer import hf_tokenizer

from scripts.qwen36_semantic_delta_margin_dataset import Qwen36SemanticDeltaMarginDataset
from scripts.prepare_semantic_delta_margin_gate import sha256_file


def check(data_file: Path, contract_file: Path, model_path: str, max_length: int) -> dict:
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if contract.get("contract") != "semantic-delta-margin-gate-dataset-v1":
        raise ValueError("unexpected semantic-delta dataset contract")
    if contract.get("output_sha256") != sha256_file(data_file):
        raise ValueError("semantic-delta parquet hash differs from its contract")
    tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
    config = OmegaConf.create(
        {
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "enable_thinking_default": False,
            "pad_mode": "no_padding",
            "max_length": max_length,
            "truncation": "error",
            "ignore_input_ids_mismatch": False,
            "apply_chat_template_kwargs": {},
        }
    )
    dataset = Qwen36SemanticDeltaMarginDataset(str(data_file), tokenizer, config)
    if len(dataset) != 32:
        raise ValueError(f"semantic-delta token gate requires 32 rows, got {len(dataset)}")
    evidence = {str(row["task_id"]): row for row in contract["evidence"]}
    labels: Counter[str] = Counter()
    samples = []
    for index in range(len(dataset)):
        row = dataset.dataframe.iloc[index]
        task_id = str(row["source_task_id"])
        label = str(row["candidate_label"])
        item = dataset[index]
        sql_mask = item["sql_shell_mask"].tolist()
        delta_mask = item["semantic_delta_mask"].tolist()
        if not all(not delta or sql for delta, sql in zip(delta_mask, sql_mask, strict=True)):
            raise ValueError(f"{task_id}::{label}: delta mask extends outside SQL")
        delta_tokens = int(sum(delta_mask))
        if delta_tokens <= 0:
            raise ValueError(f"{task_id}::{label}: empty delta mask")
        if label == "chosen":
            positions = [position for position, value in enumerate(sql_mask) if value]
            offset = int(evidence[task_id]["critical_sql_token_offset"])
            actual_id = int(item["input_ids"][positions[offset]].item())
            if actual_id != int(evidence[task_id]["critical_sql_target_id"]):
                raise ValueError(f"{task_id}: frozen critical target changed")
        labels[label] += 1
        samples.append(
            {
                "task_id": task_id,
                "candidate_label": label,
                "sql_tokens": int(sum(sql_mask)),
                "semantic_delta_tokens": delta_tokens,
                "sequence_tokens": int(item["input_ids"].numel()),
            }
        )
    if labels != Counter({"chosen": 16, "rejected": 16}):
        raise ValueError(f"semantic-delta candidate imbalance: {dict(labels)}")
    return {
        "contract": "semantic-delta-margin-token-gate-v1",
        "rows": len(dataset),
        "pairs": 16,
        "candidate_rows": dict(sorted(labels.items())),
        "all_delta_masks_nonempty": True,
        "all_delta_masks_subset_of_sql": True,
        "all_chosen_critical_targets_match_frozen_step120": True,
        "truncation": "error",
        "max_length": max_length,
        "samples": samples,
        "npu_required": False,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.data_file, args.contract, args.model_path, args.max_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
