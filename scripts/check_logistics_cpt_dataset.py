#!/usr/bin/env python3
"""Fail-closed tokenization gate for the private logistics CPT dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from scripts.qwen36_causal_lm_dataset import Qwen36CausalLMDataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--expected-records", type=int, default=116)
    parser.add_argument("--expected-content-tokens", type=int, default=336586)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_json = args.model_path / "tokenizer.json"
    actual_tokenizer_hash = sha256_file(tokenizer_json)
    if actual_tokenizer_hash != args.expected_tokenizer_sha256:
        raise ValueError(
            f"tokenizer hash mismatch: expected={args.expected_tokenizer_sha256} actual={actual_tokenizer_hash}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = Qwen36CausalLMDataset(
        parquet_files=str(args.train_file),
        tokenizer=tokenizer,
        config=OmegaConf.create(
            {
                "text_key": "text",
                "pad_mode": "no_padding",
                "max_length": args.max_length,
                "truncation": "error",
                "shuffle": False,
            }
        ),
    )
    if len(dataset) != args.expected_records:
        raise ValueError(f"record count mismatch: expected={args.expected_records} actual={len(dataset)}")

    sequence_lengths: list[int] = []
    loss_tokens = 0
    eos_terminated = 0
    for index in range(len(dataset)):
        item = dataset[index]
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        if input_ids.shape != loss_mask.shape:
            raise ValueError(f"sample {index} tensor shape mismatch")
        if int(loss_mask[0]) != 0 or not bool((loss_mask[1:] == 1).all()):
            raise ValueError(f"sample {index} loss mask is not all-token causal-LM aligned")
        if int(input_ids[-1]) != tokenizer.eos_token_id:
            raise ValueError(f"sample {index} does not end in EOS")
        sequence_lengths.append(int(input_ids.numel()))
        loss_tokens += int(loss_mask.sum().item())
        eos_terminated += 1

    if loss_tokens != args.expected_content_tokens:
        raise ValueError(
            f"loss token count mismatch: expected={args.expected_content_tokens} actual={loss_tokens}"
        )
    summary = {
        "schema_version": 1,
        "dataset_type": "causal_lm_continued_pretraining",
        "train_file_sha256": sha256_file(args.train_file),
        "tokenizer_json_sha256": actual_tokenizer_hash,
        "records": len(dataset),
        "sequence_tokens_with_eos": sum(sequence_lengths),
        "loss_tokens": loss_tokens,
        "minimum_sequence_tokens": min(sequence_lengths),
        "maximum_sequence_tokens": max(sequence_lengths),
        "eos_terminated_records": eos_terminated,
        "max_length": args.max_length,
        "chat_template_applied": False,
        "first_token_wraparound_masked": True,
        "all_real_next_token_targets_supervised": True,
        "source_content_included": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, args.output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
