#!/usr/bin/env python3
"""Run paired private book-continuation cases with one offline vLLM replica."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from scripts.audit_book_memorization import (
    PROMPT_VERSION,
    aggregate,
    build_messages,
    load_cases,
    longest_exact_prefix,
    normalize_tokens,
    token_f1,
    write_json,
    write_private_rows,
)


def parse_named_paths(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw in values:
        label, separator, path = raw.partition("=")
        if not separator:
            raise ValueError(f"expected PREFIX=PATH, got {raw!r}")
        prefix = int(label)
        if prefix in result:
            raise ValueError(f"duplicate prefix {prefix}")
        result[prefix] = Path(path)
    return result


def normalize_template_token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not value or any(isinstance(item, (dict, list)) for item in value):
        raise ValueError("chat template did not return a flat token-id list")
    return [int(item) for item in value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--cases", action="append", required=True, metavar="PREFIX=PRIVATE_JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--safe-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    case_paths = parse_named_paths(args.cases)
    grouped = {prefix: load_cases(path) for prefix, path in case_paths.items()}
    flattened = [(prefix, case) for prefix in sorted(grouped) for case in grouped[prefix]]
    if len({(prefix, case.case_id) for prefix, case in flattened}) != len(flattened):
        raise ValueError("duplicate prefixed case id")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts: list[dict[str, list[int]]] = []
    for _, case in flattened:
        rendered = tokenizer.apply_chat_template(
            build_messages(case.prefix),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str):
            raise ValueError("chat template did not return rendered text")
        token_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
        if len(token_ids) + args.max_output_tokens > args.max_model_len:
            raise ValueError(f"prompt too long: {case.case_id}")
        prompts.append({"prompt_token_ids": token_ids})

    started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enforce_eager=True,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens, seed=args.seed),
        use_tqdm=True,
    )
    by_id = {int(output.request_id): output for output in outputs}
    rows_by_prefix: dict[int, list[dict[str, Any]]] = {prefix: [] for prefix in grouped}
    for index, (prefix, case) in enumerate(flattened):
        output = by_id.get(index)
        if output is None or not output.outputs:
            raise ValueError(f"missing output {index}")
        prediction = output.outputs[0].text or ""
        rows_by_prefix[prefix].append(
            {
                "case_id": case.case_id,
                "source_hash": hashlib.sha256((case.prefix + "\0" + case.target).encode("utf-8")).hexdigest(),
                "prompt_version": PROMPT_VERSION,
                "chat_template_disable_thinking": True,
                "prediction": prediction,
                "reasoning": "",
                "target": case.target,
                "exact_prefix_tokens": longest_exact_prefix(case.target, prediction),
                "target_tokens": len(normalize_tokens(case.target)),
                "token_f1": token_f1(case.target, prediction),
                "empty_prediction": not bool(prediction.strip()),
                "reasoning_only": False,
                "elapsed_sec": None,
                "usage": None,
                "error": None,
            }
        )

    total_elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.safe_dir.mkdir(parents=True, exist_ok=True)
    for prefix, rows in rows_by_prefix.items():
        private_path = args.output_dir / f"p{prefix}_n{len(rows)}.jsonl"
        safe_path = args.safe_dir / f"p{prefix}_n{len(rows)}.safe.json"
        write_private_rows(private_path, rows)
        summary = aggregate(
            rows,
            model=args.model_label,
            source_name="handbook8e-private-fixed-cases",
            concurrency=args.max_num_seqs,
            prefix_tokens=prefix,
            target_tokens=32,
            seed=20260901,
            chat_template_disable_thinking=True,
        )
        summary["runtime"] = {
            "backend": "offline_vllm",
            "dtype": "bfloat16",
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": 1,
            "async_scheduling": True,
            "prefix_caching": False,
            "seed": args.seed,
            "shared_model_load_total_elapsed_sec": round(total_elapsed, 3),
        }
        write_json(safe_path, summary)
    print(json.dumps({
        "model": args.model_label,
        "cases": len(flattened),
        "prefixes": sorted(grouped),
        "elapsed_sec": round(total_elapsed, 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
