#!/usr/bin/env python3
"""Evaluate fixed private logistics MCQs with one offline vLLM replica."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scripts.evaluate_logistics_knowledge import (
    PROMPT_VERSION,
    EvalItem,
    _make_item,
    build_messages,
    build_safe_result,
    parse_answers,
    sha256_bytes,
    write_private_rows,
)


def load_items(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        item = _make_item(
            dataset=str(row["dataset"]),
            source_id=str(row.get("source_id") or line_number),
            category=str(row.get("category") or "unknown"),
            question_type=str(row.get("question_type") or "single_choice"),
            question=str(row["question"]),
            options=row["options"],
            expected=row["expected"],
        )
        supplied_hash = row.get("item_hash")
        if supplied_hash is not None and str(supplied_hash) != item.item_hash:
            raise ValueError(f"item hash mismatch at line {line_number}")
        items.append(item)
    if not items or len({item.item_hash for item in items}) != len(items):
        raise ValueError("cases are empty or contain duplicate item hashes")
    return items


def repeat_path(path: Path, repeat_index: int) -> Path:
    if repeat_index == 0:
        return path
    return path.with_name(f"{path.stem}.repeat{repeat_index + 1}{path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    items = load_items(args.cases)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts: list[dict[str, list[int]]] = []
    for item in items:
        rendered = tokenizer.apply_chat_template(
            build_messages(item), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(token_ids) + args.max_output_tokens > args.max_model_len:
            raise ValueError(f"prompt too long: {item.item_hash}")
        prompts.append({"prompt_token_ids": [int(value) for value in token_ids]})

    started = time.perf_counter()
    llm = LLM(
        model=str(args.model), tensor_parallel_size=args.tensor_parallel_size, dtype="bfloat16",
        trust_remote_code=True, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed, enforce_eager=True,
    )
    repeated_rows: list[list[dict[str, Any]]] = []
    repeated_results: list[dict[str, Any]] = []
    for repeat_index in range(args.repeats):
        repeat_started = time.perf_counter()
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens, seed=args.seed),
            use_tqdm=True,
        )
        ordered_outputs = sorted(outputs, key=lambda output: int(output.request_id))
        if len(ordered_outputs) != len(items):
            raise ValueError(f"output count mismatch in repeat {repeat_index + 1}")
        rows: list[dict[str, Any]] = []
        for item, output in zip(items, ordered_outputs):
            if not output.outputs:
                raise ValueError(f"missing output for {item.item_hash}")
            prediction = output.outputs[0].text or ""
            parsed, parse_ok = parse_answers(prediction, len(item.options))
            rows.append(
                {
                    "prompt_version": PROMPT_VERSION,
                    "chat_template_disable_thinking": True,
                    "item_hash": item.item_hash,
                    "dataset": item.dataset,
                    "source_id": item.source_id,
                    "category": item.category,
                    "question_type": item.question_type,
                    "question": item.question,
                    "options": list(item.options),
                    "expected": list(item.expected),
                    "prediction": prediction,
                    "reasoning": "",
                    "parsed": list(parsed),
                    "parse_ok": parse_ok,
                    "correct": parse_ok and parsed == item.expected,
                    "elapsed_sec": 0.0,
                    "usage": None,
                    "error": None,
                }
            )
        write_private_rows(repeat_path(args.private_output, repeat_index), rows)
        result = build_safe_result(
            rows,
            model=args.model_label,
            endpoint_label="m05-offline-vllm-single-replica",
            concurrency=args.max_num_seqs,
            input_hashes={"cases_jsonl": sha256_bytes(args.cases.read_bytes())},
            elapsed_sec=time.perf_counter() - repeat_started,
            chat_template_disable_thinking=True,
        )
        result["request"]["max_output_tokens"] = args.max_output_tokens
        result["runtime"] = {
            "backend": "offline_vllm",
            "dtype": "bfloat16",
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": 1,
            "async_scheduling": True,
            "prefix_caching": False,
            "seed": args.seed,
            "repeat_index": repeat_index + 1,
        }
        safe_path = repeat_path(args.safe_output, repeat_index)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        repeated_rows.append(rows)
        repeated_results.append(result)

    if args.repeats > 1:
        first = repeated_results[0]
        first["repeat_stability"] = {
            "repeats": args.repeats,
            "same_seed_each_repeat": True,
            "parsed_answer_identical_items": sum(
                len({tuple(rows[index]["parsed"]) for rows in repeated_rows}) == 1 for index in range(len(items))
            ),
            "correctness_identical_items": sum(
                len({bool(rows[index]["correct"]) for rows in repeated_rows}) == 1 for index in range(len(items))
            ),
            "prediction_text_identical_items": sum(
                len({str(rows[index]["prediction"]) for rows in repeated_rows}) == 1 for index in range(len(items))
            ),
            "accuracy_by_repeat": [result["accuracy"] for result in repeated_results],
        }
        first["runtime"]["all_repeats_elapsed_sec"] = round(time.perf_counter() - started, 3)
        args.safe_output.write_text(json.dumps(first, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": args.model_label,
        "items": len(items),
        "repeats": args.repeats,
        "accuracy_by_repeat": [result["accuracy"] for result in repeated_results],
        "repeat_stability": repeated_results[0].get("repeat_stability"),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
