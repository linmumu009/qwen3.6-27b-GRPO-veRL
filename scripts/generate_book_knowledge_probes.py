#!/usr/bin/env python3
"""Generate private, paraphrased MCQ probes from CPT text with offline vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.evaluate_logistics_knowledge import _canonical_hash
from scripts.prepare_book_nll_cases import sha256_file, write_json


CONTRACT = "handbook8e-paraphrased-mcq-probes-v1"
WORD_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalized_words(text: str) -> list[str]:
    return [value.casefold().replace("’", "'") for value in WORD_RE.findall(text)]


def has_shared_ngram(source: str, candidate: str, n: int = 4) -> bool:
    source_words = normalized_words(source)
    candidate_words = normalized_words(candidate)
    if len(source_words) < n or len(candidate_words) < n:
        return False
    source_ngrams = {tuple(source_words[index : index + n]) for index in range(len(source_words) - n + 1)}
    return any(
        tuple(candidate_words[index : index + n]) in source_ngrams
        for index in range(len(candidate_words) - n + 1)
    )


def validate_probe(value: Any, source: str) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(value, dict):
        return None, "not_object"
    if not {"question", "options", "answer"} <= set(value):
        return None, "missing_keys"
    question = str(value["question"]).strip()
    options = value["options"]
    answer = value["answer"]
    if not question or not isinstance(options, list) or len(options) != 4:
        return None, "invalid_shape"
    cleaned_options = [str(option).strip() for option in options]
    if any(not option for option in cleaned_options) or len({option.casefold() for option in cleaned_options}) != 4:
        return None, "invalid_options"
    if isinstance(answer, bool) or not isinstance(answer, int) or answer not in range(4):
        return None, "invalid_answer"
    if len(normalized_words(cleaned_options[answer])) < 6:
        return None, "answer_too_short"
    if has_shared_ngram(source, question, 4):
        return None, "question_4gram_overlap"
    if has_shared_ngram(source, cleaned_options[answer], 4):
        return None, "answer_4gram_overlap"
    return {"question": question, "options": cleaned_options, "answer": answer}, None


def rotate_options(probe: dict[str, Any], seed: int) -> dict[str, Any]:
    order = list(range(4))
    random.Random(seed).shuffle(order)
    inverse = {old: new for new, old in enumerate(order)}
    return {
        "question": probe["question"],
        "options": [probe["options"][old] for old in order],
        "answer": inverse[int(probe["answer"])],
    }


def build_windows(texts: list[str], tokenizer: Any, *, window_tokens: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for record_index, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, len(token_ids) - window_tokens + 1, window_tokens):
            selected = token_ids[start : start + window_tokens]
            encoded = ",".join(str(int(value)) for value in selected).encode("ascii")
            windows.append(
                {
                    "source_id": f"r{record_index:04d}-o{start:06d}",
                    "source_hash": hashlib.sha256(encoded).hexdigest(),
                    "text": tokenizer.decode(selected, skip_special_tokens=True),
                }
            )
    return windows


def build_messages(passage: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Create one rigorous closed-book English logistics multiple-choice question from the supplied passage. "
                "Test a reusable concept, relationship, decision rule, or trade-off rather than wording recall. "
                "Do not say 'according to the passage'. Paraphrase both the question and correct option: neither may "
                "copy four consecutive words from the passage. The correct option must contain at least six words. "
                "Provide exactly four plausible, mutually exclusive options and exactly one correct answer. Return "
                "only JSON: {\"question\":\"...\",\"options\":[\"...\",\"...\",\"...\",\"...\"],\"answer\":0}."
            ),
        },
        {"role": "user", "content": f"Source passage:\n{passage}"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--candidate-count", type=int, default=400)
    parser.add_argument("--target-count", type=int, default=200)
    parser.add_argument("--window-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=1536)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-output-tokens", type=int, default=320)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    args = parser.parse_args()
    if args.target_count > args.candidate_count:
        raise ValueError("target count cannot exceed candidate count")

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    table = pq.read_table(args.parquet, columns=[args.text_key])
    texts = [str(value) for value in table.column(args.text_key).to_pylist()]
    available = build_windows(texts, tokenizer, window_tokens=args.window_tokens)
    if len(available) < args.candidate_count:
        raise ValueError(f"only {len(available)} source windows are available")
    selected = random.Random(args.seed).sample(available, args.candidate_count)
    prompts: list[dict[str, list[int]]] = []
    for row in selected:
        rendered = tokenizer.apply_chat_template(
            build_messages(row["text"]), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(token_ids) + args.max_output_tokens > args.max_model_len:
            raise ValueError(f"prompt too long: {row['source_id']}")
        prompts.append({"prompt_token_ids": [int(value) for value in token_ids]})

    started = time.perf_counter()
    llm = LLM(
        model=str(args.model), tensor_parallel_size=args.tensor_parallel_size, dtype="bfloat16",
        trust_remote_code=True, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed, enforce_eager=True,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.35, top_p=0.9, max_tokens=args.max_output_tokens, seed=args.seed),
        use_tqdm=True,
    )
    by_id = {int(output.request_id): output for output in outputs}
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen_questions: set[str] = set()
    for index, source in enumerate(selected):
        output = by_id.get(index)
        raw = output.outputs[0].text if output is not None and output.outputs else ""
        parsed, reason = validate_probe(extract_json_object(raw), source["text"])
        if parsed is None:
            rejected[reason or "unknown"] += 1
            continue
        normalized_question = " ".join(normalized_words(parsed["question"]))
        if normalized_question in seen_questions:
            rejected["duplicate_question"] += 1
            continue
        seen_questions.add(normalized_question)
        rotated = rotate_options(parsed, args.seed + index)
        fingerprint = {
            "dataset": "handbook8e-derived",
            "question_type": "single_choice",
            "question": rotated["question"],
            "options": tuple(rotated["options"]),
            "expected": (rotated["answer"],),
        }
        accepted.append(
            {
                "item_hash": _canonical_hash(fingerprint),
                "dataset": "handbook8e-derived",
                "source_id": source["source_id"],
                "source_window_hash": source["source_hash"],
                "category": "book_derived",
                "question_type": "single_choice",
                "question": rotated["question"],
                "options": rotated["options"],
                "expected": [rotated["answer"]],
                "generator_model": args.model_label,
            }
        )
        if len(accepted) == args.target_count:
            break
    if len(accepted) < args.target_count:
        raise ValueError(f"only {len(accepted)} valid probes; rejection counts: {dict(rejected)}")

    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.private_output.with_name(f".{args.private_output.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, args.private_output)
    safe = {
        "schema_version": 1,
        "contract": CONTRACT,
        "source_content_included": False,
        "probe_content_included": False,
        "source_parquet_sha256": sha256_file(args.parquet),
        "tokenizer_json_sha256": sha256_file(args.model / "tokenizer.json"),
        "generator_model": args.model_label,
        "source_records": len(texts),
        "available_windows": len(available),
        "candidate_windows": len(selected),
        "accepted_probes": len(accepted),
        "window_tokens": args.window_tokens,
        "exact_overlap_gate": "no shared normalized 4-gram in question or correct option",
        "correct_option_min_words": 6,
        "answer_position_counts": dict(sorted(Counter(row["expected"][0] for row in accepted).items())),
        "rejection_counts_before_target_reached": dict(sorted(rejected.items())),
        "item_hashes_sha256": hashlib.sha256(
            "\n".join(row["item_hash"] for row in accepted).encode("ascii")
        ).hexdigest(),
        "runtime": {
            "backend": "offline_vllm",
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": 1,
            "max_num_seqs": args.max_num_seqs,
            "seed": args.seed,
            "elapsed_sec": round(time.perf_counter() - started, 3),
        },
    }
    write_json(args.safe_output, safe)
    print(json.dumps(safe, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
