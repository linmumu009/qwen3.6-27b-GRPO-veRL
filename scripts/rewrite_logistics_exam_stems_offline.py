#!/usr/bin/env python3
"""Paraphrase and verify private exam stems with one local offline vLLM."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.rewrite_logistics_exam_stems import (
    REWRITE_VERSION,
    _extract_object,
    build_rewrite_messages,
    build_verify_messages,
    deterministic_validation,
    item_validation,
    normalize_space,
    normalize_rewritten_form,
    read_jsonl,
    sha256_file,
    text_hash,
    verify_payload,
    write_json,
    write_jsonl_private,
)


def generate_json_objects(
    llm: Any,
    tokenizer: Any,
    messages: list[list[dict[str, str]]],
    *,
    max_tokens: int,
    max_model_len: int,
    seed: int,
    temperature: float = 0.0,
) -> list[dict[str, Any] | None]:
    from vllm import SamplingParams

    prompts: list[dict[str, list[int]]] = []
    for message_list in messages:
        rendered = tokenizer.apply_chat_template(
            message_list,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) + max_tokens > max_model_len:
            raise ValueError(f"rewrite prompt exceeds max_model_len: {len(ids)} + {max_tokens}")
        prompts.append({"prompt_token_ids": [int(value) for value in ids]})
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=temperature,
            top_p=0.9 if temperature > 0 else 1.0,
            max_tokens=max_tokens,
            seed=seed,
        ),
        use_tqdm=True,
    )
    ordered = sorted(outputs, key=lambda output: int(output.request_id))
    if len(ordered) != len(messages):
        raise ValueError("offline output count mismatch")
    parsed: list[dict[str, Any] | None] = []
    for output in ordered:
        try:
            content = output.outputs[0].text if output.outputs else ""
            parsed.append(_extract_object(content or ""))
        except (IndexError, ValueError):
            parsed.append(None)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--rewrite-max-tokens", type=int, default=256)
    parser.add_argument("--verify-max-tokens", type=int, default=256)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_attempts < 1 or args.max_num_seqs < 1:
        raise ValueError("max_attempts and max_num_seqs must be positive")
    source_rows = read_jsonl(args.source)
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(source_rows):
            raise ValueError("limit must be within source row count")
        source_rows = source_rows[: args.limit]
    if len({str(row["item_hash"]) for row in source_rows}) != len(source_rows):
        raise ValueError("source item hashes are not unique")
    if args.private_output.exists() or args.safe_output.exists():
        raise FileExistsError("refusing to overwrite a complete rewrite artifact")

    incomplete = args.private_output.with_suffix(args.private_output.suffix + ".incomplete")
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and incomplete.exists():
        completed = {str(row["item_hash"]): row for row in read_jsonl(incomplete)}
    elif incomplete.exists():
        raise FileExistsError("incomplete artifact exists; use --resume")
    source_by_hash = {str(row["item_hash"]): row for row in source_rows}
    if not set(completed) <= set(source_by_hash):
        raise ValueError("incomplete artifact contains unknown item hashes")

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
    pending = [row for row in source_rows if str(row["item_hash"]) not in completed]
    feedback: dict[str, list[str]] = {str(row["item_hash"]): [] for row in pending}
    accepted_attempt: dict[str, int] = {}
    failure_reason_counts: Counter[str] = Counter()
    verifier_diagnostic_counts: Counter[str] = Counter()
    private_diagnostics: list[dict[str, Any]] = []

    for attempt in range(1, args.max_attempts + 1):
        if not pending:
            break
        # For the difficult tail, sample several alternatives per item.  The
        # engine's max_num_seqs is a concurrency limit rather than a cap on the
        # total request batch, so up to 4 * 64 requests is safe here.
        variants_per_item = 4 if len(pending) <= args.max_num_seqs else 1
        expanded_rows = [row for row in pending for _variant in range(variants_per_item)]
        rewrite_payloads = generate_json_objects(
            llm,
            tokenizer,
            [build_rewrite_messages(row, feedback[str(row["item_hash"])]) for row in expanded_rows],
            max_tokens=args.rewrite_max_tokens,
            max_model_len=args.max_model_len,
            seed=args.seed + attempt,
            temperature=0.2 if variants_per_item > 1 else 0.0,
        )
        verifier_candidates: list[tuple[dict[str, Any], str]] = []
        for row, payload in zip(expanded_rows, rewrite_payloads):
            item_hash = str(row["item_hash"])
            if payload is None:
                feedback[item_hash] = ["invalid_rewrite_json"]
                failure_reason_counts["invalid_rewrite_json"] += 1
                private_diagnostics.append(
                    {"item_hash": item_hash, "attempt": attempt, "stage": "rewrite", "reason": "invalid_json"}
                )
                continue
            rewritten = normalize_rewritten_form(row, payload.get("rewritten_question") or "")
            reasons = item_validation(row, rewritten)
            if reasons:
                feedback[item_hash] = reasons
                failure_reason_counts.update(reasons)
                private_diagnostics.append(
                    {
                        "item_hash": item_hash,
                        "attempt": attempt,
                        "stage": "deterministic_validation",
                        "rewritten_question": rewritten,
                        "reasons": reasons,
                    }
                )
                continue
            verifier_candidates.append((row, rewritten))

        verifier_payloads = generate_json_objects(
            llm,
            tokenizer,
            [build_verify_messages(row, rewritten) for row, rewritten in verifier_candidates],
            max_tokens=args.verify_max_tokens,
            max_model_len=args.max_model_len,
            seed=args.seed + 1000 + attempt,
        ) if verifier_candidates else []
        accepted_this_attempt: set[str] = set()
        for (row, rewritten), payload in zip(verifier_candidates, verifier_payloads):
            item_hash = str(row["item_hash"])
            if item_hash in accepted_this_attempt:
                continue
            passed, issues = verify_payload(payload or {})
            if not passed:
                feedback[item_hash] = ["semantic_verifier_rejected"] + issues[:4]
                failure_reason_counts["semantic_verifier_rejected"] += 1
                if payload is None:
                    verifier_diagnostic_counts["invalid_json"] += 1
                else:
                    for key in ("equivalent", "same_correct_answer"):
                        if payload.get(key) is not True:
                            verifier_diagnostic_counts[f"not_true:{key}"] += 1
                    if payload.get("issues"):
                        verifier_diagnostic_counts["nonempty_issues"] += 1
                private_diagnostics.append(
                    {
                        "item_hash": item_hash,
                        "attempt": attempt,
                        "stage": "semantic_verification",
                        "rewritten_question": rewritten,
                        "verifier_payload": payload,
                    }
                )
                continue
            completed[item_hash] = {
                "rewrite_version": REWRITE_VERSION,
                "generation_backend": "offline_vllm",
                "generation_model": args.model_label,
                "item_hash": item_hash,
                "original_question_sha256": text_hash(normalize_space(row["question"])),
                "rewritten_question_sha256": text_hash(rewritten),
                "rewritten_question": rewritten,
                "deterministic_validation_passed": True,
                "semantic_validation_passed": True,
                "attempts": attempt,
            }
            accepted_attempt[item_hash] = attempt
            accepted_this_attempt.add(item_hash)

        pending = [row for row in pending if str(row["item_hash"]) not in accepted_this_attempt]
        ordered_partial = [
            completed[str(row["item_hash"])] for row in source_rows if str(row["item_hash"]) in completed
        ]
        write_jsonl_private(incomplete, ordered_partial)
        print(
            json.dumps(
                {
                    "attempt": attempt,
                    "variants_per_item": variants_per_item,
                    "validated": len(completed),
                    "remaining": len(pending),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if pending:
        diagnostics_path = args.private_output.with_suffix(args.private_output.suffix + ".diagnostics.jsonl")
        write_jsonl_private(diagnostics_path, private_diagnostics)
        safe_failure = {
            "schema_version": 1,
            "status": "incomplete",
            "private_content_included": False,
            "source_items": len(source_rows),
            "validated_rewrites": len(completed),
            "remaining": len(pending),
            "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
            "verifier_diagnostic_counts": dict(sorted(verifier_diagnostic_counts.items())),
        }
        failure_path = args.safe_output.with_suffix(args.safe_output.suffix + ".incomplete")
        write_json(failure_path, safe_failure)
        return 2

    ordered = [completed[str(row["item_hash"])] for row in source_rows]
    os.replace(incomplete, args.private_output)
    os.chmod(args.private_output, 0o600)
    attempts = Counter(int(row["attempts"]) for row in ordered)
    generation_models = Counter(str(row["generation_model"]) for row in ordered)
    summary = {
        "schema_version": 1,
        "rewrite_version": REWRITE_VERSION,
        "status": "complete",
        "private_content_included": False,
        "generation_backend": "offline_vllm",
        "model": args.model_label if len(generation_models) == 1 else "mixed_local_models",
        "generation_model_counts": dict(sorted(generation_models.items())),
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": args.max_num_seqs,
        "source_items": len(source_rows),
        "validated_rewrites": len(ordered),
        "unchanged_questions": 0,
        "semantic_validation_passed": len(ordered),
        "attempt_distribution": {str(key): value for key, value in sorted(attempts.items())},
        "retry_failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "source_sha256": sha256_file(args.source),
        "private_output_sha256": sha256_file(args.private_output),
    }
    write_json(args.safe_output, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
