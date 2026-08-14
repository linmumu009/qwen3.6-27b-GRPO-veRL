#!/usr/bin/env python3
"""Measure single-request vLLM latency and decode throughput by context length.

The benchmark deliberately stores no prompt or generated text.  Every supported
point uses the same fixed output length, concurrency one, and prefix caching off.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TARGETS = (2048, 4096, 8192, 16384, 32768, 40960, 49152, 65536)


@dataclass(frozen=True)
class BenchmarkCase:
    target_total_tokens: int
    prompt_tokens: int
    output_tokens: int
    supported: bool
    reason: str | None = None


def build_cases(
    targets: Iterable[int], output_tokens: int, max_model_len: int
) -> list[BenchmarkCase]:
    if output_tokens < 2:
        raise ValueError("output_tokens must be at least 2 for decode-rate timing")
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")

    cases: list[BenchmarkCase] = []
    for target in targets:
        if target <= output_tokens:
            cases.append(
                BenchmarkCase(
                    target,
                    max(0, target - output_tokens),
                    output_tokens,
                    False,
                    "target_must_exceed_output_tokens",
                )
            )
        elif target > max_model_len:
            cases.append(
                BenchmarkCase(
                    target,
                    target - output_tokens,
                    output_tokens,
                    False,
                    "exceeds_max_model_len",
                )
            )
        else:
            cases.append(
                BenchmarkCase(
                    target,
                    target - output_tokens,
                    output_tokens,
                    True,
                )
            )
    return cases


def _finite_median(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def summarize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    unsupported: dict[int, dict[str, Any]] = {}
    for row in rows:
        target = int(row["target_total_tokens"])
        if row["status"] == "ok":
            grouped.setdefault(target, []).append(row)
        else:
            unsupported.setdefault(target, row)

    summary: list[dict[str, Any]] = []
    for target in sorted(set(grouped) | set(unsupported)):
        valid = grouped.get(target, [])
        if not valid:
            row = unsupported[target]
            summary.append(
                {
                    "target_total_tokens": target,
                    "prompt_tokens": row["prompt_tokens"],
                    "output_tokens": row["output_tokens"],
                    "status": row["status"],
                    "reason": row.get("reason"),
                    "n": 0,
                }
            )
            continue
        summary.append(
            {
                "target_total_tokens": target,
                "prompt_tokens": valid[0]["prompt_tokens"],
                "output_tokens": valid[0]["output_tokens"],
                "status": "ok",
                "n": len(valid),
                "ttft_s_median": _finite_median(row["ttft_s"] for row in valid),
                "decode_s_median": _finite_median(
                    row["decode_s"] for row in valid
                ),
                "total_s_median": _finite_median(row["total_s"] for row in valid),
                "decode_tps_median": _finite_median(
                    row["decode_tps"] for row in valid
                ),
                "e2e_tps_median": _finite_median(row["e2e_tps"] for row in valid),
                "prompt_tokens_per_ttft_s_median": _finite_median(
                    row["prompt_tokens_per_ttft_s"] for row in valid
                ),
            }
        )
    return summary


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


async def _measure_one(
    engine: Any,
    sampling_params: Any,
    filler_token_id: int,
    case: BenchmarkCase,
    repeat: int,
    run_id: str,
) -> dict[str, Any]:
    prompt_token_ids = [filler_token_id] * case.prompt_tokens
    request_id = f"{run_id}-{case.target_total_tokens}-{repeat}-{uuid.uuid4().hex}"
    start = time.perf_counter()
    first_token_at: float | None = None
    final_output_tokens = 0

    async for output in engine.generate(
        {"prompt_token_ids": prompt_token_ids}, sampling_params, request_id
    ):
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now
        if output.outputs:
            final_output_tokens = len(output.outputs[0].token_ids)

    finished = time.perf_counter()
    if first_token_at is None or final_output_tokens < 2:
        raise RuntimeError(
            f"request produced insufficient streamed output: {final_output_tokens}"
        )
    ttft_s = first_token_at - start
    decode_s = finished - first_token_at
    total_s = finished - start
    return {
        "target_total_tokens": case.target_total_tokens,
        "prompt_tokens": case.prompt_tokens,
        "output_tokens": final_output_tokens,
        "repeat": repeat,
        "status": "ok",
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "decode_tps": (final_output_tokens - 1) / decode_s,
        "e2e_tps": final_output_tokens / total_s,
        "prompt_tokens_per_ttft_s": case.prompt_tokens / ttft_s,
    }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    cases = build_cases(args.targets, args.output_tokens, args.max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=False, local_files_only=True
    )
    filler_ids = tokenizer.encode(" benchmark", add_special_tokens=False)
    if not filler_ids:
        raise RuntimeError("tokenizer produced no filler token")
    filler_token_id = int(filler_ids[0])
    if filler_token_id in set(tokenizer.all_special_ids):
        raise RuntimeError("selected filler token is special")

    engine_args = AsyncEngineArgs(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        load_format="safetensors",
        disable_custom_all_reduce=True,
        stream_interval=1,
        use_tqdm_on_load=False,
        seed=args.seed,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        min_tokens=args.output_tokens,
        ignore_eos=True,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, Any]] = []
    supported_cases = [case for case in cases if case.supported]
    if supported_cases:
        # A short excluded request pays one-time graph/runtime initialization cost.
        warmup_case = BenchmarkCase(
            target_total_tokens=args.warmup_prompt_tokens + args.output_tokens,
            prompt_tokens=args.warmup_prompt_tokens,
            output_tokens=args.output_tokens,
            supported=True,
        )
        await _measure_one(
            engine,
            sampling_params,
            filler_token_id,
            warmup_case,
            repeat=0,
            run_id=f"{run_id}-warmup",
        )

    for case in cases:
        if not case.supported:
            rows.append(
                {
                    "target_total_tokens": case.target_total_tokens,
                    "prompt_tokens": case.prompt_tokens,
                    "output_tokens": case.output_tokens,
                    "repeat": None,
                    "status": "unsupported",
                    "reason": case.reason,
                }
            )
            continue
        for repeat in range(1, args.repeats + 1):
            row = await _measure_one(
                engine,
                sampling_params,
                filler_token_id,
                case,
                repeat,
                run_id,
            )
            rows.append(row)
            partial = {
                "schema_version": 1,
                "status": "running",
                "started_at_utc": run_id,
                "measurement_contract": measurement_contract(args),
                "rows": rows,
                "summary": summarize_rows(rows),
            }
            atomic_write_json(args.output, partial)

    payload = {
        "schema_version": 1,
        "status": "complete",
        "started_at_utc": run_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_contract": measurement_contract(args),
        "rows": rows,
        "summary": summarize_rows(rows),
    }
    atomic_write_json(args.output, payload)
    return payload


def measurement_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "context_definition": "prompt_tokens_plus_generated_tokens",
        "concurrency": 1,
        "fixed_requested_output_tokens": args.output_tokens,
        "repeats_per_supported_point": args.repeats,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "prefix_caching": False,
        "chunked_prefill": True,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "temperature": 0.0,
        "ignore_eos": True,
        "generated_text_persisted": False,
        "decode_tps_definition": "(generated_tokens_minus_first_token)/time_after_first_token",
        "ttft_definition": "request_start_to_first_streamed_token",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--warmup-prompt-tokens", type=int, default=1792)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run_benchmark(args))
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
