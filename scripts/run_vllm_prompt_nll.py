#!/usr/bin/env python3
"""Compute fixed-passage prompt NLL with vLLM without exposing source text."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from scripts.prepare_book_nll_cases import CONTRACT, write_json


RESULT_CONTRACT = "book-fixed-passage-prompt-nll-v1"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def chosen_logprob(entry: dict[Any, Any] | None, token_id: int) -> float:
    if entry is None:
        raise ValueError("missing prompt logprob entry")
    value = entry.get(token_id)
    if value is None:
        value = entry.get(str(token_id))
    if value is None:
        raise ValueError(f"chosen token {token_id} absent from prompt logprobs")
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict):
        return float(value["logprob"])
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    if payload.get("contract") != CONTRACT:
        raise ValueError("case contract mismatch")
    cases = payload["cases"]
    if not cases:
        raise ValueError("no cases")

    from vllm import LLM, SamplingParams

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
    prompts = [{"prompt_token_ids": row["token_ids"]} for row in cases]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, detokenize=False),
        use_tqdm=True,
    )
    if len(outputs) != len(cases):
        raise ValueError("output count mismatch")
    by_request_id = {int(output.request_id): output for output in outputs}
    private_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        output = by_request_id.get(index)
        if output is None:
            raise ValueError(f"missing output request {index}")
        token_ids = [int(value) for value in case["token_ids"]]
        logprobs = output.prompt_logprobs
        if logprobs is None or len(logprobs) != len(token_ids):
            raise ValueError(f"prompt logprob length mismatch for {case['case_id']}")
        values = [chosen_logprob(logprobs[position], token_ids[position]) for position in range(1, len(token_ids))]
        if not all(math.isfinite(value) and value <= 1e-6 for value in values):
            raise ValueError(f"invalid logprob for {case['case_id']}")
        nll = -statistics.fmean(values)
        private_rows.append(
            {
                "case_id": case["case_id"],
                "token_ids_sha256": case["token_ids_sha256"],
                "scored_tokens": len(values),
                "nll": nll,
                "sum_logprob": sum(values),
            }
        )
    private_payload = {
        "contract": RESULT_CONTRACT,
        "model_label": args.model_label,
        "rows": private_rows,
    }
    write_json(args.private_output, private_payload)
    try:
        os.chmod(args.private_output, 0o600)
    except OSError:
        pass
    total_tokens = sum(int(row["scored_tokens"]) for row in private_rows)
    token_nll = -sum(float(row["sum_logprob"]) for row in private_rows) / total_tokens
    case_nlls = [float(row["nll"]) for row in private_rows]
    case_hashes = [str(row["token_ids_sha256"]) for row in private_rows]
    safe = {
        "schema_version": 1,
        "contract": RESULT_CONTRACT,
        "model_label": args.model_label,
        "model_path_sha256": hashlib.sha256(str(args.model).encode("utf-8")).hexdigest(),
        "source_content_included": False,
        "case_level_values_included": False,
        "cases": len(private_rows),
        "scored_tokens": total_tokens,
        "token_weighted_nll": round(token_nll, 9),
        "perplexity": round(math.exp(token_nll), 9),
        "case_nll": {
            "mean": round(statistics.fmean(case_nlls), 9),
            "median": round(statistics.median(case_nlls), 9),
            "p05": round(percentile(case_nlls, 0.05), 9),
            "p95": round(percentile(case_nlls, 0.95), 9),
        },
        "case_hashes_sha256": hashlib.sha256("\n".join(case_hashes).encode("ascii")).hexdigest(),
        "runtime": {
            "backend": "vllm_prompt_logprobs",
            "dtype": "bfloat16",
            "tensor_parallel_size": args.tensor_parallel_size,
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
