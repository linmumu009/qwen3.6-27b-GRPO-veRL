#!/usr/bin/env python3
"""Load an exported HF checkpoint in a fresh vLLM process and generate text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="Reply with exactly: HF export works")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model.resolve()),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=False,
    )
    text = outputs[0].outputs[0].text
    result = {
        "valid": bool(text.strip()),
        "model": str(args.model.resolve()),
        "tensor_parallel_size": args.tensor_parallel_size,
        "prompt": args.prompt,
        "generated_text": text,
        "generated_token_count": len(outputs[0].outputs[0].token_ids),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
