"""Measure answer-only learning on both frozen training renderings, locally."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def split_answer(document: str) -> tuple[str, str]:
    for marker in ("\nCorrect answers:", "\nCorrect answer:", "\nThis statement is"):
        if marker in document:
            head, answer = document.rsplit(marker, 1)
            if not answer.strip():
                raise ValueError("empty answer")
            return head + marker, answer
    raise ValueError("unknown historical training template")


def encode_case(tokenizer, row):
    prefix, answer = split_answer(row["training_document"])
    encoded = tokenizer(prefix + answer, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    start = next(i for i, (_, end) in enumerate(encoded["offset_mapping"]) if end > len(prefix))
    if start < 1 or start >= len(ids):
        raise ValueError("invalid answer boundary")
    return {"item_hash": row["item_hash"], "arm": row["arm"], "ids": ids,
            "start": start, "boundary_crossing": encoded["offset_mapping"][start][0] < len(prefix)}


def summarize(rows):
    result = {}
    for arm in sorted({row["arm"] for row in rows}):
        part = [row for row in rows if row["arm"] == arm]
        count = sum(row["answer_tokens"] for row in part)
        result[arm] = {"items": len(part), "answer_tokens": count,
            "answer_nll": sum(row["answer_nll_sum"] for row in part) / count,
            "prefix_nll": sum(row["prefix_nll_sum"] for row in part) / sum(row["prefix_tokens"] for row in part),
            "answer_teacher_forced_top1": sum(row["answer_top1"] for row in part) / count,
            "greedy_gold_token_prefix_exact": sum(row["greedy_gold_token_prefix_exact"] for row in part),
            "generation_censored": sum(row["generation_censored"] for row in part),
            "boundary_crossings": sum(row["boundary_crossing"] for row in part)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--items", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refusing overwrite")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from scripts.run_vllm_prompt_nll import chosen_logprob
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cases = []
    hashes = {}
    for path in args.items:
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        cases.extend(encode_case(tokenizer, row) for row in (rows[:args.limit] if args.limit else rows))
    if len({(c['arm'], c['item_hash']) for c in cases}) != len(cases):
        raise ValueError("duplicate cases")
    if max(len(c['ids']) for c in cases) + 1 > 4096:
        raise ValueError("context exceeded")
    llm = LLM(model=args.model, tensor_parallel_size=8, dtype="bfloat16", trust_remote_code=True,
              max_model_len=4096, max_num_seqs=64, gpu_memory_utilization=0.8, seed=1024, enforce_eager=True)
    rows = []
    for offset in range(0, len(cases), 64):
        batch = cases[offset:offset+64]
        scored = llm.generate([{"prompt_token_ids": c['ids']} for c in batch],
            SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1, detokenize=False))
        generated = llm.generate([{"prompt_token_ids": c['ids'][:c['start']]} for c in batch],
            [SamplingParams(temperature=0, max_tokens=min(512, len(c['ids'])-c['start']),
                            detokenize=False) for c in batch])
        if len(scored) != len(batch) or len(generated) != len(batch):
            raise ValueError("missing outputs")
        for case, score, gen in zip(batch, scored, generated):
            ids, start = case['ids'], case['start']
            if list(score.prompt_token_ids) != ids or list(gen.prompt_token_ids) != ids[:start]:
                raise ValueError("output ordering mismatch")
            logs = score.prompt_logprobs
            vals = [-chosen_logprob(logs[i], ids[i]) for i in range(1, len(ids))]
            if not all(math.isfinite(v) and v >= -1e-6 for v in vals):
                raise ValueError("invalid log probabilities")
            target = ids[start:]
            pred = list(gen.outputs[0].token_ids)
            rows.append({"arm": case['arm'], "item_hash": case['item_hash'],
                "answer_tokens": len(target), "prefix_tokens": start-1,
                "answer_nll_sum": sum(vals[start-1:]), "prefix_nll_sum": sum(vals[:start-1]),
                "answer_top1": sum(ids[i] == max(logs[i], key=lambda k: logs[i][k].logprob) for i in range(start, len(ids))),
                "greedy_gold_token_prefix_exact": pred[:len(target)] == target,
                "generation_censored": len(target) > 512, "boundary_crossing": case['boundary_crossing']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)
    args.output.write_text(json.dumps({"rows": rows, "source_hashes": hashes}, indent=2))
    safe = {"contract": "exam-answer-probe-v1", "model": args.label, "source_hashes": hashes,
        "private_content_included": False, "context": "individual document without chat template or preceding packed items",
        "generation_metric": "exact gold token prefix; extra tokens allowed; not semantic accuracy",
        "results": summarize(rows)}
    args.output.with_suffix('.safe.json').write_text(json.dumps(safe, indent=2))
    print(json.dumps(safe), flush=True)


if __name__ == '__main__':
    main()
