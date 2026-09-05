"""Score gold answer spans in the exact historical packed training blocks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from scripts.probe_logistics_exam_answers import split_answer


def answer_spans(text, separator):
    spans, offset = [], 0
    for doc in text.split(separator):
        prefix, answer = split_answer(doc)
        spans.append((offset + len(prefix), offset + len(prefix) + len(answer)))
        offset += len(doc) + len(separator)
    return spans


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--corpus-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refusing overwrite')
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from scripts.run_vllm_prompt_nll import chosen_logprob
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cases, hashes = [], {}
    for arm in ('direct','rewritten'):
        path = args.corpus_root / f'{arm}-balanced.cpt.parquet'
        hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_parquet(path)
        if len(frame) != 64:
            raise ValueError('expected historical 64 blocks')
        for text in frame['text']:
            spans = answer_spans(text, tokenizer.eos_token+'\n\n')
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = encoded['input_ids'] + [tokenizer.eos_token_id]
            positions = [i for i,(start,end) in enumerate(encoded['offset_mapping'])
                         if i > 0 and any(end > a and start < b for a,b in spans)]
            cases.append({'arm':arm,'ids':ids,'positions':positions,'items':len(spans)})
    llm = LLM(model=args.model,tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,
              max_model_len=4096,max_num_seqs=64,gpu_memory_utilization=0.8,seed=1024,enforce_eager=True)
    outputs = llm.generate([{'prompt_token_ids':c['ids']} for c in cases],
        SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1,detokenize=False))
    if len(outputs) != len(cases):
        raise ValueError('output count mismatch')
    result = {}
    for arm in ('direct','rewritten'):
        answer_loss = total_loss = 0.0
        answers = total = items = 0
        for c,o in zip(cases,outputs):
            if list(o.prompt_token_ids) != c['ids']:
                raise ValueError('ordering mismatch')
            if c['arm'] != arm:
                continue
            vals = {i:-chosen_logprob(o.prompt_logprobs[i],c['ids'][i]) for i in range(1,len(c['ids']))}
            if not all(math.isfinite(v) and v >= -1e-6 for v in vals.values()):
                raise ValueError('invalid probabilities')
            answer_loss += sum(vals[i] for i in c['positions'])
            total_loss += sum(vals.values())
            answers += len(c['positions'])
            total += len(vals)
            items += c['items']
        if items != 1672 or answers < 1:
            raise ValueError('coverage mismatch')
        result[arm] = {'items':items,'answer_tokens':answers,'scored_total_tokens':total,
                       'answer_nll':answer_loss/answers,'all_target_nll':total_loss/total}
    safe = {'contract':'exam-exact-packed-answer-nll-v1','model':args.label,
            'private_content_included':False,'parquet_hashes':hashes,'results':result}
    os.umask(0o077)
    args.output.write_text(json.dumps(safe,indent=2))
    print(json.dumps(safe),flush=True)


if __name__ == '__main__':
    main()
