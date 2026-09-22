"""Fixed-source conditional NLL: 24 requests, zero training or answer evaluation.

vLLM requires one decoded token per request; discarded and separately counted.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace/llin-verl-grpo')
OLD = ROOT/'runs/llin-mechanism-run-20260920-01'
CODE = ROOT/'runs/llin-mechanism-package-20260920-02/scripts'
OUT = ROOT/'runs/llin-source-nll-20260922-01'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def save(p, v):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(v, indent=2)+'\n')
    tmp.replace(p)


def stats(values):
    assert values and all(math.isfinite(x) and x >= -1e-5 for x in values)
    return dict(tokens=len(values), nll_sum=sum(values), nll=sum(values)/len(values))


def worker(package, label):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from run_cpt_formal_transfer import model_identity
    identity = read(OLD/(label+'_direct')/'identity.safe.json')
    model = Path(identity['path'])
    assert model_identity(model) == identity
    reg = read(package/'registration.safe.json')
    assert sha(package/'cases.private.json') == reg['cases_sha256']
    cases = read(package/'cases.private.json')
    d = OUT/label
    d.mkdir(exist_ok=True)
    completed = d/'completed.safe.json'
    result = d/'token_nll.private.json'
    reservation = d/'request.reserved.json'
    if completed.exists():
        c = read(completed)
        assert c['requests'] == 6 and sha(result) == c['result_sha256']
        return
    assert not reservation.exists(), 'Unresolved scoring reservation; inspect without resubmission'
    assert not result.exists(), 'Unfinalized result; inspect without resubmission'
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    supplied = []
    masks = []
    for c in cases:
        prefix = tok.encode(tok.apply_chat_template(c['messages'], tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False), add_special_tokens=False)
        encoded = tok(c['body'], add_special_tokens=False, return_offsets_mapping=True)
        ids = prefix + encoded['input_ids'] + [tok.eos_token_id]
        assert ids == c['input_ids'] and len(prefix) == c['answer_start']
        assert len(ids) < 8192 and len(ids)-len(prefix) == c['loss_tokens']
        boundary = c['body'].index('\n\n')+2
        body_mask = [end > boundary for start, end in encoded['offset_mapping']]
        assert any(body_mask) and not all(body_mask)
        masks.append(body_mask)
        supplied.append({'prompt_token_ids': ids})
    assert len(cases) == 6 and sum(c['loss_tokens'] for c in cases) == 357
    llm = LLM(model=str(model), tensor_parallel_size=8, dtype='bfloat16', trust_remote_code=True,
              max_model_len=8192, max_num_seqs=32, gpu_memory_utilization=.8, seed=1024,
              enforce_eager=True, enable_prefix_caching=False)
    save(reservation, dict(requests=6, scored_tokens=357, max_discarded_decode_tokens=6,
                           cases_sha256=sha(package/'cases.private.json'), identity=identity))
    outputs = sorted(llm.generate(supplied, SamplingParams(temperature=0, seed=1024, max_tokens=1,
                                 prompt_logprobs=1, detokenize=False)), key=lambda x: int(x.request_id))
    assert len(outputs) == 6
    records = []
    for c, pr, mask, o in zip(cases, supplied, masks, outputs):
        ids = pr['prompt_token_ids']
        assert list(o.prompt_token_ids) == ids
        lp = o.prompt_logprobs
        assert lp is not None and len(lp) == len(ids)
        vals = []
        for i in range(c['answer_start'], len(ids)):
            assert lp[i] is not None and ids[i] in lp[i], 'Actual reference token logprob missing'
            vals.append(-float(lp[i][ids[i]].logprob))
        assert len(o.outputs[0].token_ids) <= 1
        body = [v for v, b in zip(vals[:-1], mask) if b]
        heading = [v for v, b in zip(vals[:-1], mask) if not b]
        records.append(dict(unit_id=c['unit_id'], body_sha256=c['body_sha256'],
             all_supervised=stats(vals), content=stats(body), heading=stats(heading), eos_nll=vals[-1],
             token_nll=vals, content_mask=mask, discarded_decode_tokens=len(o.outputs[0].token_ids)))
    assert model_identity(model) == identity
    save(result, records)
    save(completed, dict(requests=6, scored_tokens=357, result_sha256=sha(result),
                         discarded_decode_tokens=sum(r['discarded_decode_tokens'] for r in records),
                         training_steps=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--worker', choices=['step120', 'p1', 'K', 'R'])
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    sys.path.insert(0, str(CODE))
    os.umask(0o077)
    if a.worker:
        worker(a.package, a.worker)
        return
    import fcntl
    from cpt_stage_b import device_idle
    from run_cpt_formal_transfer import model_identity
    from run_logistics_cpt_curve_8x import evaluation_env
    OUT.mkdir(exist_ok=True)
    reg = read(a.package/'registration.safe.json')
    assert reg['max_scoring_requests'] == 24 and reg['max_discarded_decode_tokens'] == 24
    assert sha(a.package/'cases.private.json') == reg['cases_sha256']
    assert sha(Path(__file__)) == reg['runner_sha256']
    for name, h in read(OLD/'registration.safe.json')['code'].items():
        assert sha(CODE/name) == h
    for label in ('step120', 'p1', 'K', 'R'):
        identity = read(OLD/(label+'_direct')/'identity.safe.json')
        assert model_identity(Path(identity['path'])) == identity
    with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle, memory = device_idle(subprocess.check_output(['npu-smi', 'info'], text=True))
        assert idle, 'Devices busy; no other task stopped'
        frozen = OUT/'registration.safe.json'
        if frozen.exists():
            assert read(frozen) == reg
        else:
            save(frozen, reg)
        save(OUT/'preflight.safe.json', dict(models_match=True, cases_match=True, runner_match=True,
                                            memory_mb=memory, checked_at=time.time()))
        if a.preflight_only:
            return
        status = OUT/'status.safe.json'
        try:
            for label in ('step120', 'p1', 'K', 'R'):
                save(status, dict(state='running', stage=label, training_steps=0))
                with (OUT/(label+f'.{time.time_ns()}.log')).open('x') as log:
                    subprocess.run([sys.executable, str(Path(__file__)), '--package', str(a.package),
                                    '--worker', label], env=evaluation_env(dict(os.environ)),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            assert sum(read(OUT/m/'completed.safe.json')['requests'] for m in ('step120','p1','K','R')) == 24
            save(status, dict(state='scoring_complete_analysis_pending', scoring_requests=24,
                             free_answer_calls=0, training_steps=0, promotion=False))
        except BaseException as error:
            prev = read(status) if status.exists() else {}
            save(status, dict(state='failed_preserved', stage=prev.get('stage'), error=str(error),
                              recovery='Inspect unresolved reservations; never resubmit automatically.'))
            raise


if __name__ == '__main__':
    main()
