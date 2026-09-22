"""Frozen two-context follow-up: 16 free responses plus 16 NLL requests."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/workspace/llin-verl-grpo')
OLD = ROOT/'runs/llin-mechanism-run-20260920-01'
CODE = ROOT/'runs/llin-mechanism-package-20260920-02/scripts'
OUT = ROOT/'runs/llin-context-bridge-20260922-01'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(p)


def worker(package, label):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from run_cpt_formal_transfer import model_identity
    identity = read(OLD/(label+'_direct')/'identity.safe.json')
    assert model_identity(Path(identity['path'])) == identity
    cases = read(package/'cases.private.json')
    reg = read(package/'registration.safe.json')
    assert sha(package/'cases.private.json') == reg['cases_sha256']
    assert sha(Path(identity['path'])/'chat_template.jinja') == reg['chat_template_sha256']
    folder = OUT/label
    folder.mkdir(exist_ok=True)
    pending = []
    for kind in ('nll', 'response'):
        done = folder/(kind+'.completed.safe.json')
        output = folder/(kind+'.private.json')
        if done.exists():
            assert read(done)['requests'] == 8 and sha(output) == read(done)['result_sha256']
        else:
            assert not (folder/(kind+'.reserved.json')).exists(), 'Unresolved reservation: '+kind
            assert not output.exists(), 'Unfinalized output: '+kind
            pending.append(kind)
    if not pending:
        return
    tok = AutoTokenizer.from_pretrained(identity['path'], local_files_only=True)
    prompts, answers = [], []
    for c in cases:
        ids = tok.encode(tok.apply_chat_template(c['messages'], tokenize=False,
                         add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
        answer = tok.encode(c['target_sentence'], add_special_tokens=False)
        assert ids == c['prompt_ids'] and answer == c['answer_ids']
        assert len(ids)+384 < 8192
        prompts.append(ids)
        answers.append(answer)
    llm = LLM(model=identity['path'], tensor_parallel_size=8, dtype='bfloat16',
              trust_remote_code=True, max_model_len=8192, max_num_seqs=32,
              gpu_memory_utilization=.8, seed=1024, enforce_eager=True,
              enable_prefix_caching=False)
    for kind in pending:
        full_ids = [p+a+[tok.eos_token_id] if kind == 'nll' else p for p, a in zip(prompts, answers)]
        save(folder/(kind+'.reserved.json'), dict(requests=8, kind=kind,
            cases_sha256=reg['cases_sha256'], identity=identity, max_tokens=1 if kind == 'nll' else 384))
        params = SamplingParams(temperature=0, seed=1024, max_tokens=1 if kind == 'nll' else 384,
                               prompt_logprobs=1 if kind == 'nll' else None,
                               detokenize=kind == 'response')
        outputs = llm.generate([dict(prompt_token_ids=ids) for ids in full_ids], params)
        # vLLM request IDs increase between batches, but ordering still recovers input order.
        outputs = sorted(outputs, key=lambda o: int(o.request_id))
        assert len(outputs) == 8
        rows = []
        for c, prefix, ids, o in zip(cases, prompts, full_ids, outputs):
            assert list(o.prompt_token_ids) == ids and len(o.outputs) == 1
            row = dict(id=c['id'], unit_id=c['unit_id'], condition=c['condition'],
                       fact_index=c['fact_index'], output_tokens=len(o.outputs[0].token_ids))
            if kind == 'nll':
                assert row['output_tokens'] <= 1
                values = [-float(o.prompt_logprobs[i][ids[i]].logprob) for i in range(len(prefix), len(ids))]
                assert all(math.isfinite(v) and v >= -1e-5 for v in values)
                row.update(token_nll=values, answer_tokens=len(values)-1,
                           answer_nll=sum(values[:-1])/(len(values)-1),
                           first_token_nll=values[0], eos_nll=values[-1])
            else:
                assert row['output_tokens'] <= 384
                row.update(prediction=o.outputs[0].text, finish_reason=o.outputs[0].finish_reason)
            rows.append(row)
        assert model_identity(Path(identity['path'])) == identity
        output = folder/(kind+'.private.json')
        save(output, rows)
        save(folder/(kind+'.completed.safe.json'), dict(requests=8, result_sha256=sha(output),
                    output_tokens=sum(r['output_tokens'] for r in rows), training_steps=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--worker', choices=['K', 'R'])
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
    reg = read(a.package/'registration.safe.json')
    cases = read(a.package/'cases.private.json')
    assert reg['id'] == OUT.name and reg['max_response_calls'] == reg['max_nll_requests'] == 16
    assert len(cases) == 8 and len({c['id'] for c in cases}) == 8
    assert sha(a.package/'cases.private.json') == reg['cases_sha256']
    assert sha(Path(__file__)) == reg['runner_sha256']
    for name, digest in read(OLD/'registration.safe.json')['code'].items():
        assert sha(CODE/name) == digest
    for label in ('K', 'R'):
        identity = read(OLD/(label+'_direct')/'identity.safe.json')
        assert model_identity(Path(identity['path'])) == identity
        assert sha(Path(identity['path'])/'chat_template.jinja') == reg['chat_template_sha256']
    OUT.mkdir(exist_ok=True)
    with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle, memory = device_idle(subprocess.check_output(['npu-smi', 'info'], text=True))
        assert idle, 'Devices busy; no other task stopped'
        assert shutil.disk_usage(OUT).free > 1024**3
        frozen = OUT/'registration.safe.json'
        if frozen.exists():
            assert read(frozen) == reg
        else:
            save(frozen, reg)
        save(OUT/'preflight.safe.json', dict(models_match=True, inputs_match=True,
            code_match=True, checked_at=time.time(), memory_mb=memory,
            free_bytes=shutil.disk_usage(OUT).free))
        if a.preflight_only:
            return
        status = OUT/'status.safe.json'
        try:
            for label in ('K', 'R'):
                save(status, dict(state='running', stage=label, training_steps=0))
                with (OUT/(label+f'.{time.time_ns()}.log')).open('x') as log:
                    subprocess.run([sys.executable, str(Path(__file__)), '--package', str(a.package),
                        '--worker', label], env=evaluation_env(dict(os.environ)),
                        stdout=log, stderr=subprocess.STDOUT, check=True)
            for kind in ('nll', 'response'):
                assert sum(read(OUT/m/(kind+'.completed.safe.json'))['requests'] for m in ('K','R')) == 16
            save(status, dict(state='complete_review_pending', response_calls=16, nll_requests=16,
                             training_steps=0, formal_reruns=0, promotion=False))
        except BaseException as e:
            save(status, dict(state='failed_preserved', error=str(e),
                recovery='Inspect reservations before any resubmission; preserve all completed batches.'))
            raise


if __name__ == '__main__':
    main()
