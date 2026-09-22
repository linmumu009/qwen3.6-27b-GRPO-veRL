"""Run only the 38 frozen new response calls; retain four historical observations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/workspace/llin-verl-grpo')
OLD = ROOT/'runs/llin-mechanism-run-20260920-01'
BRIDGE = ROOT/'runs/llin-context-bridge-20260922-01'
CODE = ROOT/'runs/llin-mechanism-package-20260920-02/scripts'
OUT = ROOT/'runs/llin-rule-disambiguation-20260922-01'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(p)


def worker(package, label):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from run_cpt_formal_transfer import model_identity
    reg = read(package/'registration.safe.json')
    assert sha(package/'cases.private.json') == reg['cases_sha256']
    identity = read(OLD/(label+'_direct')/'identity.safe.json')
    assert model_identity(Path(identity['path'])) == identity
    assert sha(Path(identity['path'])/'chat_template.jinja') == reg['chat_template_sha256']
    cases = [c for c in read(package/'cases.private.json') if c['reused_case_id'] is None]
    assert len(cases)==19
    folder = OUT/label
    folder.mkdir(exist_ok=True)
    output, done, reserved = [folder/n for n in ('responses.private.json','completed.safe.json','reserved.safe.json')]
    if done.exists():
        assert read(done)['requests']==19 and sha(output)==read(done)['result_sha256']
        return
    assert not reserved.exists() and not output.exists(), 'Unresolved batch; inspect without resubmission'
    tok = AutoTokenizer.from_pretrained(identity['path'],local_files_only=True)
    inputs=[]
    for c in cases:
        ids=tok.encode(tok.apply_chat_template(c['messages'],tokenize=False,add_generation_prompt=True,
                       enable_thinking=False),add_special_tokens=False)
        assert ids==c['prompt_ids'] and len(ids)+384<8192
        inputs.append(dict(prompt_token_ids=ids))
    llm=LLM(model=identity['path'],tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,
            max_model_len=8192,max_num_seqs=32,gpu_memory_utilization=.8,seed=1024,
            enforce_eager=True,enable_prefix_caching=False)
    save(reserved,dict(requests=19,cases_sha256=reg['cases_sha256'],identity=identity,max_tokens=384))
    outputs=sorted(llm.generate(inputs,SamplingParams(temperature=0,seed=1024,max_tokens=384)),
                   key=lambda o:int(o.request_id))
    assert len(outputs)==19
    rows=[]
    for c,o in zip(cases,outputs):
        assert list(o.prompt_token_ids)==c['prompt_ids'] and len(o.outputs)==1
        r={k:c[k] for k in ('id','unit_id','module','condition','repeat','fact_index')}
        r.update(prediction=o.outputs[0].text,output_tokens=len(o.outputs[0].token_ids),
                 finish_reason=o.outputs[0].finish_reason)
        assert r['output_tokens']<=384
        rows.append(r)
    assert model_identity(Path(identity['path']))==identity
    save(output,rows)
    save(done,dict(requests=19,result_sha256=sha(output),output_tokens=sum(r['output_tokens'] for r in rows),training_steps=0))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--worker',choices=['K','R'])
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    sys.path.insert(0,str(CODE))
    os.umask(0o077)
    if a.worker:
        worker(a.package,a.worker)
        return
    import fcntl
    from cpt_stage_b import device_idle
    from run_cpt_formal_transfer import model_identity
    from run_logistics_cpt_curve_8x import evaluation_env
    reg=read(a.package/'registration.safe.json')
    assert reg['id']==OUT.name and reg['max_new_response_calls']==38
    assert sha(a.package/'cases.private.json')==reg['cases_sha256'] and sha(Path(__file__))==reg['runner_sha256']
    for name,digest in read(OLD/'registration.safe.json')['code'].items():
        assert sha(CODE/name)==digest
    for label in ('K','R'):
        identity=read(OLD/(label+'_direct')/'identity.safe.json')
        assert model_identity(Path(identity['path']))==identity
        assert sha(Path(identity['path'])/'chat_template.jinja')==reg['chat_template_sha256']
        assert sha(BRIDGE/label/'response.private.json')==reg['prior_raw_sha256'][label+'/response']
    OUT.mkdir(exist_ok=True)
    with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle,memory=device_idle(subprocess.check_output(['npu-smi','info'],text=True))
        assert idle, 'Devices busy; no other task stopped'
        assert shutil.disk_usage(OUT).free>1024**3
        frozen=OUT/'registration.safe.json'
        if frozen.exists(): assert read(frozen)==reg
        else: save(frozen,reg)
        save(OUT/'preflight.safe.json',dict(checked_at=time.time(),models_match=True,inputs_match=True,
            code_match=True,prior_results_match=True,memory_mb=memory,free_bytes=shutil.disk_usage(OUT).free))
        if a.preflight_only:return
        status=OUT/'status.safe.json'
        try:
            for label in ('K','R'):
                save(status,dict(state='running',stage=label,training_steps=0))
                with (OUT/(label+f'.{time.time_ns()}.log')).open('x') as log:
                    subprocess.run([sys.executable,str(Path(__file__)),'--package',str(a.package),'--worker',label],
                        env=evaluation_env(dict(os.environ)),stdout=log,stderr=subprocess.STDOUT,check=True)
            assert sum(read(OUT/m/'completed.safe.json')['requests'] for m in ('K','R'))==38
            save(status,dict(state='complete_review_pending',new_response_calls=38,reused_observations=4,
                training_steps=0,formal_reruns=0,promotion=False))
        except BaseException as e:
            save(status,dict(state='failed_preserved',error=str(e),recovery='Inspect reservations; never automatically resubmit unresolved calls.'))
            raise


if __name__=='__main__':
    main()
