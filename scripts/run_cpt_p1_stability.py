"""Exactly one additional full P1 probe per frozen model; no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/workspace/llin-verl-grpo')
CASE_SHA = 'b74ed230308e4d0da500263817280bf4222d1f5fabf38c315c22f4fc26f8ff3c'
MODELS = {
    'baseline_repeat': ROOT/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource',
    'post_repeat': ROOT/'runs/llin-transfer-p1-run-20260916-02/training/llin-step120-p1-hf',
}


def main():
    import fcntl
    p=argparse.ArgumentParser()
    for name in ('code','cases','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();os.umask(0o077)
    assert hashlib.sha256(a.cases.read_bytes()).hexdigest()==CASE_SHA
    assert all(m.is_dir() for m in MODELS.values())
    a.out.mkdir(exist_ok=False)
    def status(stage,**kwargs):
        (a.out/'status.safe.json').write_text(json.dumps(dict(status=stage,training_running=False,**kwargs),indent=2)+'\n')
    status('waiting_for_lock')
    try:
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                VLLM_WORKER_MULTIPROC_METHOD='spawn',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
            (a.out/'registration.safe.json').write_text(json.dumps(dict(cases_sha256=CASE_SHA,
                models={k:str(v) for k,v in MODELS.items()},requests=360,repeats_per_model=1,
                training=False,original_results_preserved=True,selection='all original 180 cases in original order',
                purpose='Bounded repeat-inference stability check, not a new training seed'),indent=2)+'\n')
            for stage,model in MODELS.items():
                status(stage)
                with (a.out/(stage+'.log')).open('wb') as log:
                    subprocess.run([sys.executable,str(a.code/'run_cpt_pilot_probe.py'),
                        '--model',str(model),'--cases',str(a.cases),'--out',str(a.out/stage)],
                        env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            status('completed_pending_independent_verification',requests=360)
    except BaseException as exc:
        status('failed',error=type(exc).__name__,detail=str(exc));raise


if __name__=='__main__':main()
