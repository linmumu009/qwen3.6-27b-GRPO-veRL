"""One training-fit probe per existing model; never trains or promotes a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/workspace/llin-verl-grpo')
MODELS=dict(base=ROOT/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource',
    p1=ROOT/'runs/llin-transfer-p1-run-20260916-02/training/llin-step120-p1-hf',
    p2=ROOT/'runs/llin-transfer-p2-run-20260917-01/training/llin-step120-p2-hf')

def main():
    import fcntl
    p=argparse.ArgumentParser()
    for k in ('packet','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();code=Path(__file__).resolve().parent;os.umask(0o077)
    reg=json.loads((a.packet/'coverage.safe.json').read_text())
    assert hashlib.sha256((a.packet/'cases.private.jsonl').read_bytes()).hexdigest()==reg['cases_sha256']
    assert a.out.parent.resolve()==ROOT/'runs' and a.out.name.startswith('llin-replay-fit-')
    a.out.mkdir(exist_ok=False)
    def status(stage,**more):
        (a.out/'status.safe.json').write_text(json.dumps(dict(status=stage,training=False,**more),indent=2)+'\n')
    status('acquiring_lock')
    try:
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Match all 135 probes to actual P1 training messages, not merely a reconstructed label file.
            from run_vllm_logistics_mcq import load_items
            from evaluate_logistics_knowledge import build_messages
            actual={r['id']:r['messages'] for r in [json.loads(s) for s in
                (ROOT/'runs/llin-transfer-p1-run-20260916-02/data/train.messages.private.jsonl').read_text().splitlines()]}
            items=load_items(a.packet/'cases.private.jsonl');assert len(items)==135
            for item in items:
                assert build_messages(item)==actual[item.source_id][:-1]
                assert json.loads(actual[item.source_id][-1]['content'])=={'answers':list(item.expected)}
            (a.out/'registration.safe.json').write_text(json.dumps(dict(cases_sha256=reg['cases_sha256'],
                models={k:str(v) for k,v in MODELS.items()},requests=405,training=False,
                actual_training_messages_matched=135,purpose='training fit only; no generalization or retention claim'),indent=2)+'\n')
            env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                VLLM_WORKER_MULTIPROC_METHOD='spawn',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
            for name,model in MODELS.items():
                status(name)
                with (a.out/(name+'.log')).open('wb') as log:
                    subprocess.run([sys.executable,str(code/'run_cpt_pilot_probe.py'),'--replay-fit',
                        '--model',str(model),'--cases',str(a.packet/'cases.private.jsonl'),'--out',str(a.out/name)],
                        env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            status('completed_pending_verification',requests=405)
    except BaseException as exc:
        status('failed',error=type(exc).__name__,detail=str(exc));raise

if __name__=='__main__':main()
