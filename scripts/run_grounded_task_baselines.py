"""Run frozen textbook learning/development baselines, without training."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    root=Path('/workspace/llin-verl-grpo')
    data=root/'runs/supply-chain-task-training-data-20260909-01'
    out=root/'runs/supply-chain-task-baselines-20260909-01'
    code=Path(__file__).parent
    os.umask(0o077)
    with (root/'runs/.logistics-exam-cpt.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir(exist_ok=False)
        def status(s): (out/'status.txt').write_text(s+'\n')
        try:
            cases=out/'cases.private.jsonl'
            cases.write_bytes((data/'train.cases.private.jsonl').read_bytes()+(data/'dev.cases.private.jsonl').read_bytes())
            models={'step120':root/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource',
                    'cpt116':root/'runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/hf_export_step_116'}
            (out/'protocol.safe.json').write_text(json.dumps(dict(cases_sha256=hashlib.sha256(cases.read_bytes()).hexdigest(),models={k:str(v) for k,v in models.items()},temperature=0,seed=1024,repeats=1,max_output_tokens=96,max_model_len=4096,training=False,checkpoint_selection=False,scope='402 learning checks and 74 held-out chapter objective questions; excludes 31 free-answer development questions'),indent=2))
            env=dict(os.environ,PYTHONPATH=f'/vllm:{root}/runtime:{root}',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',TOKENIZERS_PARALLELISM='true')
            for label,model in models.items():
                status('evaluating_'+label)
                command=['python3',str(code/'run_vllm_logistics_mcq.py'),'--model',str(model),'--model-label',label,'--cases',str(cases),'--tensor-parallel-size','8','--max-model-len','4096','--max-num-seqs','64','--max-output-tokens','96','--gpu-memory-utilization','0.8','--seed','1024','--repeats','1','--private-output',str(out/(label+'.private.jsonl')),'--safe-output',str(out/(label+'.safe.json'))]
                with (out/(label+'.log')).open('w') as log:
                    subprocess.run(['bash','-c','source /usr/local/Ascend/ascend-toolkit/set_env.sh && exec "$@"','bash',*command],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            status('baselines_complete_training_not_started')
        except BaseException:
            status('failed');raise


if __name__=='__main__':main()
