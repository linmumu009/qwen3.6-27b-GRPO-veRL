"""Wait for this exact SFT export then run locked, bounded validation stages."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root=Path('/workspace/llin-verl-grpo');code=Path(__file__).parent
    training=Path('/opt/llin-grounded-supply-sft-cpt-20260909-01')
    out=root/'runs/supply-chain-grounded-sft-validation-20260909-01'
    os.umask(0o077);out.mkdir(exist_ok=False)
    def status(s):(out/'status.txt').write_text(s+'\n')
    status('waiting_for_training_export')
    try:
        official=root/'runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl'
        assert hashlib.sha256(official.read_bytes()).hexdigest()=='b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99'
        deadline=time.monotonic()+8*3600
        while time.monotonic()<deadline:
            state=(training/'status.txt').read_text().strip()
            if state.startswith('failed'):raise RuntimeError('Training failed: '+state)
            if state=='training_export_complete_validation_pending':break
            time.sleep(20)
        else:raise TimeoutError('Training/export exceeded 8-hour validation wait limit')
        with (root/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            env=dict(os.environ,PYTHONPATH=f'/vllm:{root}/runtime:{root}',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',TOKENIZERS_PARALLELISM='true')
            models={'step120':root/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource','cpt116':root/'runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/hf_export_step_116','sft73':training/'hf_export'}
            def run(label,command):
                status(label)
                with (out/(label+'.log')).open('w') as log:
                    subprocess.run(['bash','-c','source /usr/local/Ascend/ascend-toolkit/set_env.sh && exec "$@"','bash',*command],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            for label,model in models.items():
                for stage,cases,repeats in [('official',root/'runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl',3),('tasks',root/'runs/supply-chain-task-baselines-20260909-01/cases.private.jsonl',1)]:
                    if stage=='tasks' and label!='sft73':continue # Frozen baselines already completed.
                    name=label+'_'+stage
                    run(name,['python3',str(code/'run_vllm_logistics_mcq.py'),'--model',str(model),'--model-label',label,'--cases',str(cases),'--tensor-parallel-size','8','--max-model-len','4096','--max-num-seqs','64','--max-output-tokens','96','--gpu-memory-utilization','0.8','--seed','1024','--repeats',str(repeats),'--private-output',str(out/(name+'.private.jsonl')),'--safe-output',str(out/(name+'.safe.json'))])
                run(label+'_general',['python3',str(code/'evaluate_supply_general_regression.py'),'--model',str(model),'--cases',str(root/'runs/supply-chain-general-regression-data-20260909-01/cases.private.jsonl'),'--out',str(out/(label+'_general'))])
            status('domain_tasks_general_complete_agent_and_analysis_pending')
    except BaseException:
        status('failed');raise


if __name__=='__main__':main()
