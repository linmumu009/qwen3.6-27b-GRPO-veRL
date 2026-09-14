"""Gate LoRA-CPT with frozen-weight audits and export smoke before one full epoch."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/workspace/llin-verl-grpo')
BASE = ROOT/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource'
DIST = ROOT/'runs/llin-step120-opensource-20260825-02/checkpoints/global_step_120/actor/model/dist_ckpt'


def stage_environment(environment, stage):
    result=dict(environment)
    for key in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'):
        result.pop(key,None)
    if stage in ('gate_inference','official_evaluation'):
        from run_logistics_cpt_curve_8x import evaluation_env
        result=evaluation_env(result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--gate-pid', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--passed-gate', type=Path, help='Reuse completed gate after a pre-training initialization failure')
    a = p.parse_args()
    assert a.out.parent == Path('/opt') and a.out.name.startswith('llin-lora-cpt-')
    code = Path(__file__).resolve().parent
    sys.path[:0] = [str(code.parent), str(code)]
    os.umask(0o077)
    a.out.mkdir(exist_ok=False)

    def save(name, value):
        (a.out/name).write_text(json.dumps(value, indent=2))

    save('registration.safe.json',dict(rank=64,alpha=128,dropout=0.0,
        target_scope='language_model attention projections and MLP projections',
        starting_model=str(BASE),fresh_adapter_and_optimizer=True,records=4912,
        steps=614,epochs=1,batch_size=8,sequence_tokens=1872924,loss_tokens=1868012,
        learning_rate=1e-5,minimum_learning_rate=1e-6,
        lr_note='First registered LoRA candidate, not an empirically optimized learning rate',
        requires_successful_gate_and_export_smoke=True,formal_training_started=False))

    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(code.parent),str(ROOT),
        str(ROOT/'reference/Megatron-Bridge-de93536e/src'),str(ROOT/'runtime'),'/verl',os.environ.get('PYTHONPATH','')]))

    def run(stage, command, extra=None):
        save('status.safe.json', dict(status=stage, promotion=False))
        with (a.out/(stage+'.log')).open('x') as log:
            subprocess.run(['bash','-c','source /usr/local/Ascend/ascend-toolkit/set_env.sh; exec "$@"','--',*command],
                           env=stage_environment(dict(env,**(extra or {})),stage),stdout=log,stderr=subprocess.STDOUT,check=True)

    def audit_training(directory, steps):
        from summarize_logistics_cpt_run import parse_metrics
        logs='\n'.join(f.read_text(errors='replace') for f in (directory/'torchrun_logs').glob('*/attempt_0/*/stdout.log'))
        metrics=parse_metrics(logs)
        assert sorted(metrics)==list(range(1,steps+1)), 'Incomplete training steps'
        for metric in metrics.values():
            assert all(math.isfinite(metric[k]) for k in ['train/loss','train/grad_norm','train/lr'])
        audits=[]
        for rank in range(16):
            before=json.loads((directory/f'weight_audit/rank{rank}.before.safe.json').read_text())
            after=json.loads((directory/f'weight_audit/rank{rank}.after.safe.json').read_text())
            assert before['base_sha256']==after['base_sha256'] and after['base_unchanged']
            assert before['adapter_sha256']!=after['adapter_sha256'] and after['adapter_changed']
            audits.append(after)
        total=sum(int(m['train/global_tokens']) for m in metrics.values())
        if steps==614:assert total==1872924
        return dict(steps=steps,sequence_tokens=total,all_16_ranks_frozen_base_unchanged=True,
                    all_16_ranks_adapter_updated=True,rank_audits=audits)

    def export(stage, checkpoint, output):
        run(stage,[sys.executable,str(code/'export_lora_cpt_to_hf.py'),'--actor-checkpoint',str(checkpoint),
            '--base-model',str(BASE),'--base-dist',str(DIST),'--output-dir',str(output)])

    try:
        save('status.safe.json',dict(status='waiting_for_two_step_gate'))
        start=time.monotonic()
        while True:
            proc=Path(f'/proc/{a.gate_pid}')
            if not proc.exists() or proc.joinpath('stat').read_text().split(') ')[1].startswith('Z'):break
            assert b'run_lora_cpt_gate.sh' in proc.joinpath('cmdline').read_bytes(), 'Gate PID reused'
            if time.monotonic()-start>1800:raise TimeoutError('Gate still active; do not restart it')
            time.sleep(5)
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            save('gate_training_audit.safe.json',audit_training(a.gate,2))
            assert shutil.disk_usage(a.out).free>180_000_000_000, 'Insufficient export reserve'
            if a.passed_gate:
                previous=a.passed_gate
                assert json.loads((previous/'status.safe.json').read_text())['status']=='failed'
                assert not list((previous/'training/checkpoints').glob('global_step_*'))
                assert not list((previous/'training/weight_audit').glob('*.before.safe.json')), 'Training initialized; use a proper resume instead'
                gate=json.loads((previous/'gate_pass.safe.json').read_text())
                assert all(gate.get(k) is True for k in ['training','base_frozen','adapter_updated','export_verified','inference_smoke'])
                manifest=json.loads((previous/'llin-lora-cpt-gate-hf/llin_export_manifest.json').read_text())
                assert manifest['verification']['valid'] and manifest['merge_math_verified']
                rows=[json.loads(x) for x in (previous/'smoke.private.jsonl').read_text().splitlines()]
                assert len(rows)==8 and all(r['parse_ok'] and not r['error'] for r in rows)
                save('gate_pass.safe.json',dict(gate,reused_from=str(previous)))
            else:
                gate_model=a.out/'llin-lora-cpt-gate-hf'
                export('gate_export',a.gate/'checkpoints/global_step_2',gate_model)
                source=Path('/opt/llin-s3-data-20260914-01/cases.private.jsonl')
                smoke=a.out/'smoke_cases.private.jsonl'
                smoke.write_text('\n'.join(source.read_text().splitlines()[:8])+'\n')
                from run_logistics_cpt_curve_8x import evaluation_env
                # Keep device selection identical to the existing evaluation engine.
                # Evaluation visibility is scoped by stage_environment; never mutate training env.
                run('gate_inference',[sys.executable,str(code/'run_vllm_logistics_mcq.py'),
                    '--model',str(gate_model),'--model-label','llin-lora-gate','--cases',str(smoke),
                    '--tensor-parallel-size','8','--max-model-len','8192','--max-num-seqs','32',
                    '--max-output-tokens','96','--gpu-memory-utilization','0.8','--seed','1024','--repeats','1',
                    '--private-output',str(a.out/'smoke.private.jsonl'),'--safe-output',str(a.out/'smoke.safe.json')])
                rows=[json.loads(x) for x in (a.out/'smoke.private.jsonl').read_text().splitlines()]
                assert len(rows)==8 and all(r['parse_ok'] and not r['error'] for r in rows)
                save('gate_pass.safe.json',dict(training=True,base_frozen=True,adapter_updated=True,
                    export_verified=True,inference_smoke=True,official_score_claim=False))
            training=a.out/'training'
            run('formal_training',['bash',str(code/'run_lora_cpt_gate.sh')],dict(
                TRAIN_FILE=str(ROOT/'runs/llin-knowledge-complete-20260911/train.parquet'),
                EXPECTED_CONTENT_TOKENS='1868012',EXPECTED_TRAIN_SHA='6ffa16684e617f657293918bf1d89ae8930ec24e0e08da346f175c4710c1f1c8',
                OUTPUT_DIR=str(training),RUN_NAME='llin-step120-lora-cpt-r64-1epoch',TOTAL_STEPS='614',
                CHECKPOINT_STEPS='614',LEARNING_RATE='1e-5',MIN_LEARNING_RATE='1e-6'))
            save('formal_training_audit.safe.json',audit_training(training,614))
            model=a.out/'llin-step120-lora-cpt-r64-1epoch-hf'
            export('formal_export',training/'checkpoints/global_step_614',model)
            run('official_evaluation',[sys.executable,str(code/'run_logistics_cpt_curve_8x.py'),
                '--evaluate',str(model),'--output',str(a.out/'evaluation')])
            run('independent_verification',[sys.executable,str(code/'verify_sft_search_result.py'),
                '--candidate',str(a.out/'evaluation')])
            save('status.safe.json',dict(status='candidate_evaluated_review_pending',promotion=False))
    except BaseException as exc:
        save('status.safe.json',dict(status='failed',error=type(exc).__name__,detail=str(exc)))
        raise


if __name__=='__main__':main()
