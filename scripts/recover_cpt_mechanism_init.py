"""Archive only a proven zero-step HCCL initialization failure, never training.

The recipe and code package remain unchanged. All ranks must fail in engine
initialization with the specific port-bind error. Unknown failures stop here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from summarize_logistics_cpt_run import parse_metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_failure(directory):
    assert directory.name=='K_training'
    assert not (directory/'checkpoints').exists(),'checkpoint directory exists'
    assert not list(directory.rglob('*.pt')) and not list(directory.rglob('*.safetensors')),'weight/state artifact exists'
    stderr=list(directory.glob('torchrun_logs/*/attempt_0/*/stderr.log'))
    stdout=list(directory.glob('torchrun_logs/*/attempt_0/*/stdout.log'))
    assert len(stderr)==len(stdout)==16
    assert {p.parent.name for p in stderr}=={str(i) for i in range(16)}
    assert {p.parent.name for p in stdout}=={str(i) for i in range(16)}
    for p in stderr:
        content=p.read_text(errors='replace')
        assert '_init_engine' in content and 'port 16666 have already been bound' in content,'unproven initialization failure'
    text='\n'.join(p.read_text(errors='replace') for p in stdout)
    assert not parse_metrics(text),'training metrics present'
    assert 'train/global_tokens' not in text and 'train/loss' not in text,'training progress present'
    assert (directory/'status.txt').read_text().strip()=='failed_at_line_29'
    return dict(ranks=16,all_ranks_engine_initialization_port_bind_failure=True,training_steps=0,
                checkpoint_exists=False,completed_training_repeated=False,
                files={p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*')) if p.is_file()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--archive',action='store_true')
    a=p.parse_args();out=a.out.resolve();report=inspect_failure(out/'K_training')
    for name in ('step120','p1'):
        done=json.loads((out/(name+'_direct')/'completed.safe.json').read_text())
        assert done['complete'] and done['calls']==36
        # Snapshot-only audit may omit private predictions; live archive may not.
        if a.archive:assert sha(out/(name+'_direct')/'predictions.private.jsonl')==done['predictions_sha256']
        report[name+'_completed']=done
    assert not (out/'R_training').exists()
    if a.archive:
        import fcntl
        import os
        root=Path('/workspace/llin-verl-grpo/runs').resolve()
        assert out.parent==root and out.name=='llin-mechanism-run-20260920-01'
        with (out/'wait.lock').open('a') as wait_lock, (root/'.logistics-exam-cpt.lock').open('a') as run_lock:
            for lock in (wait_lock,run_lock):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
                try:cmd=(proc/'cmdline').read_bytes().decode(errors='replace')
                except (FileNotFoundError,PermissionError,ProcessLookupError):continue
                if 'python' in cmd:
                    assert not any(name in cmd for name in ('run_cpt_mechanism.py','train_cpt_mechanism_sft.py','llin-mechanism-wait.py')),'active mechanism process'
            failed=json.loads((out/'status.safe.json').read_text())
            assert failed['state']=='failed_preserved' and failed['last_stage']=='K_training_export'
            archived=out/'K_initialization_failure_01';assert not archived.exists()
            # Recheck under locks, and keep every original log in the same run.
            assert inspect_failure(out/'K_training')=={k:v for k,v in report.items() if not k.endswith('_completed')}
            (out/'K_training').rename(archived)
            report['archived_directory']=archived.name
            report['recipe_changed']=False
            (out/'initialization_recovery.safe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
