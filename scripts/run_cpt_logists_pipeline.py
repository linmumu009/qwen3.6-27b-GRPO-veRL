"""One exact epoch of released logistics corpus followed by both frozen benchmarks."""
import hashlib
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_logistics_cpt_curve_8x import BASE, ROOT, checkpoint_gate, evaluation_env, majority, paired, save

OUT=ROOT/'runs/cpt-stage2-storage-20260910/llin/llin-cpt-logists-one-epoch-20260910'
CORPUS=ROOT/'runs/llin-logists-cpt-v4-20260910'
RETENTION=ROOT/'runs/cpt-stage2-storage-20260910/llin/cpt-retention-20260910'
BASELINE=ROOT/'runs/cpt-controlled-storage-20260910/llin/cpt-controlled-20260910/curve8x/eval_epoch_0'
TRAIN_SHA='f79c1cf5c7418f73fa6ca099c397a66915bec7366f010dbbb2366b3d02881c5e'


def audit_training(metrics, lengths):
    assert len(lengths)==3016 and sum(lengths)==1183197
    assert sorted(metrics)==list(range(1,378)), 'Missing or extra training steps'
    assert sum(int(m['train/global_tokens']) for m in metrics.values())==sum(lengths)
    for m in metrics.values():
        assert all(math.isfinite(m[k]) for k in ('train/loss','train/grad_norm','train/lr'))
        assert 8 <= m['train/global_tokens'] <= 8*4096
    return dict(steps=377,epochs=1,records=3016,batch_size=8,sequence_tokens=1183197,
        exact_divisible_epoch=True,mean_step_loss=sum(m['train/loss'] for m in metrics.values())/377)


def audit_evaluation(directory):
    from collections import defaultdict
    groups=defaultdict(list)
    rows=[json.loads(x) for x in (directory/'predictions.private.jsonl').read_text().splitlines()]
    assert len(rows)==5016
    for row in rows:groups[row['item_hash']].append(row)
    assert len(groups)==1672
    scores=json.loads((directory/'scores.safe.json').read_text())
    assert scores.keys()==groups.keys()
    for key,group in groups.items():
        assert scores[key]['correct']==majority(group)
        assert all(r['dataset']==scores[key]['dataset'] for r in group)
    return scores


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--after-training',action='store_true',help='Continue from completed step377 without repeating training')
    args=parser.parse_args()
    import fcntl
    os.umask(0o077)
    code=Path(__file__).resolve().parent
    status=OUT/'status.safe.json'
    with (OUT/'pipeline.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            assert hashlib.sha256((CORPUS/'train.parquet').read_bytes()).hexdigest()==TRAIN_SHA
            save(status,dict(status='waiting_for_resource_lock'))
            with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as resource:
                fcntl.flock(resource,fcntl.LOCK_EX)
                if shutil.disk_usage(OUT).free<550_000_000_000:raise RuntimeError('Need550GB checkpoint/export headroom')
                env=dict(os.environ,TRAIN_FILE=str(CORPUS/'train.parquet'),EXPECTED_TRAIN_SHA=TRAIN_SHA,
                    EXPECTED_CONTENT_TOKENS='1180181',OUTPUT_DIR=str(OUT/'llin-training'),RUN_NAME='llin-cpt-logists-one-epoch-20260910')
                def run(label,command,environment):
                    save(status,dict(status=label))
                    with (OUT/(label+'.log')).open('x') as log:
                        subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
                if args.after_training:
                    checkpoint_gate(OUT/'llin-training/checkpoints/global_step_377')
                else:
                    run('training',['bash',str(code/'run_cpt_logists_one_epoch.sh')],env)
                from summarize_logistics_cpt_run import parse_metrics
                logs='\n'.join(p.read_text(errors='replace') for p in (OUT/'llin-training').glob('torchrun_logs/*/attempt_0/*/stdout.log'))
                import pyarrow.parquet as pq
                lengths=pq.read_table(CORPUS/'train.parquet',columns=['token_count']).column('token_count').to_pylist()
                save(OUT/'training_audit.safe.json',audit_training(parse_metrics(logs),lengths))
                checkpoint=OUT/'llin-training/checkpoints/global_step_377'
                checkpoint_gate(checkpoint)
                run('export',[sys.executable,str(code/'export_megatron_dist_to_hf.py'),
                    '--actor-checkpoint',str(checkpoint),'--base-model',str(BASE),'--output-dir',str(OUT/'llin-step120-logists-cpt-1epoch-20260910')],env)
                run('evaluation',[sys.executable,str(code/'run_logistics_cpt_curve_8x.py'),
                    '--evaluate',str(OUT/'llin-step120-logists-cpt-1epoch-20260910'),'--output',str(OUT/'evaluation')],evaluation_env(env))
                before=audit_evaluation(BASELINE);after=audit_evaluation(OUT/'evaluation')
                protocols=[json.loads((d/'protocol.safe.json').read_text()) for d in (BASELINE,OUT/'evaluation')]
                for key in ('cases_sha256','cases','repeats','temperature','seed','max_tokens','max_model_len','tp','max_num_seqs','prompt','prompt_hashes'):
                    assert protocols[0][key]==protocols[1][key],key
                table=[]
                for dataset in ('SC-bench-knowledge','LogistikaBench','all'):
                    a={k:v['correct'] for k,v in before.items() if dataset=='all' or v['dataset']==dataset}
                    b={k:v['correct'] for k,v in after.items() if dataset=='all' or v['dataset']==dataset}
                    table.append(dict(dataset=dataset,**paired(a,b)))
                save(OUT/'comparison.safe.json',dict(table=table,baseline_reused=True,
                    baseline=str(BASELINE),candidate_requests=5016,prompt_identity_verified=True,
                    independent_test=False,promotion=False))
                save(status,dict(status='complete',training_epochs=1,training_steps=377,benchmark_requests=5016))
        except BaseException as exc:
            save(status,dict(status='failed',error=type(exc).__name__,detail=str(exc)))
            raise


if __name__=='__main__':main()
