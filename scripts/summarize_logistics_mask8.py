"""Aggregate the reviewed-eight positive-control experiment without private text."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.compare_logistics_answer_probes import compare
from scripts.probe_logistics_exam_answers import summarize
from scripts.prepare_logistics_reviewed8 import REVIEWED
from scripts.summarize_logistics_cpt_run import parse_metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    args=p.parse_args()
    models={name:json.loads((args.root/f'{name}.json').read_text()) for name in ('step120','all','answer')}
    for payload in models.values():
        rows=payload['rows']
        if len(rows)!=8 or {r['item_hash'] for r in rows}!=set(REVIEWED):
            raise ValueError('reviewed item coverage mismatch')
        metrics=summarize(rows)
        if set(metrics)!={'reviewed8'} or metrics['reviewed8']['answer_tokens']!=39:
            raise ValueError('answer scoring scope mismatch')
        if metrics['reviewed8']['generation_censored'] or metrics['reviewed8']['boundary_crossings']:
            raise ValueError('censored generation or ambiguous boundary')
    training={}
    for arm in ('all','answer'):
        run=args.runs/f'logistics-mask8-{arm}-20260905'
        report=json.loads((run/'training_summary.safe.json').read_text())
        if report['total_steps'] != 64 or report['total_sequence_tokens_with_eos'] != 15712:
            raise ValueError('training coverage mismatch')
        if len(report['checkpoints'])!=1 or not all(
            c['valid'] and c['optimizer_declared'] and c['global_step']==64
            for c in report['checkpoints']
        ):
            raise ValueError('missing valid optimizer checkpoint')
        export=json.loads((run/'hf_export_step_64/llin_export_manifest.json').read_text())
        if not export['verification']['valid']:
            raise ValueError('invalid HF export')
        metrics=parse_metrics('\n'.join(path.read_text(errors='replace') for path in
            sorted(run.glob('torchrun_logs/*/attempt_0/*/stdout.log'))))
        validation={str(step):row['val/loss'] for step,row in metrics.items() if 'val/loss' in row}
        if set(validation)!={'16','32','48','64'}:
            raise ValueError('training-set validation coverage mismatch')
        training[arm]={'steps':64,'epochs':32,'sequence_tokens_with_eos':15712,
            'mean_loss_by_epoch':[r['loss']['mean'] for r in report['exposures']],
            'training_set_validation_loss_by_step':validation,
            'hf_export_verification':export['verification'],
            'checkpoints':report['checkpoints'],'peak_memory_gb':report['peak_memory_gb']}
    result={'contract':'reviewed8-mask-ablation-v1','private_content_included':False,
        'manifest':json.loads((args.root/'manifest.safe.json').read_text()),
        'mask_gate':json.loads((args.root/'mask-gate.safe.json').read_text()),
        'eval':{name:summarize(payload['rows']) for name,payload in models.items()},
        'all_vs_step120':compare(models['step120'],models['all']),
        'answer_vs_step120':compare(models['step120'],models['answer']),
        'answer_vs_all':compare(models['all'],models['answer']),
        'training':training,
        'input_hashes':{name:hashlib.sha256((args.root/f'{name}.json').read_bytes()).hexdigest() for name in models},
        'interpretation_limit':'eight training items, agent-reviewed entailment only, no false judgment cases; not generalization evidence'}
    output=args.root/'comparison.safe.json'
    if output.exists():
        raise ValueError('refusing overwrite')
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
