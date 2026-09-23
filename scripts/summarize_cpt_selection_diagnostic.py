"""Verify the fixed run ledger, reconstruct all decision sets, export aggregates."""
import argparse
from pathlib import Path
from cpt_two_family import read,sha,freeze
from cpt_selection_diagnostic import score


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit-output',type=Path)
    a=p.parse_args()
    root=a.root
    reg=read(root/'execution.safe.json')
    run=root/'remote_run'
    assert read(run/'execution.safe.json')==reg
    assert read(run/'status.safe.json')['state']=='complete_pending_local_verification'
    assert sha(root/'packet.private.json')==reg['packet_sha256']
    assert sha(root/'quality.safe.json')==reg['quality_sha256']
    folder=run/'p1'
    done=read(folder/'completed.safe.json')
    assert done['calls']==144 and sha(folder/'predictions.private.json')==done['predictions_sha256']
    rows=read(folder/'predictions.private.json')
    collected=[]
    for start in range(0,144,16):
        name=f'batch-{start:03d}'
        raw,reserved,completed=(folder/(name+suffix) for suffix in ('.private.json','.reserved.safe.json','.completed.safe.json'))
        r,d=read(reserved),read(completed)
        assert r['start']==start and r['calls']==d['calls']==16
        assert r['packet_sha256']==reg['packet_sha256'] and sha(raw)==d['sha256']
        collected.extend(read(raw))
    assert rows==collected
    packet=read(root/'packet.private.json')
    ledger,summary=score(packet,rows)
    option_audit=[]
    for group in packet['groups']:
        for form in ('S','V','B'):
            selected=[r for r in ledger if r['task']==group['id'] and r['form']==form]
            stable_options=0
            for original in range(4):
                observations=[]
                for row in selected:
                    condition=group['conditions'][row['shift']]
                    position=condition['order'].index(original)
                    observations.append(dict(shift=row['shift'],repeat=row['repeat'],
                        observed=row['values'][position],expected=position in condition['expected']))
                repeated=sorted((r for r in observations if r['shift']==0),key=lambda r:r['repeat'])
                stable=all(type(r['observed']) is bool for r in repeated) and all(r['observed']==repeated[0]['observed'] for r in repeated)
                stable_options+=stable
                option_audit.append(dict(task=group['id'],form=form,original_option=original,
                    observations=observations,first_three_stable=stable,
                    omissions=sum(r['expected'] and r['observed'] is False for r in observations),
                    false_positives=sum(not r['expected'] and r['observed'] is True for r in observations)))
            task=next(t for t in summary[form]['tasks'] if t['id']==group['id'])
            task['first_three_stable_options']=stable_options
    for form in summary:
        retention=[t for t in summary[form]['tasks'] if t['id'] in ('MT2','MT4')]
        summary[form]['retention']=dict(tasks=2,stable_tasks=sum(t['stable_four'] for t in retention),
            main_correct=sum(t['main_correct'] for t in retention),main_total=8)
    result=dict(forms=summary,calls=144,completed_batches=9,training_runs=0,formal_calls=0,
                packet_sha256=reg['packet_sha256'],predictions_sha256=done['predictions_sha256'],
                interpretation='Local development output-form comparison; B requires four calls per complete set. No promotion or training.')
    freeze(root/'selection_ledger.private.json',ledger)
    freeze(root/'option_audit.private.json',option_audit)
    freeze(a.output,result)
    if a.audit_output:
        audit_path=root/'independent_execution_audit.private.json'
        audit=read(audit_path)
        assert audit['passed'] is True and audit['no_primary_summary_or_score_read'] is True
        assert audit['input_sha256']['packet']==reg['packet_sha256']
        assert audit['input_sha256']['predictions']==done['predictions_sha256']
        assert audit['input_sha256']['frozen_blind_review']==sha(root.parent/'llin-operation-matrix-20260923-01/development_blind_review.private.json')
        safe=dict(passed=True,input_sha256=audit['input_sha256'],private_audit_sha256=sha(audit_path),forms={},paired_main={})
        keys=('calls','output_tokens','main_correct','main_total','all_correct','all_total',
              'four_permutations_correct_tasks','invalid_complete_units','main_omissions','main_false_positives','all_omissions','all_false_positives','keeping_items')
        for form in ('S','V','B'):
            independent=audit['counts'][form]
            ours=summary[form]
            for left,right in [('calls','calls'),('output_tokens','output_tokens'),('main_correct','main_complete_sets_correct'),
                               ('all_correct','all_complete_sets_correct'),('four_permutations_correct_tasks','stable_tasks'),
                               ('all_omissions','observed_omissions'),('all_false_positives','observed_false_positives')]:
                assert independent[left]==ours[right]
            safe['forms'][form]={key:independent[key] for key in keys}
        main={(r['task'],r['shift'],r['form']):r['correct'] for r in ledger if r['repeat']==0}
        for form in ('V','B'):
            independent=audit['counts']['paired_against_same_session_S'][form]['main']
            repairs=sum(not main[(task,shift,'S')] and main[(task,shift,form)] for task in ('MT1','MT2','MT3','MT4') for shift in range(4))
            losses=sum(main[(task,shift,'S')] and not main[(task,shift,form)] for task in ('MT1','MT2','MT3','MT4') for shift in range(4))
            assert (repairs,losses)==(independent['repairs'],independent['new_losses'])
            safe['paired_main'][form]={key:independent[key] for key in ('paired_units','repairs','new_losses','both_correct','both_wrong')}
        freeze(a.audit_output,safe)
    print(result)


if __name__=='__main__':main()
