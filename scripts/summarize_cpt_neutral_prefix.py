"""Verify all new journals and historical predictions, then publish fixed cohorts."""
import argparse
import json
from pathlib import Path
from prepare_cpt_neutral_prefix import BASE
from prepare_cpt_presentation import read,sha,digest,freeze
from summarize_cpt_presentation import score


def aggregate(ledger):
    table=[]
    for name,selected in [('all',ledger),('baseline_wrong',[r for r in ledger if not r['before']['correct']]),
                          ('baseline_correct',[r for r in ledger if r['before']['correct']]),
                          ('source_A_known_errors',[r for r in ledger if r['source_status']=='A']),
                          ('old_25_targets',[r for r in ledger if r['old_target']])]:
        gains=sum(not r['before']['correct'] and r['after']['correct'] for r in selected)
        losses=sum(r['before']['correct'] and not r['after']['correct'] for r in selected)
        row=dict(cohort=name,n=len(selected),gains=gains,losses=losses,net=gains-losses)
        for condition in ('before','after'):
            row[condition]={k:sum(r[condition][k] for r in selected) for k in ('correct','stable_correct','unstable','invalid','truncated')}
        table.append(row)
    return table


def main(export_audit=False):
    packet=read(BASE/'packet.private.json');reg=read(BASE/'execution.safe.json');run=BASE/'remote_run';folder=run/'p1'
    assert sha(BASE/'packet.private.json')==reg['packet_sha256'] and sha(BASE/'quality.safe.json')==reg['quality_sha256']
    assert read(run/'execution.safe.json')==reg and read(run/'status.safe.json')['state']=='complete_pending_local_verification'
    done=read(folder/'completed.safe.json');assert done['calls']==915 and sha(folder/'predictions.private.json')==done['predictions_sha256']
    rows=read(folder/'predictions.private.json');assert len(rows)==915
    collected=[]
    for start in range(0,915,16):
        stem=folder/f'batch-{start:03d}'
        path=Path(str(stem)+'.private.json');r=read(Path(str(stem)+'.reserved.safe.json'));d=read(Path(str(stem)+'.completed.safe.json'))
        assert r['start']==start and r['calls']==d['calls']==min(16,915-start)
        assert r['packet_sha256']==reg['packet_sha256'] and sha(path)==d['sha256']
        collected+=read(path)
    assert collected==rows and len({r['id'] for r in rows})==915
    for row,request in zip(rows,packet['requests']):
        assert row['id']==request['id'] and row['messages_sha256']==digest(request['messages'])
        assert 0<=row['output_tokens']<=96
    by_id={r['id']:r for r in rows};old={r['group']:r for r in packet['reuse']};ledger=[]
    for g in packet['groups']:
        before=score(old[g['id']]['rows'],g['expected'],269)
        after=score([by_id[f"{g['id']}:{i}"] for i in range(3)],g['expected'],269)
        assert before['correct']==g['baseline_correct']
        ledger.append(dict(id=g['id'],source_status=g['source_status'],old_target=g['old_target'],before=before,after=after))
    assert len(ledger)==305
    table=aggregate(ledger)
    freeze(BASE/'results.private.json',ledger)
    safe=dict(status='closed_local_verification_complete',new_calls=915,reused_calls=915,completed_batches=58,
        table=table,training_runs=0,official_results_unchanged=True,training_ready=False,
        decision='positive_development_signal_only' if table[0]['net']>0 else 'no_net_gain_close_this_recipe',
        packet_sha256=reg['packet_sha256'],predictions_sha256=done['predictions_sha256'],private_ledger_sha256=sha(BASE/'results.private.json'),
        limitations=['Historical baseline, not contemporaneous randomized control','Existing labels are not all source-validated',
                    'Source A cohort contains only previously audited errors, not a fully audited retention set',
                    'Full original question meaning is changed; matching definition/options are preserved','Development evidence only; no promotion'])
    freeze(Path('docs/cpt_neutral_prefix_results_20260923.safe.json'),safe)
    if export_audit:
        path=BASE/'independent_execution_audit.private.json';audit=read(path)
        assert audit['passed'] is True
        assert audit['input_sha256']['packet']==reg['packet_sha256']
        assert audit['input_sha256']['predictions']==done['predictions_sha256']
        independent={r['cohort']:r for r in audit['cohorts']}
        assert len(audit['cohorts'])==len(table) and set(independent)=={r['cohort'] for r in table}
        exported=[]
        for row in table:
            other=independent[row['cohort']]
            fields=['cohort','n','gains','losses','net']
            for key in fields:assert other[key]==row[key]
            for condition in ('before','after'):
                for key,ours in [('correct','correct'),('stable','stable_correct'),('unstable','unstable'),('invalid','invalid'),('truncated','truncated')]:
                    field=f'{condition}_{key}';assert other[field]==row[condition][ours];fields.append(field)
            exported.append({key:other[key] for key in fields})
        freeze(Path('docs/cpt_neutral_prefix_execution_audit_20260923.safe.json'),dict(passed=True,
            input_sha256=audit['input_sha256'],private_audit_sha256=sha(path),cohorts=exported))
    print(json.dumps(safe,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');a=p.parse_args();main(a.audit)
