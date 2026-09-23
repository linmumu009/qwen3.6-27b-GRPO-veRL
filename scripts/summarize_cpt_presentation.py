"""Verify every journalled request and reparse answers; report all four cells."""
from collections import Counter
import json
from pathlib import Path
from prepare_cpt_presentation import BASE,read,sha,digest,freeze
from evaluate_logistics_knowledge import parse_answers


def score(rows,expected,count):
    parsed=[];invalid=0;truncated=0;correct=0
    for r in rows:
        answer,ok=parse_answers(r.get('text',r.get('prediction','')),count)
        cut=r.get('finish_reason')=='length' or r.get('truncated',False)
        valid=ok and not cut
        parsed.append(tuple(answer) if valid else None)
        invalid+=not valid;truncated+=cut;correct+=valid and list(answer)==sorted(expected)
    assert len(rows)==3
    winner,n=Counter(parsed).most_common(1)[0]
    return dict(correct=winner is not None and n>=2 and list(winner)==sorted(expected),stable_correct=correct==3,
        unstable=len(set(parsed))>1,invalid=invalid,truncated=truncated,answer=list(winner) if winner is not None and n>=2 else None)


def main():
    reg=read(BASE/'execution.safe.json');packet=read(BASE/'packet.private.json');out=BASE/'remote_run'
    assert sha(BASE/'packet.private.json')==reg['packet_sha256'] and read(out/'execution.safe.json')==reg
    assert read(out/'status.safe.json')['state']=='complete_pending_local_verification'
    done=read(out/'p1/completed.safe.json');path=out/'p1/predictions.private.json'
    assert done['calls']==300 and sha(path)==done['predictions_sha256']
    raw=read(path);plan=packet['requests'];assert len(raw)==len(plan)==300
    merged=[]
    for start in range(0,300,16):
        stem=out/'p1'/f'batch-{start:03d}'
        result=Path(str(stem)+'.private.json');reservation=read(Path(str(stem)+'.reserved.safe.json'))
        complete=read(Path(str(stem)+'.completed.safe.json'))
        n=len(plan[start:start+16]);assert complete['calls']==reservation['calls']==n
        assert reservation['start']==start and reservation['packet_sha256']==reg['packet_sha256'] and sha(result)==complete['sha256']
        merged+=read(result)
    assert merged==raw and len({r['id'] for r in raw})==300
    for r,s in zip(raw,plan):
        assert r['id']==s['id'] and r['messages_sha256']==digest(s['messages']) and 0<=r['output_tokens']<=96
    by_id={r['id']:r for r in raw};reuse={(r['group'],r['condition']):r for r in packet['reuse']}
    ledger=[]
    for group in packet['groups']:
        cells={}
        for condition,c in group['conditions'].items():
            rows=reuse[(group['id'],condition)]['rows'] if condition in ('LL','SS') else [by_id[f"{group['id']}:{condition}:{i}"] for i in range(3)]
            cells[condition]=score(rows,c['expected'],len(c['options']))
        ledger.append(dict(id=group['id'],cohort=group['cohort'],cells=cells))
    table=[];effects=[]
    for cohort in ['target','retention']:
        group=[r for r in ledger if r['cohort']==cohort]
        for condition in ['LL','SL','LS','SS']:
            table.append(dict(cohort=cohort,condition=condition,n=len(group),**{k:sum(r['cells'][condition][k] for r in group)
                for k in ['correct','stable_correct','unstable','invalid','truncated']}))
        for label,before,after in [('prefix_full','LL','SL'),('prefix_small','LS','SS'),('pool_original_prefix','LL','LS'),('pool_specific_prefix','SL','SS')]:
            gain=sum(not r['cells'][before]['correct'] and r['cells'][after]['correct'] for r in group)
            loss=sum(r['cells'][before]['correct'] and not r['cells'][after]['correct'] for r in group)
            effects.append(dict(cohort=cohort,contrast=label,gains=gain,losses=loss,net=gain-loss))
    freeze(BASE/'results.private.json',ledger)
    safe=dict(status='closed_validation_complete',new_calls=300,reused_calls=300,training_runs=0,table=table,effects=effects,
        input_packet_sha256=reg['packet_sha256'],predictions_sha256=sha(path),private_ledger_sha256=sha(BASE/'results.private.json'),
        original_stage_c_gate_unchanged=True,training_ready=False,official_results_unchanged=True,limitations=reg['limitations'])
    freeze(Path('docs/cpt_presentation_results_20260923.safe.json'),safe)
    print(json.dumps(safe,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
