"""Verify the bounded source/structure diagnostic without changing formal labels."""
from pathlib import Path
import json
from prepare_cpt_small_choice import BASE
from prepare_cpt_presentation import read,sha,digest,freeze
from summarize_cpt_presentation import score


def main():
    reg=read(BASE/'execution.safe.json');packet=read(BASE/'packet.private.json');out=BASE/'remote_run'
    assert sha(BASE/'packet.private.json')==reg['packet_sha256'] and read(out/'execution.safe.json')==reg
    assert read(out/'status.safe.json')['state']=='complete_pending_local_verification'
    path=out/'p1/predictions.private.json';done=read(out/'p1/completed.safe.json')
    assert done['calls']==36 and sha(path)==done['predictions_sha256']
    rows=read(path);plan=packet['requests'];assert len(rows)==len(plan)==36
    merged=[]
    for start in range(0,36,16):
        stem=out/'p1'/f'batch-{start:03d}';p=Path(str(stem)+'.private.json')
        r=read(Path(str(stem)+'.reserved.safe.json'));c=read(Path(str(stem)+'.completed.safe.json'))
        assert r['start']==start and r['packet_sha256']==reg['packet_sha256']
        assert r['calls']==c['calls']==len(plan[start:start+16]) and sha(p)==c['sha256']
        merged+=read(p)
    assert merged==rows and len({r['id'] for r in rows})==36
    for r,s in zip(rows,plan):assert r['id']==s['id'] and r['messages_sha256']==digest(s['messages']) and 0<=r['output_tokens']<=96
    raw={r['id']:r for r in rows};reuse={r['group']:r for r in packet['reuse']};ledger=[]
    for group in packet['groups']:
        cells={k:score([raw[f"{group['id']}:{k}:{i}"] for i in range(3)],c['expected'],len(c['options'])) for k,c in group['conditions'].items()}
        c=group['conditions']['scoped_closed']
        cells['original_closed']=score(reuse[group['id']]['rows'],c['expected'],len(c['options']))
        ledger.append(dict(id=group['id'],family=group['family'],cells=cells))
    freeze(BASE/'results.private.json',ledger)
    table=[]
    for family in sorted({r['family'] for r in ledger}):
        subset=[r for r in ledger if r['family']==family]
        for condition in ['original_closed','scoped_closed','source','structured']:
            table.append(dict(family=family,condition=condition,n=len(subset),**{k:sum(r['cells'][condition][k] for r in subset)
                for k in ['correct','stable_correct','unstable','invalid','truncated']}))
    safe=dict(status='closed_validation_complete',new_calls=36,reused_calls=12,training_runs=0,table=table,
        excluded_before_calls=1,packet_sha256=reg['packet_sha256'],predictions_sha256=sha(path),
        private_ledger_sha256=sha(BASE/'results.private.json'),training_ready=False,stage_c_gate_unchanged=True,
        limitations=reg['limitations'])
    freeze(Path('docs/cpt_small_choice_results_20260923.safe.json'),safe)
    print(json.dumps(safe,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
