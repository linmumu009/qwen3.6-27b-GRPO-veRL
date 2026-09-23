"""Freeze four independently reviewed existing tasks and two complete rule families."""
import json
from pathlib import Path
from prepare_cpt_presentation import read,sha,freeze,item,HISTORY,CASES,MAP
from audit_cpt_transfer import read_rows,verify_predictions,verify_protocols,CASE_SHA
from evaluate_logistics_knowledge import build_messages

BASE=Path('CPT_resources/llin-small-choice-20260923-01')
REVIEW=Path('CPT_resources/llin-presentation-20260923-01/small_choice_blind_review.private.json')


def main():
    BASE.mkdir(exist_ok=True)
    review=read(REVIEW);families={f['family_id']:f for f in review['families']}
    cases_list=read_rows(CASES);cases={c['item_hash']:c for c in cases_list}
    assert sha(CASES)==CASE_SHA
    raw=read_rows(HISTORY/'predictions.private.jsonl')
    scores=verify_predictions(cases,raw,read(HISTORY/'scores.safe.json'))
    protocol=read(HISTORY/'protocol.safe.json');verify_protocols({'p1':protocol},CASE_SHA,1672)
    hashes={c['item_hash']:h for c,h in zip(cases_list,protocol['prompt_hashes'])}
    queue=read(MAP/'decision_queues.private.json')['queues']['small_choice_rule_discrimination']
    assert set(queue)=={r['item_hash'] for r in review['items']}
    groups=[];requests=[];reuse=[]
    for r in sorted(review['items'],key=lambda x:x['item_hash']):
        if not r['release']:continue
        key=r['item_hash'];c=cases[key];f=families[r['family_id']]
        assert r['derived_indices']==sorted(c['expected']) and not scores[key]['correct']
        assert sha(Path(f['primary_snapshot_path']))==f['primary_snapshot_sha256']
        assert f['supplied_rule_review']['corrected_pair_equivalent']
        original=build_messages(item(c));conditions={}
        for condition,context in [('scoped_closed',''),('source',f['evidence_text']),('structured',f['structured_text'])]:
            msg=[dict(m) for m in original]
            msg[1]['content']=f['scope_preamble']+'\n\n'+('Reference material:\n'+context+'\n\n' if context else '')+msg[1]['content']
            conditions[condition]=dict(messages=msg,expected=c['expected'],options=c['options'])
            for repeat in range(3):requests.append(dict(id=f'{key}:{condition}:{repeat}',group=key,condition=condition,repeat=repeat,messages=msg))
        groups.append(dict(id=key,family=f['family_id'],conditions=conditions))
        reuse.append(dict(group=key,condition='original_closed',messages=original,historical_prompt_token_sha256=hashes[key],
            rows=[row for row in raw if row['item_hash']==key]))
    assert len(groups)==4 and len(requests)==36
    freeze(BASE/'packet.private.json',dict(groups=groups,requests=requests,reuse=reuse,training_allowed=False))
    previous=read(Path('CPT_resources/llin-presentation-20260923-01/execution.safe.json'))
    reg=dict(id='llin-small-choice-20260923-01',max_calls=36,reused_calls=12,groups=4,families=2,
        excluded_before_generation=1,packet_sha256=sha(BASE/'packet.private.json'),runner_sha256=sha(Path('scripts/run_cpt_small_choice.py')),
        protocol_code_sha256=previous['protocol_code_sha256'],model_path=protocol['model'],training_allowed=False,
        repeats=3,temperature=0,seed=1024,max_tokens=96,max_model_len=8192,tp=8,max_num_seqs=32,
        inputs={str(p):sha(p) for p in [REVIEW,CASES,HISTORY/'predictions.private.jsonl',HISTORY/'protocol.safe.json']},
        conditions=['scoped_closed','source','structured'],
        primary='Report all four tasks and three conditions; source/structured gains versus identically scoped closed prompts.',
        limitations=['Four development tasks, two rule families, not independent transfer','Source and structure lengths differ',
            'Scope preamble changes original prompt; original closed results are historical reference only',
            'Formal closed-book system text retained across all conditions; added reference is a diagnostic intervention',
            'One disputed competitor excluded by blind review before calls'],
        training_gate='Original Stage C unchanged; no automatic training from this diagnostic.')
    freeze(BASE/'execution.safe.json',reg);freeze(Path('docs/cpt_small_choice_registration_20260923.safe.json'),reg)
    print(json.dumps({k:reg[k] for k in ['max_calls','reused_calls','groups','families','excluded_before_generation']}))


if __name__=='__main__':main()
