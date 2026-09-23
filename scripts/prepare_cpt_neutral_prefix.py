"""Freeze a label-independent prefix deletion over the entire fixed 269-option pool."""
import argparse
from pathlib import Path
from audit_cpt_transfer import CASE_SHA,read_rows,verify_predictions,verify_protocols
from prepare_cpt_presentation import read,sha,digest,freeze,item
from evaluate_logistics_knowledge import build_messages,EvalItem

BASE=Path('CPT_resources/llin-neutral-prefix-20260923-01')
CASES=Path('CPT_resources/llin-transfer-audit-20260915-01/frozen_cases.private.jsonl')
HISTORY=Path('CPT_resources/llin-transfer-formal-20260917-01/p1')
MAP=Path('CPT_resources/llin-p1-error-map-20260923-01')
PREFIX='What are pros and cons of containerisation? '
MARKER='Which of the following best matches:'


def neutral_question(question):
    if not question.startswith(PREFIX+MARKER) or question.count(MARKER)!=1:
        raise ValueError('Input does not match the frozen structure')
    result=question[len(PREFIX):]
    if not result[len(MARKER):].strip():
        raise ValueError('Missing definition')
    return result


def neutral_messages(question,options):
    # No label/prediction/case object accepted by this boundary.
    return build_messages(EvalItem(item_hash='',dataset='',source_id='',category='',question_type='multiple_choice',
        question=neutral_question(question),options=tuple(options),expected=()))


def prepare():
    assert sha(CASES)==CASE_SHA
    rows=read_rows(CASES)
    selected=sorted((c for c in rows if c['dataset']=='LogistikaBench' and len(c['options'])==269),key=lambda c:c['item_hash'])
    assert len(selected)==305 and len({c['item_hash'] for c in selected})==305
    assert all(c['options']==selected[0]['options'] for c in selected)
    # The full inference message plan is built before reading outcome/source ledgers.
    requests=[dict(id=f"{c['item_hash']}:{repeat}",group=c['item_hash'],repeat=repeat,
                   messages=neutral_messages(c['question'],c['options'])) for c in selected for repeat in range(3)]
    semantic=read(BASE/'semantic_review_input.private.json')
    assert semantic['items']==[dict(id=c['item_hash'],original_question=c['question'],neutral_question=neutral_question(c['question'])) for c in selected]
    cases={c['item_hash']:c for c in rows}
    raw=read_rows(HISTORY/'predictions.private.jsonl')
    scores=verify_predictions(cases,raw,read(HISTORY/'scores.safe.json'))
    protocol=read(HISTORY/'protocol.safe.json');verify_protocols({'p1':protocol},CASE_SHA,1672)
    tokens={c['item_hash']:h for c,h in zip(rows,protocol['prompt_hashes'])}
    source={r['item_hash']:r for r in read(MAP/'ledger.private.json')}
    targets=set(read(MAP/'decision_queues.private.json')['queues']['presentation_control_first'])
    groups=[];reuse=[]
    for c in selected:
        key=c['item_hash']
        groups.append(dict(id=key,expected=c['expected'],option_count=269,baseline_correct=scores[key]['correct'],
                           source_status=source[key]['source_status'] if key in source else 'not_independently_reviewed',old_target=key in targets))
        reuse.append(dict(group=key,messages=build_messages(item(c)),historical_prompt_token_sha256=tokens[key],
                          rows=[r for r in raw if r['item_hash']==key]))
    assert sum(g['baseline_correct'] for g in groups)==224 and sum(g['old_target'] for g in groups)==25
    freeze(BASE/'packet.private.json',dict(groups=groups,requests=requests,reuse=reuse,training_allowed=False))
    freeze(BASE/'offline.safe.json',dict(items=305,new_calls=915,historical_calls=915,baseline_correct=224,baseline_wrong=81,
        source_A_errors=sum(g['source_status']=='A' for g in groups),old_targets=25,unique_pool=True,unique_prefix=True,
        exact_definition_and_options_preserved=True,whole_prompt_semantic_equivalence_claimed=False,
        input_sha256={str(p):sha(p) for p in [CASES,HISTORY/'predictions.private.jsonl',HISTORY/'protocol.safe.json',HISTORY/'scores.safe.json',MAP/'ledger.private.json',MAP/'decision_queues.private.json']}))


def release():
    semantic=read(BASE/'semantic_review.private.json')
    assert semantic['passed'] is True and semantic['blockers']==[]
    assert semantic['input_sha256']==sha(BASE/'semantic_review_input.private.json')
    ids=[g['id'] for g in read(BASE/'packet.private.json')['groups']]
    assert len(semantic['reviewed_ids'])==305 and set(semantic['reviewed_ids'])==set(ids)
    review=read(BASE/'protocol_review.private.json')
    assert review['passed'] is True and review['remaining_blockers']==[]
    paths={'packet':BASE/'packet.private.json','registration':Path('docs/cpt_neutral_prefix_registration_20260923.md'),
           'builder':Path(__file__),'runner':Path('scripts/run_cpt_neutral_prefix.py')}
    for key,path in paths.items():assert review['input_sha256'][key]==sha(path),key
    quality=dict(passed=True,semantic_review_sha256=sha(BASE/'semantic_review.private.json'),protocol_review_sha256=sha(BASE/'protocol_review.private.json'),inputs={k:sha(p) for k,p in paths.items()})
    freeze(BASE/'quality.safe.json',quality)
    old=read(Path('CPT_resources/llin-presentation-20260923-01/execution.safe.json'))
    reg=dict(id='llin-neutral-prefix-20260923-01',max_calls=915,reused_calls=915,groups=305,training_allowed=False,
             quality_released=True,quality_sha256=sha(BASE/'quality.safe.json'),packet_sha256=sha(BASE/'packet.private.json'),
             runner_sha256=sha(paths['runner']),registered_plan_sha256=sha(paths['registration']),
             protocol_code_sha256=old['protocol_code_sha256'],model_path=old['model_path'],
             temperature=0,seed=1024,max_tokens=96,max_model_len=8192,tp=8,max_num_seqs=32,repeats=3)
    freeze(BASE/'execution.safe.json',reg)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',action='store_true');a=p.parse_args()
    release() if a.release else prepare()
