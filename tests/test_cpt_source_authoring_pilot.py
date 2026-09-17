import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_cpt_source_authoring_pilot import answer, closed_prompt, review_prompt, specs, summarize


def row():
    return dict(id='example',split='train',question='Independent scenario.',options=['a','b','c','d'],
        correct_indices=[0,2],option_reasons=['SECRET_RATIONALE']*4,authoring_scope='archived',
        reasoning_requirement='Combine two conditions.',sources=[dict(title='source',scope='archive',source_text='SOURCE_ONLY')])


def test_full_answer_set_is_valid_but_bad_indices_are_not():
    assert answer('{"answers":[3,2,1,0]}') == [0,1,2,3]
    for text in ('{"answers":[true]}','{"answers":[0,0]}','{"answers":[4]}','{"answers":[]}',
                 '{"answers":[0],"extra":1}','{"answers":[0]} {"answers":[0]}'):
        assert answer(text) is None


def test_blind_payload_and_closed_book_separation():
    r=row();payload=json.loads(review_prompt(r).split('INPUT:\n',1)[1])
    assert set(payload['task'])=={'question','options'}
    assert 'SECRET_RATIONALE' not in review_prompt(r)
    assert 'correct_indices' not in payload
    text=closed_prompt(r,specs([r])[0])
    assert 'SOURCE_ONLY' not in text and 'SECRET_RATIONALE' not in text


def test_permutations_relocate_every_option_and_keep_answer_identity():
    r=row();a,b=specs([r])
    assert all(x!=y for x,y in zip(a['order'],b['order']))
    for s in (a,b):
        assert sorted(s['order'][i] for i in s['expected'])==r['correct_indices']


def records(r,review_key=None,flag=True):
    raw=[]
    for s in specs([r]):
        raw.append(dict(id=r['id'],variant=s['variant'],order=s['order'],text=json.dumps({'answers':s['expected']}),
            finish_reason='stop',output_tokens=10,prompt_tokens=20,
            prompt_text_sha256=hashlib.sha256(closed_prompt(r,s).encode()).hexdigest()))
    v=dict(correct_indices=r['correct_indices'] if review_key is None else review_key,
        supported=flag,unambiguous=True,self_contained=True,scope_preserved=True,not_answer_leaking=True,
        design_satisfied=True,option_reasons=['x']*4)
    review=[dict(id=r['id'],text=json.dumps(v),finish_reason='stop',output_tokens=20,prompt_tokens=30,
        prompt_text_sha256=hashlib.sha256(review_prompt(r).encode()).hexdigest())]
    return raw,review


def test_review_disagreement_never_hides_baseline_answers():
    r=row();raw,review=records(r,review_key=[1]);result=summarize([r],raw,review)
    assert result['blind_review_agrees']==0
    assert result['closed_book_calls']==2 and result['both_orders_correct']==1
    assert result['training_allowed'] is False


def test_negative_review_flag_rejects_even_with_matching_key():
    r=row();raw,review=records(r,flag=False)
    assert summarize([r],raw,review)['blind_review_agrees']==0


def test_option_order_instability_maps_back_to_authored_indices():
    r=row();raw,review=records(r)
    assert summarize([r],raw,review)['option_order_changed_answer_ids']==[]
    raw[1]['text']='{"answers":[0,1,2,3]}'
    result=summarize([r],raw,review)
    assert result['option_order_changed_answer_ids']==['example']
    assert result['both_orders_correct']==0
