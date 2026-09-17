import copy
import hashlib
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cpt_remediation_packet import validate_row
from run_cpt_remediation_review import review_prompt, review_decision
from verify_cpt_remediation_result import independent_prediction, replace_amended


def row():
    return dict(id='example',question='A complete case. Select all correct statements.',options=['a','b','c','d'],
        option_truth=[True,False,True,False],correct_indices=[0,2],option_reasons=['PRIVATE']*4,
        sources=[dict(title='Title',scope='Archived',source_text='Evidence',evidence=[{'key':'archive'}],source_text_sha256=hashlib.sha256(b'Evidence').hexdigest())],
        arithmetic_checks=[['7-3','4']],training_allowed=False,reasoning_requirement='Combine constraints.')


def raw(indices=(0,2),truths=(True,False,True,False),flag=True):
    obj=dict(option_judgments=[dict(correct=t,reason='Reason') for t in truths],
        correct_indices=list(indices),supported=flag,unambiguous=True,self_contained=True,scope_preserved=True,
        not_answer_leaking=True,design_satisfied=True)
    return dict(text=json.dumps(obj),finish_reason='stop')


def test_equation_and_key_corruption_are_rejected():
    r=row();validate_row(r)
    for key,value in [('correct_indices',[0]),('arithmetic_checks',[['7-3','5']]),('option_truth',[1,False,True,False])]:
        changed=copy.deepcopy(r);changed[key]=value
        with pytest.raises(AssertionError):validate_row(changed)


def test_source_mutation_is_rejected():
    r=row();r['sources'][0]['source_text']='Modified'
    with pytest.raises(AssertionError):validate_row(r)


def test_review_is_blind_to_keys_and_rationales():
    payload=json.loads(review_prompt(row()).split('INPUT:\n',1)[1])
    assert set(payload['task'])=={'question','options'}
    assert 'PRIVATE' not in review_prompt(row())


def test_matching_key_with_contradictory_truths_is_rejected():
    r=review_decision(row(),raw(truths=(True,True,True,False)))
    assert not r['accepted'] and not r['internally_consistent']


def test_internal_agreement_does_not_override_author_or_flags():
    assert not review_decision(row(),raw(indices=(1,),truths=(False,True,False,False)))['accepted']
    assert not review_decision(row(),raw(flag=False))['accepted']
    assert review_decision(row(),raw())['accepted']


def test_all_true_is_allowed_and_truncation_is_not_repaired():
    r=row();r['correct_indices']=[0,1,2,3]
    v=raw(indices=(0,1,2,3),truths=(True,True,True,True))
    assert review_decision(r,v)['accepted']
    v['finish_reason']='length'
    assert not review_decision(r,v)['accepted']


def test_independent_parser_rejects_invalid_indices_and_extra_fields():
    assert independent_prediction(dict(text='{"answers":[3,0]}',finish_reason='stop'))==[0,3]
    for text in ('{"answers":[true]}','{"answers":[0,0]}','{"answers":[4]}',
                 '{"answers":[]}','{"answers":[0],"reason":"extra"}'):
        assert independent_prediction(dict(text=text,finish_reason='stop')) is None
    assert independent_prediction(dict(text='{"answers":[0]}',finish_reason='length')) is None


def test_amendments_replace_both_orders_without_score_selection():
    old=[dict(id=i,variant=v,text='old') for i in ('a','b') for v in (0,1)]
    new=[dict(id='b',variant=v,text='new') for v in (0,1)]
    assert replace_amended(old,new,{'b'})==old[:2]+new
    with pytest.raises(AssertionError):replace_amended(old,new[:1],{'b'})
    with pytest.raises(AssertionError):replace_amended(old,new+new[:1],{'b'})
    with pytest.raises(AssertionError):replace_amended(old,new,{'a'})
