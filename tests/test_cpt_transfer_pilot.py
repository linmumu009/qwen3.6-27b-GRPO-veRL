from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_cpt_transfer_pilot import UNITS,RETENTION,forward,conditional,sealed,retention_task,value
from verify_cpt_transfer_pilot import check,reference
from run_cpt_transfer_pilot import comparison
from verify_cpt_pilot_result import independently_parse


@pytest.mark.parametrize('unit',list(UNITS))
def test_all_frozen_family_designs_and_independent_calculations(unit):
    tasks=[forward(unit,'train',i) for i in range(6)]+[forward(unit,'dev_expression',i) for i in range(2)]+[conditional(unit,i) for i in range(2)]+[sealed(unit)]
    for t in tasks:
        t=json.loads(json.dumps(t));check(t)
        for r in t['premises']['rows']:assert value(unit,r)==reference(unit,r)
        bad=deepcopy(t);bad['correct_indices']=[0,1,2,3]
        with pytest.raises(AssertionError):check(bad)


@pytest.mark.parametrize('unit',list(RETENTION))
def test_retention_arithmetic_and_exclusion_from_training_units(unit):
    assert unit not in UNITS
    count={'abc':3,'packages':5,'moment':6,'payload':6}.get(unit,10)
    for i in range(count):check(retention_task(unit,i))


def test_actual_boundaries_not_observed_usage():
    assert value('injury',dict(hours=24,accident=True,suicide=False))=='excluded'
    assert value('injury',dict(hours=25,accident=True,suicide=False))=='included'
    assert value('injury',dict(hours=40,accident=True,suicide=True))=='excluded'
    assert value('capacity',dict(capacity=20,design='inland freight',load=0))=='included'
    assert value('capacity',dict(capacity=40,design='inland passengers',load=20))=='excluded'
    assert value('derailment',dict(collision=True,wheels=1))=='collision'
    assert value('railcar',dict(powered=1,pairs=2,trailers=7,cabs=0))==5
    assert value('propulsion',dict(propulsion='auxiliary-only',tanker=False,tow=True))=='excluded'
    assert value('pipeline',dict(length=20,active=True,national=True,seabed=True))==20


def prediction_rows():
    counts={'train':60,'dev_expression':20,'dev_conditions':20,'retention':80}
    rows=[]
    for split,count in counts.items():
        for i in range(count):
            rows.append(dict(source_id=f'{split}-{i}',dataset='p1_'+split,category=f'family-{i%10}',item_hash=f'h-{split}-{i}',prediction='{"answers":[0]}',parsed=[0],expected=[0],valid=True,finish_reason='stop',correct=True))
    return rows


def test_no_change_does_not_pass_transfer_gate():
    rows=prediction_rows();result=comparison(rows,deepcopy(rows))
    assert result['next_action']=='hold_expansion_review_failure_strata'
    assert result['prospective_signal_gates']['retention']


def test_truncated_output_cannot_count_as_correct():
    rows=prediction_rows();post=deepcopy(rows)
    post[0]['finish_reason']='length'
    with pytest.raises(AssertionError):comparison(rows,post)
    post[0]['correct']=False
    result=comparison(rows,post)
    assert result['strata']['p1_train']['losses']==1


def test_positive_signal_requires_two_families_in_both_strata():
    post=prediction_rows();before=deepcopy(post)
    for r in before:
        if r['dataset'] in ('p1_dev_expression','p1_dev_conditions') and r['source_id'].endswith(('-0','-1')):
            r.update(prediction='{"answers":[1]}',parsed=[1],correct=False)
    assert comparison(before,post)['next_action']=='broader_data_and_confirmation_review'


@pytest.mark.parametrize('text,expected,valid',[
    ('{"answers":[3,0,3]}',[0,3],True),
    ('prefix {"answers":"0, 2"} suffix',[0,2],True),
    ('{"answers":[true]}',[],False),
    ('{"answers":[4]}',[],False),
    ('{"answers":[]}',[],False),
    ('{"answers":[0],"extra":1}',[],False)])
def test_independent_answer_parser_matches_frozen_semantics(text,expected,valid):
    assert independently_parse(text,4)==(expected,valid)
