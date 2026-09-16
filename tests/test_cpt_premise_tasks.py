"""Boundary, invariance and nontrivial inverse/optimization task checks."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_cpt_premise_tasks import compile_task, measure, specifications, speed_row, track_row
import build_cpt_premise_tasks as compiler
from verify_cpt_premise_tasks import normalize, verify_calculation


@pytest.mark.parametrize('build,speed,category', [
    ('new_special',249,'N'),('new_special',250,'D'),('new_parallel',250,'D'),
    ('upgraded_conventional',199,'N'),('upgraded_conventional',200,'U'),
    ('upgraded_conventional',270,'U'),('ordinary_conventional',300,'N')])
def test_inclusive_boundaries_and_construction(build,speed,category):
    totals,trace=measure('rail_speed',[speed_row('A',build,speed,100,60)])
    assert totals[category]==1 and trace[0]['category']==category
    assert sum(totals[k] for k in ('D','U','N'))==1


def test_operating_and_connector_speeds_do_not_replace_design():
    for build in ('new_special','upgraded_conventional','ordinary_conventional'):
        for observed,connector in itertools.product((0,100,200,250),repeat=2):
            totals,_=measure('rail_speed',[speed_row('A',build,250,observed,connector)])
            assert totals=={'D':int(build=='new_special'),'U':int(build=='upgraded_conventional'),
                            'N':int(build=='ordinary_conventional'),'eligible':int(build!='ordinary_conventional')}


@pytest.mark.parametrize('constructed,public,current',itertools.product((False,True),repeat=3))
def test_original_construction_and_public_access_conjunction(constructed,public,current):
    totals,trace=measure('rail_network_length',[track_row('A',7,3,built_installation=constructed,public=public,current_mine=current)])
    expected=0 if constructed and not public else 1
    assert totals==dict(route=7*expected,track=21*expected,included=expected,extra=14*expected)
    assert trace[0]['included']==bool(expected)


@pytest.mark.parametrize('mode,tracks',[('metro',2),('tram',1),('light_rail',1),('water',0),('road',0)])
def test_excluded_modes(mode,tracks):
    assert measure('rail_network_length',[track_row('A',5,tracks,mode=mode)])[0]==dict(route=0,track=0,included=0,extra=0)


def test_heritage_and_track_versus_route():
    rows=[track_row('A',4,3),track_row('B',7,2),track_row('C',50,4,heritage=True)]
    assert measure('rail_network_length',rows)[0]==dict(route=11,track=26,included=2,extra=15)


@pytest.mark.parametrize('family,row,field,value',[
    ('rail_speed',speed_row('A','new_special',250,200,90),'main_design_kmh',True),
    ('rail_speed',speed_row('A','new_special',250,200,90),'observed_kmh',251),
    ('rail_network_length',track_row('A',4,2),'open_to_public_traffic','false'),
    ('rail_network_length',track_row('A',4,2),'tracks',0),
    ('rail_network_length',track_row('A',4,2),'length_km',-1)])
def test_rejects_invalid_premises(family,row,field,value):
    row=deepcopy(row);row[field]=value
    with pytest.raises(ValueError):measure(family,[row])


def test_missing_fields_and_duplicate_names():
    row=track_row('A',4,2);del row['constructed_solely_for_installation']
    with pytest.raises(ValueError):measure('rail_network_length',[row])
    with pytest.raises(ValueError):measure('rail_network_length',[track_row('A',3),track_row('A',4)])


def test_inverse_cost_and_branch_answers_without_using_compiler_oracle():
    tasks={s['id']:compile_task(s) for s in specifications()}
    def selected(suffix):
        t=tasks['premise-v1-'+suffix]
        return [t['options'][i] for i in t['correct_indices']]
    assert set(selected('rail_speed-dev-inverse'))=={
        'The unrecorded construction was a specially built new line.',
        'It was new tracks beside existing tracks that remained unupgraded.'}
    assert selected('rail_network_length-dev-inverse')==['Xanthic has three rail pairs.']
    assert selected('rail_speed-dev-minimum')==['Raise Clover main-segment design capability to 250 km/h. Total cost: 8 units.']
    assert selected('rail_network_length-dev-minimum')==['Open Bronze to public traffic, preserving its original construction history and all other premises. Total cost: 7 units.']
    assert selected('rail_speed-sealed-policy')==['if branch L, report 1; if branch R, report 0.']
    assert selected('rail_network_length-sealed-policy')==['if branch L, report 14; if branch R, report 8.']


def test_serialization_keeps_truth_tables_and_split_contract():
    tasks=[compile_task(json.loads(json.dumps(s))) for s in specifications()]
    assert len(tasks)==20 and len({t['id'] for t in tasks})==20
    for family in ('rail_speed','rail_network_length'):
        subset=[t for t in tasks if t['unit']==family]
        assert {s:sum(t['split']==s for t in subset) for s in ('train','dev_expression','dev_conditions','sealed')}=={
            'train':6,'dev_expression':1,'dev_conditions':2,'sealed':1}
    for t in tasks:
        assert t['correct_indices']==[i for i,p in enumerate(t['option_proofs']) if p['true']]
        assert len(set(t['options']))==4 and not t['training_allowed'] and not t['model_baseline_queried']
        assert compile_task(t['premise_spec'])==t


def test_second_calculator_checks_every_registered_design():
    for spec in specifications():verify_calculation(compile_task(spec))


def test_certificate_revision_preserves_other_splits():
    old=specifications();new=specifications('certificate')
    assert [s for s in old if s['split']!='dev_expression']==[s for s in new if s['split']!='dev_expression']
    expected=[dict(Upland='D',Vale='U',Willow='N',Yew='N'),
              dict(Teal='included',Umber='included',Violet='excluded',White='excluded')]
    expressions=[s for s in new if s['split']=='dev_expression']
    for spec,answer in zip(expressions,expected):
        task=compile_task(spec);verify_calculation(task)
        assert spec['operation']=='certificate'
        assert [task['option_proofs'][i]['claimed'] for i in task['correct_indices']]==[answer]
        broken=deepcopy(spec);broken['certificates'][0].pop(next(iter(answer)))
        with pytest.raises(ValueError,match='incomplete certificate'):compile_task(broken)


@pytest.mark.parametrize('field',['question','correct_indices','option_proofs'])
def test_verifier_rejects_tampered_render_labels_and_proofs(field):
    task=compile_task(specifications()[0])
    if field=='question':task[field]+=' Unsupported extra assumption.'
    elif field=='correct_indices':task[field]=[0,1,2,3]
    else:task[field][0]['true']=not task[field][0]['true']
    with pytest.raises(AssertionError):verify_calculation(task)


def test_second_calculator_detects_consistent_wrong_compiler_oracle(monkeypatch):
    spec=specifications()[0]
    original=compiler.measure
    def broken(family,rows):
        totals,trace=original(family,rows);totals['D']+=1
        return totals,trace
    monkeypatch.setattr(compiler,'measure',broken)
    task=compiler.compile_task(spec)
    # Renderer, label and attached proof agree with each other but are all wrong.
    assert compiler.compile_task(spec)==task
    with pytest.raises(AssertionError):verify_calculation(task)


def test_number_normalization_is_only_a_lexical_screen():
    assert normalize('Track length: 13.5 km!')==normalize('Track length: 42 km.')
    assert normalize('route length')!=normalize('track length')
