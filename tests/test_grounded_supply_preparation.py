import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_grounded_supply_training import clean, input_key, messages


def row():
    return dict(id='x',unit_id='u',chapter=1,split='train',source_unit_ids=['u'],kind='single_choice',question='Select',options=['one','two'],correct_indices=[0],answer='',explanation='Supported explanation.')


def test_input_includes_options():
    a=row();b=row();b['options']=['three','four']
    assert input_key(a)!=input_key(b)


def test_duplicate_and_conflict():
    a=row();b=row();b['id']='y'
    units=[dict(id='u',chapter=1,split='train')]
    kept,rejected=clean([a,b],units)
    assert len(kept)==len(rejected)==1
    b['correct_indices']=[1]
    with pytest.raises(AssertionError):clean([a,b],units)


def test_cross_split_reference_rejected():
    a=row();a['source_unit_ids'].append('v')
    with pytest.raises(AssertionError):
        clean([a],[dict(id='u',chapter=1,split='train'),dict(id='v',chapter=2,split='dev')])


def test_format_does_not_change_answer():
    a=row();m=messages(a)
    assert m[-1]['content'].endswith('{"answers":[0]}')
    a.update(kind='short_answer',answer='Original answer')
    assert messages(a)[-1]['content']=='Original answer'
