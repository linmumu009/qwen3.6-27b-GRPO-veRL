import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_grounded_supply_tasks import validate


def row():
    return dict(id='Q1',kind='multiple_choice',question='Select all correct statements.',
      options=['one','two','three','four'],correct_indices=[0,2],answer='',explanation='Both apply.',
      rubric=['Select both'],source_unit_ids=['u'])


def test_valid_multiple():assert validate(row(),{'u'},'u')


def test_duplicate_options():
    q=row();q['options'][1]='ONE'
    assert not validate(q,{'u'},'u')


def test_rejects_out_of_range():
    q=row();q['correct_indices']=[4]
    assert not validate(q,{'u'},'u')


def test_rejects_cross_split_reference():
    q=row();q['source_unit_ids'].append('dev-unit')
    assert not validate(q,{'u'},'u')
