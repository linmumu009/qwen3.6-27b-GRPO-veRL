import pytest
from scripts.extract_book_learning_units import valid_unit


SOURCE = {'S001': 'A unit load combines items into a single load for handling.'}


def example():
    return dict(id='U01', domain='material_handling', concept='Unit load',
                definition='Items combined into one load for handling.',
                distinguishing_features=[], conditions=[], contrasts=[], rule_or_formula='',
                source_ids=['S001'], source_quotes=[{'source_id':'S001', 'quote':SOURCE['S001']}])


def test_valid_grounded_unit():
    assert valid_unit(example(), SOURCE)


@pytest.mark.parametrize('change', [dict(source_ids=['S002']), dict(source_ids=['S001','S001']),
    dict(source_quotes=[{'source_id':'S001','quote':'This sentence is not actually in the source.'}]),
    dict(contrasts=[{'concept':'Pallet'}]), dict(domain='unknown'), dict(id='U13'),
    dict(definition='word '*221)])
def test_reject_invalid_unit(change):
    assert not valid_unit(dict(example(), **change), SOURCE)
