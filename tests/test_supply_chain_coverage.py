import pytest
from scripts.audit_supply_chain_coverage import validate


def row():
    return dict(id='x', primary_domain='warehousing', secondary_domains=[],
                capabilities=['calculation'], specific_topics=['capacity'],
                usable_evidence='partial', limitation='Missing figure')


def test_valid_classification():
    assert validate({'items': [row()]}, ['x'])[0]['usable_evidence'] == 'partial'


@pytest.mark.parametrize('change', [dict(id='wrong'), dict(primary_domain='invented'),
                                    dict(capabilities=[]), dict(specific_topics=[]),
                                    dict(usable_evidence='guaranteed')])
def test_rejects_invalid_classification(change):
    with pytest.raises(ValueError):
        validate({'items': [dict(row(), **change)]}, ['x'])
