from scripts.extract_book_learning_units import valid_unit
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recover_book_learning_units import repair


def make(quote):
    return dict(id='U01',domain='warehousing',concept='test',definition='test definition',
                distinguishing_features=[],conditions=[],contrasts=[],rule_or_formula='',
                source_ids=['S001'],source_quotes=[dict(source_id='S001',quote=quote)])


def test_cross_boundary_exact_span():
    quote='A verified statement spanning a source boundary.'
    full='x'*790+quote+'y'*800
    fixed,refs=repair(make(quote),full)
    assert fixed['definition']=='test definition'
    assert fixed['source_quotes'][0]['quote']==quote
    assert fixed['source_quotes'][0]['span_start']==790
    assert valid_unit(fixed,refs)


def test_missing_quote_rejected():
    assert repair(make('An invented quotation not in source.'),'x'*1600) is None


def test_ambiguous_relocation_rejected():
    quote='An exact repeated quotation.'
    assert repair(make(quote),'x'*801+quote+'x'*801+quote) is None
