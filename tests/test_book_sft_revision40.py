import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_book_sft_revision40 import valid_ids, audit_status


def test_ids():
    assert valid_ids(['S001'],{'S001':'source'})
    assert not valid_ids(['S001','S001'],{'S001':'source'})
    assert not valid_ids(['wrong'],{'S001':'source'})


def test_material_and_minor_are_distinct():
    v=dict.fromkeys(('closed_book_answerable','answers_question','scope_correct','all_answer_claims_covered','type_match'),True)
    v.update(claims=[{'claim':'supported fact','supported':True,'source_ids':['S001']}],material_issues=[],minor_notes=['Style only'])
    assert audit_status(v,{'S001':'source'})=='auto_pass'
    v['closed_book_answerable']=False
    assert audit_status(v,{'S001':'source'})=='audit_reject'
    v['closed_book_answerable']=True
    assert audit_status(v,{})=='invalid_audit_evidence'
    assert audit_status({}, {})=='invalid_audit'
