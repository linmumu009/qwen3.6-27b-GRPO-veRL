import json
import hashlib
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_cpt_p3_arithmetic import build,generate
from audit_cpt_p3_arithmetic import audit,canonical,evaluate
import ast

def test_release(tmp_path):
    p=tmp_path/'packet';build(p)
    assert audit(p)['unique_canonical_expressions']==120
    assert generate(71437,'train')==generate(71437,'train')

@pytest.mark.parametrize('expr,answer',[('(-12) / (-3)',4),('(4) - (8) - (9)',-13),('(-5) * (3) + (2) * (-4)',-23)])
def test_exact_negative_arithmetic(expr,answer):
    assert evaluate(ast.parse(expr,mode='eval').body)==answer

def test_commutative_duplicates():
    for a,b in [('2+3+4','4+2+3'),('2*3+4*5','5*4+3*2'),('9-3-2','9-2-3')]:
        assert canonical(ast.parse(a,mode='eval').body)==canonical(ast.parse(b,mode='eval').body)

def test_reject_wrong_label_even_with_updated_hash(tmp_path):
    p=tmp_path/'packet';build(p);f=p/'train.private.jsonl'
    rows=[json.loads(x) for x in f.read_text().splitlines()]
    rows[0]['correct_indices']=[(rows[0]['correct_indices'][0]+1)%4]
    f.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    mf=p/'manifest.safe.json';m=json.loads(mf.read_text());m['files'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest();mf.write_text(json.dumps(m))
    with pytest.raises(AssertionError):audit(p)
