from copy import deepcopy
import json
from pathlib import Path
import pytest
from scripts.audit_supply_sft_rethink import collapse,paired


def base():
    return [dict(item_hash=str(i),question='synthetic',options=['x','y'],expected=[0],parsed=[0],parse_ok=True,correct=True,dataset='synthetic',category='synthetic',question_type='single_choice') for i in range(1672)]


def test_no_majority_is_wrong():
    r=base();s=deepcopy(r);t=deepcopy(r)
    s[0].update(parsed=[1],correct=False);t[0].update(parsed=[],parse_ok=False,correct=False)
    out=collapse([r,s,t])
    assert out['0']['nomajority'] and not out['0']['correct'] and not out['0']['stable']
    assert out['1']['correct'] and out['1']['stable']


def test_mismatched_case_rejected():
    a=base();b=deepcopy(a);b[0]['expected']=[1]
    with pytest.raises(AssertionError):collapse([a,b,a])


def test_pair_conservation():
    a={0:{'correct':True},1:{'correct':False},2:{'correct':True}}
    b={0:{'correct':False},1:{'correct':True},2:{'correct':False}}
    p=paired(a,b,a)
    assert p==dict(n=3,before=2,after=1,gains=1,losses=2)


def test_saved_audit_conservation():
    d=json.loads((Path(__file__).resolve().parents[1]/'docs/supply_chain_rethink_20260909.safe.json').read_text())
    total=d['pairs']['all']
    for field in ('dataset','category','kind','long'):
        for key in ('n','before','after','gains','losses'):
            assert sum(p[key] for p in d['pairs'][field].values())==total[key]
    assert 96*3*(96+96+2048+2048)==1234944
