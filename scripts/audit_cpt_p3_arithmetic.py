"""Independently evaluate published integer expressions using AST and Fraction."""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

def evaluate(n):
    if isinstance(n,ast.Constant) and type(n.value) is int:return Fraction(n.value)
    if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -evaluate(n.operand)
    assert isinstance(n,ast.BinOp)
    a,b=evaluate(n.left),evaluate(n.right)
    if isinstance(n.op,ast.Add):return a+b
    if isinstance(n.op,ast.Sub):return a-b
    if isinstance(n.op,ast.Mult):return a*b
    assert isinstance(n.op,ast.Div) and b!=0
    return a/b

def canonical(n):
    if isinstance(n,(ast.Constant,ast.UnaryOp)):return ('n',int(evaluate(n)))
    if isinstance(n.op,ast.Add):
        def terms(x):return terms(x.left)+terms(x.right) if isinstance(x,ast.BinOp) and isinstance(x.op,ast.Add) else [canonical(x)]
        return ('add',*sorted(terms(n)))
    if isinstance(n.op,ast.Mult):return ('mul',*sorted((canonical(n.left),canonical(n.right))))
    if isinstance(n.op,ast.Sub):return ('sub',canonical(n.left.left),*sorted((canonical(n.left.right),canonical(n.right))))
    assert isinstance(n.op,ast.Div)
    return ('div',canonical(n.left),canonical(n.right))

def audit(packet):
    manifest=json.loads((packet/'manifest.safe.json').read_text());allkeys=set();records={}
    for name,digest in manifest['files'].items():assert hashlib.sha256((packet/name).read_bytes()).hexdigest()==digest
    for split in ('train','check'):
        rows=[json.loads(x) for x in (packet/(split+'.private.jsonl')).read_text().splitlines()]
        assert len(rows)==60 and len({x['id'] for x in rows})==60
        assert Counter(x['operation'] for x in rows)==Counter(dict(add3=15,div_exact=15,products_sum=15,sub2=15))
        for x in rows:
            assert x['split']==split
            tree=ast.parse(x['question'],mode='eval').body;answer=evaluate(tree)
            assert answer.denominator==1 and int(answer)==x['answer']
            values=[int(v) for v in x['options']];assert len(values)==len(set(values))==4
            assert [i for i,v in enumerate(values) if v==answer]==x['correct_indices']
            key=canonical(tree);assert key not in allkeys;allkeys.add(key)
        assert any(x['answer']<0 for x in rows) and any(x['answer']>100 for x in rows)
        records[split]=rows
    cases=[json.loads(x) for x in (packet/'cases.private.jsonl').read_text().splitlines()]
    assert len(cases)==60
    for c,x in zip(cases,records['check']):
        assert c['source_id']==x['id'] and c['question']==x['question'] and c['options']==x['options'] and c['expected']==x['correct_indices']
    result=dict(passed=True,train=60,check=60,unique_canonical_expressions=120,
        manifest_sha256=hashlib.sha256((packet/'manifest.safe.json').read_bytes()).hexdigest(),model_queries=0)
    (packet/'release.safe.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);print(json.dumps(audit(p.parse_args().packet)))
