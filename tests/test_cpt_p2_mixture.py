from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cpt_p2_mixture import mixture,rank

def fixture():
    tasks=[dict(id=f'family-{f}-task-{i}',unit=str(f)) for f in range(10) for i in range(6)]
    messages=[dict(id=f'p1-replay-{i}',messages=[{'content':str(i)}]) for i in range(135)]
    messages += [dict(id=t['id']+f'-exposure{e}',messages=[{'content':t['id']+str(e)}]) for t in tasks for e in range(3)]
    random.Random(20260916).shuffle(messages)
    return messages,tasks

def test_preregistered_mixture_and_no_rewrite():
    messages,tasks=fixture();frozen=deepcopy(messages)
    new,mapping=mixture(messages,tasks)
    assert messages==frozen and len(new)==315 and len(mapping)==60
    assert len([x for x in new if x['id'].startswith('p1-replay-')])==135
    assert len([x for x in new if x['id'].startswith('p2-extra-')])==60
    old={r['id']:r for r in messages}
    for m in mapping:assert new[m['position']]['messages']==old[m['copied_id']]['messages']
    assert sum(a==b for a,b in zip(new,messages))==255
    removed=Counter(m['removed_id'].split('-exposure')[1] for m in mapping)
    assert removed=={'0':20,'1':20,'2':20}
    expected=sorted((x['id'] for x in messages if x['id'].startswith('p1-replay-')),key=rank)[:60]
    assert [m['copied_id'] for m in mapping]==expected
    assert mixture(messages,list(reversed(tasks)))==(new,mapping)

def test_missing_exposure_fails_closed():
    messages,tasks=fixture();messages[0]['id']='unexpected'
    with pytest.raises(AssertionError):mixture(messages,tasks)

def test_unbalanced_family_fails_closed():
    messages,tasks=fixture();tasks[0]['unit']='1'
    with pytest.raises(AssertionError):mixture(messages,tasks)
