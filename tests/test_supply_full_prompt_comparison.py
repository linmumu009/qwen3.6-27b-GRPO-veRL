import pytest
from scripts.run_supply_full_prompt_comparison import partition,vote


def test_partition_disjoint_balanced_complete():
    rows=[dict(item_hash=str(i),dataset='a' if i%3 else 'b',options=list(range(269 if i%5==0 else 4))) for i in range(1672)]
    a,b=partition(rows)
    assert len(a)==len(b)==836
    assert {r['item_hash'] for r in a+b}=={str(i) for i in range(1672)}
    assert partition(rows)==partition(list(reversed(rows)))


def test_vote_counts_truncated_wrong():
    rows=[dict(repeat=i,parsed=[0],valid=False,correct=False) for i in range(3)]
    assert not vote(rows)['correct']
    rows[0].update(valid=True,correct=True);rows[1].update(valid=True,correct=True)
    assert vote(rows)['correct'] and not vote(rows)['stable']


def test_vote_rejects_duplicate_repeat():
    with pytest.raises(AssertionError):vote([dict(repeat=0)]*3)
