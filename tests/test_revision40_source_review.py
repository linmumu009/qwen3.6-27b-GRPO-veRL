import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('review40',Path(__file__).parents[1]/'scripts/record_revision40_source_review.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_reproducible_stratification():
    kinds=('definition','comparison','conditions','application')
    rows=[{'id':f'revision40-{i+1:03d}','kind':kinds[i%4]} for i in range(40)]
    result=m.selected(rows)
    assert result==m.selected(list(reversed(rows)))
    assert set(r['id'] for r in result)==set(m.DECISIONS)
    assert all(sum(r['kind']==k for r in result)==2 for k in kinds)
