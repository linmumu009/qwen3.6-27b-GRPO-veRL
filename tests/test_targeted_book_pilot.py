import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from prepare_targeted_book_pilot import select_training


def test_fixed_pilot_preserves_handling_and_maintenance():
    rows=[{'id':f'{topic}-{i}','topic':topic} for topic,n in
          [('general',23),('warehousing',25),('transport',13),('material_handling',6)] for i in range(n)]
    maintenance=[{'id':f'm-{i}','topic':'general'} for i in range(16)]
    train,reserve=select_training(rows,maintenance)
    assert len(train)==80 and len(reserve)==3
    assert len({r['id'] for r in train})==80
    assert sum(r['topic']=='material_handling' for r in train)==6
    assert not ({r['id'] for r in train}&{r['id'] for r in reserve})
    assert select_training(list(reversed(rows)),maintenance)==(train,reserve)


def test_launcher_budget_and_no_benchmark_training():
    s=(Path(__file__).parents[1]/'scripts/run_targeted_book_pilot_m05.sh').read_text()
    assert 'trainer.total_training_steps=10' in s and 'trainer.save_freq=5' in s
    assert 'data.val_files=null' in s and 'optim.lr=5e-7' in s
    assert "'checkpoint.load_contents=[]'" in s
    assert 'flock -n 9' in s and 'trainer.resume_mode=disable' in s
