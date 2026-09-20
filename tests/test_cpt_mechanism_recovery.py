import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recover_cpt_mechanism_init import inspect_failure
from wait_cpt_mechanism import stable_idle


def fixture(tmp_path):
    d=tmp_path/'K_training';d.mkdir()
    (d/'status.txt').write_text('failed_at_line_29\n')
    for i in range(16):
        r=d/f'torchrun_logs/attempt/attempt_0/{i}';r.mkdir(parents=True)
        (r/'stderr.log').write_text('_init_engine\nport 16666 have already been bound')
        (r/'stdout.log').write_text('initializing')
    return d


def test_proven_initialization_only(tmp_path):
    assert inspect_failure(fixture(tmp_path))['training_steps']==0


@pytest.mark.parametrize('failure',['checkpoint','progress','missing_rank','unknown_error'])
def test_reject_uncertain_recovery(tmp_path,failure):
    d=fixture(tmp_path);r=d/'torchrun_logs/attempt/attempt_0/0'
    if failure=='checkpoint':(d/'checkpoints').mkdir()
    elif failure=='progress':(r/'stdout.log').write_text('train/global_tokens: 300')
    elif failure=='missing_rank':(r/'stderr.log').unlink()
    else:(r/'stderr.log').write_text('unrelated error')
    with pytest.raises(AssertionError):inspect_failure(d)


def test_idle_window_resets_on_activity():
    since,ready=stable_idle(None,0,True);assert not ready
    since,ready=stable_idle(since,60,True);assert not ready
    since,ready=stable_idle(since,120,True);assert ready
    since,ready=stable_idle(since,121,False);assert since is None and not ready
    since,ready=stable_idle(since,180,True);assert since==180 and not ready
