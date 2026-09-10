import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_cpt_logists_pipeline import audit_training


def fixture():
    lengths=[392]*3015+[1183197-392*3015]
    metrics={i+1:{'train/global_tokens':sum(lengths[i*8:(i+1)*8]),'train/loss':2.,'train/grad_norm':1.,'train/lr':5e-7} for i in range(377)}
    return metrics,lengths


def test_complete_exact_epoch():
    metrics,lengths=fixture()
    assert audit_training(metrics,lengths)['steps']==377


def test_reject_missing_step():
    metrics,lengths=fixture();metrics.pop(377)
    with pytest.raises(AssertionError):audit_training(metrics,lengths)


def test_reject_wrong_token_budget():
    metrics,lengths=fixture();metrics[1]['train/global_tokens']-=1
    with pytest.raises(AssertionError):audit_training(metrics,lengths)
