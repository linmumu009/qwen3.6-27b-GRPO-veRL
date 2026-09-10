import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_cpt_erp_pipeline import audit_training


def fixture_metrics():
    lengths=[690]*46+[775]
    return {i+1:{'train/global_tokens':n,'train/loss':2.,'train/grad_norm':1.,'train/lr':1e-7} for i,n in enumerate(lengths)}, lengths


def test_exact_epoch_budget():
    metrics,lengths=fixture_metrics()
    assert audit_training(metrics,lengths)['sequence_tokens']==32515


def test_missing_sample_rejected():
    metrics,lengths=fixture_metrics();metrics.pop(47)
    with pytest.raises(AssertionError):audit_training(metrics,lengths)


def test_same_total_wrong_exposure_rejected():
    metrics,lengths=fixture_metrics()
    metrics[1]['train/global_tokens']+=1;metrics[2]['train/global_tokens']-=1
    with pytest.raises(AssertionError):audit_training(metrics,lengths)
