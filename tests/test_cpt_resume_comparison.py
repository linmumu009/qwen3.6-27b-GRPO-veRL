from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_cpt_confirmation_pipeline import check_replay


def test_full_match_and_token_order_failure():
    report = dict(replayed_step=30, input_tokens=9768,
        metrics=dict(lr=4.855502637151969e-7, loss=1.8916434049606323, grad_norm=2.6089337369868244))
    assert check_replay(report)['passed']
    report['input_tokens'] -= 1
    assert not check_replay(report)['passed']


def test_incorrect_scheduler_rejected_even_if_loss_matches():
    report = dict(replayed_step=30, input_tokens=9768,
        metrics=dict(lr=5e-7, loss=1.8916434049606323, grad_norm=2.6089337369868244))
    assert not check_replay(report)['checks']['lr']
