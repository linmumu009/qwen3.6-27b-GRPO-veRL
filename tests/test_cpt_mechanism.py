"""Budget and isolation failures must stop before training."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cpt_mechanism import validate_budgets


def b(seq=89599,loss=13273,n=333):
    return dict(sequence_tokens=seq,supervised_tokens=loss,records=n)


def test_registered_actual_budgets():
    validate_budgets(b(),b(94954,13438),6)


@pytest.mark.parametrize('other,g',[ (b(seq=120001),6),(b(loss=30001),6),(b(loss=14000),6),
                                   (b(seq=99000),6),(b(n=334),6),(b(),7),(b(n=315),0)])
def test_fail_closed(other,g):
    with pytest.raises(AssertionError):validate_budgets(b(),other,g)


def test_no_formal_inputs_in_builder():
    import inspect,prepare_cpt_mechanism
    source=inspect.getsource(prepare_cpt_mechanism.prepare)
    assert 'frozen_cases' not in source and 'correct_indices' not in source and 'prediction' not in source


def test_no_training_restart_or_score_branch():
    import inspect,run_cpt_mechanism
    source=inspect.getsource(run_cpt_mechanism.main)
    assert "if not training.exists():" in source
    assert "for arm in ('K','R')" in source
    assert "['correct']" not in source


def test_seed_contract_replaces_hidden_sampler_default():
    from types import SimpleNamespace
    from train_cpt_mechanism_sft import install_seed_contract
    class Sampler:
        seed=0
        def set_epoch(self,e):self.epoch=e
    class Trainer:
        def _build_dataloader(self):self.train_sampler=Sampler()
    install_seed_contract(Trainer)
    t=Trainer();t.config=SimpleNamespace(trainer=SimpleNamespace(seed=1),engine=SimpleNamespace(seed=1))
    t._build_dataloader()
    assert (t.train_sampler.seed,t.train_sampler.epoch)==(1,0)
    t.config.engine.seed=42
    with pytest.raises(AssertionError):t._build_dataloader()


def test_blind_packet_has_no_arm_or_repeat_identity(tmp_path):
    import json
    from summarize_cpt_mechanism import blind
    prompts=[dict(id='u-rule',messages=[dict(role='user',content='Explain')],required=['fact'],forbidden=['overclaim'])]
    q=tmp_path/'probes.json';q.write_text(json.dumps(prompts))
    for model in ('step120','p1','K','R'):
        d=tmp_path/(model+'_direct');d.mkdir()
        rows=[dict(id='u-rule',unit_id='u',kind='rule',repeat=i,prediction='reply',finish_reason='stop') for i in range(3)]
        (d/'predictions.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    blind(tmp_path,q,tmp_path/'blind')
    packet=json.loads((tmp_path/'blind/review.private.json').read_text())
    assert len(packet)==12 and len({r['blind_id'] for r in packet})==12
    assert all(not {'model','arm','repeat'} & set(r) for r in packet)
