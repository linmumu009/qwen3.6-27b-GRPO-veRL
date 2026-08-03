from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_baseline_is_read_only_and_uses_full_pi_contract():
    script = (ROOT / "scripts" / "run_pi_frozen_baseline.sh").read_text(encoding="utf-8")

    assert "MAX_ASSISTANT_TURNS:-26" in script
    assert "MAX_USER_TURNS:-25" in script
    assert "pi_workspace_tools.yaml" in script
    assert "pi_agent_loops.yaml" in script
    assert "trainer.val_only=True" in script
    assert "trainer.val_before_train=True" in script
    assert "actor_rollout_ref.actor.megatron.forward_only=True" in script
    assert "actor_rollout_ref.actor.megatron.optimizer_offload=False" in script
    assert "actor_rollout_ref.actor.megatron.grad_offload=False" in script
    assert "trainer.save_freq=-1" in script
    assert "actor_rollout_ref.rollout.val_kwargs.n=1" in script
    assert "actor_rollout_ref.rollout.val_kwargs.do_sample=False" in script


def test_formal_builder_emits_combined_baseline_file():
    script = (ROOT / "scripts" / "prepare_pi_formal_dataset.py").read_text(encoding="utf-8")

    assert '"pi_formal_all.parquet"' in script
    assert "all_records.extend(records)" in script
