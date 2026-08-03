from pathlib import Path

import importlib.util


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


def test_val_only_patch_skips_only_unresumed_base_model_sync(tmp_path):
    patch_path = ROOT / "scripts" / "patch_verl_val_only_skip_initial_sync.py"
    spec = importlib.util.spec_from_file_location("val_only_sync_patch", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = (
        ROOT
        / "reference"
        / "verl"
        / "verl"
        / "experimental"
        / "one_step_off_policy"
        / "ray_trainer.py"
    )
    target = tmp_path / "ray_trainer.py"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    assert module.patch(target) == "patched"
    assert module.patch(target) == "already-patched"
    patched = target.read_text(encoding="utf-8")
    assert "LLIN_VAL_ONLY_SKIP_INITIAL_SYNC" in patched
    assert 'self.config.trainer.get("val_only", False)' in patched
    assert 'self.config.trainer.get("resume_mode", "disable") == "disable"' in patched
    assert "else:\n            self._fit_update_weights()" in patched
    compile(patched, str(target), "exec")
