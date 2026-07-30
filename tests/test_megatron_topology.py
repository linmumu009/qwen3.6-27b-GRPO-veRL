from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pi_grpo_megatron_tp4_pp2_cp2.sh"


def test_megatron_topology_and_cpu_optimizer_are_explicit() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    expected = (
        'TRAIN_TP="${TRAIN_TP:-4}"',
        'TRAIN_PP="${TRAIN_PP:-2}"',
        'TRAIN_CP="${TRAIN_CP:-2}"',
        'TRAIN_NPUS="${TRAIN_NPUS:-16}"',
        'SAVE_FREQ="${SAVE_FREQ:--1}"',
        "actor_rollout_ref.actor.strategy=megatron",
        "actor_rollout_ref.actor.megatron.param_offload=False",
        "actor_rollout_ref.actor.megatron.optimizer_offload=True",
        "optimizer_cpu_offload=True",
        "optimizer_offload_fraction=1",
        "context_parallel_algo=kvallgather_cp_algo",
        "actor_rollout_ref.model.lora_rank=0",
        "Megatron-Bridge-de93536e/src",
        "MEGATRON_BRIDGE_ROOT}/megatron/bridge",
        "trainer.save_freq=\"${SAVE_FREQ}\"",
        "export CUDA_DEVICE_MAX_CONNECTIONS=1",
    )

    for item in expected:
        assert item in text


def test_megatron_topology_has_world_size_guard() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "TRAIN_TP * TRAIN_PP * TRAIN_CP != TRAIN_NPUS" in text
    assert "ROLLOUT_NPUS % ROLLOUT_TP != 0" in text


def test_ascend_bridge_compatibility_patch_is_applied_before_launch() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    patch_call = 'python3 "${PROJECT_ROOT}/scripts/patch_verl_megatron_bridge_compat.py"'
    launch_call = "python3 -m verl.experimental.one_step_off_policy.main_ppo"

    assert patch_call in text
    assert text.index(patch_call) < text.index(launch_call)


def test_bridge_patch_is_idempotent(tmp_path: Path) -> None:
    import importlib.util

    patch_path = ROOT / "scripts" / "patch_verl_megatron_bridge_compat.py"
    spec = importlib.util.spec_from_file_location("bridge_patch", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = tmp_path / "bridge.py"
    target.write_text(module.OLD, encoding="utf-8")

    assert module.patch(target) == "patched"
    assert module.patch(target) == "already-patched"
    assert "llin_verl.megatron_bridge_compat" in target.read_text(encoding="utf-8")

    peft_target = tmp_path / "megatron_utils.py"
    peft_target.write_text(module.PEFT_OLD + module.PEFT_HOOK_OLD, encoding="utf-8")
    assert module.patch_peft_import(peft_target) == "patched"
    assert module.patch_peft_import(peft_target) == "already-patched"
    peft_text = peft_target.read_text(encoding="utf-8")
    assert peft_text.index("if peft_cls is not None") < peft_text.index("create_peft_hook")


def test_bridge_compat_backports_provider_finalization() -> None:
    text = (ROOT / "llin_verl" / "megatron_bridge_compat.py").read_text(encoding="utf-8")

    assert "_install_model_provider_compatibility()" in text
    assert 'hasattr(ModelProviderMixin, "apply_overrides_and_finalize")' in text
    assert "self.params_dtype = dtype" in text
    assert "setattr(self, name, value)" in text
    assert "self.finalize()" in text
    assert "def create_ddp_config(" in text
    assert "DistributedDataParallelConfig(**values)" in text
