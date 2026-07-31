from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pi_grpo_megatron_tp4_pp2_cp2.sh"
FULLY_ASYNC_SCRIPT = ROOT / "scripts" / "run_pi_grpo_fully_async_tp4_pp2_cp2.sh"


def test_megatron_topology_and_cpu_optimizer_are_explicit() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    expected = (
        'TRAIN_TP="${TRAIN_TP:-4}"',
        'TRAIN_PP="${TRAIN_PP:-2}"',
        'TRAIN_CP="${TRAIN_CP:-2}"',
        'TRAIN_NPUS="${TRAIN_NPUS:-16}"',
        'ROLLOUT_NPUS="${ROLLOUT_NPUS:-16}"',
        'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"',
        'SAVE_FREQ="${SAVE_FREQ:-20}"',
        'WEIGHT_BUCKET_MB="${WEIGHT_BUCKET_MB:-3072}"',
        'MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-49152}"',
        'MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"',
        'MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-45056}"',
        'MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-25}"',
        'MAX_USER_TURNS="${MAX_USER_TURNS:-24}"',
        'MAX_PARALLEL_TOOL_CALLS="${MAX_PARALLEL_TOOL_CALLS:-4}"',
        'MAX_TOOL_RESPONSE_CHARS="${MAX_TOOL_RESPONSE_CHARS:-32768}"',
        'VLLM_ROOT="${VLLM_ROOT:-/vllm}"',
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
        "trainer.total_training_steps=\"${TOTAL_TRAINING_STEPS}\"",
        "trainer.total_epochs=\"${TOTAL_TRAINING_STEPS}\"",
        "trainer.max_actor_ckpt_to_keep=1",
        "'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]'",
        "data.continuous_token.enable=True",
        "data.continuous_token.model_family=qwen35",
        "actor_rollout_ref.rollout.enable_prefix_caching=True",
        "actor_rollout_ref.rollout.disable_log_stats=False",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        'actor_rollout_ref.rollout.max_model_len="${MAX_CONTEXT_TOKENS}"',
        'actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS}"',
        'actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS}"',
        'actor_rollout_ref.rollout.multi_turn.max_parallel_calls="${MAX_PARALLEL_TOOL_CALLS}"',
        'checkpoint_engine.update_weights_bucket_megabytes="${WEIGHT_BUCKET_MB}"',
        "export CUDA_DEVICE_MAX_CONNECTIONS=1",
    )

    for item in expected:
        assert item in text


def test_megatron_topology_has_world_size_guard() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "TRAIN_TP * TRAIN_PP * TRAIN_CP != TRAIN_NPUS" in text
    assert "ROLLOUT_NPUS % ROLLOUT_TP != 0" in text


def test_hccl_fanout_defaults_to_one_trainer_and_sixteen_rollout_ranks() -> None:
    text = (ROOT / "scripts" / "check_hccl_fanout.py").read_text(encoding="utf-8")

    assert 'ROLLOUT_RANKS = int(os.getenv("ROLLOUT_RANKS", "16"))' in text
    assert "WORLD_SIZE = 1 + ROLLOUT_RANKS" in text


def test_ascend_bridge_compatibility_patch_is_applied_before_launch() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    patch_call = 'python3 "${PROJECT_ROOT}/scripts/patch_verl_megatron_bridge_compat.py"'
    launch_call = "python3 -m verl.experimental.one_step_off_policy.main_ppo"

    assert patch_call in text
    assert text.index(patch_call) < text.index(launch_call)


def test_ascend_vllm_dp_weight_sync_patch_is_applied_before_launch() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    patch_call = 'python3 "${PROJECT_ROOT}/scripts/patch_verl_vllm_dp_weight_sync.py"'
    launch_call = "python3 -m verl.experimental.one_step_off_policy.main_ppo"

    assert patch_call in text
    assert text.index(patch_call) < text.index(launch_call)


def test_one_step_dump_executor_patch_is_applied_before_launch() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    patch_call = 'python3 "${PROJECT_ROOT}/scripts/patch_verl_one_step_dump_executor.py"'
    launch_call = "python3 -m verl.experimental.one_step_off_policy.main_ppo"

    assert patch_call in text
    assert text.index(patch_call) < text.index(launch_call)


def test_one_step_continuous_token_patch_is_applied_before_launch() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    patch_call = 'python3 "${PROJECT_ROOT}/scripts/patch_verl_one_step_continuous_token.py"'
    launch_call = "python3 -m verl.experimental.one_step_off_policy.main_ppo"

    assert patch_call in text
    assert text.index(patch_call) < text.index(launch_call)


def test_agent_loop_continuous_token_patch_is_applied_before_launch() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    patch_call = 'python3 "${PROJECT_ROOT}/scripts/patch_verl_agent_loop_continuous_token.py"'
    launch_call = "python3 -m verl.experimental.one_step_off_policy.main_ppo"

    assert patch_call in text
    assert text.index(patch_call) < text.index(launch_call)


def test_rollout_node_applies_worker_side_patches_before_ray_start() -> None:
    text = (ROOT / "scripts" / "start_ray_m06.sh").read_text(encoding="utf-8")
    ray_start = "ray start"
    for patch_call in (
        'python3 "${PROJECT_ROOT}/scripts/patch_verl_vllm_dp_weight_sync.py"',
        'python3 "${PROJECT_ROOT}/scripts/patch_verl_agent_loop_continuous_token.py"',
    ):
        assert patch_call in text
        assert text.index(patch_call) < text.index(ray_start)


def test_fully_async_launch_preserves_topology_groups_and_token_bound() -> None:
    text = FULLY_ASYNC_SCRIPT.read_text(encoding="utf-8")
    expected = (
        "verl.experimental.fully_async_policy.fully_async_main",
        'TRAIN_TP="${TRAIN_TP:-4}"',
        'TRAIN_PP="${TRAIN_PP:-2}"',
        'TRAIN_CP="${TRAIN_CP:-2}"',
        'ROLLOUT_TP="${ROLLOUT_TP:-8}"',
        'ROLLOUT_NPUS="${ROLLOUT_NPUS:-16}"',
        'GROUPS_PER_STEP="${GROUPS_PER_STEP:-4}"',
        'MAX_QUEUE_TOKENS="${MAX_QUEUE_TOKENS:-$((GROUPS_PER_STEP * 4 * MAX_CONTEXT_TOKENS))}"',
        'MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-49152}"',
        'MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-25}"',
        'MAX_PARALLEL_TOOL_CALLS="${MAX_PARALLEL_TOOL_CALLS:-4}"',
        'actor_rollout_ref.rollout.multi_turn.max_parallel_calls="${MAX_PARALLEL_TOOL_CALLS}"',
        "actor_rollout_ref.rollout.n=4",
        "rollout.n=4",
        "async_training.staleness_threshold=0.5",
        "async_training.trigger_parameter_sync_step=1",
        "async_training.partial_rollout=True",
        '+async_training.max_queue_tokens="${MAX_QUEUE_TOKENS}"',
        "actor_rollout_ref.actor.use_rollout_log_probs=True",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.actor.optim.lr_decay_style=constant",
        'actor_rollout_ref.actor.optim.lr_decay_steps="${TOTAL_TRAINING_STEPS}"',
        "data.continuous_token.enable=True",
        "actor_rollout_ref.rollout.enable_prefix_caching=True",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
    )
    for item in expected:
        assert item in text


def test_both_ray_nodes_apply_fully_async_queue_patch_before_start() -> None:
    for name in ("start_ray_m05.sh", "start_ray_m06.sh"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        patch_call = 'python3 "${PROJECT_ROOT}/scripts/patch_verl_fully_async_group_token_queue.py"'
        assert patch_call in text
        assert text.index(patch_call) < text.index("ray start")


def test_vllm_dp_weight_sync_patch_is_idempotent(tmp_path: Path) -> None:
    import importlib.util

    patch_path = ROOT / "scripts" / "patch_verl_vllm_dp_weight_sync.py"
    spec = importlib.util.spec_from_file_location("vllm_dp_patch", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = tmp_path / "utils.py"
    target.write_text(module.OLD, encoding="utf-8")

    assert module.patch(target) == "patched"
    assert module.patch(target) == "already-patched"
    patched = target.read_text(encoding="utf-8")
    assert 'os.environ.get("ASCEND_RT_VISIBLE_DEVICES")' in patched
    assert "visible_ranks[worker_local_rank % len(visible_ranks)]" in patched


def test_one_step_dump_executor_patch_is_idempotent(tmp_path: Path) -> None:
    import importlib.util

    patch_path = ROOT / "scripts" / "patch_verl_one_step_dump_executor.py"
    spec = importlib.util.spec_from_file_location("one_step_dump_patch", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = tmp_path / "ray_trainer.py"
    target.write_text(module.OLD, encoding="utf-8")

    assert module.patch(target) == "patched"
    assert module.patch(target) == "already-patched"
    patched = target.read_text(encoding="utf-8")
    assert "self._init_dump_executor()" in patched
    assert patched.index("self._init_dump_executor()") < patched.index("SeparateRayPPOTrainer config")


def test_one_step_continuous_token_patch_is_idempotent(tmp_path: Path) -> None:
    import importlib.util

    patch_path = ROOT / "scripts" / "patch_verl_one_step_continuous_token.py"
    spec = importlib.util.spec_from_file_location("one_step_continuous_token_patch", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = tmp_path / "main_ppo.py"
    target.write_text(module.OLD, encoding="utf-8")

    assert module.patch(target) == "patched"
    assert module.patch(target) == "already-patched"
    patched = target.read_text(encoding="utf-8")
    assert 'continuous_token.get("enable", False)' in patched
    assert 'not config.data.get("return_multi_modal_inputs", True)' in patched
    assert "processor = None" in patched


def test_agent_loop_continuous_token_patch_is_idempotent(tmp_path: Path) -> None:
    import importlib.util

    patch_path = ROOT / "scripts" / "patch_verl_agent_loop_continuous_token.py"
    spec = importlib.util.spec_from_file_location("agent_loop_continuous_token_patch", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = tmp_path / "agent_loop.py"
    target.write_text(module.OLD, encoding="utf-8")

    assert module.patch(target) == "patched"
    assert module.patch(target) == "already-patched"
    patched = target.read_text(encoding="utf-8")
    assert 'not self.data_config.get("return_multi_modal_inputs", True)' in patched
    assert "self.processor = None" in patched
    assert patched.index("self.processor = None") < patched.index(
        "continuous_token_config.enable and self.processor is None"
    )


def test_fully_async_group_token_queue_patch_is_idempotent(tmp_path: Path) -> None:
    import importlib.util

    patch_path = ROOT / "scripts" / "patch_verl_fully_async_group_token_queue.py"
    spec = importlib.util.spec_from_file_location("fully_async_group_queue_patch", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    queue_target = tmp_path / "message_queue.py"
    queue_target.write_text(
        (
            ROOT
            / "reference"
            / "verl"
            / "verl"
            / "experimental"
            / "fully_async_policy"
            / "message_queue.py"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    rollouter_target = tmp_path / "fully_async_rollouter.py"
    rollouter_target.write_text(
        (
            ROOT
            / "reference"
            / "verl"
            / "verl"
            / "experimental"
            / "fully_async_policy"
            / "fully_async_rollouter.py"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert module.patch_message_queue(queue_target) == "patched"
    assert module.patch_message_queue(queue_target) == "already-patched"
    assert module.patch_rollouter(rollouter_target) == "patched"
    assert module.patch_rollouter(rollouter_target) == "already-patched"

    queue_text = queue_target.read_text(encoding="utf-8")
    assert "LLIN_GROUP_TOKEN_QUEUE" in queue_text
    assert "await self._producer_condition.wait()" in queue_text
    assert "self.queue.popleft()" in queue_text
    assert "self.dropped_samples += 1" not in queue_text
    rollouter_text = rollouter_target.read_text(encoding="utf-8")
    assert "LLIN_GROUP_TOKEN_COUNT" in rollouter_text
    assert 'full_batch.batch["attention_mask"].sum()' in rollouter_text


def test_fully_async_continuous_token_patch_is_idempotent(tmp_path: Path) -> None:
    import importlib.util

    patch_path = ROOT / "scripts" / "patch_verl_fully_async_continuous_token.py"
    spec = importlib.util.spec_from_file_location("fully_async_continuous_token_patch", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = tmp_path / "fully_async_main.py"
    target.write_text(module.OLD, encoding="utf-8")
    assert module.patch(target) == "patched"
    assert module.patch(target) == "already-patched"
    patched = target.read_text(encoding="utf-8")
    assert 'continuous_token.get("enable", False)' in patched
    assert "processor = None" in patched


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
