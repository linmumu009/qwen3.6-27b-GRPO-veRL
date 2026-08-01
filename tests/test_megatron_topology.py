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
        'python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_oversampling.py"',
        'python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_abort_observability.py"',
        'python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_abort_retry.py"',
        'python3 "${PROJECT_ROOT}/scripts/patch_verl_vllm_abort_api.py"',
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
        'FASTEST_K="${FASTEST_K:-4}"',
        'OVERSAMPLE_CANDIDATES="${OVERSAMPLE_CANDIDATES:-6}"',
        'PREWARM_GROUPS="${PREWARM_GROUPS:-0}"',
        'STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-0.5}"',
        'AGENT_WORKERS="${AGENT_WORKERS:-8}"',
        'actor_rollout_ref.rollout.multi_turn.max_parallel_calls="${MAX_PARALLEL_TOOL_CALLS}"',
        "actor_rollout_ref.rollout.n=4",
        "rollout.n=4",
        "async_training.trigger_parameter_sync_step=1",
        "async_training.partial_rollout=True",
        '+async_training.max_queue_tokens="${MAX_QUEUE_TOKENS}"',
        '+async_training.fastest_k="${FASTEST_K}"',
        '+async_training.oversample_candidates="${OVERSAMPLE_CANDIDATES}"',
        '+async_training.prewarm_groups="${PREWARM_GROUPS}"',
        'async_training.staleness_threshold="${STALENESS_THRESHOLD}"',
        'actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS}"',
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
        for patch_call in (
            'python3 "${PROJECT_ROOT}/scripts/patch_verl_fully_async_group_token_queue.py"',
            'python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_oversampling.py"',
            'python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_abort_observability.py"',
            'python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_abort_retry.py"',
            'python3 "${PROJECT_ROOT}/scripts/patch_verl_vllm_abort_api.py"',
            'python3 "${PROJECT_ROOT}/scripts/patch_verl_fully_async_observability.py"',
        ):
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


def test_fastest_k_oversampling_patch_is_idempotent_and_preserves_group_size(tmp_path: Path) -> None:
    import importlib.util

    patch_path = ROOT / "scripts" / "patch_verl_fastest_k_oversampling.py"
    spec = importlib.util.spec_from_file_location("fastest_k_oversampling_patch", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source_root = ROOT / "reference" / "verl" / "verl"
    targets = {
        "rollouter": (
            tmp_path / "fully_async_rollouter.py",
            source_root / "experimental" / "fully_async_policy" / "fully_async_rollouter.py",
            module.patch_rollouter,
        ),
        "agent": (
            tmp_path / "agent_loop.py",
            source_root / "experimental" / "agent_loop" / "agent_loop.py",
            module.patch_agent_loop,
        ),
        "tool": (
            tmp_path / "tool_agent_loop.py",
            source_root / "experimental" / "agent_loop" / "tool_agent_loop.py",
            module.patch_tool_agent_loop,
        ),
        "client": (
            tmp_path / "llm_server.py",
            source_root / "workers" / "rollout" / "llm_server.py",
            module.patch_llm_client,
        ),
    }
    for target, source, patch in targets.values():
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        assert patch(target) == "patched"
        assert patch(target) == "already-patched"
        compile(target.read_text(encoding="utf-8"), str(target), "exec")

    rollouter_text = targets["rollouter"][0].read_text(encoding="utf-8")
    assert "LLIN_FASTEST_K_OVERSAMPLE_BATCH" in rollouter_text
    assert "full_batch[:1].repeat(" in rollouter_text
    assert "fastest_k != expected_group_size" in rollouter_text

    agent_text = targets["agent"][0].read_text(encoding="utf-8")
    assert "LLIN_FASTEST_K_QUORUM" in agent_text
    assert "return_when=asyncio.FIRST_COMPLETED" in agent_text
    assert "reset_prefix_cache=False" in agent_text
    assert "selected_non_tensor_batch" in agent_text

    tool_text = targets["tool"][0].read_text(encoding="utf-8")
    assert 'kwargs.get("__llin_request_id")' in tool_text

    client_text = targets["client"][0].read_text(encoding="utf-8")
    assert "LLIN_FASTEST_K_PHYSICAL_ABORT" in client_text
    assert "server.abort_request.remote(physical_id, reset_prefix_cache)" in client_text
    assert '"reset_prefix_cache": bool(reset_prefix_cache)' in client_text


def test_fastest_k_abort_observability_upgrade_is_idempotent(tmp_path: Path) -> None:
    import importlib.util

    base_path = ROOT / "scripts" / "patch_verl_fastest_k_oversampling.py"
    base_spec = importlib.util.spec_from_file_location("fastest_k_base_patch", base_path)
    assert base_spec and base_spec.loader
    base = importlib.util.module_from_spec(base_spec)
    base_spec.loader.exec_module(base)

    upgrade_path = ROOT / "scripts" / "patch_verl_fastest_k_abort_observability.py"
    upgrade_spec = importlib.util.spec_from_file_location("fastest_k_abort_upgrade", upgrade_path)
    assert upgrade_spec and upgrade_spec.loader
    upgrade = importlib.util.module_from_spec(upgrade_spec)
    upgrade_spec.loader.exec_module(upgrade)

    retry_path = ROOT / "scripts" / "patch_verl_fastest_k_abort_retry.py"
    retry_spec = importlib.util.spec_from_file_location("fastest_k_abort_retry", retry_path)
    assert retry_spec and retry_spec.loader
    retry = importlib.util.module_from_spec(retry_spec)
    retry_spec.loader.exec_module(retry)

    source_root = ROOT / "reference" / "verl" / "verl"
    agent = tmp_path / "agent_loop.py"
    client = tmp_path / "llm_server.py"
    agent.write_text(
        (source_root / "experimental" / "agent_loop" / "agent_loop.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    client.write_text(
        (source_root / "workers" / "rollout" / "llm_server.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert base.patch_agent_loop(agent) == "patched"
    assert base.patch_llm_client(client) == "patched"
    assert upgrade.patch_agent_loop(agent) == "patched"
    assert upgrade.patch_client(client) == "patched"
    assert upgrade.patch_agent_loop(agent) == "already-patched"
    assert upgrade.patch_client(client) == "already-patched"
    assert retry.patch_agent_loop(agent) == "patched"
    assert retry.patch_client(client) == "patched"
    assert retry.patch_agent_loop(agent) == "already-patched"
    assert retry.patch_client(client) == "already-patched"
    assert upgrade.patch_agent_loop(agent) == "already-patched"
    assert upgrade.patch_client(client) == "already-patched"

    agent_text = agent.read_text(encoding="utf-8")
    client_text = client.read_text(encoding="utf-8")
    assert "LLIN_FASTEST_K_QUORUM_V3" in agent_text
    assert "active_requests=" in agent_text
    assert "abort_failures=" in agent_text
    assert "LLIN_FASTEST_K_PHYSICAL_ABORT_V3" in client_text
    assert '"acknowledged_count"' in client_text
    assert '"retry_exhausted_count"' in client_text
    compile(agent_text, str(agent), "exec")
    compile(client_text, str(client), "exec")


def test_vllm_abort_patch_uses_public_external_id_api(tmp_path: Path) -> None:
    import importlib.util

    patch_path = ROOT / "scripts" / "patch_verl_vllm_abort_api.py"
    spec = importlib.util.spec_from_file_location("vllm_abort_api_patch", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = (
        ROOT
        / "reference"
        / "verl"
        / "verl"
        / "workers"
        / "rollout"
        / "vllm_rollout"
        / "vllm_async_server.py"
    )
    target = tmp_path / "vllm_async_server.py"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    assert module.patch_server(target) == "patched"
    assert module.patch_server(target) == "already-patched"
    patched = target.read_text(encoding="utf-8")
    assert "LLIN_VLLM_PUBLIC_ABORT_V4" in patched
    assert "external_req_ids" in patched
    assert "await self.engine.abort(request_id)" in patched
    assert "request_states.get(request_id)" not in patched
    compile(patched, str(target), "exec")


def test_fully_async_observability_patch_is_idempotent(tmp_path: Path) -> None:
    import importlib.util

    patch_path = ROOT / "scripts" / "patch_verl_fully_async_observability.py"
    spec = importlib.util.spec_from_file_location("fully_async_observability_patch", patch_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source_root = ROOT / "reference" / "verl" / "verl" / "experimental" / "fully_async_policy"
    trainer_target = tmp_path / "fully_async_trainer.py"
    main_target = tmp_path / "fully_async_main.py"
    trainer_target.write_text(
        (source_root / "fully_async_trainer.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    main_target.write_text(
        (source_root / "fully_async_main.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert module.patch_trainer(trainer_target) == "patched"
    assert module.patch_trainer(trainer_target) == "already-patched"
    assert module.patch_main(main_target) == "patched"
    assert module.patch_main(main_target) == "already-patched"

    trainer_text = trainer_target.read_text(encoding="utf-8")
    main_text = main_target.read_text(encoding="utf-8")
    compile(trainer_text, str(trainer_target), "exec")
    compile(main_text, str(main_target), "exec")
    assert "LLIN_FULLY_ASYNC_STAGE_TIMING" in trainer_text
    assert "[LLIN_QUEUE_STAGE]" in trainer_text
    assert "[LLIN_TRAIN_STAGE]" in trainer_text
    assert "update_actor_s=" in trainer_text
    assert "LLIN_FULLY_ASYNC_PREWARM" in main_text
    assert "[LLIN_PREWARM]" in main_text
    assert "rollouter exited before prewarm completed" in main_text


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
