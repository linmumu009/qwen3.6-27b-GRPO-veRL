from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_standalone_runner_keeps_sampling_and_context_contract_variable_but_explicit():
    text = (ROOT / "scripts" / "run_runtime_parity_verl_standalone.py").read_text(
        encoding="utf-8"
    )
    for fragment in (
        '"--samples-per-task", type=int, default=8',
        '"--max-num-seqs", type=int, default=24',
        '"--max-prompt-tokens", type=int, default=4096',
        '"--max-response-tokens", type=int, default=45056',
        '"--max-context-tokens", type=int, default=49152',
        'actor_rollout_ref.rollout.val_kwargs.temperature=1.0',
        'actor_rollout_ref.rollout.val_kwargs.top_p=0.95',
        'actor_rollout_ref.rollout.val_kwargs.top_k=20',
        'actor_rollout_ref.rollout.val_kwargs.do_sample=True',
        'prompt + response token budgets must equal max context',
        '"--trajectory-timeout-seconds", type=float, default=900.0',
        'trajectory_abort_acknowledged_count',
        '"--model-label", default="step120"',
        '"--policy-step", type=int, default=120',
        '"global_steps": args.policy_step',
        '"native_hf_checkpoint"',
        '"llin_megatron_to_hf_export"',
        'tail_batch_padding_policy',
        'contract["trajectory_admission"] = trajectory_admission_contract(',
        'contract["rolling_admission"] = rolling_admission_contract(',
        'worker.generate_sequences.remote(unit)',
        'ray.wait(list(inflight), num_returns=1)',
        'batch.padding(padding_rows, padding_candidate="last")',
        'output = output[:expected]',
        'stamp_trajectory_enqueue(unit, epoch_ns=enqueued_epoch_ns)',
        'stamp_trajectory_enqueue(batch, epoch_ns=enqueued_epoch_ns)',
        'enqueued_epoch_ns = time.time_ns()',
        '"__num_turns__"',
        'trajectory_queue_wait_seconds',
        'trajectory_generation_seconds',
        'trajectory_tool_seconds',
        'trajectory_total_seconds',
        'trajectory_timeout_partial_response_tokens',
    ):
        assert fragment in text


def test_ray_start_defaults_unchanged_and_allow_distinct_dual_rollout_resources():
    m05 = (ROOT / "scripts" / "start_ray_m05.sh").read_text(encoding="utf-8")
    m06 = (ROOT / "scripts" / "start_ray_m06.sh").read_text(encoding="utf-8")
    assert 'RAY_ROLE_RESOURCE="${RAY_ROLE_RESOURCE:-llin_trainer}"' in m05
    assert 'RAY_ROLE_RESOURCE="${RAY_ROLE_RESOURCE:-llin_rollout}"' in m06
    assert '--resources="{\\"${RAY_ROLE_RESOURCE}\\": 1}"' in m05
    assert '--resources="{\\"${RAY_ROLE_RESOURCE}\\": 1}"' in m06
    for text in (m05, m06):
        assert 'python3 "${PROJECT_ROOT}/scripts/patch_verl_abort_partial_tokens.py"' in text


def test_launcher_passes_concurrency_context_and_sampling_shape_to_runner():
    text = (ROOT / "scripts" / "launch_multisandbox_dwh_standalone.sh").read_text(
        encoding="utf-8"
    )
    for argument in (
        "--expected-tasks",
        "--model-label",
        "--policy-step",
        "--samples-per-task",
        "--task-batch-size",
        "--max-num-seqs",
        "--agent-workers",
        "--max-prompt-tokens",
        "--max-response-tokens",
        "--max-context-tokens",
        "--trajectory-timeout-seconds",
        "--rolling-admission",
        "--rolling-window-trajectories",
    ):
        assert argument in text
    assert 'export LLIN_ROLLOUT_RESOURCE="${ROLLOUT_RESOURCE}"' in text
    assert "analyze_multisandbox_dwh_rollout.py" in text
    assert '[[ "${code}" == "0" && "${ANALYZE_ON_SUCCESS}" == "1" ]]' in text
    assert "monitor_npu_utilization.py" in text
    assert '--until-file "${OUTPUT_DIR}/exit_code"' in text
    assert 'MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"' in text
    assert 'TRAJECTORY_TIMEOUT_SECONDS="${TRAJECTORY_TIMEOUT_SECONDS:-900}"' in text
    assert 'ROLLING_ADMISSION="${ROLLING_ADMISSION:-0}"' in text
    assert 'MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"' in text


def test_plan_first_finalizer_recovers_remote_per_task_and_runs_paired_comparison():
    finalizer = (ROOT / "scripts" / "finalize_plan_first_dwh_model_comparison.py").read_text(
        encoding="utf-8"
    )
    launcher = (
        ROOT / "scripts" / "launch_plan_first_dwh_model_comparison_finalizer.sh"
    ).read_text(encoding="utf-8")
    assert "NodeAffinitySchedulingStrategy" in finalizer
    assert '"per_task": per_task.read_bytes()' in finalizer
    assert "compare(" in finalizer
    assert "step120_mixed_training_candidates" in finalizer
    assert "finalizer_exit_code" in launcher


def test_pi_loop_physically_aborts_before_cancelling_timed_out_task():
    text = (ROOT / "llin_verl" / "pi_agent_loop.py").read_text(encoding="utf-8")
    abort = text.index("await self.server_manager.abort_request")
    cancel = text.index("task.cancel()")
    assert abort < cancel
    assert "trajectory_timeout\": True" in text
    assert "await WORKSPACES.release(request_id)" in text
    assert 'kwargs["__llin_request_id"] = request_id' in text
    assert "__llin_request_id=request_id, **kwargs" not in text
    assert "ContextVar" in text
    assert "self._llin_trajectory_telemetry" not in text
    assert 'abort_report.get("partial_response_tokens", 0)' in text
    assert "telemetry.timeout_active_generation_tokens" in text


def test_pi_loop_timeout_placeholder_preserves_request_version_columns_for_mixed_batches():
    text = (ROOT / "llin_verl" / "pi_agent_loop.py").read_text(encoding="utf-8")
    defaults = text[text.index("        defaults = {") : text.index("        for key, value in defaults.items():")]

    assert '"global_steps": None' in defaults
    assert '"min_global_steps": None' in defaults
    assert '"max_global_steps": None' in defaults


def test_dual_finalizer_waits_on_remote_node_without_consuming_npu_resource():
    text = (ROOT / "scripts" / "finalize_multisandbox_dwh_dual_server.py").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts" / "launch_multisandbox_dwh_dual_finalizer.sh").read_text(
        encoding="utf-8"
    )
    assert "NodeAffinitySchedulingStrategy" in text
    assert "ray.remote(num_cpus=0.1)(wait_for_arm)" in text
    assert "timeout_trajectories" in text
    assert "finalizer_exit_code" in launcher
    assert "FINALIZER_TIMEOUT_SECONDS" in launcher
    assert "export PROJECT_ROOT LOCAL_ARM_DIR REMOTE_ARM_DIR OUTPUT_DIR" in launcher
    assert "export RAY_ADDRESS REMOTE_RESOURCE" in launcher
