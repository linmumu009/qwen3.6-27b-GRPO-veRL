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
    ):
        assert fragment in text


def test_ray_start_defaults_unchanged_and_allow_distinct_dual_rollout_resources():
    m05 = (ROOT / "scripts" / "start_ray_m05.sh").read_text(encoding="utf-8")
    m06 = (ROOT / "scripts" / "start_ray_m06.sh").read_text(encoding="utf-8")
    assert 'RAY_ROLE_RESOURCE="${RAY_ROLE_RESOURCE:-llin_trainer}"' in m05
    assert 'RAY_ROLE_RESOURCE="${RAY_ROLE_RESOURCE:-llin_rollout}"' in m06
    assert '--resources="{\\"${RAY_ROLE_RESOURCE}\\": 1}"' in m05
    assert '--resources="{\\"${RAY_ROLE_RESOURCE}\\": 1}"' in m06


def test_launcher_passes_concurrency_context_and_sampling_shape_to_runner():
    text = (ROOT / "scripts" / "launch_multisandbox_dwh_standalone.sh").read_text(
        encoding="utf-8"
    )
    for argument in (
        "--expected-tasks",
        "--samples-per-task",
        "--task-batch-size",
        "--max-num-seqs",
        "--agent-workers",
        "--max-prompt-tokens",
        "--max-response-tokens",
        "--max-context-tokens",
    ):
        assert argument in text
    assert 'export LLIN_ROLLOUT_RESOURCE="${ROLLOUT_RESOURCE}"' in text
    assert "analyze_multisandbox_dwh_rollout.py" in text
    assert '[[ "${code}" == "0" && "${ANALYZE_ON_SUCCESS}" == "1" ]]' in text
    assert "monitor_npu_utilization.py" in text
    assert '--until-file "${OUTPUT_DIR}/exit_code"' in text
