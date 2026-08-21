from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sealed6_eval_is_one_host_strict_2plus2plus2_and_fail_closed() -> None:
    source = (ROOT / "scripts" / "run_qwen38_mixed27_sealed6_eval_host.py").read_text(
        encoding="utf-8"
    )
    assert 'POLICY_STEP = 54' in source
    assert 'SEALED_TASKS = 6' in source
    assert '"--tensor-parallel-size",\n        "4"' in source
    assert '"--data-parallel-size",\n        "4"' in source
    assert '"--task-batch-size",\n        "6"' in source
    assert '"--rolling-window-trajectories",\n        "80"' in source
    assert '"max_context_tokens": 94208' in source
    assert '"trajectory_timeout_seconds": 1800' in source
    assert '"training_allowed": False' in source
    assert '"promotion_allowed": False' in source


def test_sealed6_eval_compares_pretraining_and_posttraining_aggregates() -> None:
    source = (ROOT / "scripts" / "run_qwen38_mixed27_sealed6_eval_host.py").read_text(
        encoding="utf-8"
    )
    assert 'baseline_before_mixed27_training' in source
    assert 'tasks_with_any_correct_delta' in source
    assert 'correct_trajectories_delta' in source
    assert 'contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths' in source
