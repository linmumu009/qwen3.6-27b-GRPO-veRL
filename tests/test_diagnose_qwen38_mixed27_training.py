import json

from scripts.diagnose_qwen38_mixed27_training import summarize_log, summarize_rollouts


def _row(prompt: str, score: float, correct: bool) -> dict:
    return {
        "input": [{"role": "user", "content": prompt}],
        "output": "answer",
        "score": score,
        "base_score": score,
        "final_answer_correct": float(correct),
        "has_final_answer": 1.0,
        "online_eligible": 1.0,
        "safe": 1.0,
        "valid_tool_protocol": 1.0,
        "successful_bash": 1.0,
        "gold_sql_verified": 1.0,
    }


def test_diagnosis_counts_proxy_only_groups_and_logged_updates(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    step_rows = {
        1: [_row("a", 0.1, False), _row("a", 0.2, False), _row("b", 0.1, False), _row("b", 0.7, True)],
        2: [_row("a", 0.15, False), _row("a", 0.25, False), _row("b", 0.2, False), _row("b", 0.3, False)],
    }
    for step, rows in step_rows.items():
        (rollout_dir / f"{step}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    summary, any_correct_steps = summarize_rollouts(rollout_dir, expected_steps=2, rows_per_step=4)
    assert summary["integrity"] == {
        "files": 2,
        "rows": 8,
        "groups": 4,
        "missing_steps": [],
        "wrong_rows_per_step": {},
        "group_sizes": [2],
        "unique_prompts": 2,
        "prompt_exposure_counts": [2],
    }
    assert summary["correctness"]["all_wrong_groups"] == 3
    assert summary["correctness"]["all_wrong_groups_with_nonzero_reward_variance"] == 3
    assert summary["correctness"]["optimizer_steps_with_no_correct"] == 1
    assert any_correct_steps == {1}

    driver = tmp_path / "driver.log"
    driver.write_text(
        "step:1 - training/global_step:0 - actor/ppo_kl:0.001 - actor/grad_norm:1.0\n"
        "step:2 - training/global_step:1 - actor/ppo_kl:0.002 - actor/grad_norm:2.0\n",
        encoding="utf-8",
    )
    log_summary = summarize_log(driver, expected_steps=2, steps_any_correct=any_correct_steps)
    assert log_summary["records"] == 2
    assert log_summary["steps_with_no_correct"]["count"] == 1
    assert log_summary["nonzero_grad_norm_no_correct_steps"] == 1
    assert log_summary["steps_with_no_correct"]["means"]["actor/grad_norm"] == 2.0
