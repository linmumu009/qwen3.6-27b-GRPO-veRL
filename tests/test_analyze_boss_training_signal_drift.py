import pytest

from scripts.analyze_boss_training_signal_drift import analyze


def _row(step, prompt, score, correct):
    return {
        "_file_step": step,
        "input": prompt,
        "score": score,
        "boss_reward": score,
        "boss_result_score": score,
        "boss_process_score": score,
        "boss_answer_correct": correct,
        "boss_numbers_match": correct,
        "boss_fields_used": 1.0,
        "boss_task_fit": 1.0,
        "has_final_answer": 1.0,
        "required_table_used": 1.0,
        "evidence_reward": score,
        "final_answer_correct": correct,
        "sql_evidence_correct": correct,
        "bash_command_count": 1.0,
    }


def test_analyze_uses_observed_step_range_for_quartiles():
    rows = []
    for step in range(102, 202):
        score = 0.2 if step < 127 else (0.8 if step >= 177 else 0.5)
        rows.extend(_row(step, f"prompt-{step % 2}", score, score > 0.5) for _ in range(4))

    result = analyze(rows)

    assert result["step_range"] == [102, 201]
    assert result["first_quartile"]["step_range"] == [102, 126]
    assert result["last_quartile"]["step_range"] == [177, 201]
    assert result["last_minus_first_mean_deltas"]["score_mean"] == pytest.approx(0.6)
    assert result["prompt_exposure"]["unique_prompts"] == 2
    assert result["prompt_exposure"]["prompt_count_by_group_exposure"] == {"50": 2}
    assert result["group_count"] == 100
    assert result["within_group_signal"]["total_groups"] == 100
    assert result["within_group_signal"]["invalid_group_size_count"] == 0
    assert result["within_group_signal"]["zero_score_variance_count"] == 100
