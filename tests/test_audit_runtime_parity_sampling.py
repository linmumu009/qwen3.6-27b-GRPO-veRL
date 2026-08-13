import json

from scripts.audit_runtime_parity_sampling import audit_verl, summarize


def test_summarize_detects_eight_unique_samples_per_task():
    groups = {
        f"task-{task}": [f"trajectory-{task}-{sample}" for sample in range(8)]
        for task in range(10)
    }

    result = summarize(groups, expected_tasks=10, expected_n=8)

    assert result["rows"] == 80
    assert result["group_size_histogram"] == {8: 10}
    assert result["unique_trajectories_per_task_histogram"] == {8: 10}
    assert result["all_samples_unique_groups"] == 10
    assert result["duplicate_pair_fraction"] == 0
    assert result["every_group_all_samples_unique"] is True


def test_audit_verl_groups_by_prompt_without_exporting_content(tmp_path):
    path = tmp_path / "0.jsonl"
    rows = [
        {"input": f"prompt-{task}", "output": f"answer-{task}-{sample}"}
        for task in range(2)
        for sample in range(3)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = audit_verl(path, expected_tasks=2, expected_n=3)

    assert result["complete_shape"] is True
    assert result["every_group_all_samples_unique"] is True
    assert "prompt-0" not in str(result)
    assert "answer-0-0" not in str(result)


def test_summarize_detects_copied_or_greedy_identical_groups():
    groups = {f"task-{task}": [f"same-{task}"] * 8 for task in range(10)}

    result = summarize(groups, expected_tasks=10, expected_n=8)

    assert result["all_samples_identical_groups"] == 10
    assert result["duplicate_pair_fraction"] == 1
    assert result["every_group_all_samples_unique"] is False
