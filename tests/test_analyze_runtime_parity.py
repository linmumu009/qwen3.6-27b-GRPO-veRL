from scripts.analyze_runtime_parity import analyze


def rows(runtime: str, counts: list[int]):
    output = []
    for task_index, correct in enumerate(counts):
        for sample_index in range(8):
            hit = sample_index < correct
            output.append(
                {
                    "runtime": runtime,
                    "task_key": f"task-{task_index}",
                    "sample_index": sample_index,
                    "final_answer_correct": float(hit),
                    "has_final_answer": 1.0,
                    "completed": True,
                    "timeout": False,
                    "runtime_error": False,
                }
            )
    return output


def test_identical_distributions_pass_and_route_uniform_groups():
    counts = [0, 8, 4, 3, 5, 2, 6, 7, 1, 4]
    result = analyze(rows("pi", counts), rows("verl", counts))
    assert result["parity_smoke_passed"] is True
    assert result["group_routing"]["fresh_grpo_eligible_mixed_tasks"] == 8
    assert result["group_routing"]["exclude_from_that_optimizer_update_all_correct"] == 1
    assert result["group_routing"]["exclude_from_that_optimizer_update_all_wrong"] == 1
    assert result["group_routing"]["permanent_deletion_allowed"] is False


def test_material_accuracy_shift_fails():
    result = analyze(rows("pi", [0] * 10), rows("verl", [4] * 10))
    assert result["parity_smoke_passed"] is False
    assert result["group_routing"]["decision_enabled"] is False
