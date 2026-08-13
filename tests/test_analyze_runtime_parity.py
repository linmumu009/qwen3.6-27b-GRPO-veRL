from scripts.analyze_runtime_parity import analyze, safe_summary


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
    result = analyze(
        rows("pi", counts),
        rows("verl", counts),
        semantic_review_passed=True,
        diagnostic_prompts_training_eligible=True,
    )
    assert result["parity_smoke_passed"] is True
    assert result["group_routing"]["fresh_grpo_eligible_mixed_tasks"] == 8
    assert result["group_routing"]["exclude_from_that_optimizer_update_all_correct"] == 1
    assert result["group_routing"]["exclude_from_that_optimizer_update_all_wrong"] == 1
    assert result["group_routing"]["permanent_deletion_allowed"] is False


def test_material_accuracy_shift_fails():
    result = analyze(rows("pi", [0] * 10), rows("verl", [4] * 10))
    assert result["parity_smoke_passed"] is False
    assert result["group_routing"]["runtime_bucket_decision_enabled"] is False
    assert result["group_routing"]["training_selection_enabled"] is False
    assert result["group_routing"]["observed_mixed_tasks"] == 10
    assert result["group_routing"]["candidate_mixed_tasks_after_runtime_parity"] == 0
    assert result["group_routing"]["fresh_grpo_eligible_mixed_tasks"] == 0
    assert result["group_routing"]["exclude_from_that_optimizer_update_all_correct"] == 0
    assert result["group_routing"]["exclude_from_that_optimizer_update_all_wrong"] == 0


def test_runtime_parity_does_not_bypass_semantic_review():
    counts = [0, 8, 4, 3, 5, 2, 6, 7, 1, 4]
    result = analyze(rows("pi", counts), rows("verl", counts))

    assert result["parity_smoke_passed"] is True
    assert result["group_routing"]["candidate_mixed_tasks_after_runtime_parity"] == 8
    assert result["group_routing"]["semantic_review_passed"] is False
    assert result["group_routing"]["fresh_grpo_eligible_mixed_tasks"] == 0


def test_evaluation_only_diagnostics_never_route_directly_to_training():
    counts = [0, 8, 4, 3, 5, 2, 6, 7, 1, 4]
    result = analyze(
        rows("pi", counts),
        rows("verl", counts),
        semantic_review_passed=True,
    )

    assert result["parity_smoke_passed"] is True
    assert result["group_routing"]["diagnostic_prompts_training_eligible"] is False
    assert result["group_routing"]["fresh_grpo_eligible_mixed_tasks"] == 0
    assert result["group_routing"]["parity_pass_only_licenses_screening_on_a_separate_training_pool"] is True


def test_resumed_timeouts_remain_a_permanent_parity_failure():
    counts = [0, 8, 4, 3, 5, 2, 6, 7, 1, 4]
    result = analyze(
        rows("pi", counts),
        rows("verl", counts),
        semantic_review_passed=True,
        diagnostic_prompts_training_eligible=True,
        pi_first_pass_timeout_count=12,
        pi_first_pass_timeout_seconds=1800,
    )

    assert result["parity_checks"]["no_timeouts"] is False
    assert result["parity_smoke_passed"] is False
    assert result["group_routing"]["training_selection_enabled"] is False
    assert result["group_routing"]["exclude_from_that_optimizer_update_all_correct"] == 0
    assert result["group_routing"]["exclude_from_that_optimizer_update_all_wrong"] == 0
    assert result["runtime_audit"]["pi_first_pass_timeout_count"] == 12


def test_safe_summary_has_no_task_identifiers_or_per_task_rows():
    counts = [0, 8, 4, 3, 5, 2, 6, 7, 1, 4]
    result = analyze(rows("pi", counts), rows("verl", counts))

    safe = safe_summary(result)

    assert "per_task" not in safe["pi_agent"]
    assert "per_task" not in safe["verl_rollout"]
    assert "per_task_comparison" not in safe
    assert "task-0" not in str(safe)
    assert safe["parity_smoke_passed"] is True
