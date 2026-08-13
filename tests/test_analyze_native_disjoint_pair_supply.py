import pytest

from scripts.analyze_native_disjoint_pair_supply import (
    EVAL_CONTRACT,
    eval_task_ids,
    forbidden_task_id,
    summarize_supply,
)


def test_forbidden_task_id_prefers_derived_source_identity():
    row = {
        "source_task_id": "source-task",
        "task_id": "display-task::chosen",
        "reward_model": {"ground_truth": {"task_id": "truth-task"}},
    }
    assert forbidden_task_id(row) == "source-task"


def test_supply_excludes_frozen_eval_tasks_and_stays_fail_closed():
    classified = {
        "eval-a": {
            "outcome": "observed_first_query_error",
            "first_error_category": "executable_wrong_or_insufficient_evidence",
        },
        "fresh-b": {
            "outcome": "observed_first_query_error",
            "first_error_category": "schema_syntax_or_execution_error",
        },
        "fresh-c": {"outcome": "no_readonly_query"},
    }
    result = summarize_supply(classified, {"eval-a"}, {"fresh-b"})
    assert result["native_observed_first_query_errors"] == 2
    assert result["native_error_overlap_with_eval22"] == 1
    assert result["native_error_states_outside_eval22"] == 1
    assert result["additional_frozen_overlap_outside_eval22"] == 1
    assert result["native_error_states_outside_all_frozen_sets"] == 0
    assert result["outside_all_frozen_first_error_category_counts"] == {}
    assert result["states_are_training_ready_pairs"] is False
    assert result["training_allowed"] is False
    assert result["contains_prompts_sql_answers_task_ids_tool_outputs_or_server_paths"] is False


def test_eval_contract_requires_training_prohibition_and_exact_id_count():
    contract = {
        "contract": EVAL_CONTRACT,
        "evaluation_only": True,
        "may_be_used_as_training_data": False,
        "pairs": 2,
        "evidence": [{"task_id": "a"}, {"task_id": "b"}],
    }
    assert eval_task_ids(contract) == {"a", "b"}

    contract["may_be_used_as_training_data"] = True
    with pytest.raises(ValueError, match="prohibition"):
        eval_task_ids(contract)
