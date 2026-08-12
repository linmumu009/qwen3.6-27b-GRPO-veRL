import pytest

from scripts.analyze_critical_token_canary import analyze_critical_token_recovery


def diagnostic(rows):
    return {
        "contract": "repair-sft-teacher-forced-component-diagnostic-v3",
        "task_ids": [row["task_id"] for row in rows],
        "data_sha256": "same-data",
        "per_task": rows,
    }


def rank_row(task_id, offset, rank, target_id=101, probability=0.1):
    return {
        "task_id": task_id,
        "sql_token_rank": {
            "first_nongreedy_offset": offset,
            "first_nongreedy_rank": rank,
            "first_nongreedy_target_id": target_id,
            "first_nongreedy_target_probability": probability,
        },
    }


def comparison(post_above_half=2):
    return {
        "contract": "repair-sft-teacher-forced-prepost-comparison-v2",
        "task_ids_identical": True,
        "data_sha256_identical": True,
        "forward_only_both": True,
        "optimizer_initialized_either": False,
        "components": {
            "sql_shell": {
                "step120_mean_nll": 1.3,
                "post_sft_mean_nll": 1.1,
                "per_task_nll": {"improved": 3},
                "per_task_probability": {
                    "step120_above_0_5": 1,
                    "post_sft_above_0_5": post_above_half,
                },
            }
        },
    }


def test_audit_classifies_recovered_still_nongreedy_and_earlier_regression():
    critical_rows = [
        {
            "task_id": task_id,
            "critical_sql_token_offset": 2,
            "critical_sql_target_id": 101,
            "critical_token_family": family,
        }
        for task_id, family in (
            ("a", "aggregation_function"),
            ("b", "query_start"),
            ("c", "identifier_or_literal"),
        )
    ]
    baseline = diagnostic(
        [rank_row("a", 2, 9), rank_row("b", 2, 8), rank_row("c", 2, 7)]
    )
    post = diagnostic(
        [rank_row("a", 4, 3, target_id=202), rank_row("b", 2, 4), rank_row("c", 1, 5)]
    )

    result = analyze_critical_token_recovery(
        critical_rows, baseline, post, comparison(), required_full_sql_probability_tasks=3
    )

    assert result["critical_token_recovery"]["status_counts"] == {
        "new_earlier_nongreedy_blocks_direct_attribution": 1,
        "original_critical_became_greedy": 1,
        "original_critical_still_first_nongreedy": 1,
    }
    assert result["critical_token_recovery"]["still_nongreedy_rank_direction"] == {
        "improved": 1
    }
    assert result["full_correction_sql_gate"]["passed"] is False
    assert result["decision"] == "stop_no_replay_probability_gate_failed"


def test_audit_allows_short_replay_only_after_full_sql_probability_gate():
    rows = [
        {
            "task_id": "a",
            "critical_sql_token_offset": 2,
            "critical_sql_target_id": 101,
            "critical_token_family": "aggregation_function",
        }
    ]
    result = analyze_critical_token_recovery(
        rows,
        diagnostic([rank_row("a", 2, 9)]),
        diagnostic([rank_row("a", None, None, target_id=None, probability=None)]),
        comparison(post_above_half=1),
        required_full_sql_probability_tasks=1,
    )

    assert result["critical_token_recovery"]["status_counts"] == {
        "original_critical_became_greedy": 1
    }
    assert result["decision"] == "eligible_for_short_replay"


def test_audit_rejects_frozen_offset_drift():
    rows = [
        {
            "task_id": "a",
            "critical_sql_token_offset": 3,
            "critical_sql_target_id": 101,
            "critical_token_family": "query_start",
        }
    ]

    with pytest.raises(ValueError, match="frozen offset"):
        analyze_critical_token_recovery(
            rows,
            diagnostic([rank_row("a", 2, 9)]),
            diagnostic([rank_row("a", 2, 7)]),
            comparison(),
        )
