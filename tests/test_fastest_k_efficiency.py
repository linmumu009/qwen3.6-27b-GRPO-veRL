import json
from pathlib import Path

import pytest

from scripts.analyze_fastest_k_efficiency import parse_driver_text, summarize_rollouts


def test_parse_driver_text_separates_queue_wait_from_actor_compute():
    text = """
[LLIN_PREWARM] groups=8 queued_tokens=104761 wait_s=375.756814
[LLIN_FASTEST_K] candidates=6 selected=4 discarded=2 completed_discarded=1 physical_aborts=1 quorum_s=120.0 active_requests=1 abort_acks=1 abort_not_active=1 abort_completed=0 abort_retry_exhausted=0 abort_failures=0 reset_prefix_cache=False
[LLIN_TRAIN_STAGE] step=1 queue_wait_s=0.05 deserialize_s=0.01 assemble_s=0.02 reward_s=0.001 old_log_prob_s=0.0 ref_log_prob_s=0.0 adv_s=0.002 update_actor_s=176.0 step_s=184.0
[LLIN_TRAIN_STAGE] step=2 queue_wait_s=10.0 deserialize_s=0.01 assemble_s=0.02 reward_s=0.001 old_log_prob_s=0.0 ref_log_prob_s=0.0 adv_s=0.002 update_actor_s=20.0 step_s=40.0
timing_s/timing_s/param_sync:7.5 - fully_async/count/dropped_stale_samples:0.0
"""

    result = parse_driver_text(text)

    assert result["prewarm"] == {"groups": 8, "queued_tokens": 104761, "wait_s": 375.756814}
    assert result["step_count"] == 2
    assert result["steady_state"]["steps"] == 1
    assert result["steady_state"]["queue_wait_s"]["mean"] == 10.0
    assert result["steady_state"]["actor_update_s"]["mean"] == 20.0
    assert result["steady_state"]["trainer_idle_ratio"] == pytest.approx(0.25)
    assert result["fastest_k"]["physical_aborts_total"] == 1
    assert result["fastest_k"]["active_requests_total"] == 1
    assert result["fastest_k"]["abort_acks_total"] == 1
    assert result["fastest_k"]["abort_not_active_total"] == 1
    assert result["fastest_k"]["abort_completed_total"] == 0
    assert result["fastest_k"]["abort_retry_exhausted_total"] == 0
    assert result["fastest_k"]["abort_failures_total"] == 0
    assert result["param_sync_s"]["mean"] == 7.5
    assert result["dropped_stale_samples_final"] == 0.0


def test_summarize_rollouts_keeps_historical_and_strict_accuracy_separate(tmp_path: Path):
    rows = [
        {
            "gts": {"verifier_id": "task-a"},
            "score": 1.0,
            "answer_correct": 1.0,
            "evidence_contains_expected": 1.0,
            "final_answer_correct": 0.0,
            "required_table_used": 1.0,
            "tool_used": 1.0,
            "output": "tool evidence contains the number but final answer is wrong",
        },
        {
            "gts": {"verifier_id": "task-b"},
            "score": 1.0,
            "answer_correct": 1.0,
            "evidence_contains_expected": 1.0,
            "final_answer_correct": 1.0,
            "required_table_used": 1.0,
            "tool_used": 1.0,
            "output": "correct final answer",
        },
    ]
    (tmp_path / "1.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = summarize_rollouts(tmp_path)

    assert result["rows"] == 2
    assert result["score"]["mean"] == 1.0
    assert result["historical_answer_correct"] == 2.0
    assert result["evidence_contains_expected"] == 2.0
    assert result["strict_final_answer_correct"] == 1.0
    assert result["strict_full_reward_count"] == 1
    assert result["strict_reward_replay"]["mean"] == pytest.approx(0.6)
    assert result["strict_reward_by_verifier"]["task-a"]["full_reward_rate"] == 0.0
    assert result["strict_reward_by_verifier"]["task-b"]["full_reward_rate"] == 1.0
