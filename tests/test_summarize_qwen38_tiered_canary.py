from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_qwen38_tiered_canary import summarize


def _row(task: str, trajectory: str, reward: float, success: int, **values):
    row = {
        "task_identity_sha256": task,
        "trajectory_identity_sha256": trajectory,
        "tiered_reward": reward,
        "score": reward,
        "train_mask": 1,
        "success": success,
        "final_answer_correct": success,
        "attempted_relevant_readonly_sql": 1,
        "successful_relevant_readonly_sql": 1,
        "query_attempt_count": 2,
        "tool_response_tokens": 1000,
        "irrelevant_query_ratio": 0,
        "duplicate_query_ratio": 0,
        "E": 0,
        "trajectory_advantage_mean": 1 if success else -1,
        "trajectory_total_response_tokens": 100,
        "judge_state": "PASS" if success else "FAIL",
        "judge_reason": "synthetic",
        "reward_layer": "success_correct_final" if success else "success_wrong_final",
    }
    row.update(values)
    return row


def test_safe_summary_reports_mixed_groups_and_reward_boundaries(tmp_path: Path):
    task = "a" * 64
    rows = [
        _row(
            task,
            f"{index:064x}",
            0.9 if index < 4 else 0.2,
            int(index < 4),
            input="SENSITIVE_PROMPT_SENTINEL",
            output="SENSITIVE_OUTPUT_SENTINEL",
        )
        for index in range(8)
    ]
    directory = tmp_path / "private-rollouts"
    directory.mkdir()
    (directory / "1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = summarize(directory)

    assert report["totals"]["rows"] == 8
    assert report["totals"]["nominal_batches"] == 1
    assert report["totals"]["actual_optimizer_steps_implied"] == 1
    assert report["totals"]["strict_mixed_groups"] == 1
    assert not any(report["totals"]["boundary_violations"].values())
    group = report["steps"][0]["groups"][0]
    assert group["success_count"] == 4
    assert group["strict_should_update_actor"] is True
    serialized = json.dumps(report)
    assert "SENSITIVE_PROMPT_SENTINEL" not in serialized
    assert "SENSITIVE_OUTPUT_SENTINEL" not in serialized


def test_unknown_or_uniform_groups_never_report_trainable(tmp_path: Path):
    task_a = "a" * 64
    task_b = "b" * 64
    rows = [_row(task_a, f"{index:064x}", 0.9, 1, trajectory_advantage_mean=0) for index in range(8)]
    rows += [
        _row(
            task_b,
            f"{index + 8:064x}",
            0,
            0,
            train_mask=0,
            trajectory_advantage_mean=0,
            judge_state="UNKNOWN",
            reward_layer="unknown",
        )
        for index in range(8)
    ]
    directory = tmp_path / "private-rollouts"
    directory.mkdir()
    (directory / "2.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = summarize(directory)

    assert report["totals"]["strict_mixed_groups"] == 0
    assert report["totals"]["actual_optimizer_steps_implied"] == 0
    assert report["totals"]["strict_skipped_groups"] == 2
    assert report["totals"]["boundary_violations"]["unknown_nonzero_advantage"] == 0
    assert report["totals"]["boundary_violations"]["uniform_group_nonzero_advantage"] == 0


def test_safe_summary_fails_closed_on_non_hash_identity(tmp_path: Path):
    row = _row("raw-sensitive-identity", "raw-trajectory", 0, 0)
    directory = tmp_path / "private-rollouts"
    directory.mkdir()
    (directory / "3.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = summarize(directory)

    step = report["steps"][0]
    assert step["missing_task_identity"] == 1
    assert step["unique_trajectory_identities"] == 0
    assert step["groups"][0]["task_identity_sha256"] == "missing"
    assert step["highest_reward_wrong"][0]["trajectory_identity_sha256"] == ""


def test_training_stage_parser_emits_only_numeric_timings(tmp_path: Path):
    directory = tmp_path / "private-rollouts"
    directory.mkdir()
    log = tmp_path / "training.log"
    log.write_text(
        "sensitive context before\n"
        "[LLIN_TRAIN_STAGE] step=3 queue_wait_s=12.5 deserialize_s=0.1 "
        "assemble_s=0.2 reward_s=0.3 old_log_prob_s=0.4 ref_log_prob_s=4.5 "
        "adv_s=0.6 update_actor_s=0.0 step_s=18.6\n"
        "sensitive context after\n",
        encoding="utf-8",
    )

    report = summarize(directory, log)

    assert report["training_stages"] == [
        {
            "step": 3,
            "queue_wait_s": 12.5,
            "deserialize_s": 0.1,
            "assemble_s": 0.2,
            "reward_s": 0.3,
            "old_log_prob_s": 0.4,
            "ref_log_prob_s": 4.5,
            "adv_s": 0.6,
            "update_actor_s": 0.0,
            "step_s": 18.6,
        }
    ]
    assert "sensitive context" not in json.dumps(report)
