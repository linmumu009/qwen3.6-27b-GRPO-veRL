from __future__ import annotations

from llin_verl.outcome_gated_contract import (
    HardGateResampleBuffer,
    TristateResampleBuffer,
    audit_mixed_group_advantages,
    evidence_binding_hash,
)


def test_every_wrong_outcome_has_strictly_negative_advantage() -> None:
    labels = [1, 0, 0, 0, 0, 0, 0, 0]
    rewards = [1.1, 0, 0, 0, 0, 0, 0, 0]
    audit = audit_mixed_group_advantages(labels, rewards)

    assert audit["all_incorrect_strictly_negative"] is True
    assert audit["incorrect_positive_advantage_count"] == 0
    assert audit["incorrect_nonnegative_advantage_count"] == 0


def test_hard_gate_resampler_accepts_eight_or_skips_at_cap() -> None:
    buffer = HardGateResampleBuffer(target_size=8, max_attempts=10)
    status = ""
    for index in range(10):
        status = buffer.observe("pass", index, hard_gate_passed=index not in {1, 4})
    assert status == "ready"
    assert buffer.result("pass")["accepted_count"] == 8

    buffer = HardGateResampleBuffer(target_size=8, max_attempts=10)
    for index in range(10):
        status = buffer.observe("skip", index, hard_gate_passed=index < 7)
    assert status == "skip"
    assert buffer.result("skip")["skipped"] is True


def test_tristate_resampler_accepts_pass_and_fail_but_not_unknown() -> None:
    buffer = TristateResampleBuffer(target_size=3, max_attempts=5)
    assert buffer.observe("group", "pass", train_mask=True) == "resample"
    assert buffer.observe("group", "unknown", train_mask=False) == "resample"
    assert buffer.observe("group", "fail", train_mask=True) == "resample"
    assert buffer.observe("group", "pass-2", train_mask=True) == "ready"
    assert buffer.result("group")["accepted"] == ["pass", "fail", "pass-2"]


def test_evidence_binding_hash_changes_on_plan_or_fields() -> None:
    truth = {
        "environment_id": "sft/v1",
        "verification_sql": "SELECT 1",
        "evidence_plan": {"task_type": "aggregation"},
        "required_tables": ["fact"],
        "must_use_fields": ["value"],
    }
    baseline = evidence_binding_hash(truth)
    assert evidence_binding_hash({**truth, "must_use_fields": ["units"]}) != baseline
    assert evidence_binding_hash({**truth, "evidence_plan": {"task_type": "topn"}}) != baseline
