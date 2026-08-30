from pathlib import Path

import pytest

from llin_verl.validation_identity import (
    align_returned_validation,
    apply_judge_states,
    build_validation_identities,
    mark_padding_identities,
    write_identity_status,
)
from scripts.patch_verl_fastest_k_oversampling import patch_agent_loop


def _expected_20x4() -> list[str]:
    repeated = []
    for task_index in range(20):
        extra = {
            "task_id": f"task-{task_index:02d}",
            "prefix_state_id": f"prefix-{task_index:02d}",
        }
        repeated.extend([extra] * 4)
    identities, slots, policies = build_validation_identities(
        repeated,
        samples_per_state=4,
        policy_version=0,
    )
    assert slots == [0, 1, 2, 3] * 20
    assert policies == [0] * 80
    return identities


def test_expected_80_actual_44_are_joined_without_padding_duplication_or_truncation() -> None:
    expected = _expected_20x4()
    # Reproduce the observed cardinalities without pretending that the failed
    # run persisted its unavailable per-slot completion order.  These 44 IDs
    # are deliberately reordered to prove that positional slicing is invalid.
    returned_indices = [(index * 37) % 80 for index in range(44)]
    returned = [expected[index] for index in returned_indices]

    source_indices, output_indices, status = align_returned_validation(expected, returned)

    assert len(source_indices) == len(output_indices) == 44
    assert source_indices == returned_indices
    assert output_indices == list(range(44))
    assert len({row["validation_identity"] for row in status}) == 80
    assert sum(row["returned"] for row in status) == 44
    assert sum(row["resample_required"] for row in status) == 36
    assert {row["status"] for row in status} == {"RETURNED", "NOT_RETURNED"}


def test_divisor_padding_and_cancelled_slots_remain_explicit() -> None:
    expected = _expected_20x4()
    padded, padding_mask = mark_padding_identities(
        expected + expected[:4],
        expected_count=80,
    )
    # A partial runtime return can include a divisor-padding row.  It is
    # excluded by identity, while missing real slots remain explicit.
    returned = [padded[index] for index in [0, 9, 18, 27, 36, 45, 54, 63, 72, 80]]
    returned_padding = [padding_mask[index] for index in [0, 9, 18, 27, 36, 45, 54, 63, 72, 80]]

    source_indices, output_indices, status = align_returned_validation(
        expected,
        returned,
        returned_padding=returned_padding,
    )

    assert len(source_indices) == len(output_indices) == 9
    assert 9 not in output_indices
    assert status[0]["padding_rows_returned_and_excluded"] == 1
    assert sum(row["returned"] for row in status) == 9
    assert sum(row["resample_required"] for row in status) == 71


def test_unknown_and_cancelled_slots_are_resampled_without_reward_or_outcome_rewrite() -> None:
    expected = _expected_20x4()
    returned = [expected[index] for index in (3, 8, 17, 29)]
    _, _, status = align_returned_validation(expected, returned)
    original_rewards = [0.92, 0.0, 0.18, 0.81]

    judged = apply_judge_states(status, returned, ["PASS", "FAIL", "UNKNOWN", "PASS"])

    assert original_rewards == [0.92, 0.0, 0.18, 0.81]
    by_id = {row["validation_identity"]: row for row in judged}
    assert by_id[returned[0]]["status"] == "PASS"
    assert by_id[returned[1]]["status"] == "FAIL"
    assert by_id[returned[2]]["status"] == "UNKNOWN"
    assert by_id[returned[2]]["resample_required"] is True
    assert by_id[returned[3]]["status"] == "PASS"
    assert sum(row["status"] == "NOT_RETURNED" for row in judged) == 76


@pytest.mark.parametrize(
    "returned",
    [
        ["outside::prefix::policy-0::slot-0"],
        ["duplicate", "duplicate"],
    ],
)
def test_outside_or_duplicate_returned_identity_fails_closed(returned: list[str]) -> None:
    expected = _expected_20x4()
    if returned == ["duplicate", "duplicate"]:
        returned = [expected[0], expected[0]]
    with pytest.raises(ValueError):
        align_returned_validation(expected, returned)


def test_fastest_k_equal_to_candidate_count_does_not_trim_worker_shards(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "reference" / "verl" / "verl" / "experimental" / "agent_loop" / "agent_loop.py"
    target = tmp_path / "agent_loop.py"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    assert patch_agent_loop(target) == "patched"
    assert patch_agent_loop(target) == "already-patched"
    patched = target.read_text(encoding="utf-8")
    assert "LLIN_FASTEST_K_PER_PROMPT_GROUP_V4" in patched
    assert "if oversample_candidates > fastest_k > 0 and len(tasks) > fastest_k:" in patched
    assert "if fastest_k > 0 and len(tasks) > fastest_k:" not in patched
    compile(patched, str(target), "exec")


def test_private_status_ledger_rejects_duplicate_identities(tmp_path: Path) -> None:
    target = tmp_path / "status.jsonl"
    with pytest.raises(ValueError, match="duplicate identities"):
        write_identity_status(
            target,
            [
                {"validation_identity": "task::prefix::policy-0::slot-0"},
                {"validation_identity": "task::prefix::policy-0::slot-0"},
            ],
        )
    assert not target.exists()
