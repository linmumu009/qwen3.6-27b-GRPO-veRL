import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.compare_plan_first_dwh_model_rollouts import compare


def dataset_row(identity: str, band: int, split: str, answer_type: str = "numeric") -> dict:
    return {
        "reward_model": {"ground_truth": {"answer_type": answer_type}},
        "extra_info": {
            "instruction_sha256": identity,
            "difficulty_band": band,
            "comparison_split": split,
        },
    }


def outcome(identity: str, correct: int, bucket: str) -> dict:
    return {
        "instruction_sha256": identity,
        "correct_count": correct,
        "runtime_error_count": 0,
        "trajectory_timeout_count": 0,
        "bucket": bucket,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_compare_pairs_models_and_selects_only_step120_mixed_candidate(tmp_path: Path):
    dataset = tmp_path / "dataset.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                dataset_row("a", 1, "frozen_evaluation"),
                dataset_row("b", 6, "training_candidate", "table"),
                dataset_row("c", 3, "training_candidate", "table"),
            ]
        ),
        dataset,
    )
    native = tmp_path / "native.jsonl"
    step120 = tmp_path / "step120.jsonl"
    write_jsonl(
        native,
        [outcome("a", 8, "all_correct"), outcome("b", 0, "all_wrong"), outcome("c", 4, "mixed")],
    )
    write_jsonl(
        step120,
        [outcome("a", 7, "mixed"), outcome("b", 3, "mixed"), outcome("c", 8, "all_correct")],
    )

    result = compare(dataset, native, step120, tmp_path / "out")

    assert result["tasks"] == 3
    assert result["paired_task_outcomes"] == {
        "native_win": 1,
        "step120_win": 2,
    }
    assert result["step120_mixed_training_candidates"] == 1
    assert result["bucket_transition_counts"] == {
        "all_correct->mixed": 1,
        "all_wrong->mixed": 1,
        "mixed->all_correct": 1,
    }
    assert result["strata"]["band"]["6"]["step120_minus_native"] == 3 / 8
    candidates = (tmp_path / "out" / "candidate_hashes.sensitive.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"instruction_sha256": "b"' in candidates
    assert '"instruction_sha256": "a"' not in candidates


def test_compare_rejects_arm_identity_mismatch(tmp_path: Path):
    dataset = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist([dataset_row("a", 1, "frozen_evaluation")]), dataset)
    native = tmp_path / "native.jsonl"
    step120 = tmp_path / "step120.jsonl"
    write_jsonl(native, [outcome("a", 0, "all_wrong")])
    write_jsonl(step120, [outcome("b", 0, "all_wrong")])

    try:
        compare(dataset, native, step120, tmp_path / "out")
    except ValueError as exc:
        assert "identities differ" in str(exc)
    else:
        raise AssertionError("expected identity mismatch")
