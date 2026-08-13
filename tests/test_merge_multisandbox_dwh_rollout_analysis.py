import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.merge_multisandbox_dwh_rollout_analysis import merge


def arm(tmp_path: Path, name: str, verifier: str, bucket: str, correct: int):
    root = tmp_path / name
    root.mkdir()
    summary = {
        "contract": "boss-multisandbox-dwh-rollout-outcomes-v1",
        "tasks": 1,
        "samples_per_task": 2,
        "trajectories": 2,
        "correct_trajectories": correct,
        "completed_trajectories": 2,
        "runtime_error_trajectories": 0,
        "timeout_trajectories": 0,
        "timeout_abort_acknowledged_count": 0,
        "timeout_abort_physical_request_count": 0,
        "timeout_abort_error_count": 0,
        "bucket_counts": {bucket: 1},
        "correct_count_histogram": {"0": int(correct == 0), "1": int(correct == 1), "2": int(correct == 2)},
        "answer_type_bucket_counts": {"numeric": {bucket: 1}},
        "version_bucket_counts": {"v1": {bucket: 1}},
        "mixed_screening_rows": int(bucket == "mixed"),
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    mixed_path = root / "mixed.parquet"
    rows = [{"extra_info": {"verifier_id": verifier}}] if bucket == "mixed" else []
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([("extra_info", pa.struct([("verifier_id", pa.string())]))])), mixed_path)
    return summary_path, mixed_path


def test_merge_sums_safe_counts_and_keeps_mixed_union_disjoint(tmp_path: Path):
    summary1, mixed1 = arm(tmp_path, "a", "v:1", "mixed", 1)
    summary2, mixed2 = arm(tmp_path, "b", "v:2", "all_wrong", 0)

    result = merge([summary1, summary2], [mixed1, mixed2], tmp_path / "merged")

    assert result["tasks"] == 2
    assert result["trajectories"] == 4
    assert result["bucket_counts"] == {"all_wrong": 1, "mixed": 1}
    assert result["mixed_screening_rows"] == 1
    assert result["training_allowed"] is False
