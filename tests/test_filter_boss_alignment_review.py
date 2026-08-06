from scripts.filter_boss_alignment_review import filter_review, parse_rejections


def test_filter_review_removes_only_requested_task_and_preserves_splits():
    rows = [
        {"task_id": "task_1", "split": "train", "instruction_sha256": "i1", "gold_sha256": "g1"},
        {"task_id": "task_2", "split": "train", "instruction_sha256": "i1", "gold_sha256": "g2"},
        {"task_id": "task_3", "split": "val", "instruction_sha256": "i3", "gold_sha256": "g3"},
    ]

    kept, audit = filter_review(rows, {"task_1": "conflicting_duplicate_less_aligned"})

    assert [row["task_id"] for row in kept] == ["task_2", "task_3"]
    assert audit["split_before"] == {"train": 2, "val": 1}
    assert audit["split_after"] == {"train": 1, "val": 1}
    assert audit["rejections"][0]["gold_sha256"] == "g1"


def test_parse_rejections_requires_reason_and_unique_task():
    assert parse_rejections(["task_1=bad label"]) == {"task_1": "bad label"}
