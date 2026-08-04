from scripts.prepare_boss_aligned_dataset import canonical_hash
from scripts.select_v15_dwh_batch import select_batch


def queue_row(index: int, instruction: str | None = None) -> dict:
    instruction = instruction or f"计算第 {index} 类运单数量。"
    gold = {
        "answer_type": "numeric",
        "value": index,
        "verification_sql": f"SELECT COUNT(*) FROM metric_{index}",
    }
    return {
        "source_label": "v15",
        "source_join_method": "task_id",
        "task_id": f"task_{index}",
        "instruction": instruction,
        "instruction_sha256": canonical_hash(instruction),
        "gold": gold,
        "gold_sha256": canonical_hash(gold),
        "source_instruction_in_current_task_definition": True,
    }


def manifest_row(index: int) -> dict:
    return {
        "task_id": f"task_{index}",
        "type": "dwh",
        "expected_tables": [f"metric_{index}"],
    }


def shadow_row(index: int, score: float = 0.2) -> dict:
    return {
        "task_id": f"task_{index}",
        "score": score,
        "online_eligible": 1.0,
        "gold_sql_verified": 1.0,
    }


def test_selector_emits_dwh_only_unique_split_approvals():
    queue = [queue_row(index) for index in range(1, 7)]
    manifests = [manifest_row(index) for index in range(1, 7)]
    shadow = [shadow_row(index, 1.0 if index == 1 else 0.2) for index in range(1, 7)]

    approvals, audit = select_batch(
        queue,
        manifests,
        shadow,
        {"train": 2, "val": 1, "test": 1},
        "seed",
        ["task_1"],
    )

    assert [row["split"] for row in approvals] == ["train", "train", "val", "test"]
    assert approvals[0]["task_id"] == "task_1"
    assert len({row["task_id"] for row in approvals}) == 4
    assert audit["invariants"]["dwh_only"] is True
    assert audit["invariants"]["kb_rows"] == 0
    assert audit["selected"] == {"train": 2, "val": 1, "test": 1}


def test_selector_records_semantic_warnings_but_only_excludes_failed_gold_gate():
    broad = queue_row(1, "请分析当前运输问题并给出改进建议。")
    failed = queue_row(2)
    valid_rows = [queue_row(index) for index in range(3, 8)]
    queue = [broad, failed, *valid_rows]
    manifests = [manifest_row(index) for index in range(1, 8)]
    shadow = [
        shadow_row(1),
        {**shadow_row(2), "online_eligible": 0.0},
        *(shadow_row(index) for index in range(3, 8)),
    ]

    approvals, audit = select_batch(
        queue,
        manifests,
        shadow,
        {"train": 2, "val": 1, "test": 1},
        "seed",
        [],
    )

    selected_ids = {row["task_id"] for row in approvals}
    assert "task_2" not in selected_ids
    assert audit["eligible_rows"] == 6
    assert audit["warning_counts"]["broad_instruction_exact_hidden_target"] == 1
    assert audit["warning_counts"]["broad_instruction_reduced_to_row_count"] == 1
    assert audit["excluded"]["shadow_gold_gate_failed"] == 1


def test_selector_can_use_every_eligible_row_across_splits_and_preserves_source_drift():
    queue = [queue_row(index) for index in range(1, 8)]
    queue[0]["source_instruction_in_current_task_definition"] = False
    manifests = [manifest_row(index) for index in range(1, 8)]
    shadow = [shadow_row(index) for index in range(1, 8)]

    approvals, audit = select_batch(
        queue,
        manifests,
        shadow,
        {"train": 5, "val": 1, "test": 1},
        "seed",
        [],
    )

    assert len(approvals) == 7
    assert audit["eligible_rows"] == 7
    assert audit["unselected_eligible_rows"] == 0
    assert audit["invariants"]["uses_all_eligible_rows"] is True
    drifted = next(row for row in approvals if row["task_id"] == "task_1")
    assert "later_task_definition_drift" in drifted["audit_warnings"]


def test_selector_keeps_duplicate_prompts_in_one_split():
    shared_instruction = "查询同一个运单指标。"
    queue = [
        queue_row(1, shared_instruction),
        queue_row(2, shared_instruction),
        *(queue_row(index) for index in range(3, 8)),
    ]
    manifests = [manifest_row(index) for index in range(1, 8)]
    shadow = [shadow_row(index) for index in range(1, 8)]

    approvals, audit = select_batch(
        queue,
        manifests,
        shadow,
        {"train": 3, "val": 2, "test": 2},
        "seed",
        [],
    )

    duplicate_splits = {
        row["split"] for row in approvals if row["task_id"] in {"task_1", "task_2"}
    }
    assert len(duplicate_splits) == 1
    assert audit["invariants"]["unique_instruction_hashes"] is False
    assert audit["invariants"]["no_cross_split_instruction_overlap"] is True
