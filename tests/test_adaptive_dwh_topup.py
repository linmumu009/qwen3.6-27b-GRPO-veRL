import json
from argparse import Namespace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.adaptive_dwh_topup import (
    FINAL_CONTRACT,
    PROFILE_CONTRACT,
    SELECTION_CONTRACT,
    canonical_hash,
    finalize,
    prepare_topup,
    profile_reference,
)
from scripts.run_adaptive_dwh_topup_queue import launch_topup


def task(index: int, *, level: int, family: str, joins: int) -> dict:
    instruction = f"instruction-{index}"
    return {
        "task_id": f"task-{index}",
        "natural_language_instruction": instruction,
        "task_type": family,
        "difficulty_level": level,
        "expected_tables": ["a", "b"][: max(1, joins)],
        "expected_operations": ["select", "join"] if joins else ["select"],
        "gold_answer": {"answer_type": "numeric", "value": index, "verification_sql": "select 1"},
        "query_plan": {"feature_counts": {"essential_joins": joins, "evidence_steps": level}},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def dataset_record(source: dict, index: int) -> dict:
    identity = canonical_hash(source["natural_language_instruction"])
    return {
        "prompt": [{"role": "user", "content": source["natural_language_instruction"]}],
        "reward_model": {
            "ground_truth": {
                "answer_type": "numeric",
                "expected_value_json": str(source["gold_answer"]["value"]),
                "verification_sql": "select 1",
            }
        },
        "extra_info": {
            "verifier_id": f"v:{index}",
            "instruction_sha256": identity,
            "source_version": "v20",
            "difficulty_level": source["difficulty_level"],
            "task_type": source["task_type"],
            "training_allowed": False,
        },
    }


def result_row(source: dict, index: int, *, correct: int, completed: int, timeout: int = 0) -> dict:
    return {
        "source_task_index": index,
        "instruction_sha256": canonical_hash(source["natural_language_instruction"]),
        "correct_count": correct,
        "completed_count": completed,
        "trajectory_timeout_count": timeout,
    }


def test_profile_and_prepare_select_hits_structure_and_exploration(tmp_path: Path):
    reference_tasks = [
        task(0, level=4, family="comparison", joins=2),
        task(1, level=2, family="aggregate", joins=1),
    ]
    reference_source = tmp_path / "reference.jsonl"
    write_jsonl(reference_source, reference_tasks)
    reference_results = tmp_path / "reference-results.jsonl"
    write_jsonl(
        reference_results,
        [
            result_row(reference_tasks[0], 0, correct=1, completed=6, timeout=2),
            result_row(reference_tasks[1], 1, correct=0, completed=8),
        ],
    )
    profile_path = tmp_path / "profile.json"
    profile = profile_reference(reference_source, reference_results, profile_path)
    assert profile["contract"] == PROFILE_CONTRACT
    assert profile["reference_explicit_mixed_tasks"] == 1
    assert profile["reference_explicit_mixed_with_timeout"] == 1

    target_tasks = [
        task(10, level=1, family="aggregate", joins=0),
        task(11, level=4, family="comparison", joins=2),
        task(12, level=3, family="other", joins=3),
        task(13, level=5, family="other", joins=4),
    ]
    target_source = tmp_path / "target.jsonl"
    write_jsonl(target_source, target_tasks)
    dataset_path = tmp_path / "screen.parquet"
    pq.write_table(
        pa.Table.from_pylist([dataset_record(row, index) for index, row in enumerate(target_tasks)]),
        dataset_path,
    )
    screen_results = tmp_path / "screen-results.jsonl"
    write_jsonl(
        screen_results,
        [
            result_row(target_tasks[0], 0, correct=1, completed=2),
            result_row(target_tasks[1], 1, correct=0, completed=2),
            result_row(target_tasks[2], 2, correct=0, completed=2),
            result_row(target_tasks[3], 3, correct=0, completed=1, timeout=1),
        ],
    )
    topup_path = tmp_path / "topup.parquet"
    manifest_path = tmp_path / "selection.json"
    manifest = prepare_topup(
        dataset_path,
        screen_results,
        target_source,
        profile_path,
        topup_path,
        manifest_path,
        exploration_per_level=1,
        seed="test",
    )

    assert manifest["contract"] == SELECTION_CONTRACT
    assert manifest["selected_tasks"] == 4
    assert manifest["selection_reason_counts_nonexclusive"] == {
        "exploration": 2,
        "reference_structure": 1,
        "screen_correct": 1,
    }
    selected = pq.read_table(topup_path).to_pylist()
    assert [row["extra_info"]["adaptive_original_task_index"] for row in selected] == [0, 1, 2, 3]
    assert all(row["extra_info"]["training_allowed"] is False for row in selected)


def test_finalize_merges_two_plus_six_and_keeps_timeout_mixed_relaxed(tmp_path: Path):
    sources = [
        task(0, level=4, family="comparison", joins=2),
        task(1, level=1, family="aggregate", joins=0),
    ]
    screen_dataset = tmp_path / "screen.parquet"
    screen_rows = [dataset_record(row, index) for index, row in enumerate(sources)]
    pq.write_table(pa.Table.from_pylist(screen_rows), screen_dataset)
    screen_shards = tmp_path / "screen-shards"
    screen_shards.mkdir()
    write_jsonl(
        screen_shards / "tasks_00000_00002.jsonl",
        [
            {"source_task_index": 0, "sample_index": 0, "output": "最终答案 0", "response_tokens": 5},
            {"source_task_index": 0, "sample_index": 1, "output": "最终答案 9", "response_tokens": 5},
            {"source_task_index": 1, "sample_index": 0, "output": "最终答案 9", "response_tokens": 5},
            {"source_task_index": 1, "sample_index": 1, "output": "最终答案 9", "response_tokens": 5},
        ],
    )
    selected = dataset_record(sources[0], 0)
    selected["extra_info"].update(
        {
            "adaptive_contract": SELECTION_CONTRACT,
            "adaptive_original_task_index": 0,
        }
    )
    topup_dataset = tmp_path / "topup.parquet"
    pq.write_table(pa.Table.from_pylist([selected]), topup_dataset)
    topup_shards = tmp_path / "topup-shards"
    topup_shards.mkdir()
    topup_outputs = [
        {"source_task_index": 0, "sample_index": sample, "output": "最终答案 9", "response_tokens": 5}
        for sample in range(5)
    ]
    topup_outputs.append(
        {
            "source_task_index": 0,
            "sample_index": 5,
            "output": "",
            "response_tokens": 0,
            "trajectory_timeout": True,
        }
    )
    write_jsonl(topup_shards / "tasks_00000_00001.jsonl", topup_outputs)

    summary = finalize(
        screen_dataset,
        screen_shards,
        topup_dataset,
        topup_shards,
        tmp_path / "final",
        expected_screen_tasks=2,
    )

    assert summary["contract"] == FINAL_CONTRACT
    assert summary["actual_sampling_trajectories"] == 10
    assert summary["avoided_trajectories"] == 6
    assert summary["strict_mixed_tasks"] == 0
    assert summary["relaxed_explicit_mixed_tasks"] == 1
    assert summary["relaxed_explicit_mixed_with_timeout"] == 1
    relaxed = pq.read_table(
        tmp_path / "final" / "outcomes" / "relaxed_mixed_candidates.sensitive.parquet"
    ).to_pylist()
    assert len(relaxed) == 1


def test_topup_launcher_fills_capacity_without_changing_frozen_contract(
    tmp_path: Path, monkeypatch
):
    captured = {}

    def fake_run(command, *, env, check):
        captured.update({"command": command, "env": env, "check": check})

    args = Namespace(
        topup_run_dir=tmp_path / "run",
        topup_task_batch_size=8,
        max_num_seqs=24,
        project_root=tmp_path,
        model=tmp_path / "model",
        topup_dataset=tmp_path / "topup.parquet",
        ray_address="ray",
        rollout_resource="llin_rollout_m05",
    )
    monkeypatch.setattr(
        "scripts.run_adaptive_dwh_topup_queue.subprocess.run", fake_run
    )

    launch_topup(args, selected_tasks=12)

    environment = captured["env"]
    assert environment["EXPECTED_TASKS"] == "12"
    assert environment["SAMPLES_PER_TASK"] == "6"
    assert environment["TASK_BATCH_SIZE"] == "8"
    assert environment["ROLLING_WINDOW_TRAJECTORIES"] == "48"
    assert environment["MAX_RESPONSE_TOKENS"] == "90112"
    assert environment["MAX_CONTEXT_TOKENS"] == "94208"
    assert environment["TRAJECTORY_TIMEOUT_SECONDS"] == "1800"
    assert captured["check"] is True
