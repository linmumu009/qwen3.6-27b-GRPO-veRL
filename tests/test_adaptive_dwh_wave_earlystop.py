import json
from argparse import Namespace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.adaptive_dwh_wave_earlystop import (
    CONTRACT,
    finalize,
    finalize_four_wave,
    prepare_initial_pool,
    prepare_remaining_pool,
    select_after_wave,
)
from scripts.run_adaptive_dwh_wave_earlystop_queue import launch_wave


def record(index: int, *, level: int = 1) -> dict:
    return {
        "prompt": [{"role": "user", "content": f"instruction-{index}"}],
        "reward_model": {
            "ground_truth": {
                "answer_type": "numeric",
                "expected_value_json": str(index),
                "verification_sql": "select 1",
            }
        },
        "extra_info": {
            "instruction_sha256": f"hash-{index}",
            "verifier_id": f"v-{index}",
            "source_version": "v20",
            "difficulty_level": level,
            "training_allowed": False,
        },
    }


def write_parquet(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def result(index: int, *, correct: int, completed: int, timeout: int = 0) -> dict:
    return {
        "source_task_index": index,
        "instruction_sha256": f"hash-{index}",
        "correct_count": correct,
        "completed_count": completed,
        "trajectory_timeout_count": timeout,
        "runtime_error_count": 0,
    }


def test_prepare_and_two_waves_stop_only_after_explicit_variance(tmp_path: Path):
    screen_rows = [record(index, level=index % 2 + 1) for index in range(5)]
    screen_path = tmp_path / "screen.parquet"
    write_parquet(screen_path, screen_rows)
    screen_results = tmp_path / "screen.jsonl"
    write_jsonl(
        screen_results,
        [
            result(0, correct=1, completed=2),
            result(1, correct=0, completed=2),
            result(2, correct=0, completed=2),
            result(3, correct=0, completed=2),
            result(4, correct=2, completed=2),
        ],
    )
    probes = tmp_path / "probes.parquet"
    direct = tmp_path / "direct.parquet"
    write_parquet(probes, [screen_rows[1]])
    write_parquet(direct, [screen_rows[0]])
    pool = tmp_path / "pool.parquet"
    pool_manifest = tmp_path / "pool.safe.json"
    prepared = prepare_remaining_pool(
        screen_path,
        screen_results,
        probes,
        direct,
        pool,
        pool_manifest,
        expected_remaining_tasks=3,
    )

    assert prepared["contract"] == CONTRACT
    assert prepared["remaining_tasks"] == 3
    assert prepared["excluded_direct_mixed_tasks"] == 1
    assert prepared["excluded_prior_probe_tasks"] == 1
    pool_rows = pq.read_table(pool).to_pylist()
    assert [row["extra_info"]["adaptive_original_task_index"] for row in pool_rows] == [2, 3, 4]

    wave4_results = tmp_path / "wave4.jsonl"
    write_jsonl(
        wave4_results,
        [
            {**result(0, correct=1, completed=2), "instruction_sha256": "hash-2"},
            {**result(1, correct=0, completed=2), "instruction_sha256": "hash-3"},
            {**result(2, correct=2, completed=2), "instruction_sha256": "hash-4"},
        ],
    )
    unresolved4 = tmp_path / "unresolved4.parquet"
    mixed4 = tmp_path / "mixed4.parquet"
    manifest4 = select_after_wave(
        pool,
        wave4_results,
        unresolved4,
        mixed4,
        tmp_path / "wave4.safe.json",
        expected_prior_samples=2,
    )
    assert manifest4["new_mixed_tasks"] == 1
    assert manifest4["unresolved_tasks"] == 2
    assert pq.read_table(mixed4).to_pylist()[0]["extra_info"]["adaptive_mixed_after_samples"] == 4

    wave6_results = tmp_path / "wave6.jsonl"
    write_jsonl(
        wave6_results,
        [
            {**result(0, correct=1, completed=2, timeout=1), "instruction_sha256": "hash-3"},
            {**result(1, correct=2, completed=2), "instruction_sha256": "hash-4"},
        ],
    )
    unresolved6 = tmp_path / "unresolved6.parquet"
    mixed6 = tmp_path / "mixed6.parquet"
    manifest6 = select_after_wave(
        unresolved4,
        wave6_results,
        unresolved6,
        mixed6,
        tmp_path / "wave6.safe.json",
        expected_prior_samples=4,
    )
    assert manifest6["new_mixed_tasks"] == 1
    assert manifest6["unresolved_tasks"] == 1
    assert manifest6["cumulative_timeout_count_over_input_tasks"] == 1

    summary = finalize(pool, mixed4, mixed6, unresolved6, tmp_path / "final")
    assert summary["variance_candidate_tasks"] == 2
    assert summary["sample_count_distribution"] == {"2": 0, "4": 1, "6": 2}
    assert summary["actual_trajectories_including_existing_two"] == 16
    assert summary["full_six_sampling_baseline_trajectories"] == 18
    assert summary["avoided_trajectories_vs_full_six"] == 2
    assert len(
        pq.read_table(
            tmp_path / "final" / "grpo_variance_candidates.sensitive.parquet"
        ).to_pylist()
    ) == 2

def test_wave_launcher_keeps_physical_capacity_and_125_percent_agent_window(
    tmp_path: Path, monkeypatch
):
    captured = {}

    def fake_run(command, *, env, check):
        captured.update({"command": command, "env": env, "check": check})

    args = Namespace(
        project_root=tmp_path,
        model=tmp_path / "model",
        task_batch_size=24,
        max_num_seqs=24,
        rolling_window_trajectories=60,
        rolling_window_max_multiplier=1.25,
        ray_address="ray",
        rollout_resource="llin_rollout_m05",
    )
    monkeypatch.setattr(
        "scripts.run_adaptive_dwh_wave_earlystop_queue.subprocess.run", fake_run
    )

    launch_wave(args, tmp_path / "wave.parquet", tmp_path / "run", 215)

    environment = captured["env"]
    assert environment["EXPECTED_TASKS"] == "215"
    assert environment["SAMPLES_PER_TASK"] == "2"
    assert environment["TASK_BATCH_SIZE"] == "24"
    assert environment["MAX_NUM_SEQS"] == "24"
    assert environment["ROLLING_WINDOW_TRAJECTORIES"] == "60"
    assert environment["ROLLING_WINDOW_MAX_MULTIPLIER"] == "1.25"
    assert environment["MAX_CONTEXT_TOKENS"] == "94208"
    assert environment["TRAJECTORY_TIMEOUT_SECONDS"] == "1800"
    assert captured["check"] is True


def test_full_four_wave_screen_stops_at_two_four_and_eight(tmp_path: Path):
    source_rows = [record(index, level=index + 1) for index in range(4)]
    source = tmp_path / "source.parquet"
    write_parquet(source, source_rows)
    initial = tmp_path / "initial.parquet"
    prepared = prepare_initial_pool(
        source,
        initial,
        tmp_path / "pool.safe.json",
        expected_tasks=4,
    )
    assert prepared["remaining_tasks"] == 4

    def select_wave(
        dataset: Path,
        rows: list[dict],
        prior: int,
        label: str,
    ) -> tuple[Path, Path]:
        per_task = tmp_path / f"{label}.jsonl"
        write_jsonl(per_task, rows)
        unresolved = tmp_path / f"unresolved{label}.parquet"
        mixed = tmp_path / f"mixed{label}.parquet"
        select_after_wave(
            dataset,
            per_task,
            unresolved,
            mixed,
            tmp_path / f"wave{label}.safe.json",
            expected_prior_samples=prior,
            max_samples=8,
        )
        return unresolved, mixed

    unresolved2, mixed2 = select_wave(
        initial,
        [
            result(0, correct=1, completed=2),
            result(1, correct=0, completed=2),
            result(2, correct=2, completed=2),
            result(3, correct=0, completed=1, timeout=1),
        ],
        0,
        "2",
    )
    unresolved4, mixed4 = select_wave(
        unresolved2,
        [
            {**result(0, correct=1, completed=2), "instruction_sha256": "hash-1"},
            {**result(1, correct=0, completed=2), "instruction_sha256": "hash-2"},
            {**result(2, correct=0, completed=1, timeout=1), "instruction_sha256": "hash-3"},
        ],
        2,
        "4",
    )
    unresolved6, mixed6 = select_wave(
        unresolved4,
        [{**result(0, correct=0, completed=2), "instruction_sha256": "hash-3"}],
        4,
        "6",
    )
    unresolved8, mixed8 = select_wave(
        unresolved6,
        [{**result(0, correct=1, completed=2), "instruction_sha256": "hash-3"}],
        6,
        "8",
    )
    summary = finalize_four_wave(
        initial,
        mixed2,
        mixed4,
        mixed6,
        mixed8,
        unresolved8,
        tmp_path / "final8",
    )
    assert summary["variance_candidate_tasks"] == 4
    assert summary["sample_count_distribution"] == {
        "2": 1,
        "4": 2,
        "6": 0,
        "8": 1,
    }
    assert summary["actual_sampling_trajectories"] == 18
    assert summary["full_eight_sampling_baseline_trajectories"] == 32
    assert summary["avoided_trajectories_vs_full_eight"] == 14


def test_wave_launcher_uses_h06_full_physical_capacity_with_125_percent_window(
    tmp_path: Path, monkeypatch
):
    captured = {}

    def fake_run(command, *, env, check):
        captured.update({"command": command, "env": env, "check": check})

    args = Namespace(
        project_root=tmp_path,
        model=tmp_path / "model",
        task_batch_size=32,
        max_num_seqs=32,
        rolling_window_trajectories=80,
        rolling_window_max_multiplier=1.25,
        ray_address="ray",
        rollout_resource="llin_rollout_m06",
    )
    monkeypatch.setattr(
        "scripts.run_adaptive_dwh_wave_earlystop_queue.subprocess.run", fake_run
    )

    launch_wave(args, tmp_path / "wave.parquet", tmp_path / "run", 209)

    environment = captured["env"]
    assert environment["EXPECTED_TASKS"] == "209"
    assert environment["SAMPLES_PER_TASK"] == "2"
    assert environment["TASK_BATCH_SIZE"] == "32"
    assert environment["MAX_NUM_SEQS"] == "32"
    assert environment["ROLLING_WINDOW_TRAJECTORIES"] == "80"
    assert environment["ROLLING_WINDOW_MAX_MULTIPLIER"] == "1.25"
    assert environment["ROLLOUT_RESOURCE"] == "llin_rollout_m06"
    assert captured["check"] is True
