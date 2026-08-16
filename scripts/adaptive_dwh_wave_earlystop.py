#!/usr/bin/env python3
"""Prepare and reconcile a strict 2+2+2 DWH variance screen.

The first two trajectories already exist from the full 500-task screen.  This
module freezes the tasks that were neither direct mixed candidates nor prior
targeted probes, then adds two trajectories at a time.  A task stops as soon
as its accumulated completed answers contain at least one correct and one
wrong result.  Sensitive task payloads remain in private Parquet files; JSON
manifests contain aggregate counts only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CONTRACT = "llin-adaptive-dwh-2plus2plus2-v1"
WAVE_SAMPLES = 2
MAX_SAMPLES = 6


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_private_parquet(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    empty_from: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(rows) if rows else pq.read_table(empty_from).slice(0, 0)
    pq.write_table(table, temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _identity(record: dict[str, Any]) -> str:
    return str(record["extra_info"]["instruction_sha256"])


def _results_by_identity(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["instruction_sha256"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("per-task result identities are not unique")
    return result


def _explicit_mixed(*, correct: int, completed: int) -> bool:
    return correct > 0 and completed - correct > 0


def _difficulty_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(row["extra_info"].get("difficulty_level", "unknown"))
                for row in rows
            ).items()
        )
    )


def prepare_remaining_pool(
    screen_dataset_path: Path,
    screen_per_task_path: Path,
    excluded_probe_dataset_path: Path,
    excluded_direct_dataset_path: Path,
    output_dataset_path: Path,
    safe_manifest_path: Path,
    *,
    expected_remaining_tasks: int,
) -> dict[str, Any]:
    screen = pq.read_table(screen_dataset_path).to_pylist()
    probes = pq.read_table(excluded_probe_dataset_path).to_pylist()
    direct = pq.read_table(excluded_direct_dataset_path).to_pylist()
    results = _results_by_identity(screen_per_task_path)
    screen_identities = [_identity(row) for row in screen]
    if len(set(screen_identities)) != len(screen_identities):
        raise ValueError("screen dataset identities are not unique")
    if set(results) != set(screen_identities):
        raise ValueError("screen results do not exactly cover the screen dataset")
    probe_identities = {_identity(row) for row in probes}
    direct_identities = {_identity(row) for row in direct}
    if probe_identities & direct_identities:
        raise ValueError("prior probes and direct mixed candidates overlap")
    if not (probe_identities | direct_identities) <= set(screen_identities):
        raise ValueError("excluded datasets are not subsets of the screen")
    computed_direct = {
        identity
        for identity, row in results.items()
        if _explicit_mixed(
            correct=int(row["correct_count"]),
            completed=int(row["completed_count"]),
        )
    }
    if computed_direct != direct_identities:
        raise ValueError("direct mixed dataset does not match the two-sample results")

    remaining: list[dict[str, Any]] = []
    for original_index, source in enumerate(screen):
        identity = screen_identities[original_index]
        if identity in probe_identities or identity in direct_identities:
            continue
        result = results[identity]
        record = deepcopy(source)
        extra = dict(record["extra_info"])
        extra.update(
            {
                "adaptive_wave_contract": CONTRACT,
                "adaptive_original_task_index": original_index,
                "adaptive_samples_observed": WAVE_SAMPLES,
                "adaptive_correct_count": int(result["correct_count"]),
                "adaptive_completed_count": int(result["completed_count"]),
                "adaptive_timeout_count": int(result.get("trajectory_timeout_count", 0)),
                "adaptive_runtime_error_count": int(result.get("runtime_error_count", 0)),
                "adaptive_decision": "continue_after_2",
                "training_allowed": False,
                "promotion_allowed": False,
            }
        )
        record["extra_info"] = extra
        remaining.append(record)
    if len(remaining) != expected_remaining_tasks:
        raise ValueError(
            f"expected {expected_remaining_tasks} remaining tasks, got {len(remaining)}"
        )
    write_private_parquet(output_dataset_path, remaining, empty_from=screen_dataset_path)
    manifest = {
        "contract": CONTRACT,
        "stage": "remaining_after_existing_two_sample_screen",
        "screen_tasks": len(screen),
        "screen_samples_per_task": WAVE_SAMPLES,
        "excluded_direct_mixed_tasks": len(direct),
        "excluded_prior_probe_tasks": len(probes),
        "remaining_tasks": len(remaining),
        "remaining_difficulty_counts": _difficulty_counts(remaining),
        "screen_dataset_sha256": file_sha256(screen_dataset_path),
        "remaining_dataset_sha256": file_sha256(output_dataset_path),
        "identity_is_instruction_sha256": True,
        "selection_is_exact_set_difference": True,
        "contains_prompts_gold_sql_task_ids_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    write_json(safe_manifest_path, manifest)
    return manifest


def prepare_initial_pool(
    source_dataset_path: Path,
    output_dataset_path: Path,
    safe_manifest_path: Path,
    *,
    expected_tasks: int,
) -> dict[str, Any]:
    source_rows = pq.read_table(source_dataset_path).to_pylist()
    identities = [_identity(row) for row in source_rows]
    if len(source_rows) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} source tasks, got {len(source_rows)}")
    if len(set(identities)) != len(identities):
        raise ValueError("source dataset identities are not unique")
    prepared: list[dict[str, Any]] = []
    for original_index, source in enumerate(source_rows):
        record = deepcopy(source)
        extra = dict(record["extra_info"])
        extra.update(
            {
                "adaptive_wave_contract": CONTRACT,
                "adaptive_original_task_index": original_index,
                "adaptive_samples_observed": 0,
                "adaptive_correct_count": 0,
                "adaptive_completed_count": 0,
                "adaptive_timeout_count": 0,
                "adaptive_runtime_error_count": 0,
                "adaptive_decision": "start_two_sample_screen",
                "training_allowed": False,
                "promotion_allowed": False,
            }
        )
        record["extra_info"] = extra
        prepared.append(record)
    write_private_parquet(output_dataset_path, prepared, empty_from=source_dataset_path)
    manifest = {
        "contract": CONTRACT,
        "stage": "initial_two_sample_pool",
        "source_tasks": len(source_rows),
        "remaining_tasks": len(prepared),
        "remaining_difficulty_counts": _difficulty_counts(prepared),
        "source_dataset_sha256": file_sha256(source_dataset_path),
        "remaining_dataset_sha256": file_sha256(output_dataset_path),
        "identity_is_instruction_sha256": True,
        "contains_prompts_gold_sql_task_ids_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    write_json(safe_manifest_path, manifest)
    return manifest


def select_after_wave(
    input_dataset_path: Path,
    wave_per_task_path: Path,
    unresolved_dataset_path: Path,
    mixed_dataset_path: Path,
    safe_manifest_path: Path,
    *,
    expected_prior_samples: int,
    max_samples: int = MAX_SAMPLES,
) -> dict[str, Any]:
    if max_samples not in (6, 8):
        raise ValueError("maximum samples must be 6 or 8")
    if expected_prior_samples not in (0, 2, 4, 6) or expected_prior_samples >= max_samples:
        raise ValueError("expected prior samples must be a valid earlier two-sample wave")
    dataset = pq.read_table(input_dataset_path).to_pylist()
    results = _results_by_identity(wave_per_task_path)
    identities = [_identity(row) for row in dataset]
    if len(set(identities)) != len(identities) or set(results) != set(identities):
        raise ValueError("wave results do not exactly cover the wave dataset")
    mixed_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    cumulative_timeouts = cumulative_runtime_errors = 0
    observed_samples = expected_prior_samples + WAVE_SAMPLES
    for source in dataset:
        record = deepcopy(source)
        extra = dict(record["extra_info"])
        if str(extra.get("adaptive_wave_contract")) != CONTRACT:
            raise ValueError("wave dataset contract mismatch")
        if int(extra.get("adaptive_samples_observed", -1)) != expected_prior_samples:
            raise ValueError("wave dataset prior sample count mismatch")
        result = results[_identity(record)]
        correct = int(extra["adaptive_correct_count"]) + int(result["correct_count"])
        completed = int(extra["adaptive_completed_count"]) + int(result["completed_count"])
        timeouts = int(extra["adaptive_timeout_count"]) + int(
            result.get("trajectory_timeout_count", 0)
        )
        runtime_errors = int(extra["adaptive_runtime_error_count"]) + int(
            result.get("runtime_error_count", 0)
        )
        is_mixed = _explicit_mixed(correct=correct, completed=completed)
        extra.update(
            {
                "adaptive_samples_observed": observed_samples,
                "adaptive_correct_count": correct,
                "adaptive_completed_count": completed,
                "adaptive_timeout_count": timeouts,
                "adaptive_runtime_error_count": runtime_errors,
                "adaptive_decision": (
                    f"confirmed_mixed_stop_after_{observed_samples}"
                    if is_mixed
                    else (
                        f"continue_after_{observed_samples}"
                        if observed_samples < max_samples
                        else "max_samples_reached_unresolved"
                    )
                ),
                "adaptive_mixed_after_samples": observed_samples if is_mixed else None,
                "training_allowed": False,
                "promotion_allowed": False,
            }
        )
        record["extra_info"] = extra
        (mixed_rows if is_mixed else unresolved_rows).append(record)
        cumulative_timeouts += timeouts
        cumulative_runtime_errors += runtime_errors
    write_private_parquet(mixed_dataset_path, mixed_rows, empty_from=input_dataset_path)
    write_private_parquet(
        unresolved_dataset_path,
        unresolved_rows,
        empty_from=input_dataset_path,
    )
    manifest = {
        "contract": CONTRACT,
        "stage": f"decision_after_{observed_samples}_samples",
        "input_tasks": len(dataset),
        "wave_samples_per_task": WAVE_SAMPLES,
        "samples_observed_per_input_task": observed_samples,
        "maximum_samples_per_task": max_samples,
        "new_mixed_tasks": len(mixed_rows),
        "unresolved_tasks": len(unresolved_rows),
        "new_mixed_difficulty_counts": _difficulty_counts(mixed_rows),
        "unresolved_difficulty_counts": _difficulty_counts(unresolved_rows),
        "cumulative_timeout_count_over_input_tasks": cumulative_timeouts,
        "cumulative_runtime_error_count_over_input_tasks": cumulative_runtime_errors,
        "input_dataset_sha256": file_sha256(input_dataset_path),
        "mixed_dataset_sha256": file_sha256(mixed_dataset_path),
        "unresolved_dataset_sha256": file_sha256(unresolved_dataset_path),
        "explicit_mixed_requires_correct_and_completed_wrong": True,
        "timeouts_do_not_prevent_explicit_mixed": True,
        "contains_prompts_gold_sql_task_ids_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    write_json(safe_manifest_path, manifest)
    return manifest


def finalize_four_wave(
    initial_pool_path: Path,
    mixed_after_two_path: Path,
    mixed_after_four_path: Path,
    mixed_after_six_path: Path,
    mixed_after_eight_path: Path,
    unresolved_after_eight_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    initial = pq.read_table(initial_pool_path).to_pylist()
    mixed2 = pq.read_table(mixed_after_two_path).to_pylist()
    mixed4 = pq.read_table(mixed_after_four_path).to_pylist()
    mixed6 = pq.read_table(mixed_after_six_path).to_pylist()
    mixed8 = pq.read_table(mixed_after_eight_path).to_pylist()
    unresolved8 = pq.read_table(unresolved_after_eight_path).to_pylist()
    initial_identities = {_identity(row) for row in initial}
    partitions = [mixed2, mixed4, mixed6, mixed8, unresolved8]
    identity_sets = [{_identity(row) for row in rows} for rows in partitions]
    if any(
        identity_sets[i] & identity_sets[j]
        for i in range(len(identity_sets))
        for j in range(i + 1, len(identity_sets))
    ):
        raise ValueError("eight-sample partitions overlap")
    if set().union(*identity_sets) != initial_identities:
        raise ValueError("eight-sample partitions do not cover the initial pool")
    candidates = [*mixed2, *mixed4, *mixed6, *mixed8]
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "grpo_variance_candidates.sensitive.parquet"
    unresolved_path = output_dir / "unresolved_after_eight.sensitive.parquet"
    write_private_parquet(candidate_path, candidates, empty_from=initial_pool_path)
    write_private_parquet(unresolved_path, unresolved8, empty_from=initial_pool_path)
    unresolved_after_two = len(initial) - len(mixed2)
    unresolved_after_four = unresolved_after_two - len(mixed4)
    unresolved_after_six = unresolved_after_four - len(mixed6)
    if unresolved_after_six != len(mixed8) + len(unresolved8):
        raise ValueError("four-wave partition counts are inconsistent")
    actual = 2 * (
        len(initial) + unresolved_after_two + unresolved_after_four + unresolved_after_six
    )
    full_eight = len(initial) * 8
    summary = {
        "contract": CONTRACT,
        "stage": "complete_after_eight_samples",
        "initial_tasks": len(initial),
        "mixed_after_two_tasks": len(mixed2),
        "mixed_after_four_tasks": len(mixed4),
        "mixed_after_six_tasks": len(mixed6),
        "mixed_after_eight_tasks": len(mixed8),
        "variance_candidate_tasks": len(candidates),
        "unresolved_after_eight_tasks": len(unresolved8),
        "sample_count_distribution": {
            "2": len(mixed2),
            "4": len(mixed4),
            "6": len(mixed6),
            "8": len(mixed8) + len(unresolved8),
        },
        "actual_sampling_trajectories": actual,
        "actual_trajectories_including_existing_two": actual,
        "full_eight_sampling_baseline_trajectories": full_eight,
        "avoided_trajectories_vs_full_eight": full_eight - actual,
        "candidate_difficulty_counts": _difficulty_counts(candidates),
        "candidate_dataset_sha256": file_sha256(candidate_path),
        "unresolved_dataset_sha256": file_sha256(unresolved_path),
        "candidate_payload_is_prompt_and_gold_not_sampled_trajectory": True,
        "contains_prompts_gold_sql_task_ids_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    write_json(output_dir / "adaptive_final_safe_summary.json", summary)
    return summary


def finalize(
    initial_pool_path: Path,
    mixed_after_four_path: Path,
    mixed_after_six_path: Path,
    unresolved_after_six_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    initial = pq.read_table(initial_pool_path).to_pylist()
    mixed4 = pq.read_table(mixed_after_four_path).to_pylist()
    mixed6 = pq.read_table(mixed_after_six_path).to_pylist()
    unresolved6 = pq.read_table(unresolved_after_six_path).to_pylist()
    partitions = [mixed4, mixed6, unresolved6]
    identity_sets = [{_identity(row) for row in rows} for rows in partitions]
    if any(identity_sets[i] & identity_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("final wave partitions overlap")
    if set().union(*identity_sets) != {_identity(row) for row in initial}:
        raise ValueError("final wave partitions do not cover the initial pool")
    candidates = [*mixed4, *mixed6]
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "grpo_variance_candidates.sensitive.parquet"
    write_private_parquet(candidate_path, candidates, empty_from=initial_pool_path)
    unresolved_path = output_dir / "unresolved_after_six.sensitive.parquet"
    write_private_parquet(unresolved_path, unresolved6, empty_from=initial_pool_path)
    initial_tasks = len(initial)
    observed_at_four = len(mixed4)
    observed_at_six = len(mixed6) + len(unresolved6)
    actual_trajectories_including_existing_two = initial_tasks * 4 + observed_at_six * 2
    baseline = initial_tasks * MAX_SAMPLES
    summary = {
        "contract": CONTRACT,
        "stage": "complete",
        "initial_tasks": initial_tasks,
        "mixed_after_four_tasks": len(mixed4),
        "mixed_after_six_tasks": len(mixed6),
        "variance_candidate_tasks": len(candidates),
        "unresolved_after_six_tasks": len(unresolved6),
        "sample_count_distribution": {
            "2": 0,
            "4": observed_at_four,
            "6": observed_at_six,
        },
        "new_trajectories_after_existing_screen": (
            initial_tasks * WAVE_SAMPLES + observed_at_six * WAVE_SAMPLES
        ),
        "actual_trajectories_including_existing_two": actual_trajectories_including_existing_two,
        "full_six_sampling_baseline_trajectories": baseline,
        "avoided_trajectories_vs_full_six": baseline - actual_trajectories_including_existing_two,
        "candidate_difficulty_counts": _difficulty_counts(candidates),
        "candidate_dataset_sha256": file_sha256(candidate_path),
        "unresolved_dataset_sha256": file_sha256(unresolved_path),
        "candidate_payload_is_prompt_and_gold_not_sampled_trajectory": True,
        "contains_prompts_gold_sql_task_ids_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    write_json(output_dir / "adaptive_final_safe_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-remaining")
    prepare.add_argument("--screen-dataset", type=Path, required=True)
    prepare.add_argument("--screen-per-task", type=Path, required=True)
    prepare.add_argument("--excluded-probe-dataset", type=Path, required=True)
    prepare.add_argument("--excluded-direct-dataset", type=Path, required=True)
    prepare.add_argument("--output-dataset", type=Path, required=True)
    prepare.add_argument("--safe-manifest", type=Path, required=True)
    prepare.add_argument("--expected-remaining-tasks", type=int, required=True)

    prepare_initial = subparsers.add_parser("prepare-initial")
    prepare_initial.add_argument("--source-dataset", type=Path, required=True)
    prepare_initial.add_argument("--output-dataset", type=Path, required=True)
    prepare_initial.add_argument("--safe-manifest", type=Path, required=True)
    prepare_initial.add_argument("--expected-tasks", type=int, required=True)

    select = subparsers.add_parser("select-after-wave")
    select.add_argument("--input-dataset", type=Path, required=True)
    select.add_argument("--wave-per-task", type=Path, required=True)
    select.add_argument("--unresolved-dataset", type=Path, required=True)
    select.add_argument("--mixed-dataset", type=Path, required=True)
    select.add_argument("--safe-manifest", type=Path, required=True)
    select.add_argument("--expected-prior-samples", type=int, required=True)
    select.add_argument("--max-samples", type=int, default=MAX_SAMPLES)

    finish = subparsers.add_parser("finalize")
    finish.add_argument("--initial-pool", type=Path, required=True)
    finish.add_argument("--mixed-after-four", type=Path, required=True)
    finish.add_argument("--mixed-after-six", type=Path, required=True)
    finish.add_argument("--unresolved-after-six", type=Path, required=True)
    finish.add_argument("--output-dir", type=Path, required=True)

    finish_eight = subparsers.add_parser("finalize-four-wave")
    finish_eight.add_argument("--initial-pool", type=Path, required=True)
    finish_eight.add_argument("--mixed-after-two", type=Path, required=True)
    finish_eight.add_argument("--mixed-after-four", type=Path, required=True)
    finish_eight.add_argument("--mixed-after-six", type=Path, required=True)
    finish_eight.add_argument("--mixed-after-eight", type=Path, required=True)
    finish_eight.add_argument("--unresolved-after-eight", type=Path, required=True)
    finish_eight.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare-remaining":
        result = prepare_remaining_pool(
            args.screen_dataset,
            args.screen_per_task,
            args.excluded_probe_dataset,
            args.excluded_direct_dataset,
            args.output_dataset,
            args.safe_manifest,
            expected_remaining_tasks=args.expected_remaining_tasks,
        )
    elif args.command == "prepare-initial":
        result = prepare_initial_pool(
            args.source_dataset,
            args.output_dataset,
            args.safe_manifest,
            expected_tasks=args.expected_tasks,
        )
    elif args.command == "select-after-wave":
        result = select_after_wave(
            args.input_dataset,
            args.wave_per_task,
            args.unresolved_dataset,
            args.mixed_dataset,
            args.safe_manifest,
            expected_prior_samples=args.expected_prior_samples,
            max_samples=args.max_samples,
        )
    elif args.command == "finalize":
        result = finalize(
            args.initial_pool,
            args.mixed_after_four,
            args.mixed_after_six,
            args.unresolved_after_six,
            args.output_dir,
        )
    else:
        result = finalize_four_wave(
            args.initial_pool,
            args.mixed_after_two,
            args.mixed_after_four,
            args.mixed_after_six,
            args.mixed_after_eight,
            args.unresolved_after_eight,
            args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
