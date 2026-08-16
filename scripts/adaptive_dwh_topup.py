#!/usr/bin/env python3
"""Build and reconcile a two-stage DWH rollout screen.

Stage one samples every task twice.  Stage two adds six trajectories only for
tasks that already produced a correct answer, match an anonymous structural
signature learned from earlier mixed tasks, or enter a small deterministic
exploration reserve.  Sensitive prompts and labels stay in private Parquet;
all JSON manifests contain aggregate counts and structural metadata only.
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

from scripts.analyze_multisandbox_dwh_rollout import analyze
from scripts.plan_first_dwh_timeout_retry import load_complete_shards
from scripts.prepare_plan_first_dwh_model_comparison import canonical_hash, file_sha256
from scripts.standalone_rollout_shards import shard_ranges, write_jsonl_atomic


PROFILE_CONTRACT = "llin-adaptive-dwh-reference-profile-v1"
SELECTION_CONTRACT = "llin-adaptive-dwh-topup-selection-v1"
FINAL_CONTRACT = "llin-adaptive-dwh-eight-trajectory-final-v1"
SCREEN_SAMPLES = 2
TOPUP_SAMPLES = 6
FINAL_SAMPLES = SCREEN_SAMPLES + TOPUP_SAMPLES


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_private_parquet(path: Path, rows: list[dict[str, Any]], *, empty_from: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if rows:
        table = pa.Table.from_pylist(rows)
    elif empty_from is not None:
        table = pq.read_table(empty_from).slice(0, 0)
    else:
        raise ValueError("cannot infer an empty Parquet schema")
    pq.write_table(table, temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def task_instruction_hash(task: dict[str, Any]) -> str:
    return canonical_hash(str(task["natural_language_instruction"]))


def structural_payload(task: dict[str, Any]) -> dict[str, Any]:
    plan = task.get("query_plan") or {}
    raw_counts = plan.get("feature_counts") or {}
    counts: dict[str, int | float] = {}
    for key, value in sorted(raw_counts.items()):
        if isinstance(value, bool):
            counts[str(key)] = int(value)
        elif isinstance(value, int):
            counts[str(key)] = value
        elif isinstance(value, float):
            counts[str(key)] = value
        else:
            raise ValueError(f"non-numeric structural feature: {key}")
    return {
        "task_type": str(task["task_type"]),
        "difficulty_level": int(task["difficulty_level"]),
        "answer_type": str(task["gold_answer"]["answer_type"]),
        "expected_table_count": len(task.get("expected_tables") or []),
        "expected_operations": sorted(str(value) for value in (task.get("expected_operations") or [])),
        "feature_counts": counts,
    }


def structural_signature(task: dict[str, Any]) -> str:
    return canonical_hash(structural_payload(task))


def is_explicit_mixed(row: dict[str, Any]) -> bool:
    correct = int(row["correct_count"])
    completed = int(row["completed_count"])
    return correct > 0 and completed - correct > 0


def profile_reference(
    source_tasks_path: Path,
    per_task_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    source_tasks = read_jsonl(source_tasks_path)
    tasks_by_hash = {task_instruction_hash(task): task for task in source_tasks}
    if len(tasks_by_hash) != len(source_tasks):
        raise ValueError("reference source instructions are not unique")
    per_task = read_jsonl(per_task_path)
    mixed = [row for row in per_task if is_explicit_mixed(row)]
    signature_counts: Counter[str] = Counter()
    signature_payloads: dict[str, dict[str, Any]] = {}
    difficulty_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    timeout_mixed = 0
    for row in mixed:
        identity = str(row["instruction_sha256"])
        if identity not in tasks_by_hash:
            raise ValueError("reference result cannot be joined to source task")
        task = tasks_by_hash[identity]
        signature = structural_signature(task)
        signature_counts[signature] += 1
        signature_payloads[signature] = structural_payload(task)
        difficulty_counts[str(task["difficulty_level"])] += 1
        task_type_counts[str(task["task_type"])] += 1
        timeout_mixed += int(int(row.get("trajectory_timeout_count", 0)) > 0)
    if not mixed:
        raise ValueError("reference profile contains no explicit mixed tasks")
    payload = {
        "contract": PROFILE_CONTRACT,
        "reference_source_tasks": len(source_tasks),
        "reference_observed_tasks": len(per_task),
        "reference_explicit_mixed_tasks": len(mixed),
        "reference_explicit_mixed_with_timeout": timeout_mixed,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "signatures": [
            {
                "signature_sha256": signature,
                "mixed_tasks": signature_counts[signature],
                "structure": signature_payloads[signature],
            }
            for signature in sorted(signature_counts)
        ],
        "contains_prompts_gold_sql_task_ids_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    write_json(output_path, payload)
    return payload


def _stable_exploration_key(seed: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def prepare_topup(
    screen_dataset_path: Path,
    screen_per_task_path: Path,
    source_tasks_path: Path,
    reference_profile_path: Path,
    output_dataset_path: Path,
    safe_manifest_path: Path,
    *,
    exploration_per_level: int,
    seed: str,
) -> dict[str, Any]:
    if exploration_per_level < 0:
        raise ValueError("exploration_per_level cannot be negative")
    dataset = pq.read_table(screen_dataset_path).to_pylist()
    results = read_jsonl(screen_per_task_path)
    if len(results) != len(dataset):
        raise ValueError("screen result count does not match dataset")
    by_index = {int(row["source_task_index"]): row for row in results}
    if set(by_index) != set(range(len(dataset))):
        raise ValueError("screen per-task indices are not an exact partition")
    source_tasks = read_jsonl(source_tasks_path)
    tasks_by_hash = {task_instruction_hash(task): task for task in source_tasks}
    if len(tasks_by_hash) != len(source_tasks):
        raise ValueError("target source instructions are not unique")
    profile = json.loads(reference_profile_path.read_text(encoding="utf-8"))
    if str(profile.get("contract")) != PROFILE_CONTRACT:
        raise ValueError("reference profile contract mismatch")
    reference_signatures = {
        str(row["signature_sha256"]) for row in profile.get("signatures", [])
    }
    if not reference_signatures:
        raise ValueError("reference profile has no structural signatures")

    candidates: list[dict[str, Any]] = []
    for index, record in enumerate(dataset):
        extra = record["extra_info"]
        identity = str(extra["instruction_sha256"])
        if identity not in tasks_by_hash:
            raise ValueError("target dataset cannot be joined to source task")
        result = by_index[index]
        if str(result["instruction_sha256"]) != identity:
            raise ValueError("screen result identity mismatch")
        if int(result["correct_count"]) > SCREEN_SAMPLES:
            raise ValueError("screen correct count exceeds two-sample contract")
        signature = structural_signature(tasks_by_hash[identity])
        reasons: list[str] = []
        if int(result["correct_count"]) > 0:
            reasons.append("screen_correct")
        if signature in reference_signatures:
            reasons.append("reference_structure")
        candidates.append(
            {
                "index": index,
                "identity": identity,
                "signature": signature,
                "difficulty": str(extra["difficulty_level"]),
                "reasons": reasons,
                "result": result,
                "record": record,
            }
        )

    selected_indices = {row["index"] for row in candidates if row["reasons"]}
    for level in sorted({row["difficulty"] for row in candidates}):
        remaining = [
            row
            for row in candidates
            if row["difficulty"] == level and row["index"] not in selected_indices
        ]
        remaining.sort(key=lambda row: _stable_exploration_key(seed, row["identity"]))
        for row in remaining[:exploration_per_level]:
            row["reasons"].append("exploration")
            selected_indices.add(row["index"])

    selected_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    screen_correct_histogram: Counter[str] = Counter()
    selected_with_screen_timeout = 0
    for row in candidates:
        if row["index"] not in selected_indices:
            continue
        record = deepcopy(row["record"])
        result = row["result"]
        extra = dict(record["extra_info"])
        extra.update(
            {
                "adaptive_contract": SELECTION_CONTRACT,
                "adaptive_original_task_index": int(row["index"]),
                "adaptive_structural_signature_sha256": row["signature"],
                "adaptive_selection_reasons": sorted(row["reasons"]),
                "adaptive_screen_correct_count": int(result["correct_count"]),
                "adaptive_screen_completed_count": int(result["completed_count"]),
                "adaptive_screen_timeout_count": int(result["trajectory_timeout_count"]),
                "adaptive_screen_samples": SCREEN_SAMPLES,
                "adaptive_topup_samples": TOPUP_SAMPLES,
                "training_allowed": False,
                "promotion_allowed": False,
            }
        )
        record["extra_info"] = extra
        selected_rows.append(record)
        reason_counts.update(row["reasons"])
        difficulty_counts[row["difficulty"]] += 1
        screen_correct_histogram[str(result["correct_count"])] += 1
        selected_with_screen_timeout += int(int(result["trajectory_timeout_count"]) > 0)
    if not selected_rows:
        raise ValueError("adaptive selector produced no top-up tasks")
    write_private_parquet(output_dataset_path, selected_rows)
    manifest = {
        "contract": SELECTION_CONTRACT,
        "screen_tasks": len(dataset),
        "screen_samples_per_task": SCREEN_SAMPLES,
        "selected_tasks": len(selected_rows),
        "topup_samples_per_selected_task": TOPUP_SAMPLES,
        "selection_reason_counts_nonexclusive": dict(sorted(reason_counts.items())),
        "selected_difficulty_counts": dict(sorted(difficulty_counts.items())),
        "selected_screen_correct_count_histogram": dict(sorted(screen_correct_histogram.items())),
        "selected_with_screen_timeout": selected_with_screen_timeout,
        "reference_structural_signatures": len(reference_signatures),
        "exploration_per_level": exploration_per_level,
        "screen_dataset_sha256": file_sha256(screen_dataset_path),
        "topup_dataset_sha256": file_sha256(output_dataset_path),
        "selection_identity_is_instruction_sha256": True,
        "contains_prompts_gold_sql_task_ids_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    write_json(safe_manifest_path, manifest)
    return manifest


def finalize(
    screen_dataset_path: Path,
    screen_shards_dir: Path,
    topup_dataset_path: Path,
    topup_shards_dir: Path,
    output_dir: Path,
    *,
    expected_screen_tasks: int,
) -> dict[str, Any]:
    screen_dataset = pq.read_table(screen_dataset_path).to_pylist()
    topup_dataset = pq.read_table(topup_dataset_path).to_pylist()
    if len(screen_dataset) != expected_screen_tasks:
        raise ValueError("screen dataset shape mismatch")
    screen_observations, _ = load_complete_shards(
        screen_shards_dir,
        expected_tasks=expected_screen_tasks,
        samples_per_task=SCREEN_SAMPLES,
    )
    topup_observations, _ = load_complete_shards(
        topup_shards_dir,
        expected_tasks=len(topup_dataset),
        samples_per_task=TOPUP_SAMPLES,
    )
    screen_hashes = [str(row["extra_info"]["instruction_sha256"]) for row in screen_dataset]
    if len(set(screen_hashes)) != len(screen_hashes):
        raise ValueError("screen dataset identities are not unique")
    selected_original_indices: set[int] = set()
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for selected_index, record in enumerate(topup_dataset):
        extra = record["extra_info"]
        if str(extra.get("adaptive_contract")) != SELECTION_CONTRACT:
            raise ValueError("top-up row contract mismatch")
        original_index = int(extra["adaptive_original_task_index"])
        if original_index in selected_original_indices:
            raise ValueError("duplicate original task in top-up dataset")
        selected_original_indices.add(original_index)
        identity = str(extra["instruction_sha256"])
        if not 0 <= original_index < len(screen_hashes) or screen_hashes[original_index] != identity:
            raise ValueError("top-up row does not map to screen dataset")
        for sample_index in range(SCREEN_SAMPLES):
            row = dict(screen_observations[(original_index, sample_index)])
            row["source_task_index"] = selected_index
            row["sample_index"] = sample_index
            row["adaptive_phase"] = "screen"
            merged[(selected_index, sample_index)] = row
        for sample_index in range(TOPUP_SAMPLES):
            row = dict(topup_observations[(selected_index, sample_index)])
            row["source_task_index"] = selected_index
            row["sample_index"] = SCREEN_SAMPLES + sample_index
            row["adaptive_phase"] = "topup"
            merged[(selected_index, SCREEN_SAMPLES + sample_index)] = row

    for start, stop in shard_ranges(len(topup_dataset), 48):
        rows = [
            merged[(task_index, sample_index)]
            for task_index in range(start, stop)
            for sample_index in range(FINAL_SAMPLES)
        ]
        write_jsonl_atomic(
            output_dir / "shards" / f"tasks_{start:05d}_{stop:05d}.jsonl", rows
        )
    outcome = analyze(
        topup_dataset_path,
        output_dir / "shards",
        output_dir / "outcomes",
        expected_tasks=len(topup_dataset),
        samples_per_task=FINAL_SAMPLES,
    )
    per_task = read_jsonl(output_dir / "outcomes" / "per_task.sensitive.jsonl")
    relaxed_indices = [
        int(row["source_task_index"]) for row in per_task if is_explicit_mixed(row)
    ]
    relaxed_rows = [topup_dataset[index] for index in relaxed_indices]
    relaxed_path = output_dir / "outcomes" / "relaxed_mixed_candidates.sensitive.parquet"
    write_private_parquet(relaxed_path, relaxed_rows, empty_from=topup_dataset_path)
    relaxed_difficulty = Counter(
        str(topup_dataset[index]["extra_info"]["difficulty_level"])
        for index in relaxed_indices
    )
    relaxed_with_timeout = sum(
        int(int(per_task[index]["trajectory_timeout_count"]) > 0)
        for index in relaxed_indices
    )
    selected = len(topup_dataset)
    baseline = expected_screen_tasks * FINAL_SAMPLES
    actual = expected_screen_tasks * SCREEN_SAMPLES + selected * TOPUP_SAMPLES
    summary = {
        "contract": FINAL_CONTRACT,
        "screen_tasks": expected_screen_tasks,
        "screen_trajectories": expected_screen_tasks * SCREEN_SAMPLES,
        "selected_tasks": selected,
        "topup_trajectories": selected * TOPUP_SAMPLES,
        "final_selected_trajectories": selected * FINAL_SAMPLES,
        "actual_sampling_trajectories": actual,
        "full_eight_sampling_baseline_trajectories": baseline,
        "avoided_trajectories": baseline - actual,
        "strict_mixed_tasks": int(outcome["mixed_screening_rows"]),
        "relaxed_explicit_mixed_tasks": len(relaxed_indices),
        "relaxed_explicit_mixed_with_timeout": relaxed_with_timeout,
        "relaxed_explicit_mixed_difficulty_counts": dict(sorted(relaxed_difficulty.items())),
        "timeout_trajectories": int(outcome["timeout_trajectories"]),
        "runtime_error_trajectories": int(outcome["runtime_error_trajectories"]),
        "correct_trajectories": int(outcome["correct_trajectories"]),
        "relaxed_candidate_dataset_sha256": file_sha256(relaxed_path),
        "selection_biased_screening": True,
        "contains_prompts_gold_sql_task_ids_outputs_or_server_paths": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    write_json(output_dir / "adaptive_final_safe_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile-reference")
    profile_parser.add_argument("--source-tasks", type=Path, required=True)
    profile_parser.add_argument("--per-task", type=Path, required=True)
    profile_parser.add_argument("--output", type=Path, required=True)

    prepare_parser = subparsers.add_parser("prepare-topup")
    prepare_parser.add_argument("--screen-dataset", type=Path, required=True)
    prepare_parser.add_argument("--screen-per-task", type=Path, required=True)
    prepare_parser.add_argument("--source-tasks", type=Path, required=True)
    prepare_parser.add_argument("--reference-profile", type=Path, required=True)
    prepare_parser.add_argument("--output-dataset", type=Path, required=True)
    prepare_parser.add_argument("--safe-manifest", type=Path, required=True)
    prepare_parser.add_argument("--exploration-per-level", type=int, default=2)
    prepare_parser.add_argument("--seed", default="adaptive-dwh-topup-v1")

    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--screen-dataset", type=Path, required=True)
    final_parser.add_argument("--screen-shards-dir", type=Path, required=True)
    final_parser.add_argument("--topup-dataset", type=Path, required=True)
    final_parser.add_argument("--topup-shards-dir", type=Path, required=True)
    final_parser.add_argument("--output-dir", type=Path, required=True)
    final_parser.add_argument("--expected-screen-tasks", type=int, required=True)

    args = parser.parse_args()
    if args.command == "profile-reference":
        result = profile_reference(args.source_tasks, args.per_task, args.output)
    elif args.command == "prepare-topup":
        result = prepare_topup(
            args.screen_dataset,
            args.screen_per_task,
            args.source_tasks,
            args.reference_profile,
            args.output_dataset,
            args.safe_manifest,
            exploration_per_level=args.exploration_per_level,
            seed=args.seed,
        )
    else:
        result = finalize(
            args.screen_dataset,
            args.screen_shards_dir,
            args.topup_dataset,
            args.topup_shards_dir,
            args.output_dir,
            expected_screen_tasks=args.expected_screen_tasks,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
