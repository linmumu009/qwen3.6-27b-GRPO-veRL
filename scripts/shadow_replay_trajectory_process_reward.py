#!/usr/bin/env python3
"""Replay the trajectory-level process reward on the frozen 100x8 run.

Sensitive trajectory-level rows and manual-audit packets stay under the run's
private output directory with mode 0600.  Only aggregate counts, distributions,
hashes, and pass/fail evidence are written to ``safe_summary.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable

import pyarrow.parquet as pq

from llin_verl.trajectory_process_reward import (
    REWARD_CONTRACT,
    compute_trajectory_process_reward,
    hard_gate_reason_counts,
    legacy_boss_reward_total_shadow,
    parse_qwen_tool_events,
    private_event_fingerprint,
    stable_hash,
)


CONTRACT = "qwen38-approved43-trajectory-process-shadow-v1"
EXPECTED_PARQUET_SHA256 = "d86b53d906806b150d43a508dce9b0dd6d05105c07e03961e8e7bf9439ccd944"
EXPECTED_MANIFEST_SHA256 = "1426bc09a3dbaf4709fd89227790603afb7a2bf11beeba80946057d490e0f424"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_private_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_safe_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None, "std": None}

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": round(ordered[0], 8),
        "p25": round(percentile(0.25), 8),
        "median": round(percentile(0.50), 8),
        "p75": round(percentile(0.75), 8),
        "max": round(ordered[-1], 8),
        "mean": round(statistics.fmean(ordered), 8),
        "std": round(statistics.pstdev(ordered), 8),
    }


def population_variance(values: list[float]) -> float:
    return statistics.pvariance(values) if len(values) > 1 else 0.0


def _task_ground_truth(dataset_row: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    truth = json.loads(
        json.dumps(dataset_row["reward_model"]["ground_truth"], ensure_ascii=False)
    )
    truth["evidence_plan"] = task.get("evidence_plan") or {}
    criteria = task.get("verification_criteria") or {}
    truth["must_use_fields"] = criteria.get("must_use_fields") or truth.get(
        "must_use_fields", []
    )
    truth["required_tables"] = task.get("expected_tables") or truth.get(
        "required_tables", []
    )
    return truth


def _manual_sample_rows(rows: list[dict[str, Any]], limit: int = 16) -> list[dict[str, Any]]:
    approved = [row for row in rows if row["approved43"]]
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for answer_type in ("numeric", "table"):
        typed = [row for row in approved if row["answer_type"] == answer_type]
        pools = (
            sorted((row for row in typed if row["correctness"] == 1), key=lambda row: row["process_score"]),
            sorted((row for row in typed if row["correctness"] == 1), key=lambda row: -row["process_score"]),
            sorted((row for row in typed if row["correctness"] == 0), key=lambda row: row["process_score"]),
            sorted((row for row in typed if row["correctness"] == 0), key=lambda row: -row["process_score"]),
        )
        for pool in pools:
            for row in pool[:2]:
                identity = (row["source_task_index"], row["sample_index"])
                if identity not in seen:
                    selected.append(row)
                    seen.add(identity)
    for row in approved:
        if len(selected) >= limit:
            break
        identity = (row["source_task_index"], row["sample_index"])
        if identity not in seen:
            selected.append(row)
            seen.add(identity)
    return selected[:limit]


def _safe_manual_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_identity_sha256": row["trajectory_identity_sha256"],
        "answer_type": row["answer_type"],
        "correctness": row["correctness"],
        "reward": row["reward"],
        "process_score": row["process_score"],
        "process_sql": row["process_sql"],
        "process_table": row["process_table"],
        "process_field": row["process_field"],
        "process_field_applicable": row["process_field_applicable"],
        "process_fit": row["process_fit"],
        "process_efficiency": row["process_efficiency"],
        "tool_event_count": row["tool_event_count"],
        "successful_sql_count": row["successful_sql_count"],
        "matching_sql_count": row["matching_sql_count"],
        "required_table_count": row["required_table_count"],
        "queried_required_table_count": row["queried_required_table_count"],
        "must_use_field_count": row["must_use_field_count"],
        "used_required_field_count": row["used_required_field_count"],
        "hard_gate_passed": row["hard_gate_passed"],
        "valid_tool_protocol": row["valid_tool_protocol"],
        "safe_readonly_tools": row["safe_readonly_tools"],
        "formula_recomputed": row["formula_recomputed"],
    }


def replay(
    *,
    dataset_path: Path,
    tasks_path: Path,
    shards_dir: Path,
    database_path: Path,
    approved_path: Path,
    manifest_path: Path,
    task_audit_path: Path,
    output_dir: Path,
    expected_tasks: int = 100,
    samples_per_task: int = 8,
) -> dict[str, Any]:
    actual_parquet_hash = file_sha256(approved_path)
    actual_manifest_hash = file_sha256(manifest_path)
    if actual_parquet_hash != EXPECTED_PARQUET_SHA256:
        raise ValueError("approved43 Parquet hash mismatch")
    if actual_manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("approved43 manifest hash mismatch")

    dataset = pq.read_table(dataset_path).to_pylist()
    approved = pq.read_table(approved_path).to_pylist()
    tasks = read_jsonl(tasks_path)
    manifest = read_jsonl(manifest_path)
    task_audit = read_jsonl(task_audit_path)
    if not (
        len(dataset) == len(tasks) == expected_tasks
        and len(approved) == len(manifest) == 43
    ):
        raise ValueError("dataset/task/approved43 row counts do not match contract")

    dataset_by_instruction = {
        str(row["extra_info"]["instruction_sha256"]): index
        for index, row in enumerate(dataset)
    }
    approved_hashes = [str(row["extra_info"]["instruction_sha256"]) for row in approved]
    if len(set(approved_hashes)) != 43 or any(
        value not in dataset_by_instruction for value in approved_hashes
    ):
        raise ValueError("approved43 instructions are not 43 unique source members")
    approved_indices = {dataset_by_instruction[value] for value in approved_hashes}
    manifest_by_instruction = {
        str(row["instruction_sha256"]): row for row in manifest
    }
    if set(manifest_by_instruction) != set(approved_hashes):
        raise ValueError("approved43 manifest and Parquet identities differ")
    if any(bool(row.get("training_allowed")) for row in approved):
        raise ValueError("approved43 source unexpectedly changed training_allowed")

    observations: dict[tuple[int, int], dict[str, Any]] = {}
    shard_paths = sorted(shards_dir.glob("tasks_*.jsonl"))
    for path in shard_paths:
        for row in read_jsonl(path):
            key = (int(row["source_task_index"]), int(row["sample_index"]))
            if key in observations:
                raise ValueError(f"duplicate trajectory slot: {key}")
            observations[key] = row
    expected_slots = {
        (task_index, sample_index)
        for task_index in range(expected_tasks)
        for sample_index in range(samples_per_task)
    }
    if set(observations) != expected_slots:
        raise ValueError("trajectory slots are not exactly 100x8")

    audit_by_index = {int(row["source_task_index"]): row for row in task_audit}
    if not approved_indices.issubset(audit_by_index):
        raise ValueError("approved43 task is missing from the private readiness audit")
    for task_index in approved_indices:
        instruction = str(dataset[task_index]["extra_info"]["instruction_sha256"])
        audit_row = audit_by_index[task_index]
        manifest_row = manifest_by_instruction[instruction]
        if (
            int(audit_row["audited_correct_count"])
            != int(manifest_row["audited_correct_count"])
            or not bool(audit_row["gold_replay_passed"])
            or not bool(audit_row["semantic_and_plan_passed"])
            or not bool(audit_row["dataset_binding_passed"])
        ):
            raise ValueError("approved43 manifest no longer matches readiness audit evidence")
    scored_rows: list[dict[str, Any]] = []
    hard_gate_reasons: Counter[str] = Counter()
    for task_index in range(expected_tasks):
        dataset_row = dataset[task_index]
        task = tasks[task_index]
        truth = _task_ground_truth(dataset_row, task)
        instruction_hash = str(dataset_row["extra_info"]["instruction_sha256"])
        manifest_row = manifest_by_instruction.get(instruction_hash)
        for sample_index in range(samples_per_task):
            source = observations[(task_index, sample_index)]
            output = str(source.get("output") or "")
            parsed = parse_qwen_tool_events(output)
            protocol_complete = bool(
                parsed["protocol_complete"]
                and not source.get("trajectory_timeout")
                and not source.get("runtime_error")
            )
            extra = {
                "pi_tool_events": parsed["events"],
                "pi_tool_protocol_complete": protocol_complete,
                "pi_reward_database_path": str(database_path),
                "pi_reward_database_root": str(database_path.parent.parent),
                "auto_retry_count": int(source.get("auto_retry_count", 0) or 0),
                "force_final_retry_count": int(
                    source.get("force_final_retry_count", 0) or 0
                ),
            }
            result = compute_trajectory_process_reward(
                str(dataset_row.get("data_source") or ""),
                output,
                truth,
                extra,
            )
            hard_gate_reasons.update(hard_gate_reason_counts(result))
            legacy = legacy_boss_reward_total_shadow(
                output,
                truth,
                parsed["events"],
                complete=protocol_complete and bool(result["has_final_answer"]),
                executable_answer_ok=bool(result["process_sql"]),
            )
            expected_reward = (
                float(result["acc"]) + 0.20 * float(result["process_score"])
                if result["hard_gate_passed"]
                else 0.0
            )
            identity = stable_hash(
                {"instruction_sha256": instruction_hash, "sample_index": sample_index}
            )
            scored = {
                "source_task_index": task_index,
                "sample_index": sample_index,
                "trajectory_identity_sha256": identity,
                "instruction_sha256": instruction_hash,
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "event_fingerprint_sha256": private_event_fingerprint(parsed["events"]),
                "answer_type": str(truth.get("answer_type") or ""),
                "approved43": task_index in approved_indices,
                "approval_source": manifest_row.get("approval_source") if manifest_row else None,
                "correctness": int(result["acc"]),
                "reward": float(result["score"]),
                "ungated_formula_reward": float(result["acc"])
                + 0.20 * float(result["process_score"]),
                "old_boss_reward_total_shadow": float(legacy["reward_total"]),
                "process_score": float(result["process_score"]),
                "process_sql": float(result["process_sql"]),
                "process_table": float(result["process_table"]),
                "process_field": float(result["process_field"]),
                "process_field_applicable": float(result["process_field_applicable"]),
                "process_fit": float(result["process_fit"]),
                "process_efficiency": float(result["process_efficiency"]),
                "hard_gate_passed": float(result["hard_gate_passed"]),
                "gold_sql_self_consistent": float(result["gold_sql_self_consistent"]),
                "database_available": float(result["database_available"]),
                "valid_tool_protocol": float(result["valid_tool_protocol"]),
                "safe_readonly_tools": float(result["safe_readonly_tools"]),
                "tool_event_count": int(result["tool_event_count"]),
                "successful_sql_count": int(result["successful_sql_count"]),
                "matching_sql_count": int(result["matching_sql_count"]),
                "required_table_count": int(result["required_table_count"]),
                "queried_required_table_count": int(result["queried_required_table_count"]),
                "must_use_field_count": int(result["must_use_field_count"]),
                "used_required_field_count": int(result["used_required_field_count"]),
                "efficiency_full_scan_count": int(result["efficiency_full_scan_count"]),
                "efficiency_duplicate_sql_count": int(result["efficiency_duplicate_sql_count"]),
                "efficiency_duplicate_command_count": int(result["efficiency_duplicate_command_count"]),
                "efficiency_auto_retry_count": int(result["efficiency_auto_retry_count"]),
                "table_comparison_mode": result["table_comparison_mode"],
                "table_order_semantics_source": result["table_order_semantics_source"],
                "tool_call_count": int(parsed["tool_call_count"]),
                "tool_response_count": int(parsed["tool_response_count"]),
                "malformed_tool_call_count": int(parsed["malformed_tool_call_count"]),
                "trajectory_timeout": bool(source.get("trajectory_timeout")),
                "runtime_error": bool(source.get("runtime_error")),
                "formula_recomputed": math.isclose(
                    float(result["score"]), expected_reward, abs_tol=1e-8, rel_tol=0.0
                ),
            }
            scored_rows.append(scored)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        grouped[row["source_task_index"]].append(row)
    if any(len(rows) != samples_per_task for rows in grouped.values()):
        raise ValueError("scored groups are not all size eight")

    approved_group_checks: list[dict[str, Any]] = []
    for task_index in sorted(approved_indices):
        rows = grouped[task_index]
        correct = [row for row in rows if row["correctness"] == 1]
        wrong = [row for row in rows if row["correctness"] == 0]
        if not correct or not wrong:
            raise ValueError("approved43 no longer forms a corrected mixed group")
        min_correct = min(row["reward"] for row in correct)
        max_wrong = max(row["reward"] for row in wrong)
        min_correct_formula = min(row["ungated_formula_reward"] for row in correct)
        max_wrong_formula = max(row["ungated_formula_reward"] for row in wrong)
        all_hard_gates_passed = all(row["hard_gate_passed"] for row in rows)
        approved_group_checks.append(
            {
                "task_identity_sha256": stable_hash(
                    {"instruction_sha256": rows[0]["instruction_sha256"]}
                ),
                "correct_count": len(correct),
                "incorrect_count": len(wrong),
                "all_hard_gates_passed": all_hard_gates_passed,
                "min_correct_reward": min_correct,
                "max_incorrect_reward": max_wrong,
                "strict_separation": min_correct > max_wrong,
                "separation_margin": min_correct - max_wrong,
                "min_correct_formula_reward": min_correct_formula,
                "max_incorrect_formula_reward": max_wrong_formula,
                "formula_strict_separation": min_correct_formula > max_wrong_formula,
                "formula_separation_margin": min_correct_formula - max_wrong_formula,
                "new_reward_variance": population_variance([row["reward"] for row in rows]),
                "old_reward_variance": population_variance(
                    [row["old_boss_reward_total_shadow"] for row in rows]
                ),
            }
        )

    repaired_table_indices = {
        dataset_by_instruction[instruction]
        for instruction, row in manifest_by_instruction.items()
        if row.get("approval_source") == "reward_false_negative_repair"
    }
    table_replay_checks = []
    for task_index in sorted(repaired_table_indices):
        instruction = str(dataset[task_index]["extra_info"]["instruction_sha256"])
        expected_count = int(manifest_by_instruction[instruction]["audited_correct_count"])
        actual_count = sum(row["correctness"] for row in grouped[task_index])
        table_replay_checks.append(
            {
                "task_identity_sha256": stable_hash({"instruction_sha256": instruction}),
                "expected_correct_count": expected_count,
                "shadow_correct_count": actual_count,
                "count_matches": expected_count == actual_count,
                "remains_mixed": 0 < actual_count < samples_per_task,
            }
        )

    component_names = (
        "process_score",
        "process_sql",
        "process_table",
        "process_field",
        "process_fit",
        "process_efficiency",
    )
    distributions = {
        component: {
            "all": describe(row[component] for row in scored_rows),
            "correct": describe(
                row[component] for row in scored_rows if row["correctness"] == 1
            ),
            "incorrect": describe(
                row[component] for row in scored_rows if row["correctness"] == 0
            ),
            "approved43": describe(
                row[component] for row in scored_rows if row["approved43"]
            ),
        }
        for component in component_names
    }
    reward_distributions = {
        "new": {
            "all": describe(row["reward"] for row in scored_rows),
            "correct": describe(
                row["reward"] for row in scored_rows if row["correctness"] == 1
            ),
            "incorrect": describe(
                row["reward"] for row in scored_rows if row["correctness"] == 0
            ),
            "approved43": describe(
                row["reward"] for row in scored_rows if row["approved43"]
            ),
        },
        "old_boss_shadow": {
            "all": describe(row["old_boss_reward_total_shadow"] for row in scored_rows),
            "correct": describe(
                row["old_boss_reward_total_shadow"]
                for row in scored_rows
                if row["correctness"] == 1
            ),
            "incorrect": describe(
                row["old_boss_reward_total_shadow"]
                for row in scored_rows
                if row["correctness"] == 0
            ),
            "approved43": describe(
                row["old_boss_reward_total_shadow"]
                for row in scored_rows
                if row["approved43"]
            ),
        },
    }

    corrected_bucket_counts: Counter[str] = Counter()
    for rows in grouped.values():
        correct_count = sum(row["correctness"] for row in rows)
        bucket = "all_wrong" if correct_count == 0 else "all_correct" if correct_count == 8 else "mixed"
        corrected_bucket_counts[bucket] += 1

    manual_samples = _manual_sample_rows(scored_rows)
    private_manual_rows = []
    for row in manual_samples:
        source = observations[(row["source_task_index"], row["sample_index"])]
        parsed = parse_qwen_tool_events(str(source.get("output") or ""))
        private_manual_rows.append(
            {
                **row,
                "final_answer": str(source.get("output") or ""),
                "tool_events": parsed["events"],
                "audit_checklist": {
                    "formula_matches": row["formula_recomputed"],
                    "matching_sql_requires_successful_sql": (
                        row["matching_sql_count"] == 0 or row["successful_sql_count"] > 0
                    ),
                    "table_credit_requires_all_required_tables": (
                        row["process_table"] == 0
                        or row["required_table_count"]
                        == row["queried_required_table_count"]
                    ),
                    "fit_requires_successful_sql": (
                        row["process_fit"] == 0 or row["successful_sql_count"] > 0
                    ),
                    "tool_event_fingerprint_present": bool(row["event_fingerprint_sha256"]),
                },
            }
        )

    private_dir = output_dir / "private"
    write_private_jsonl(private_dir / "trajectory_rewards.sensitive.jsonl", scored_rows)
    write_private_jsonl(
        private_dir / "approved43_group_checks.sensitive.jsonl", approved_group_checks
    )
    write_private_jsonl(
        private_dir / "manual_audit_samples.sensitive.jsonl", private_manual_rows
    )

    safe_manual = [_safe_manual_projection(row) for row in manual_samples]
    all_formula_match = all(row["formula_recomputed"] for row in scored_rows)
    summary = {
        "contract": CONTRACT,
        "reward_contract": REWARD_CONTRACT,
        "training_status": "paused_shadow_only_no_model_or_npu",
        "reward_scope": "one_scalar_after_complete_multiturn_trajectory",
        "turn_or_token_credit_assignment_implemented": False,
        "input_gate": {
            "dataset_rows": len(dataset),
            "trajectory_rows": len(scored_rows),
            "groups": len(grouped),
            "samples_per_group": samples_per_task,
            "approved43_rows": len(approved),
            "approved43_unique_source_tasks": len(approved_indices),
            "approved43_parquet_sha256": actual_parquet_hash,
            "approved43_manifest_sha256": actual_manifest_hash,
            "approved43_source_training_allowed_all_false": all(
                not bool(row["extra_info"].get("training_allowed")) for row in approved
            ),
            "shard_files": len(shard_paths),
            "shards_set_sha256": stable_hash(
                {path.name: file_sha256(path) for path in shard_paths}
            ),
            "database_sha256": file_sha256(database_path),
            "task_audit_sha256": file_sha256(task_audit_path),
        },
        "corrected_outcomes": {
            "correct_trajectories": sum(row["correctness"] for row in scored_rows),
            "incorrect_trajectories": sum(1 - row["correctness"] for row in scored_rows),
            "task_bucket_counts": dict(sorted(corrected_bucket_counts.items())),
            "approved43_all_mixed": all(
                0 < sum(row["correctness"] for row in grouped[index]) < samples_per_task
                for index in approved_indices
            ),
        },
        "hard_gates": {
            "eligible_trajectories": sum(row["hard_gate_passed"] for row in scored_rows),
            "ineligible_trajectories": sum(1 - row["hard_gate_passed"] for row in scored_rows),
            "failure_reason_counts": dict(sorted(hard_gate_reasons.items())),
            "approved43_all_trajectories_eligible": all(
                row["hard_gate_passed"] for row in scored_rows if row["approved43"]
            ),
        },
        "correctness_dominance": {
            "approved43_groups_checked": len(approved_group_checks),
            "formula_groups_strictly_separated": sum(
                bool(row["formula_strict_separation"]) for row in approved_group_checks
            ),
            "formula_groups_failing_strict_separation": sum(
                not bool(row["formula_strict_separation"]) for row in approved_group_checks
            ),
            "minimum_formula_group_margin": min(
                row["formula_separation_margin"] for row in approved_group_checks
            ),
            "groups_strictly_separated": sum(
                bool(row["strict_separation"]) for row in approved_group_checks
            ),
            "groups_failing_strict_separation": sum(
                not bool(row["strict_separation"]) for row in approved_group_checks
            ),
            "minimum_group_margin": min(
                row["separation_margin"] for row in approved_group_checks
            ),
            "fully_hard_gate_eligible_groups": sum(
                bool(row["all_hard_gates_passed"]) for row in approved_group_checks
            ),
            "hard_gate_excluded_groups": sum(
                not bool(row["all_hard_gates_passed"]) for row in approved_group_checks
            ),
            "eligible_groups_strictly_separated": sum(
                bool(row["all_hard_gates_passed"] and row["strict_separation"])
                for row in approved_group_checks
            ),
            "correct_reward_floor_by_formula": 1.0,
            "incorrect_reward_ceiling_by_formula": 0.2,
        },
        "table_false_negative_repair": {
            "repaired_table_tasks": len(table_replay_checks),
            "correct_count_matches_audit": sum(
                bool(row["count_matches"]) for row in table_replay_checks
            ),
            "remain_mixed": sum(bool(row["remains_mixed"]) for row in table_replay_checks),
            "all_22_pass": len(table_replay_checks) == 22
            and all(row["count_matches"] and row["remains_mixed"] for row in table_replay_checks),
        },
        "reward_distributions": reward_distributions,
        "process_component_distributions": distributions,
        "group_variance": {
            "approved43_new_positive_variance_groups": sum(
                row["new_reward_variance"] > 0 for row in approved_group_checks
            ),
            "approved43_old_positive_variance_groups": sum(
                row["old_reward_variance"] > 0 for row in approved_group_checks
            ),
            "new_variance": describe(
                row["new_reward_variance"] for row in approved_group_checks
            ),
            "old_variance": describe(
                row["old_reward_variance"] for row in approved_group_checks
            ),
        },
        "tool_event_anti_forgery": {
            "process_components_read_solution_text": False,
            "process_components_source": "paired_runtime_or_shadow_tool_events_plus_readonly_database_replay",
            "final_answer_only_affects_corrected_C": True,
            "formula_recomputed_for_all_800": all_formula_match,
            "tool_call_blocks": sum(row["tool_call_count"] for row in scored_rows),
            "tool_response_blocks": sum(row["tool_response_count"] for row in scored_rows),
            "protocol_complete_trajectories": sum(
                row["valid_tool_protocol"] for row in scored_rows
            ),
        },
        "manual_sample_audit": {
            "status": "structural_packet_prepared_for_human_review",
            "sample_count": len(manual_samples),
            "sampling": "deterministic_stratified_by_answer_type_correctness_and_process_extremes",
            "safe_rows": safe_manual,
            "private_packet_mode": "0600",
        },
        "old_reward_caveat": (
            "Old boss reward_total is deterministically reconstructed from decoded paired tool "
            "events because the original boss JSONL event stream was not persisted in shards."
        ),
        "private_outputs": {
            "trajectory_reward_rows": len(scored_rows),
            "approved43_group_check_rows": len(approved_group_checks),
            "manual_audit_sample_rows": len(private_manual_rows),
            "private_paths_emitted": False,
        },
        "promotion_allowed": False,
        "formal_training_allowed": False,
    }
    write_safe_json(output_dir / "safe_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--approved43", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = replay(
        dataset_path=args.dataset,
        tasks_path=args.tasks,
        shards_dir=args.shards_dir,
        database_path=args.database,
        approved_path=args.approved43,
        manifest_path=args.manifest,
        task_audit_path=args.task_audit,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "contract": summary["contract"],
        "trajectory_rows": summary["input_gate"]["trajectory_rows"],
        "approved43_groups_checked": summary["correctness_dominance"]["approved43_groups_checked"],
        "groups_strictly_separated": summary["correctness_dominance"]["groups_strictly_separated"],
        "table_22_pass": summary["table_false_negative_repair"]["all_22_pass"],
        "training_status": summary["training_status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
