#!/usr/bin/env python3
"""Validate prefix curriculum v1 and emit veRL-native runtime datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.outcome_gated_contract import evidence_binding_hash
from llin_verl.prefix_state_curriculum import (
    CONTRACT,
    RESET_MODE,
    adapt_pi_prefix_messages,
    json_field,
    prefix_group_base,
    stable_json_sha256,
    validate_ready_state,
)


EXPECTED_SAFE_SHA256 = "7fcace126d13b0ded74bada98075ec64994f1dea800218f6bf3c9ce0f82788bf"
EXPECTED_MANIFEST_SHA256 = "3668fc60118a7f0371f7eb304881907d49eea31738daf8347e441d7ee723b37d"
EXPECTED_ALL_STATES_SHA256 = "99129edb21b64dfa9786eb56f413c813e6ecc9de6ba5b1da95f5709cb8d28c4e"
EXPECTED_FORMAL_SHA256 = "47cab569395e6015f9ae3e840526840feaa4a38379bac52bac2a11cd1192a1ac"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private(path: Path) -> None:
    path.chmod(0o700 if path.is_dir() else 0o600)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("JSONL row must be a mapping")
                values.append(value)
    return values


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pq.write_table(pa.Table.from_pylist(rows), path)
    _private(path.parent)
    _private(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _private(path.parent)
    _private(path)


def _truth_rows(formal: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(formal)
    if len(rows) != 14:
        raise ValueError("frozen formal task package must contain exactly 14 rows")
    output: dict[str, dict[str, Any]] = {}
    identities: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        identity = str(row.get("task_identity_sha256") or "")
        record = row.get("dataset_record")
        if not task_id or task_id in output or not identity or identity in identities:
            raise ValueError("formal task identities are missing or duplicated")
        if not isinstance(record, dict):
            raise TypeError("formal task is missing dataset_record")
        if row.get("gold_sql_model_visible") is not False:
            raise ValueError("formal task does not explicitly hide scoring material")
        output[task_id] = row
        identities.add(identity)
    return output


def _trajectory_selection(states: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    by_task_trajectory: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in states:
        by_task_trajectory[str(row["task_id"])][str(row["source_trajectory_id"])].append(row)

    selected: dict[str, str] = {}
    audit: dict[str, Any] = {}
    for task_id, trajectories in sorted(by_task_trajectory.items()):
        candidates: list[tuple[str, bool, int, int, int]] = []
        for trajectory_id, rows in trajectories.items():
            ready = [row for row in rows if row.get("training_ready") is True]
            full = next((row for row in rows if row.get("stage") == "stage-full"), None)
            q = int((full or rows[0]).get("teacher_suffix_query_attempt_count_baseline", 0))
            t = int((full or rows[0]).get("teacher_suffix_tool_response_tokens_baseline", 0))
            candidates.append((trajectory_id, len(ready) == len(rows), len(ready), q, t))
        fully_ready = [value for value in candidates if value[1]]
        eligible = fully_ready or candidates
        median_cost = statistics.median(value[3] * 4000 + value[4] for value in eligible)
        choice = min(
            eligible,
            key=lambda value: (
                -int(value[1]),
                abs((value[3] * 4000 + value[4]) - median_cost),
                -value[2],
                value[0],
            ),
        )
        selected[task_id] = choice[0]
        audit[task_id] = {
            "representative_fully_ready": choice[1],
            "representative_ready_state_count": choice[2],
            "candidate_trajectory_count": len(candidates),
        }
    return selected, audit


def _runtime_row(
    state: dict[str, Any], truth_row: dict[str, Any], database_root: str
) -> dict[str, Any]:
    validate_ready_state(state)
    prompt = adapt_pi_prefix_messages(state["prefix_messages"])
    record = copy.deepcopy(truth_row["dataset_record"])
    record["prompt"] = prompt
    truth = ((record.get("reward_model") or {}).get("ground_truth") or {})
    if not isinstance(truth, dict):
        raise TypeError("dataset_record ground truth is missing")
    if str(truth.get("task_id") or "") != str(state["task_id"]):
        raise ValueError("prefix and scoring task identity mismatch")
    if str(truth.get("environment_id") or "") != str(state["environment_id"]):
        raise ValueError("prefix and scoring environment identity mismatch")
    if str(truth_row.get("database_sha256") or "") != str(state["database_sha256"]):
        raise ValueError("prefix and frozen task database hashes differ")
    truth["process_evidence_binding_sha256"] = evidence_binding_hash(truth)

    extra = record.setdefault("extra_info", {})
    if not isinstance(extra, dict):
        raise TypeError("dataset_record extra_info must be a mapping")
    task_id = str(state["task_id"])
    state_id = str(state["prefix_state_id"])
    extra.update(
        {
            "prefix_curriculum_contract": CONTRACT,
            "prefix_curriculum_training_ready": True,
            "prefix_state_id": state_id,
            "task_id": task_id,
            "prefix_group_base": prefix_group_base(task_id, state_id),
            "prefix_prompt_sha256": stable_json_sha256(prompt),
            "prefix_message_count": len(prompt),
            "generated_suffix_start_message_index": len(prompt),
            "response_mask_scope": "generated_suffix_assistant_tokens_only",
            "reward_scope": "generated_suffix_only",
            "final_correctness_scope": "combined_prefix_plus_suffix",
            "prefix_counts_toward_process_or_efficiency_reward": False,
            "prefix_future_information_leakage": 0,
            "prefix_hidden_reasoning_count": 0,
            "prefix_inherited_evidence": bool(state.get("inherited_evidence")),
            "prefix_query_attempt_count_audit_only": int(state.get("prefix_query_attempt_count", 0)),
            "prefix_tool_response_tokens_audit_only": int(state.get("prefix_tool_response_tokens", 0)),
            "workspace_reset_mode": RESET_MODE,
            "workspace_identity_sha256": str(state["workspace_identity_sha256"]),
            "database_sha256": str(state["database_sha256"]),
            "environment_id": str(state["environment_id"]),
            "pi_reward_database_root": str(database_root),
            "curriculum_split": str(state["split"]),
            "curriculum_stage": str(state["stage"]),
            "remaining_assistant_decisions": int(state["remaining_assistant_decisions"]),
            "remaining_tool_rounds": int(state["remaining_tool_rounds"]),
            "source_trajectory_id": str(state["source_trajectory_id"]),
            "teacher_suffix_model_visible": False,
            "quarantine_source_read": False,
            "training_allowed": False,
        }
    )
    record["reward_model"]["ground_truth"] = truth
    return record


def prepare(
    package_root: Path,
    formal: Path,
    output_root: Path,
    database_root: str,
) -> dict[str, Any]:
    safe = package_root / "safe_summary.json"
    manifest = package_root / "curriculum_manifest.safe.json"
    all_states_path = package_root / "private" / "all_states.sensitive.parquet"
    observed = {
        "safe_summary": file_sha256(safe),
        "curriculum_manifest": file_sha256(manifest),
        "all_states_parquet": file_sha256(all_states_path),
        "formal_tasks": file_sha256(formal),
    }
    expected = {
        "safe_summary": EXPECTED_SAFE_SHA256,
        "curriculum_manifest": EXPECTED_MANIFEST_SHA256,
        "all_states_parquet": EXPECTED_ALL_STATES_SHA256,
        "formal_tasks": EXPECTED_FORMAL_SHA256,
    }
    if observed != expected:
        raise ValueError(f"frozen input hash mismatch: {observed}")

    states = pq.read_table(all_states_path).to_pylist()
    if len(states) != 332 or len({str(row["prefix_state_id"]) for row in states}) != 332:
        raise ValueError("curriculum package must contain 332 unique states")
    ready = [row for row in states if row.get("training_ready") is True]
    quarantine = [row for row in states if row.get("training_ready") is not True]
    if len(ready) != 322 or len(quarantine) != 10:
        raise ValueError("curriculum ready/quarantine count mismatch")
    if len({str(row["source_trajectory_id"]) for row in states}) != 55:
        raise ValueError("curriculum must derive from exactly 55 teacher PASS trajectories")
    task_splits: dict[str, set[str]] = defaultdict(set)
    for row in states:
        task_splits[str(row["task_id"])].add(str(row["split"]))
    if any(len(values) != 1 for values in task_splits.values()):
        raise ValueError("a task crosses the train/heldout split")
    split_tasks = Counter(next(iter(values)) for values in task_splits.values())
    if split_tasks != Counter({"train": 10, "heldout": 4}):
        raise ValueError("curriculum task split must be 10 train / 4 heldout")

    truth = _truth_rows(formal)
    if set(truth) != set(task_splits):
        raise ValueError("curriculum and formal task package membership differ")
    selected, selection_audit = _trajectory_selection(states)
    selected_states = [
        row
        for row in ready
        if selected[str(row["task_id"])] == str(row["source_trajectory_id"])
    ]
    runtime = [_runtime_row(row, truth[str(row["task_id"])], database_root) for row in ready]
    selected_runtime_by_state = {
        row["extra_info"]["prefix_state_id"]: row
        for row in runtime
        if selected[row["extra_info"]["task_id"]] == row["extra_info"]["source_trajectory_id"]
    }
    frontier_ladders = [selected_runtime_by_state[str(row["prefix_state_id"])] for row in selected_states]
    frontier_endpoints = [
        row
        for row in frontier_ladders
        if row["extra_info"]["curriculum_split"] == "train"
        and row["extra_info"]["curriculum_stage"] in {"stage-full", "stage-00"}
    ]
    heldout_endpoints = [
        row
        for row in frontier_ladders
        if row["extra_info"]["curriculum_split"] == "heldout"
        and row["extra_info"]["curriculum_stage"] in {"stage-full", "stage-00"}
    ]
    if len(frontier_endpoints) != 20 or len(heldout_endpoints) != 8:
        raise ValueError("every representative trajectory must expose full and stage-00 endpoints")

    output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    private = output_root / "private"
    private.mkdir(mode=0o700)
    _write_parquet(private / "all_ready.runtime.sensitive.parquet", runtime)
    _write_parquet(private / "representative_ladders.runtime.sensitive.parquet", frontier_ladders)
    _write_parquet(private / "frontier_endpoints_10x2.runtime.sensitive.parquet", frontier_endpoints)
    _write_parquet(private / "heldout_endpoints_4x2.runtime.sensitive.parquet", heldout_endpoints)
    _write_json(private / "representative_selection.sensitive.json", {"selected": selected, "audit": selection_audit})

    stage_ready = Counter(str(row["stage"]) for row in ready)
    stage_runtime = Counter(row["extra_info"]["curriculum_stage"] for row in runtime)
    result = {
        "schema_version": CONTRACT,
        "source_hashes": observed,
        "state_rows": len(states),
        "ready_rows": len(ready),
        "quarantine_rows_rejected": len(quarantine),
        "runtime_rows": len(runtime),
        "teacher_pass_trajectory_count": len({str(row["source_trajectory_id"]) for row in states}),
        "teacher_suffix_rows_loaded_as_model_targets": 0,
        "train_task_count": split_tasks["train"],
        "heldout_task_count": split_tasks["heldout"],
        "representative_trajectory_count": len(selected),
        "representative_fully_ready_count": sum(
            bool(value["representative_fully_ready"]) for value in selection_audit.values()
        ),
        "frontier_endpoint_rows": len(frontier_endpoints),
        "heldout_endpoint_rows": len(heldout_endpoints),
        "stage_ready_counts": dict(sorted(stage_ready.items())),
        "stage_runtime_counts": dict(sorted(stage_runtime.items())),
        "future_information_leakage_count": 0,
        "hidden_reasoning_export_count": 0,
        "prefix_role_messages_flattened": False,
        "group_key_contract": "task_id+prefix_state_id+policy_version",
        "response_mask_contract": "prompt/history=0; generated assistant suffix=1; generated tool observation=0",
        "process_reward_scope": "generated_suffix_only",
        "formal_full_training_allowed": False,
        "output_hashes": {},
    }
    for path in sorted(private.glob("*.parquet")):
        result["output_hashes"][path.name] = file_sha256(path)
    _write_json(output_root / "safe_summary.json", result)
    result["safe_summary_sha256"] = file_sha256(output_root / "safe_summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--formal-tasks", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--database-root", default="/pi_sandbox")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.package_root, args.formal_tasks, args.output_root, args.database_root),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
