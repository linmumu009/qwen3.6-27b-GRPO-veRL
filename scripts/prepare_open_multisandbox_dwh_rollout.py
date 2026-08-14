#!/usr/bin/env python3
"""Prepare a leak-free 500-task open-DWH Step120 rollout dataset.

The source sandbox and generated Parquet files are sensitive.  The model-visible
runtime projection contains only the SQLite database, schema dictionary, and an
empty documents directory.  Tasks remain disabled for training until rollout
screening and explicit semantic review are complete.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.boss_pi_contract import canonical_json, contract_hashes, load_boss_pi_contract
from scripts.audit_open_multisandbox_dwh import audit_sandbox
from scripts.prepare_plan_first_dwh_model_comparison import (
    canonical_hash,
    create_runtime_projection,
    file_sha256,
    read_jsonl,
)


CONTRACT = "llin-open-multisandbox-dwh-step120-rollout-v1"
TOOL_NAMES = ["bash", "read", "write", "edit"]
RUNTIME_SUFFIX = "_runtime"


def stable_key(task: dict[str, Any], seed: str) -> str:
    instruction_hash = canonical_hash(task["natural_language_instruction"])
    return hashlib.sha256(f"{seed}:{instruction_hash}".encode()).hexdigest()


def ordered_tasks(tasks: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    by_level = {level: [] for level in range(1, 6)}
    for task in tasks:
        level = int(task["difficulty_level"])
        if level not in by_level:
            raise ValueError(f"unsupported difficulty level: {level}")
        by_level[level].append(task)
    if any(len(rows) != 100 for rows in by_level.values()):
        raise ValueError("expected exactly 100 tasks in every difficulty level")
    by_level = {
        level: sorted(rows, key=lambda row: stable_key(row, f"{seed}:level{level}"))
        for level, rows in by_level.items()
    }
    ordered = [by_level[level][offset] for offset in range(100) for level in range(1, 6)]
    identities = [canonical_hash(row["natural_language_instruction"]) for row in ordered]
    if len(ordered) != 500 or len(set(identities)) != 500:
        raise ValueError("ordered rollout dataset must contain 500 unique tasks")
    return ordered


def build_record(
    task: dict[str, Any],
    *,
    global_index: int,
    environment_id: str,
    system_prompt: str,
    guidance_prefix: str,
) -> dict[str, Any]:
    instruction = str(task["natural_language_instruction"])
    gold = task["gold_answer"]
    verifier_id = f"{environment_id}:{task['task_id']}"
    ground_truth = {
        "verifier_id": verifier_id,
        "task_id": str(task["task_id"]),
        "environment_id": environment_id,
        "answer_type": str(gold["answer_type"]),
        "expected_value_json": canonical_json(gold["value"]),
        "verification_sql": str(gold["verification_sql"]),
        "required_tables": sorted({str(value).casefold() for value in task["expected_tables"]}),
        "must_use_fields": [],
        "task_family": "open_plan_first_dwh",
        "reward_contract": "pure-final-outcome-screening-v1",
        "abs_tol": 1e-3,
        "rel_tol": 1e-5,
    }
    tools_kwargs = {
        name: {"create_kwargs": {"environment_id": environment_id}}
        for name in TOOL_NAMES
    }
    return {
        "data_source": "llin_open_multisandbox_dwh_step120_rollout_v1",
        "agent_name": "pi_agent",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": guidance_prefix + instruction},
        ],
        "ability": "boss_pi_dwh",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": global_index,
            "global_index": global_index,
            "split": "open_multisandbox_dwh_rollout_screening",
            "source_version": str(task["source_sandbox_version"]),
            "difficulty_level": int(task["difficulty_level"]),
            "difficulty_band": int(task["difficulty_level"]),
            "task_type": str(task["task_type"]),
            "verifier_id": verifier_id,
            "environment_id": environment_id,
            "instruction_sha256": canonical_hash(instruction),
            "gold_sha256": canonical_hash(gold["value"]),
            "mechanical_screen_passed": True,
            "api_semantic_validation_passed": True,
            "explicit_semantic_reviewed": False,
            "training_allowed": False,
            "promotion_allowed": False,
            "response_messages_in_grpo_input": 0,
            "need_tools_kwargs": True,
            "tool_selection": TOOL_NAMES,
            "tools_kwargs": tools_kwargs,
        },
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def prepare(source_dir: Path, runtime_root: Path, output_dir: Path, *, seed: str) -> dict[str, Any]:
    audit = audit_sandbox(source_dir)
    if audit["task_count"] != 500 or audit["sql_gold_replay_passed_rows"] != 500:
        raise ValueError("source sandbox did not pass the 500-row exact replay gate")
    ordered = ordered_tasks(read_jsonl(source_dir / "dwh_tasks.jsonl"), seed)
    runtime_version = source_dir.name + RUNTIME_SUFFIX
    runtime_dir = runtime_root / "sft" / runtime_version
    projection = create_runtime_projection(source_dir, runtime_dir)
    environment_id = f"sft/{runtime_version}"
    boss_contract = load_boss_pi_contract()
    records = [
        build_record(
            task,
            global_index=index,
            environment_id=environment_id,
            system_prompt=str(boss_contract["system_prompt"]),
            guidance_prefix=str(boss_contract["runtime"]["guidance_prefix"]),
        )
        for index, task in enumerate(ordered)
    ]
    partitions = {"full500": records, "m05": records[:250], "m06": records[250:]}
    paths: dict[str, Path] = {}
    for label, rows in partitions.items():
        path = output_dir / f"open_multisandbox_dwh_{label}.sensitive.parquet"
        write_parquet(path, rows)
        paths[label] = path
    partition_levels = {
        label: dict(sorted(Counter(str(row["extra_info"]["difficulty_level"]) for row in rows).items()))
        for label, rows in partitions.items()
    }
    manifest = {
        "contract": CONTRACT,
        "source_contract": str(json.loads((source_dir / "generation_summary.json").read_text(encoding="utf-8"))["contract"]),
        "tasks": 500,
        "samples_per_task": 8,
        "total_trajectories": 4000,
        "partition_tasks": {"m05": 250, "m06": 250},
        "difficulty_level_counts": partition_levels["full500"],
        "partition_difficulty_level_counts": {"m05": partition_levels["m05"], "m06": partition_levels["m06"]},
        "runtime_environment_id_sha256": canonical_hash(environment_id),
        "runtime_projection": projection,
        "sampling_contract": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "context_contract": {
            "max_prompt_tokens": 4096,
            "max_response_tokens": 90112,
            "max_context_tokens": 94208,
            "trajectory_timeout_seconds": 1800,
        },
        "boss_contract_hashes": contract_hashes(boss_contract),
        "source_tasks_sha256": file_sha256(source_dir / "dwh_tasks.jsonl"),
        "dataset_sha256": {label: file_sha256(path) for label, path in paths.items()},
        "sensitive_artifacts_permissions": "0600",
        "contains_prompts_gold_sql_task_ids_or_server_paths": False,
        "rollout_screening_allowed": True,
        "explicit_semantic_review_completed": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "open_multisandbox_dwh_rollout_safe_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sandbox", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="open-multisandbox-dwh-step120-rollout-20260815-v1")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_sandbox, args.runtime_root, args.output_dir, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
