#!/usr/bin/env python3
"""Prepare a leak-free plan-first DWH base-vs-Step120 rollout dataset.

The source sandbox contains the hidden task manifest.  The model-visible
runtime projection deliberately contains only the database, schema dictionary,
and an empty documents directory.  The Parquet output is sensitive because it
contains prompts, hidden gold, SQL, and task identities.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.boss_pi_contract import canonical_json, contract_hashes, load_boss_pi_contract
from scripts.generate_plan_first_dwh_sandbox import verify_existing


CONTRACT = "llin-plan-first-dwh-base-step120-comparison-v1"
TOOL_NAMES = ["bash", "read", "write", "edit"]
RUNTIME_SUFFIX = "_runtime"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def stable_key(row: dict[str, Any], seed: str) -> str:
    instruction = str(row["natural_language_instruction"])
    return hashlib.sha256(f"{seed}:{canonical_hash(instruction)}".encode()).hexdigest()


def interleave(bands: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    width = max((len(rows) for rows in bands.values()), default=0)
    for index in range(width):
        for band in sorted(bands):
            if index < len(bands[band]):
                result.append(bands[band][index])
    return result


def ordered_tasks(
    tasks: Iterable[dict[str, Any]],
    *,
    seed: str,
    frozen_per_band: int = 10,
    pilot_per_band: int = 8,
) -> list[tuple[dict[str, Any], str, bool]]:
    by_band: dict[int, list[dict[str, Any]]] = {band: [] for band in range(1, 7)}
    for task in tasks:
        band = int(task["difficulty_band"])
        if band not in by_band:
            raise ValueError(f"unsupported difficulty band: {band}")
        by_band[band].append(task)
    if any(len(rows) != 50 for rows in by_band.values()):
        raise ValueError("expected exactly 50 tasks in every difficulty band")
    if not 0 < pilot_per_band <= frozen_per_band < 50:
        raise ValueError("pilot/frozen per-band counts are invalid")
    by_band = {
        band: sorted(rows, key=lambda row: stable_key(row, f"{seed}:band{band}"))
        for band, rows in by_band.items()
    }
    pilot = interleave({band: rows[:pilot_per_band] for band, rows in by_band.items()})
    frozen_tail = interleave(
        {band: rows[pilot_per_band:frozen_per_band] for band, rows in by_band.items()}
    )
    candidates = interleave({band: rows[frozen_per_band:] for band, rows in by_band.items()})
    ordered = [
        *((row, "frozen_evaluation", True) for row in pilot),
        *((row, "frozen_evaluation", False) for row in frozen_tail),
        *((row, "training_candidate", False) for row in candidates),
    ]
    identities = [canonical_hash(row["natural_language_instruction"]) for row, _, _ in ordered]
    if len(ordered) != 300 or len(set(identities)) != 300:
        raise ValueError("ordered comparison dataset must contain 300 unique tasks")
    return ordered


def _runtime_files(runtime_dir: Path) -> set[str]:
    return {
        str(path.relative_to(runtime_dir)).replace("\\", "/")
        for path in runtime_dir.rglob("*")
        if path.is_file()
    }


def create_runtime_projection(source_dir: Path, runtime_dir: Path) -> dict[str, Any]:
    source_files = {
        "logistics.sqlite": source_dir / "logistics.sqlite",
        "schema_dictionary.md": source_dir / "schema_dictionary.md",
    }
    for path in source_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_hashes = {name: file_sha256(path) for name, path in source_files.items()}
    allowed_files = set(source_files)
    if runtime_dir.exists():
        actual_files = _runtime_files(runtime_dir)
        if actual_files != allowed_files:
            raise ValueError(f"runtime projection contains unexpected files: {sorted(actual_files)}")
        for name, expected in expected_hashes.items():
            if file_sha256(runtime_dir / name) != expected:
                raise ValueError(f"runtime projection hash mismatch: {name}")
    else:
        temporary = runtime_dir.with_name(runtime_dir.name + f".tmp.{os.getpid()}")
        temporary.mkdir(parents=True, mode=0o700)
        try:
            for name, source in source_files.items():
                shutil.copy2(source, temporary / name)
                os.chmod(temporary / name, 0o600)
            (temporary / "documents").mkdir(mode=0o700)
            os.chmod(temporary, 0o700)
            temporary.replace(runtime_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    if _runtime_files(runtime_dir) != allowed_files:
        raise ValueError("runtime projection leak-prevention contract failed")
    return {
        "database_sha256": expected_hashes["logistics.sqlite"],
        "schema_sha256": expected_hashes["schema_dictionary.md"],
        "visible_files": sorted(allowed_files),
        "hidden_task_manifest_visible": False,
        "hidden_gold_or_verification_sql_visible": False,
    }


def build_record(
    task: dict[str, Any],
    *,
    index: int,
    comparison_split: str,
    pilot: bool,
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
        "task_family": "plan_first_dwh",
        "reward_contract": "pure-final-outcome-screening-v1",
        "abs_tol": 1e-3,
        "rel_tol": 1e-5,
    }
    tools_kwargs = {
        name: {"create_kwargs": {"environment_id": environment_id}}
        for name in TOOL_NAMES
    }
    return {
        "data_source": "llin_plan_first_dwh_model_comparison_v1",
        "agent_name": "pi_agent",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": guidance_prefix + instruction},
        ],
        "ability": "boss_pi_dwh",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": index,
            "split": "plan_first_dwh_model_comparison",
            "comparison_split": comparison_split,
            "pilot": pilot,
            "source_version": str(task["generation_contract"]),
            "difficulty_band": int(task["difficulty_band"]),
            "task_type": str(task["task_type"]),
            "verifier_id": verifier_id,
            "environment_id": environment_id,
            "instruction_sha256": canonical_hash(instruction),
            "gold_sha256": canonical_hash(gold["value"]),
            "mechanical_screen_passed": True,
            "plan_first_semantic_validation_passed": True,
            "training_allowed": False,
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


def prepare(
    source_dir: Path,
    runtime_root: Path,
    output_dir: Path,
    *,
    seed: str,
) -> dict[str, Any]:
    verification = verify_existing(source_dir)
    if int(verification["task_count"]) != 300 or int(verification["gold_replay_rows"]) != 300:
        raise ValueError("source sandbox did not pass the 300-row exact replay gate")
    tasks = read_jsonl(source_dir / "dwh_tasks.jsonl")
    ordered = ordered_tasks(tasks, seed=seed)
    runtime_version = source_dir.name + RUNTIME_SUFFIX
    runtime_dir = runtime_root / "sft" / runtime_version
    projection = create_runtime_projection(source_dir, runtime_dir)
    environment_id = f"sft/{runtime_version}"
    contract = load_boss_pi_contract()
    guidance_prefix = str(contract["runtime"]["guidance_prefix"])
    records = [
        build_record(
            task,
            index=index,
            comparison_split=comparison_split,
            pilot=pilot,
            environment_id=environment_id,
            system_prompt=str(contract["system_prompt"]),
            guidance_prefix=guidance_prefix,
        )
        for index, (task, comparison_split, pilot) in enumerate(ordered)
    ]
    dataset_path = output_dir / "plan_first_dwh_300.sensitive.parquet"
    write_parquet(dataset_path, records)
    band_counts = Counter(int(row["extra_info"]["difficulty_band"]) for row in records)
    pilot_band_counts = Counter(
        int(row["extra_info"]["difficulty_band"])
        for row in records
        if row["extra_info"]["pilot"]
    )
    answer_types = Counter(
        str(row["reward_model"]["ground_truth"]["answer_type"]) for row in records
    )
    manifest = {
        "contract": CONTRACT,
        "source_contract": str(json.loads((source_dir / "generation_summary.json").read_text(encoding="utf-8"))["contract"]),
        "tasks": len(records),
        "samples_per_task": 8,
        "total_trajectories_per_model": len(records) * 8,
        "models": 2,
        "total_trajectories": len(records) * 8 * 2,
        "difficulty_band_counts": {str(key): value for key, value in sorted(band_counts.items())},
        "answer_type_counts": dict(sorted(answer_types.items())),
        "frozen_evaluation_rows": 60,
        "training_candidate_rows": 240,
        "pilot_rows": 48,
        "pilot_band_counts": {str(key): value for key, value in sorted(pilot_band_counts.items())},
        "first_shard_is_stratified_pilot": True,
        "runtime_environment_id_sha256": canonical_hash(environment_id),
        "runtime_projection": projection,
        "sampling_contract": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "context_contract": {
            "max_prompt_tokens": 4096,
            "max_response_tokens": 45056,
            "max_context_tokens": 49152,
            "trajectory_timeout_seconds": 900,
        },
        "boss_contract_hashes": contract_hashes(contract),
        "source_tasks_sha256": file_sha256(source_dir / "dwh_tasks.jsonl"),
        "dataset_sha256": file_sha256(dataset_path),
        "sensitive_artifacts_permissions": "0600",
        "model_comparison_allowed": True,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_gold_sql_task_ids_or_server_paths": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plan_first_dwh_model_comparison_safe_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sandbox", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="plan-first-dwh-base-step120-comparison-20260814-v1")
    args = parser.parse_args()
    result = prepare(args.source_sandbox, args.runtime_root, args.output_dir, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
