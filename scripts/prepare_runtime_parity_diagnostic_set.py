#!/usr/bin/env python3
"""Freeze a stratified 10-task, evaluation-only runtime parity set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.boss_pi_contract import load_boss_pi_contract


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_key(row: dict[str, Any]) -> str:
    prompt = row["prompt"]
    payload = json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_rows(rows: list[dict[str, Any]], per_type: int, seed: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for answer_type in ("numeric", "table"):
        candidates = [
            row
            for row in rows
            if row["reward_model"]["ground_truth"].get("answer_type") == answer_type
        ]
        candidates.sort(
            key=lambda row: hashlib.sha256(f"{seed}:{task_key(row)}".encode()).hexdigest()
        )
        if len(candidates) < per_type:
            raise ValueError(f"only {len(candidates)} {answer_type} rows; need {per_type}")
        selected.extend(candidates[:per_type])
    selected.sort(key=task_key)
    return selected


def prepare(source: Path, output_dir: Path, per_type: int, seed: str) -> dict[str, Any]:
    rows = pq.read_table(source).to_pylist()
    if not rows:
        raise ValueError("source dataset is empty")
    if not all((row.get("extra_info") or {}).get("alignment_reviewed") is True for row in rows):
        raise ValueError("source contains rows without alignment_reviewed=true")
    if not all([message.get("role") for message in row["prompt"]] == ["system", "user"] for row in rows):
        raise ValueError("every diagnostic prompt must contain exactly system+user")

    selected = select_rows(rows, per_type, seed)
    contract = load_boss_pi_contract()
    guidance = str((contract.get("runtime") or {}).get("guidance_prefix") or "")
    if not guidance:
        raise ValueError("boss PI guidance prefix is missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "runtime_parity_eval10.sensitive.parquet"
    tasks_path = output_dir / "pi_tasks.sensitive.jsonl"
    manifest_path = output_dir / "runtime_parity_eval10_safe_manifest.json"
    pq.write_table(pa.Table.from_pylist(selected), dataset_path)

    task_rows = []
    for row in selected:
        truth = row["reward_model"]["ground_truth"]
        environment_id = str(truth["environment_id"])
        group, version = environment_id.split("/", 1)
        visible = str(row["prompt"][-1]["content"])
        if not visible.startswith(guidance):
            raise ValueError("diagnostic prompt does not contain the frozen PI guidance prefix")
        task_rows.append(
            {
                "task_key": task_key(row),
                "group": group,
                "version": version,
                "task_family": str(truth.get("task_family") or "dwh"),
                "instruction_without_guidance": visible[len(guidance) :],
            }
        )
    with tasks_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in task_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    os.chmod(dataset_path, 0o600)
    os.chmod(tasks_path, 0o600)
    answer_types = {
        answer_type: sum(
            row["reward_model"]["ground_truth"]["answer_type"] == answer_type
            for row in selected
        )
        for answer_type in ("numeric", "table")
    }
    manifest = {
        "contract": "runtime-parity-eval10-safe-manifest-v1",
        "role": "evaluation_only_never_training",
        "source_rows": len(rows),
        "selected_tasks": len(selected),
        "samples_per_runtime_per_task": 8,
        "answer_types": answer_types,
        "task_keys": [task_key(row) for row in selected],
        "source_sha256": file_sha256(source),
        "dataset_sha256": file_sha256(dataset_path),
        "sensitive_artifacts_permissions": "0600",
        "contains_prompts_answers_sql_or_task_ids": False,
        "training_allowed": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=5)
    parser.add_argument("--seed", default="runtime-parity-eval10-20260813-v1")
    args = parser.parse_args()
    result = prepare(args.source, args.output_dir, args.per_type, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

