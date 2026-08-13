#!/usr/bin/env python3
"""Audit 10x8 trajectory uniqueness without exporting sensitive content."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def summarize(groups: dict[str, list[str]], expected_tasks: int, expected_n: int) -> dict[str, Any]:
    unique_counts = [len(set(values)) for values in groups.values()]
    pair_total = 0
    duplicate_pairs = 0
    for values in groups.values():
        for left, right in itertools.combinations(values, 2):
            pair_total += 1
            duplicate_pairs += left == right
    rows = sum(len(values) for values in groups.values())
    return {
        "rows": rows,
        "tasks": len(groups),
        "group_size_histogram": dict(sorted(Counter(map(len, groups.values())).items())),
        "unique_trajectories_per_task_histogram": dict(sorted(Counter(unique_counts).items())),
        "all_samples_identical_groups": sum(count == 1 for count in unique_counts),
        "all_samples_unique_groups": sum(count == expected_n for count in unique_counts),
        "duplicate_pair_fraction": duplicate_pairs / pair_total if pair_total else None,
        "complete_shape": len(groups) == expected_tasks and rows == expected_tasks * expected_n,
        "every_group_all_samples_unique": (
            len(groups) == expected_tasks and all(count == expected_n for count in unique_counts)
        ),
    }


def audit_verl(path: Path, expected_tasks: int, expected_n: int) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(path):
        prompt_key = hashlib.sha256(str(row.get("input") or "").encode()).hexdigest()
        groups[prompt_key].append(str(row.get("output") or ""))
    return summarize(groups, expected_tasks, expected_n)


def assistant_trajectory(messages: list[dict[str, Any]]) -> str:
    assistant = [
        {
            "content": message.get("content"),
            "reasoning_content": message.get("reasoning_content"),
            "tool_calls": message.get("tool_calls"),
        }
        for message in messages
        if message.get("role") == "assistant"
    ]
    return json.dumps(assistant, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def audit_pi(
    manifest: Path,
    trajectory_dir: Path,
    converter: Path,
    expected_tasks: int,
    expected_n: int,
) -> dict[str, Any]:
    parser = load_module(converter, "boss_pi_to_openai_sampling_audit")
    groups: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(manifest):
        path = trajectory_dir / str(row["trajectory_file"])
        parsed = parser.parse_trajectory(str(path), "") if path.is_file() else None
        groups[str(row["task_key"])].append(
            assistant_trajectory((parsed or {}).get("messages") or [])
        )
    return summarize(groups, expected_tasks, expected_n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="runtime", required=True)
    for name in ("pi", "verl"):
        child = subparsers.add_parser(name)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--expected-tasks", type=int, default=10)
        child.add_argument("--expected-n", type=int, default=8)
    pi = subparsers.choices["pi"]
    pi.add_argument("--manifest", type=Path, required=True)
    pi.add_argument("--trajectory-dir", type=Path, required=True)
    pi.add_argument("--converter", type=Path, required=True)
    verl = subparsers.choices["verl"]
    verl.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()

    if args.runtime == "pi":
        arm = audit_pi(
            args.manifest,
            args.trajectory_dir,
            args.converter,
            args.expected_tasks,
            args.expected_n,
        )
    else:
        arm = audit_verl(args.validation, args.expected_tasks, args.expected_n)
    result = {
        "contract": "runtime-parity-sampling-uniqueness-audit-v1",
        "runtime": args.runtime,
        "contains_prompts_answers_sql_task_ids_hashes_or_trajectories": False,
        **arm,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(args.output, 0o600)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
