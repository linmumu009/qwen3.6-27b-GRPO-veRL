#!/usr/bin/env python3
"""Normalize and outcome-score PI-native or veRL parity trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

import pyarrow.parquet as pq

from llin_verl.outcome_shadow import score_final_outcome


_USER_TURN_RE = re.compile(r"(?:^|\n)user\n(.*?)(?=\nassistant\n)", re.DOTALL)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_key(prompt: list[dict[str, Any]]) -> str:
    payload = json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    truth_by_key = {}
    key_by_user = {}
    for row in pq.read_table(path).to_pylist():
        key = task_key(row["prompt"])
        if key in truth_by_key:
            raise ValueError(f"duplicate task key: {key}")
        truth_by_key[key] = row["reward_model"]["ground_truth"]
        user = str(row["prompt"][-1]["content"]).strip()
        if user in key_by_user:
            raise ValueError("duplicate visible user prompt")
        key_by_user[user] = key
    return truth_by_key, key_by_user


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def messages_to_solution(messages: list[dict[str, Any]]) -> str:
    blocks = []
    for message in messages:
        role = str(message.get("role") or "")
        if role == "system":
            continue
        if role == "assistant":
            content = ""
            reasoning = str(message.get("reasoning_content") or "")
            if reasoning:
                content += f"<think>{reasoning}</think>\n"
            content += str(message.get("content") or "")
            for call in message.get("tool_calls") or []:
                content += "\n<tool_call>" + json.dumps(call, ensure_ascii=False) + "</tool_call>"
            blocks.append("assistant\n" + content)
        elif role == "tool":
            blocks.append("user\n<tool_response>" + str(message.get("content") or "") + "</tool_response>")
        elif role == "user":
            blocks.append("user\n" + str(message.get("content") or ""))
    return "\n".join(blocks)


def pi_api_error_count(path: Path) -> int:
    count = 0
    if not path.is_file():
        return count
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = event.get("message") or {}
            if (
                event.get("type") == "message_end"
                and message.get("role") == "assistant"
                and message.get("stopReason") == "error"
            ):
                count += 1
    return count


def normalize_pi(
    dataset: Path,
    manifest: Path,
    trajectory_dir: Path,
    converter: Path,
) -> list[dict[str, Any]]:
    truth_by_key, _ = dataset_index(dataset)
    parser = load_module(converter, "boss_pi_to_openai_runtime_parity")
    rows = []
    for item in read_jsonl(manifest):
        key = str(item["task_key"])
        if key not in truth_by_key:
            raise ValueError(f"PI task not in diagnostic dataset: {key}")
        status = str(item.get("status") or "")
        path = trajectory_dir / str(item["trajectory_file"])
        parsed = parser.parse_trajectory(str(path), "") if path.is_file() else None
        messages = (parsed or {}).get("messages") or []
        solution = messages_to_solution(messages)
        score = score_final_outcome(solution, truth_by_key[key])
        api_errors = pi_api_error_count(path)
        assistant_turns = sum(message.get("role") == "assistant" for message in messages)
        tool_calls = sum(len(message.get("tool_calls") or []) for message in messages)
        rows.append(
            {
                "runtime": "pi_agent",
                "task_key": key,
                "sample_index": int(item["sample_index"]),
                **score,
                "completed": bool(score["has_final_answer"]),
                "timeout": status == "timeout",
                "runtime_error": api_errors > 0 or status.startswith(("exit", "err:")),
                "api_error_count": api_errors,
                "runner_status": "api_error" if api_errors else status,
                "assistant_turns": assistant_turns,
                "tool_call_count": tool_calls,
                "duration_seconds": float(item.get("duration_seconds") or 0),
            }
        )
    return rows


def prompt_from_verl_input(text: str) -> str:
    turns = _USER_TURN_RE.findall(text or "")
    if not turns:
        raise ValueError("veRL validation input has no user turn")
    return turns[-1].strip()


def normalize_verl(dataset: Path, validation: Path) -> list[dict[str, Any]]:
    truth_by_key, key_by_user = dataset_index(dataset)
    counters: dict[str, int] = defaultdict(int)
    rows = []
    for item in read_jsonl(validation):
        user = prompt_from_verl_input(str(item.get("input") or ""))
        key = key_by_user.get(user)
        if key is None:
            raise ValueError("veRL validation prompt is outside the diagnostic dataset")
        sample_index = counters[key]
        counters[key] += 1
        solution = str(item.get("output") or "")
        score = score_final_outcome(solution, truth_by_key[key])
        rows.append(
            {
                "runtime": "verl_rollout",
                "task_key": key,
                "sample_index": sample_index,
                **score,
                "completed": bool(score["has_final_answer"]),
                "timeout": bool(item.get("timeout", False)),
                "runtime_error": bool(item.get("error", False)),
                "runner_status": "ok",
                "assistant_turns": int(item.get("num_turns") or item.get("assistant_turns") or 0),
                "tool_call_count": int(item.get("pi_bash_command_count") or item.get("bash_command_count") or 0),
                "duration_seconds": float(item.get("latency") or item.get("duration_seconds") or 0),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="runtime", required=True)
    pi = subparsers.add_parser("pi")
    pi.add_argument("--dataset", type=Path, required=True)
    pi.add_argument("--manifest", type=Path, required=True)
    pi.add_argument("--trajectory-dir", type=Path, required=True)
    pi.add_argument("--converter", type=Path, required=True)
    pi.add_argument("--output", type=Path, required=True)
    verl = subparsers.add_parser("verl")
    verl.add_argument("--dataset", type=Path, required=True)
    verl.add_argument("--validation", type=Path, required=True)
    verl.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runtime == "pi":
        rows = normalize_pi(args.dataset, args.manifest, args.trajectory_dir, args.converter)
    else:
        rows = normalize_verl(args.dataset, args.validation)
    write_rows(args.output, rows)
    print(json.dumps({"runtime": args.runtime, "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
