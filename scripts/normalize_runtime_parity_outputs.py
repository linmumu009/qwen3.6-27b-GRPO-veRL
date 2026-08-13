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
_CONTEXT_OVERFLOW_RE = re.compile(
    r"context[_ ]length[_ ]exceeded|maximum context length is \d+ tokens|"
    r"exceeds (?:the )?(?:model'?s )?maximum context length|exceeds the context window",
    re.IGNORECASE,
)


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


def pi_error_summary(path: Path) -> dict[str, int]:
    assistant_errors = 0
    context_overflows = 0
    recovered_overflows = 0
    terminal_assistant_error = False
    terminal_assistant_error_is_overflow = False
    if not path.is_file():
        return {
            "assistant_error_events": 0,
            "context_overflow_events": 0,
            "recovered_context_overflows": 0,
            "fatal_api_errors": 0,
        }
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
            ):
                terminal_assistant_error = message.get("stopReason") == "error"
                terminal_assistant_error_is_overflow = False
                if terminal_assistant_error:
                    assistant_errors += 1
                    error_message = str(message.get("errorMessage") or message.get("error") or "")
                    terminal_assistant_error_is_overflow = bool(
                        _CONTEXT_OVERFLOW_RE.search(error_message)
                    )
                    if terminal_assistant_error_is_overflow:
                        context_overflows += 1
            if (
                event.get("type") == "compaction_end"
                and event.get("reason") == "overflow"
                and not bool(event.get("aborted", False))
                and bool(event.get("willRetry", False))
                and event.get("result") is not None
                and not event.get("errorMessage")
            ):
                recovered_overflows += 1
    # PI records transient failed assistant attempts before a later successful
    # retry.  Only a terminal assistant error is fatal; otherwise a recovered
    # retry would incorrectly poison an otherwise valid trajectory.
    terminal_overflow_recovered = (
        terminal_assistant_error
        and terminal_assistant_error_is_overflow
        and recovered_overflows >= context_overflows
    )
    fatal_api_errors = int(terminal_assistant_error and not terminal_overflow_recovered)
    return {
        "assistant_error_events": assistant_errors,
        "context_overflow_events": context_overflows,
        "recovered_context_overflows": min(recovered_overflows, context_overflows),
        "fatal_api_errors": fatal_api_errors,
    }


def pi_api_error_count(path: Path) -> int:
    """Return fatal API errors; recovered PI overflow retries are audit-only."""

    return pi_error_summary(path)["fatal_api_errors"]


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
        errors = pi_error_summary(path)
        api_errors = errors["fatal_api_errors"]
        assistant_turns = sum(message.get("role") == "assistant" for message in messages)
        tool_calls = sum(len(message.get("tool_calls") or []) for message in messages)
        rows.append(
            {
                "runtime": "pi_agent",
                "task_key": key,
                "sample_index": int(item["sample_index"]),
                **score,
                "completed": bool(score["has_final_answer"]),
                "timeout": status in {"timeout", "exit124"},
                "runtime_error": api_errors > 0 or status.startswith(("exit", "err:")),
                "api_error_count": api_errors,
                "assistant_error_event_count": errors["assistant_error_events"],
                "context_overflow_count": errors["context_overflow_events"],
                "recovered_context_overflow_count": errors["recovered_context_overflows"],
                "runner_status": (
                    "api_error"
                    if api_errors
                    else "recovered_overflow"
                    if errors["recovered_context_overflows"]
                    else status
                ),
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
                "runtime_error": bool(item.get("runtime_error", False) or item.get("error", False)),
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
