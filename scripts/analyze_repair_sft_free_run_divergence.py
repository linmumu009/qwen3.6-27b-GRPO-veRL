#!/usr/bin/env python3
"""Locate the first behavioral divergence from the 16 repair teacher trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd


def normalize_container(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [normalize_container(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: normalize_container(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_container(item) for item in value]
    return value


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def sql_from_command(command: str) -> str | None:
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    for part in reversed(parts):
        normalized = normalize_text(part).rstrip(";")
        if normalized.upper().startswith(("SELECT ", "WITH ")):
            return normalized
    return None


def arguments_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def bash_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if function.get("name") != "bash":
                continue
            arguments = arguments_mapping(function.get("arguments"))
            command = str(arguments.get("command") or "")
            calls.append(
                {
                    "message_index": message_index,
                    "call_id": str(tool_call.get("id") or ""),
                    "command": command,
                    "sql": sql_from_command(command),
                }
            )
    return calls


def tool_outputs(messages: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(message.get("tool_call_id") or ""): normalize_text(message.get("content"))
        for message in messages
        if message.get("role") == "tool"
    }


def next_assistant(messages: list[dict[str, Any]], after_index: int) -> dict[str, Any] | None:
    return next(
        (
            message
            for index, message in enumerate(messages)
            if index > after_index and message.get("role") == "assistant"
        ),
        None,
    )


def analyze_task(teacher_messages: list[dict], rollout_messages: list[dict]) -> dict[str, Any]:
    teacher_call = teacher_messages[2]["tool_calls"][0]
    teacher_command = teacher_call["function"]["arguments"]["command"]
    teacher_sql = sql_from_command(teacher_command)
    teacher_output = normalize_text(teacher_messages[3]["content"])
    teacher_final = normalize_text(teacher_messages[4]["content"])
    calls = bash_calls(rollout_messages)
    outputs = tool_outputs(rollout_messages)
    target_indices = [
        index for index, call in enumerate(calls) if teacher_sql is not None and call["sql"] == teacher_sql
    ]
    first_target_index = target_indices[0] if target_indices else None
    first_target = calls[first_target_index] if first_target_index is not None else None
    next_after_target = (
        next_assistant(rollout_messages, first_target["message_index"]) if first_target else None
    )
    target_output_observed = bool(
        first_target and outputs.get(first_target["call_id"]) == teacher_output
    )
    immediate_final = bool(next_after_target and not (next_after_target.get("tool_calls") or []))
    final_messages = [
        message for message in rollout_messages if message.get("role") == "assistant" and not (message.get("tool_calls") or [])
    ]
    final_text = normalize_text(final_messages[-1]["content"]) if final_messages else ""
    teacher_final_contained = bool(teacher_final and teacher_final in final_text)

    if not calls:
        bucket = "no_bash_call"
    elif calls[0]["sql"] != teacher_sql:
        bucket = "first_sql_diverged"
    elif not target_output_observed:
        bucket = "teacher_sql_but_tool_output_mismatch"
    elif not immediate_final:
        bucket = "teacher_evidence_then_continued"
    elif not teacher_final_contained:
        bucket = "immediate_final_but_teacher_answer_missing"
    else:
        bucket = "teacher_path_matched"

    return {
        "bash_command_count": len(calls),
        "first_bash_sql_matches_teacher": bool(calls and calls[0]["sql"] == teacher_sql),
        "teacher_sql_seen_anywhere": first_target_index is not None,
        "teacher_sql_first_seen_at": first_target_index,
        "teacher_tool_output_observed": target_output_observed,
        "continued_bash_after_teacher_sql": bool(
            first_target_index is not None and first_target_index + 1 < len(calls)
        ),
        "immediate_final_after_teacher_sql": immediate_final,
        "has_final_answer": bool(final_messages),
        "teacher_final_contained": teacher_final_contained,
        "first_divergence_bucket": bucket,
    }


def read_openai(path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task_id = str(row["task_id"])
            if task_id in rows:
                raise ValueError(f"duplicate rollout task_id: {task_id}")
            rows[task_id] = row["messages"]
    return rows


def summarize(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    boolean_fields = (
        "first_bash_sql_matches_teacher",
        "teacher_sql_seen_anywhere",
        "teacher_tool_output_observed",
        "continued_bash_after_teacher_sql",
        "immediate_final_after_teacher_sql",
        "has_final_answer",
        "teacher_final_contained",
    )
    return {
        "task_count": len(per_task),
        "bash_command_count_mean": mean(row["bash_command_count"] for row in per_task),
        **{f"{field}_count": sum(bool(row[field]) for row in per_task) for field in boolean_fields},
        "first_divergence_buckets": dict(
            sorted(Counter(row["first_divergence_bucket"] for row in per_task).items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-parquet", type=Path, required=True)
    parser.add_argument("--step120-openai", type=Path, required=True)
    parser.add_argument("--post-sft-openai", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    teacher_frame = pd.read_parquet(args.teacher_parquet)
    teachers = {
        str(row["task_id"]): normalize_container(row["messages"])
        for _, row in teacher_frame.iterrows()
    }
    rollouts = {
        "step120": read_openai(args.step120_openai),
        "post_sft": read_openai(args.post_sft_openai),
    }
    task_ids = list(teachers)
    if set(task_ids) != set(rollouts["step120"]) or set(task_ids) != set(rollouts["post_sft"]):
        raise ValueError("teacher and rollout task IDs differ")

    by_model: dict[str, Any] = {}
    for label, messages_by_task in rollouts.items():
        per_task = [
            {"task_id": task_id, **analyze_task(teachers[task_id], messages_by_task[task_id])}
            for task_id in task_ids
        ]
        by_model[label] = {"summary": summarize(per_task), "per_task": per_task}

    result = {
        "contract": "repair-sft-free-run-first-divergence-v1",
        "task_count": len(task_ids),
        "task_ids_identical": True,
        "models": by_model,
        "promotion_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({label: data["summary"] for label, data in by_model.items()}, indent=2))


if __name__ == "__main__":
    main()
