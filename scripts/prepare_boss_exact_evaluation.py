#!/usr/bin/env python3
"""Prepare veRL validation trajectories for the boss's original evaluator.

veRL stores a decoded Qwen multi-turn response as one string containing
``<tool_call>`` and ``<tool_response>`` blocks.  The boss's authoritative
``reward_judge.py --input`` entry point expects OpenAI-style messages instead.
This adapter performs only that representation change; tasks and gold labels
are copied byte-for-byte (at the JSON object level) from the original boss
manifest and joined strictly by ``task_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
USER_TOOL_RESPONSE_PREFIX = "\nuser\n"
ASSISTANT_PREFIX = "\nassistant\n"
FUNCTION_RE = re.compile(r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$", re.DOTALL)
PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
TRUNCATED_FUNCTION_RE = re.compile(
    r"^\s*<tool_call>\s*<function=([^>\s]+)>\s*(.*)$", re.DOTALL
)
TRUNCATED_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>\s*", re.DOTALL)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_tool_call(block: str, call_number: int) -> dict[str, Any]:
    match = FUNCTION_RE.fullmatch(block)
    if not match:
        raise ValueError(f"tool call {call_number}: malformed function block")
    name, parameter_blob = match.groups()
    parameters = PARAMETER_RE.findall(parameter_blob)
    if not parameters:
        raise ValueError(f"tool call {call_number}: no parameters")
    residue = PARAMETER_RE.sub("", parameter_blob).strip()
    if residue:
        raise ValueError(f"tool call {call_number}: unparsed parameter content: {residue[:80]!r}")
    arguments: dict[str, str] = {}
    for key, value in parameters:
        if key in arguments:
            raise ValueError(f"tool call {call_number}: duplicate parameter {key!r}")
        arguments[key] = value.strip()
    return {
        "id": f"call_{call_number:04d}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


def _parse_truncated_terminal_tool_call(block: str, call_number: int) -> dict[str, Any]:
    """Preserve one token-truncated final tool call as an unanswered call.

    The call remains incomplete because no tool response is synthesized.  This
    lets the original evaluator apply its normal completion gate without
    treating the preceding reasoning text as a fabricated final answer.
    """
    match = TRUNCATED_FUNCTION_RE.fullmatch(block)
    if not match or TOOL_CALL_END in block:
        raise ValueError(f"tool call {call_number}: unsupported truncated terminal block")
    name, parameter_blob = match.groups()
    starts = list(TRUNCATED_PARAMETER_RE.finditer(parameter_blob))
    if not starts:
        raise ValueError(f"tool call {call_number}: truncated block has no parameters")
    arguments: dict[str, str] = {}
    for index, parameter in enumerate(starts):
        key = parameter.group(1)
        if key in arguments:
            raise ValueError(f"tool call {call_number}: duplicate parameter {key!r}")
        value_end = starts[index + 1].start() if index + 1 < len(starts) else len(parameter_blob)
        value = parameter_blob[parameter.end() : value_end]
        value = re.sub(r"\s*</parameter>\s*(?:</function>)?\s*$", "", value).strip()
        arguments[key] = value
    return {
        "id": f"call_{call_number:04d}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


def qwen_output_to_openai_messages(output: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert one decoded Qwen response into OpenAI assistant/tool messages.

    Parsing is deliberately fail-closed.  Every tool call must have exactly one
    following tool response and every non-terminal response must return control
    to the assistant.  This prevents a plausible-looking but lossy transcript
    from being sent to the authoritative scorer.
    """
    messages: list[dict[str, Any]] = []
    cursor = 0
    call_number = 0
    response_number = 0
    missing_tool_responses = 0
    while True:
        user_start = output.find(USER_TOOL_RESPONSE_PREFIX + "<tool_response>", cursor)
        if user_start < 0:
            break
        assistant_blob = output[cursor:user_start]
        raw_call_blocks = TOOL_CALL_RE.findall(assistant_blob)
        if not raw_call_blocks:
            raise ValueError("tool-response group has no preceding tool calls")
        assistant_content = TOOL_CALL_RE.sub("", assistant_blob).strip()
        calls: list[dict[str, Any]] = []
        for block in raw_call_blocks:
            call_number += 1
            calls.append(_parse_tool_call(block, call_number))

        response_group_start = user_start + len(USER_TOOL_RESPONSE_PREFIX)
        group_suffix = "</tool_response>" + ASSISTANT_PREFIX
        response_group_end = output.find(group_suffix, response_group_start)
        if response_group_end < 0:
            raise ValueError(
                f"tool calls ending at {call_number}: missing response/assistant delimiter"
            )
        response_blob = output[
            response_group_start : response_group_end + len("</tool_response>")
        ]
        raw_response_blocks = TOOL_RESPONSE_RE.findall(response_blob)
        residue = TOOL_RESPONSE_RE.sub("", response_blob).strip()
        if residue:
            raise ValueError(f"unparsed tool-response content: {residue[:80]!r}")
        if len(raw_response_blocks) > len(calls):
            raise ValueError(
                f"parallel tool group mismatch: calls={len(calls)}, "
                f"responses={len(raw_response_blocks)}"
            )
        missing_tool_responses += len(calls) - len(raw_response_blocks)

        messages.append(
            {"role": "assistant", "content": assistant_content, "tool_calls": calls}
        )
        for call, response in zip(calls, raw_response_blocks):
            response_number += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["function"]["name"],
                    "content": response,
                }
            )
        cursor = response_group_end + len(group_suffix)

    terminal = output[cursor:]
    terminal_call_blocks = TOOL_CALL_RE.findall(terminal)
    terminal_open_calls = terminal.count(TOOL_CALL_START)
    truncated_terminal_calls = 0
    terminal_nonempty = False
    if terminal_call_blocks or terminal_open_calls:
        terminal_calls: list[dict[str, Any]] = []
        for block in terminal_call_blocks:
            call_number += 1
            terminal_calls.append(_parse_tool_call(block, call_number))
        if terminal_open_calls != len(terminal_call_blocks):
            if terminal_open_calls != len(terminal_call_blocks) + 1:
                raise ValueError(
                    "terminal assistant contains more than one truncated tool call"
                )
            truncated_start = terminal.rfind(TOOL_CALL_START)
            call_number += 1
            terminal_calls.append(
                _parse_truncated_terminal_tool_call(terminal[truncated_start:], call_number)
            )
            truncated_terminal_calls = 1
            terminal_content = TOOL_CALL_RE.sub("", terminal[:truncated_start]).strip()
        else:
            terminal_content = TOOL_CALL_RE.sub("", terminal).strip()
        missing_tool_responses += len(terminal_calls)
        messages.append(
            {
                "role": "assistant",
                "content": terminal_content,
                "tool_calls": terminal_calls,
            }
        )
    else:
        terminal_nonempty = bool(terminal.strip())
        if terminal_nonempty:
            messages.append({"role": "assistant", "content": terminal.strip()})

    raw_calls = output.count(TOOL_CALL_START)
    raw_responses = output.count("<tool_response>")
    if call_number != raw_calls or response_number != raw_responses:
        raise ValueError(
            f"lossy conversion: parsed_calls={call_number}, raw_calls={raw_calls}, "
            f"parsed_responses={response_number}, raw_responses={raw_responses}"
        )
    audit = {
        "tool_calls": call_number,
        "tool_responses": raw_responses,
        "missing_tool_responses": missing_tool_responses,
        "terminal_assistant": terminal_nonempty,
    }
    if truncated_terminal_calls:
        audit["truncated_terminal_tool_calls"] = truncated_terminal_calls
    return messages, audit


def _index_unique(rows: list[dict[str, Any]], label: str, key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "").strip()
        if not value:
            raise ValueError(f"{label}: missing {key}")
        if value in result:
            raise ValueError(f"{label}: duplicate {key}={value}")
        result[value] = row
    return result


def load_parquet_prompts(path: Path) -> dict[str, list[dict[str, Any]]]:
    import pandas as pd

    rows = pd.read_parquet(path).to_dict(orient="records")
    prompts: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        reward_model = row.get("reward_model") or {}
        ground_truth = reward_model.get("ground_truth") or {}
        task_id = str(ground_truth.get("task_id") or "").strip()
        if not task_id or task_id in prompts:
            raise ValueError(f"parquet: missing or duplicate task_id={task_id!r}")
        prompt = row.get("prompt")
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if not isinstance(prompt, list) or [m.get("role") for m in prompt] != ["system", "user"]:
            raise ValueError(f"parquet: task {task_id} does not have exactly system+user prompt")
        prompts[task_id] = [dict(message) for message in prompt]
    return prompts


def prepare(
    validation_path: Path,
    parquet_path: Path,
    task_manifest_path: Path,
    trajectory_output: Path,
    manifest_output: Path,
) -> dict[str, Any]:
    validation = read_jsonl(validation_path)
    tasks = _index_unique(read_jsonl(task_manifest_path), "task manifest", "task_id")
    prompts = load_parquet_prompts(parquet_path)

    trajectory_rows: list[dict[str, Any]] = []
    selected_tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    tool_calls = 0
    missing_tool_responses = 0
    terminal_answers = 0
    truncated_terminal_tool_calls = 0
    for row in validation:
        ground_truth = row.get("gts") or {}
        task_id = str(ground_truth.get("task_id") or "").strip()
        if not task_id or task_id in seen:
            raise ValueError(f"validation: missing or duplicate task_id={task_id!r}")
        if task_id not in tasks or task_id not in prompts:
            raise ValueError(f"validation: task_id not found in exact source assets: {task_id}")
        seen.add(task_id)
        output = row.get("output")
        if not isinstance(output, str):
            raise ValueError(f"validation: task {task_id} has non-string output")
        try:
            response_messages, audit = qwen_output_to_openai_messages(output)
        except ValueError as exc:
            raise ValueError(f"validation task {task_id}: {exc}") from exc
        trajectory_rows.append({"task_id": task_id, "messages": prompts[task_id] + response_messages})
        selected_tasks.append(tasks[task_id])
        tool_calls += int(audit["tool_calls"])
        missing_tool_responses += int(audit["missing_tool_responses"])
        terminal_answers += int(audit["terminal_assistant"])
        truncated_terminal_tool_calls += int(audit.get("truncated_terminal_tool_calls", 0))

    if set(prompts) != seen:
        raise ValueError(
            f"validation/parquet task mismatch: validation={len(seen)}, parquet={len(prompts)}"
        )
    write_jsonl(trajectory_output, trajectory_rows)
    write_jsonl(manifest_output, selected_tasks)
    return {
        "validation_rows": len(validation),
        "unique_task_ids": len(seen),
        "tool_calls": tool_calls,
        "missing_tool_responses": missing_tool_responses,
        "terminal_assistant_answers": terminal_answers,
        "truncated_terminal_tool_calls": truncated_terminal_tool_calls,
        "all_terminal": terminal_answers == len(validation),
        "trajectory_output": str(trajectory_output),
        "trajectory_sha256": sha256(trajectory_output),
        "manifest_output": str(manifest_output),
        "manifest_sha256": sha256(manifest_output),
        "source_validation_sha256": sha256(validation_path),
        "source_parquet_sha256": sha256(parquet_path),
        "source_task_manifest_sha256": sha256(task_manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    summary = prepare(
        args.validation,
        args.parquet,
        args.task_manifest,
        args.trajectory_output,
        args.manifest_output,
    )
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
