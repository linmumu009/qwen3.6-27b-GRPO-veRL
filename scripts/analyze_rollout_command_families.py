#!/usr/bin/env python3
"""Summarize rollout tool and shell-command families without leaking payloads."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from scripts.analyze_repair_sft_free_run_divergence import (
    arguments_mapping,
    read_openai,
    sql_from_command,
)


def command_family(command: str) -> str:
    value = command.strip()
    if not value:
        return "empty"
    if sql_from_command(value) is not None:
        return "recognized_readonly_sqlite"
    lowered = value.lower()
    if "sqlite" in lowered:
        if re.search(r"(?:^|[\s'\"])(?:\.tables|\.databases|\.dbinfo)\b", lowered):
            return "sqlite_schema_catalog"
        if re.search(r"(?:^|[\s'\"])\.schema\b", lowered):
            return "sqlite_schema_definition"
        if re.search(r"\bpragma\b", lowered):
            return "sqlite_schema_pragma"
        if re.search(r"\b(?:select|with)\b", lowered):
            return "sqlite_readonly_unparsed"
        if re.search(
            r"\b(?:insert|update|delete|drop|alter|create|replace|vacuum|attach|detach)\b",
            lowered,
        ):
            return "sqlite_mutating_or_unsafe"
        return "sqlite_path_or_cli_only"
    match = re.match(r"(?:cd\s+\S+\s*(?:&&|;)\s*)?([^\s;&|]+)", value)
    executable = (match.group(1) if match else "").lower().rsplit("/", 1)[-1]
    if executable.startswith("python"):
        return "python"
    if executable in {"ls", "find", "cat", "sed", "head", "tail", "grep", "rg", "pwd", "wc"}:
        return executable
    return "other"


def summarize(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    tool_names: Counter[str] = Counter()
    command_families: Counter[str] = Counter()
    command_hashes: Counter[str] = Counter()
    observed_calls = 0
    unobserved_calls = 0
    rows_with_any_bash = 0
    rows_with_any_sqlite = 0
    rows_with_schema_discovery_sqlite = 0
    rows_with_recognized_sqlite = 0
    rows_with_no_tool_calls = 0
    rows_with_unobserved_tool_calls = 0

    for messages in rows.values():
        row_calls: list[dict[str, Any]] = []
        row_observed = {
            str(message.get("tool_call_id") or "")
            for message in messages
            if message.get("role") == "tool"
        }
        row_has_bash = False
        row_has_any_sqlite = False
        row_has_schema_sqlite = False
        row_has_sqlite = False
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                row_calls.append(tool_call)
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "missing")
                tool_names[name] += 1
                if name != "bash":
                    continue
                row_has_bash = True
                arguments = arguments_mapping(function.get("arguments"))
                command = str(arguments.get("command") or "")
                family = command_family(command)
                command_families[family] += 1
                command_hashes[hashlib.sha256(command.encode("utf-8")).hexdigest()] += 1
                row_has_any_sqlite = row_has_any_sqlite or family == (
                    "recognized_readonly_sqlite"
                ) or family.startswith("sqlite_")
                row_has_schema_sqlite = row_has_schema_sqlite or family in {
                    "sqlite_schema_catalog",
                    "sqlite_schema_definition",
                    "sqlite_schema_pragma",
                }
                row_has_sqlite = row_has_sqlite or family == "recognized_readonly_sqlite"
        rows_with_any_bash += int(row_has_bash)
        rows_with_any_sqlite += int(row_has_any_sqlite)
        rows_with_schema_discovery_sqlite += int(row_has_schema_sqlite)
        rows_with_recognized_sqlite += int(row_has_sqlite)
        rows_with_no_tool_calls += int(not row_calls)
        rows_with_unobserved_tool_calls += int(
            any(str(call.get("id") or "") not in row_observed for call in row_calls)
        )
        observed_calls += sum(
            str(call.get("id") or "") in row_observed for call in row_calls
        )
        unobserved_calls += sum(
            str(call.get("id") or "") not in row_observed for call in row_calls
        )

    total_calls = sum(tool_names.values())
    bash_calls = sum(command_families.values())
    duplicate_bash_calls = sum(count - 1 for count in command_hashes.values())
    return {
        "rows": len(rows),
        "tool_calls": total_calls,
        "tool_call_name_counts": dict(sorted(tool_names.items())),
        "bash_calls": bash_calls,
        "bash_command_family_counts": dict(sorted(command_families.items())),
        "rows_with_any_bash": rows_with_any_bash,
        "rows_with_any_sqlite": rows_with_any_sqlite,
        "rows_with_schema_discovery_sqlite": rows_with_schema_discovery_sqlite,
        "rows_with_recognized_readonly_sqlite": rows_with_recognized_sqlite,
        "rows_with_no_tool_calls": rows_with_no_tool_calls,
        "rows_with_unobserved_tool_calls": rows_with_unobserved_tool_calls,
        "observed_tool_calls": observed_calls,
        "unobserved_tool_calls": unobserved_calls,
        "unique_bash_command_hashes": len(command_hashes),
        "duplicate_bash_calls": duplicate_bash_calls,
        "contains_raw_commands_sql_prompts_or_tool_outputs": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-openai", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(read_openai(args.rollout_openai))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
