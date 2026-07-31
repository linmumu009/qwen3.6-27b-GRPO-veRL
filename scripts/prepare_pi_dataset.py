#!/usr/bin/env python3
"""Convert verified PI prompt records into veRL trajectory-GRPO parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """你是一个物流数据分析师。你可以调用 query_sqlite 工具查询当前任务对应的只读 SQLite 数据库。
先分析问题需要哪些表和字段，再使用 SQL 获取证据，最后用中文给出简洁结论并明确写出数值。
禁止猜测答案；工具只接受 SELECT/WITH 查询。"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def build_records(
    prompts: list[dict[str, Any]],
    verifier_by_id: dict[str, dict[str, Any]],
    selected_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for prompt_record in prompts:
        verifier_id = prompt_record.get("verifier_id") or prompt_record.get("metadata", {}).get("verifier_id")
        if not verifier_id or verifier_id not in verifier_by_id:
            continue
        if selected_ids and verifier_id not in selected_ids:
            continue

        verifier = verifier_by_id[verifier_id]
        environment_id = verifier["environment_id"]
        source_prompt_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in prompt_record["messages"]
            if message.get("role") in {"system", "user"}
        ]
        if not any(message["role"] == "user" for message in source_prompt_messages):
            continue
        if not any(message["role"] == "system" for message in source_prompt_messages):
            source_prompt_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        output.append(
            {
                "data_source": "llin_pi_dwh",
                "agent_name": "tool_agent",
                "prompt": source_prompt_messages,
                "ability": "dwh_sql",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "verifier_id": verifier_id,
                        "expected_value": float(verifier["gold"]["value"]),
                        "answer_type": verifier["gold"]["answer_type"],
                        "required_tables": verifier["required_tables"],
                    },
                },
                "extra_info": {
                    "verifier_id": verifier_id,
                    "environment_id": environment_id,
                    "need_tools_kwargs": True,
                    "tool_selection": ["query_sqlite"],
                    "tools_kwargs": {
                        "query_sqlite": {
                            "create_kwargs": {
                                "environment_id": environment_id,
                            },
                        },
                    },
                },
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--verifier-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifier-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = read_jsonl(args.prompts)
    verifier_records = read_jsonl(args.verifier_manifest)
    verifier_by_id = {record["verifier_id"]: record for record in verifier_records}
    selected = set(args.verifier_id) or None
    records = build_records(prompts, verifier_by_id, selected)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no matching verified prompt records")

    from datasets import Dataset

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(args.output))
    print(json.dumps({"output": str(args.output), "rows": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
