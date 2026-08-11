#!/usr/bin/env python3
"""Create a tiny deterministic multi-turn tool-use dataset for veRL SFT smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_sql",
        "description": "Execute one read-only SQL query and return its rows.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A read-only SQL query."}
            },
            "required": ["query"],
        },
    },
}


def build_rows(row_count: int) -> list[dict]:
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    rows = []
    for index in range(row_count):
        left = 2 + index
        right = 3 + index
        answer = left + right
        query = f"SELECT {left} + {right} AS answer"
        rows.append(
            {
                "sample_id": f"repair-sft-smoke-{index:04d}",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful data assistant. Use the minimum necessary query, "
                            "do not repeat a successful query, and stop once the evidence is sufficient."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Use query_sql to determine the value returned by `{query}`, "
                            "then answer with the number."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "I will run one minimal query and stop after it returns the requested value.",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "query_sql", "arguments": {"query": query}},
                            }
                        ],
                    },
                    {"role": "tool", "content": json.dumps([{"answer": answer}])},
                    {"role": "assistant", "content": f"The answer is {answer}."},
                ],
                "tools": [QUERY_TOOL],
                "enable_thinking": False,
            }
        )
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1)
    args = parser.parse_args()

    rows = build_rows(args.rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "repair_sft_smoke_train.parquet"
    pd.DataFrame(rows).to_parquet(train_path, index=False)
    contract = {
        "contract": "repair-sft-smoke-dataset-v1",
        "rows": len(rows),
        "messages_key": "messages",
        "tools_key": "tools",
        "assistant_turns_per_row": 2,
        "tool_turns_per_row": 1,
        "train_file": train_path.name,
        "train_sha256": sha256(train_path),
        "synthetic_only": True,
        "promotion_allowed": False,
    }
    contract_path = args.output_dir / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
