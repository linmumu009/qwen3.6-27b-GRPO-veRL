#!/usr/bin/env python3
"""Run the real PI database tool and reward against every smoke record."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from datasets import Dataset
from verl.tools.tool_registry import load_all_tools

from llin_verl.pi_reward import compute_score


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["verifier_id"]] = record
    return records


async def check(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset = Dataset.from_parquet(str(args.dataset))
    manifest = read_manifest(args.verifier_manifest)
    tools = load_all_tools(tool_config_path=str(args.tool_config), function_tool_path=None)
    tool = next(item for item in tools if item.name == "query_sqlite")
    results: list[dict[str, Any]] = []

    for row in dataset:
        verifier_id = row["extra_info"]["verifier_id"]
        verifier = manifest[verifier_id]
        agent_data = SimpleNamespace(
            tools_kwargs=row["extra_info"]["tools_kwargs"],
            extra_fields={},
        )
        instance_id, _ = await tool.create()
        response, _, metrics = await tool.execute(
            instance_id,
            {"sql": verifier["verification_sql"]},
            agent_data=agent_data,
        )
        reward = compute_score(
            row["data_source"],
            response.text or "",
            row["reward_model"]["ground_truth"],
            agent_data.extra_fields,
        )
        if metrics["query_ok"] != 1.0 or reward["score"] != 1.0:
            raise RuntimeError(
                f"integration check failed for {verifier_id}: metrics={metrics}, reward={reward}, response={response.text}"
            )
        results.append({"verifier_id": verifier_id, "score": reward["score"]})
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--verifier-manifest", type=Path, required=True)
    parser.add_argument("--tool-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    results = asyncio.run(check(parse_args()))
    print(json.dumps({"checked": len(results), "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
