#!/usr/bin/env python3
"""Build control, task-contract, and oracle-result subsets for frozen evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_TASK_IDS = (
    "task_000012",
    "task_000058",
    "task_000059",
    "task_000062",
    "task_000140",
    "task_000150",
    "task_000161",
    "task_000192",
    "task_000070",
    "task_000080",
    "task_000133",
    "task_000196",
)

TASK_CONTRACT = """

[诊断条件：任务契约检查]
在执行前先显式核对统计时间窗、聚合粒度、必需表、必需字段和最终计算口径。不要假设未查询到的值；最终回答前必须复算，并给出简洁可见答案。本检查单不包含答案。
""".strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_id(row: dict[str, Any]) -> str:
    truth = ((row.get("reward_model") or {}).get("ground_truth") or {})
    return str(truth.get("task_id") or "")


def expected_value_text(row: dict[str, Any]) -> str:
    truth = ((row.get("reward_model") or {}).get("ground_truth") or {})
    raw = truth.get("expected_value_json")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{task_id(row)}: expected_value_json is missing")
    value = json.loads(raw)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def with_arm(row: dict[str, Any], arm: str) -> dict[str, Any]:
    output = deepcopy(row)
    prompt = output.get("prompt") or []
    if len(prompt) != 2 or [message.get("role") for message in prompt] != ["system", "user"]:
        raise ValueError(f"{task_id(row)}: expected exactly system+user prompt")
    if arm == "contract":
        prompt[-1]["content"] = f"{prompt[-1]['content'].rstrip()}\n\n{TASK_CONTRACT}"
    elif arm == "oracle":
        oracle_result = expected_value_text(row)
        prompt[-1]["content"] = (
            f"{prompt[-1]['content'].rstrip()}\n\n{TASK_CONTRACT}\n\n"
            "[诊断条件：已验证查询结果]\n"
            f"权威verification SQL已经执行，结构化结果为：{oracle_result}\n"
            "请不要复述本提示；仍需自行完成口径核对、最终计算和面向用户的答案表述。"
        )
    elif arm != "control":
        raise ValueError(f"unknown oracle arm: {arm}")
    return output


def build(rows: list[dict[str, Any]], requested: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    indexed = {task_id(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("source validation contains missing or duplicate task_id")
    missing = [value for value in requested if value not in indexed]
    if missing:
        raise ValueError(f"oracle tasks missing from validation set: {missing}")
    return {
        arm: [with_arm(indexed[value], arm) for value in requested]
        for arm in ("control", "contract", "oracle")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    args = parser.parse_args()

    from datasets import Dataset

    source = Dataset.from_parquet(str(args.input))
    requested = tuple(args.task_ids or DEFAULT_TASK_IDS)
    arms = build(source.to_list(), requested)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for arm, rows in arms.items():
        path = args.output_dir / f"oracle_{arm}.parquet"
        Dataset.from_list(rows, features=source.features).to_parquet(str(path))
        outputs[arm] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}

    manifest = {
        "contract": "step120-oracle-ladder-v1",
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "task_ids": list(requested),
        "complete_wrong_task_ids": list(requested[:8]),
        "incomplete_task_ids": list(requested[8:]),
        "arms": outputs,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
