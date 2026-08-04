#!/usr/bin/env python3
"""Validate boss-primary reward metadata in a generated GRPO parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset

from llin_verl.pi_reward import boss_reward_components


def validate(path: Path) -> dict:
    dataset = Dataset.from_parquet(str(path))
    if not dataset:
        raise ValueError("reward dataset is empty")
    contracts: set[str] = set()
    families: set[str] = set()
    missing_fields = 0
    for row in dataset:
        ground_truth = (row.get("reward_model") or {}).get("ground_truth") or {}
        contracts.add(str(ground_truth.get("reward_contract") or ""))
        families.add(str(ground_truth.get("task_family") or ""))
        missing_fields += "must_use_fields" not in ground_truth
    expected_contract = "boss-primary-70-strict-evidence-30-v1"
    if contracts != {expected_contract} or families != {"dwh"} or missing_fields:
        raise ValueError(
            f"reward metadata mismatch: contracts={contracts}, families={families}, "
            f"missing_fields={missing_fields}"
        )
    probe = boss_reward_components(
        "查询和复核已经完成，最终确认的合计结果为 10。",
        10,
        ['sqlite3 /workspace/logistics.sqlite "SELECT SUM(value) FROM metric"'],
        ["SELECT SUM(value) FROM metric"],
        {"metric"},
        {"metric"},
        [],
        True,
        [],
    )
    if probe["reward"] != 1.0:
        raise ValueError(f"boss reward probe failed: {probe}")
    return {
        "status": "passed",
        "rows": len(dataset),
        "reward_contract": expected_contract,
        "task_family": "dwh",
        "must_use_fields_missing": missing_fields,
        "reward_probe": probe["reward"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.parquet), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
