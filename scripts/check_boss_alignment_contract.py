#!/usr/bin/env python3
"""Hard gate for any new boss-aligned formal GRPO run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llin_verl.boss_pi_contract import contract_hashes, load_boss_pi_contract
from scripts.prepare_boss_aligned_dataset import CONTRACT_NAME, file_sha256


REQUIRED_ZERO_INVARIANTS = (
    "project_system_fallback_count",
    "project_tool_schema_fallback_count",
    "generated_instruction_count",
    "generated_gold_or_sql_count",
    "conflicting_instruction_gold_count",
    "unreviewed_grpo_count",
    "assistant_or_tool_messages_in_grpo_input",
)


def validate_alignment_contract(
    data_dir: Path,
    *,
    allow_pilot: bool = False,
) -> dict[str, Any]:
    path = data_dir / "boss_alignment_contract.json"
    if not path.is_file():
        raise FileNotFoundError(f"boss alignment contract not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("contract") != CONTRACT_NAME:
        raise ValueError("formal run refuses non-boss-aligned data")
    if report.get("mode") != "full" and not allow_pilot:
        raise ValueError("formal run refuses pilot data; use it only with an explicit pilot launcher")
    invariants = report.get("invariants") or {}
    for name in REQUIRED_ZERO_INVARIANTS:
        if invariants.get(name) != 0:
            raise ValueError(f"boss alignment invariant failed: {name}={invariants.get(name)!r}")
    if report.get("mode") == "full" and invariants.get("uses_all_approved_by_default") is not True:
        raise ValueError("full dataset did not retain all approved source tasks")
    if invariants.get("source_responses_exported_only_for_sft_and_regression") is not True:
        raise ValueError("source response isolation invariant failed")
    if any(item.get("task_ids") or item.get("instruction_hashes") for item in invariants.get("split_overlap", [])):
        raise ValueError("boss-aligned data has split leakage")

    expected_hashes = contract_hashes(load_boss_pi_contract())
    observed = report.get("boss_contract") or {}
    for name, expected in expected_hashes.items():
        if observed.get(name) != expected:
            raise ValueError(f"boss PI contract drift: {name}")

    artifacts = report.get("artifacts") or {}
    for required in ("boss_pi_train.parquet", "boss_pi_val.parquet"):
        item = artifacts.get(required)
        file_path = data_dir / required
        if not isinstance(item, dict) or not file_path.is_file():
            raise FileNotFoundError(f"required boss-aligned artifact missing: {required}")
        if item.get("purpose") != "grpo" or item.get("sha256") != file_sha256(file_path):
            raise ValueError(f"boss-aligned artifact integrity failed: {required}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--allow-pilot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_alignment_contract(args.data_dir, allow_pilot=args.allow_pilot)
    print(json.dumps({"status": "passed", "mode": report["mode"], "selected": report["selected"]}, indent=2))


if __name__ == "__main__":
    main()
