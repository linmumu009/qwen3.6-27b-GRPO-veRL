#!/usr/bin/env python3
"""Replay every boss GRPO label and separate mechanical validity from semantics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from llin_verl.pi_reward import execute_readonly_sql
from scripts.audit_formal_instruction_gold_alignment import classify
from scripts.prepare_pi_formal_dataset import gold_supported_by_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_rows(
    split_rows: dict[str, list[dict[str, Any]]],
    sandbox_root: Path,
    review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    review_by_task = {str(row.get("task_id") or ""): row for row in review_rows}
    errors: list[str] = []
    warning_counts: Counter[str] = Counter()
    warning_rows = 0
    instruction_gold: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    split_summary: dict[str, Any] = {}
    executable = nonempty = result_matches = 0

    for split, rows in split_rows.items():
        split_matches = 0
        split_warning_rows = 0
        for position, row in enumerate(rows):
            truth = (row.get("reward_model") or {}).get("ground_truth") or {}
            extra = row.get("extra_info") or {}
            task_id = str(truth.get("task_id") or "")
            verifier_id = str(truth.get("verifier_id") or task_id)
            instruction_hash = str(extra.get("instruction_sha256") or "")
            gold_hash = str(extra.get("gold_sha256") or "")
            instruction_gold[instruction_hash][gold_hash].add(f"{split}:{task_id}")
            try:
                expected = json.loads(str(truth["expected_value_json"]))
                database = (
                    sandbox_root / str(truth["environment_id"]) / "logistics.sqlite"
                ).resolve(strict=True)
                sql_rows = execute_readonly_sql(database, str(truth["verification_sql"]))
                executable += 1
                if sql_rows:
                    nonempty += 1
                gold = {"answer_type": str(truth["answer_type"]), "value": expected}
                if gold_supported_by_rows(gold, sql_rows):
                    result_matches += 1
                    split_matches += 1
                else:
                    errors.append(f"{verifier_id}: expected value does not match SQL result")
            except Exception as exc:
                errors.append(
                    f"{split}[{position}] {verifier_id}: {type(exc).__name__}: {exc}"
                )

            source = review_by_task.get(task_id)
            if source:
                warnings = list(classify(source))
                if source.get("source_instruction_in_current_task_definition") is False:
                    warnings.append("later_task_definition_drift")
                warnings = list(dict.fromkeys(warnings))
                warning_counts.update(warnings)
                warning_rows += bool(warnings)
                split_warning_rows += bool(warnings)
        split_summary[split] = {
            "rows": len(rows),
            "sql_result_matches": split_matches,
            "semantic_warning_rows": split_warning_rows,
        }

    conflicts = {
        instruction_hash: {
            gold_hash: sorted(task_ids) for gold_hash, task_ids in sorted(gold_tasks.items())
        }
        for instruction_hash, gold_tasks in sorted(instruction_gold.items())
        if len(gold_tasks) > 1
    }
    total = sum(len(rows) for rows in split_rows.values())
    return {
        "contract": "boss-grpo-label-quality-audit-v1",
        "passed_mechanical_gate": not errors and result_matches == total and not conflicts,
        "rows": total,
        "split_summary": split_summary,
        "verification_sql_executable": executable,
        "verification_sql_nonempty": nonempty,
        "expected_value_matches_sql": result_matches,
        "conflicting_instruction_gold_count": len(conflicts),
        "conflicting_instruction_gold": conflicts,
        "semantic_warning_rows": warning_rows,
        "semantic_warning_counts": dict(sorted(warning_counts.items())),
        "human_semantic_review_coverage": "not established by this audit",
        "interpretation": (
            "SQL execution and expected-value equality prove mechanical self-consistency; "
            "they do not prove that the hidden SQL is the unique correct interpretation of the instruction."
        ),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sandbox-root", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import Dataset

    split_rows = {
        split: Dataset.from_parquet(str(args.data_dir / f"boss_pi_{split}.parquet")).to_list()
        for split in ("train", "val", "test")
    }
    report = audit_rows(split_rows, args.sandbox_root, read_jsonl(args.review_queue))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed_mechanical_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
