#!/usr/bin/env python3
"""Approve and split the boss-v15 executable DWH pool for GRPO.

The v15 source instruction, v15 gold, v15 sandbox and boss evaluator are the
authoritative contract. Later rewritten task definitions are retained only as
diagnostic metadata and never replace or veto the source instruction.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_formal_instruction_gold_alignment import classify
from scripts.prepare_boss_aligned_dataset import canonical_hash


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def unique_index(rows: Iterable[dict[str, Any]], key_name: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(key_name) or "").strip()
        if not key:
            raise ValueError(f"{label} row missing {key_name}")
        if key in result:
            raise ValueError(f"duplicate {label} {key_name}: {key}")
        result[key] = row
    return result


def parse_split_sizes(values: list[str]) -> dict[str, int]:
    result = {"train": 0, "val": 0, "test": 0}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"split size must use SPLIT=COUNT: {raw!r}")
        split, count = raw.split("=", 1)
        if split not in result or int(count) < 0:
            raise ValueError(f"invalid split size: {raw!r}")
        result[split] = int(count)
    if not result["train"] or not result["val"]:
        raise ValueError("DWH batch requires non-empty train and val splits")
    return result


def reward_band(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.85:
        return "strict_or_near_strict"
    if score >= 0.20:
        return "sql_evidence_only"
    if score >= 0.15:
        return "answer_only"
    return "low_signal"


def stable_digest(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def round_robin_select(rows: list[dict[str, Any]], count: int, seed: str) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        key = (row["answer_type"], row["primary_table"], row["reward_band"])
        strata[key].append(row)
    for key, values in strata.items():
        strata[key] = deque(sorted(values, key=lambda row: stable_digest(seed, row["task_id"])))
    keys = sorted(strata, key=lambda key: stable_digest(seed, "|".join(key)))
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if strata[key] and len(selected) < count:
                selected.append(strata[key].popleft())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible rows for requested count {count}")
    return selected


def assign_prompt_groups_to_splits(
    rows: list[dict[str, Any]],
    split_sizes: dict[str, int],
    seed: str,
) -> list[tuple[dict[str, Any], str]]:
    """Assign identical prompts atomically while preserving exact split sizes."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_order = {row["task_id"]: index for index, row in enumerate(rows)}
    for row in rows:
        groups[row["instruction_sha256"]].append(row)
    multi = sorted(
        (group for group in groups.values() if len(group) > 1),
        key=lambda group: stable_digest(seed, group[0]["instruction_sha256"]),
    )
    singles = sorted(
        (group for group in groups.values() if len(group) == 1),
        key=lambda group: row_order[group[0]["task_id"]],
    )
    remaining = dict(split_sizes)
    assigned: list[tuple[dict[str, Any], str]] = []
    for group in [*multi, *singles]:
        candidates = [split for split in ("train", "val", "test") if remaining[split] >= len(group)]
        if not candidates:
            raise ValueError(f"cannot place prompt group of {len(group)} rows into remaining {remaining}")
        split = max(candidates, key=lambda name: (remaining[name], name == "train", name == "val"))
        for row in group:
            assigned.append((row, split))
        remaining[split] -= len(group)
    if any(remaining.values()):
        raise AssertionError(f"split assignment left capacity: {remaining}")
    return assigned


def select_batch(
    queue_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    shadow_rows: list[dict[str, Any]],
    split_sizes: dict[str, int],
    seed: str,
    include_tasks: list[str],
    allow_source_definition_drift: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queue = unique_index(queue_rows, "task_id", "review queue")
    manifest = unique_index(manifest_rows, "task_id", "manifest")
    shadow = unique_index(shadow_rows, "task_id", "shadow result") if shadow_rows else {}
    excluded: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []

    for task_id, row in queue.items():
        task = manifest.get(task_id)
        if task is None:
            excluded["missing_manifest_task"] += 1
            continue
        if str(task.get("type") or "").casefold() != "dwh":
            excluded["not_dwh"] += 1
            continue
        if str(row.get("source_join_method") or "") not in {"task_id", "exact_instruction"}:
            excluded["untrusted_source_join"] += 1
            continue
        source_current = row.get("source_instruction_in_current_task_definition") is True
        if not source_current and not allow_source_definition_drift:
            excluded["source_instruction_not_current"] += 1
            continue
        gold = row.get("gold") or {}
        if gold.get("answer_type") not in {"numeric", "table"}:
            excluded["unsupported_answer_type"] += 1
            continue
        if canonical_hash(str(row.get("instruction") or "")) != row.get("instruction_sha256"):
            excluded["instruction_hash_mismatch"] += 1
            continue
        if canonical_hash(gold) != row.get("gold_sha256"):
            excluded["gold_hash_mismatch"] += 1
            continue
        expected_tables = sorted({str(value).casefold() for value in task.get("expected_tables") or []})
        if not expected_tables:
            excluded["missing_expected_tables"] += 1
            continue
        shadow_row = shadow.get(task_id)
        if shadow_rows and (
            shadow_row is None
            or not bool(shadow_row.get("online_eligible"))
            or not bool(shadow_row.get("gold_sql_verified"))
        ):
            excluded["shadow_gold_gate_failed"] += 1
            continue
        warnings = list(classify(row))
        if not source_current:
            warnings.append("later_task_definition_drift")
        warnings = list(dict.fromkeys(warnings))
        warning_counts.update(warnings)
        score = float(shadow_row.get("score")) if shadow_row and shadow_row.get("score") is not None else None
        eligible.append(
            {
                "source_label": str(row.get("source_label") or ""),
                "task_id": task_id,
                "instruction_sha256": row["instruction_sha256"],
                "gold_sha256": row["gold_sha256"],
                "answer_type": str(gold["answer_type"]),
                "primary_table": expected_tables[0],
                "expected_tables": expected_tables,
                "shadow_score": score,
                "reward_band": reward_band(score),
                "source_instruction_current": source_current,
                "audit_warnings": warnings,
            }
        )

    eligible_by_id = {row["task_id"]: row for row in eligible}
    missing_includes = sorted(set(include_tasks) - set(eligible_by_id))
    if missing_includes:
        raise ValueError(f"explicit include tasks are not eligible: {missing_includes}")
    total = sum(split_sizes.values())
    if total > len(eligible):
        raise ValueError(f"requested {total} rows but only {len(eligible)} pass source/gold gates")

    selected = [eligible_by_id[task_id] for task_id in include_tasks]
    used = set(include_tasks)
    selected.extend(
        round_robin_select(
            [row for row in eligible if row["task_id"] not in used],
            total - len(selected),
            seed,
        )
    )
    approvals: list[dict[str, Any]] = []
    selected_audit: list[dict[str, Any]] = []
    reviewed_at = datetime.now(timezone.utc).isoformat()
    for row, split in assign_prompt_groups_to_splits(selected, split_sizes, seed):
        approvals.append(
            {
                "source_label": row["source_label"],
                "task_id": row["task_id"],
                "instruction_sha256": row["instruction_sha256"],
                "gold_sha256": row["gold_sha256"],
                "approved_for_grpo": True,
                "reviewer": "codex-boss-v15-source-contract-audit",
                "reviewed_at": reviewed_at,
                "split": split,
                "review_basis": "boss v15 source instruction + executable gold + v15 sandbox + boss evaluator",
                "audit_warnings": row["audit_warnings"],
            }
        )
        selected_audit.append({**row, "split": split})

    task_ids = [row["task_id"] for row in selected_audit]
    instruction_hashes = [row["instruction_sha256"] for row in selected_audit]
    prompt_splits: dict[str, set[str]] = defaultdict(set)
    for row in selected_audit:
        prompt_splits[row["instruction_sha256"]].add(row["split"])
    audit = {
        "contract": "boss-v15-dwh-full-pool-v1",
        "seed": seed,
        "input": {
            "review_queue_rows": len(queue_rows),
            "manifest_rows": len(manifest_rows),
            "shadow_rows": len(shadow_rows),
        },
        "eligible_rows": len(eligible),
        "excluded": dict(sorted(excluded.items())),
        "selected": dict(split_sizes),
        "unselected_eligible_rows": len(eligible) - len(selected_audit),
        "warning_rows": sum(bool(row["audit_warnings"]) for row in eligible),
        "warning_counts": dict(sorted(warning_counts.items())),
        "invariants": {
            "boss_v15_source_is_authority": True,
            "uses_all_eligible_rows": len(selected_audit) == len(eligible),
            "dwh_only": True,
            "numeric_or_table_only": True,
            "source_join_verified": True,
            "gold_hash_verified": True,
            "shadow_gold_gate_required": bool(shadow_rows),
            "unique_task_ids": len(task_ids) == len(set(task_ids)),
            "unique_instruction_hashes": len(instruction_hashes) == len(set(instruction_hashes)),
            "no_cross_split_instruction_overlap": all(len(splits) == 1 for splits in prompt_splits.values()),
            "kb_rows": 0,
            "hybrid_rows": 0,
        },
        "answer_types": dict(Counter(row["answer_type"] for row in selected_audit)),
        "primary_tables": dict(Counter(row["primary_table"] for row in selected_audit)),
        "reward_bands": dict(Counter(row["reward_band"] for row in selected_audit)),
        "selected_rows": selected_audit,
    }
    return approvals, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--shadow-results", type=Path)
    parser.add_argument("--output-review", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--split-size", action="append", default=["train=237", "val=20", "test=20"])
    parser.add_argument("--seed", default="boss-v15-dwh-full-pool-v1-20260804")
    parser.add_argument("--include-task", action="append", default=[])
    parser.add_argument("--reject-source-definition-drift", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    approvals, audit = select_batch(
        read_jsonl(args.review_queue),
        read_jsonl(args.task_manifest),
        read_jsonl(args.shadow_results) if args.shadow_results else [],
        parse_split_sizes(args.split_size),
        args.seed,
        args.include_task,
        not args.reject_source_definition_drift,
    )
    args.output_review.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_review.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in approvals),
        encoding="utf-8",
    )
    args.output_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in audit.items() if key != "selected_rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
