#!/usr/bin/env python3
"""Build a verified, leakage-controlled 200-task full-PI DWH dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from llin_verl.pi_reward import execute_readonly_sql


SYSTEM_PROMPT = (
    "你是一个物流数据分析师。当前任务唯一允许访问的工作区是 /workspace，"
    "数据库固定为 /workspace/logistics.sqlite；禁止扫描根目录或其他环境。"
    "你可以使用 Bash 工具执行 sqlite3 只读查询，也可在 /workspace 内使用 read/write/edit。"
    "分析数据时，先检查当前数据库结构，再执行必要且尽量少的 SQL，"
    "最后在可见回答中明确写出问题要求的数值、类别及结论。"
)
TOOL_NAMES = ["bash", "read", "write", "edit"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _contains_value(rows: list[tuple[Any, ...]], value: Any, abs_tol: float, rel_tol: float) -> bool:
    import math

    for row in rows:
        for actual in row:
            if isinstance(value, (int, float)) and isinstance(actual, (int, float)):
                if math.isclose(float(actual), float(value), abs_tol=abs_tol, rel_tol=rel_tol):
                    return True
            elif str(value).strip().casefold() == str(actual).strip().casefold():
                return True
    return False


def gold_supported_by_rows(gold: dict[str, Any], rows: list[tuple[Any, ...]]) -> bool:
    answer_type = gold.get("answer_type")
    value = gold.get("value")
    if answer_type == "numeric":
        return isinstance(value, (int, float)) and _contains_value(rows, value, 1e-3, 1e-5)
    if answer_type != "table" or not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict) or not _contains_value(rows, item.get("value"), 1e-3, 1e-5):
            return False
        label = item.get("category", item.get("date"))
        if label is not None and not _contains_value(rows, label, 0.0, 0.0):
            return False
    return True


def validate_candidates(
    rows: list[dict[str, Any]],
    sandbox_root: Path,
    source_manifest: Path,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    valid: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in rows:
        if row.get("type") != "dwh":
            rejected["not_dwh"] += 1
            continue
        gold = row.get("gold_answer")
        if not isinstance(gold, dict) or gold.get("answer_type") not in {"numeric", "table"}:
            rejected["unsupported_gold"] += 1
            continue
        sql = gold.get("verification_sql")
        if not isinstance(sql, str) or not sql.strip():
            rejected["missing_verification_sql"] += 1
            continue
        version = str(row.get("v") or "")
        database = sandbox_root / "sft" / version / "logistics.sqlite"
        try:
            expected_rows = execute_readonly_sql(database.resolve(strict=True), sql)
        except Exception:
            rejected["verification_sql_failed"] += 1
            continue
        if not expected_rows:
            rejected["verification_sql_empty"] += 1
            continue
        if not gold_supported_by_rows(gold, expected_rows):
            rejected["gold_result_mismatch"] += 1
            continue
        instruction = str(row.get("instruction") or "").strip()
        task_id = str(row.get("task_id") or "").strip()
        if not instruction or not task_id:
            rejected["missing_identity"] += 1
            continue
        candidate = {
            "task_id": task_id,
            "task_type": "dwh",
            "version": version,
            "environment_id": f"sft/{version}",
            "instruction": instruction,
            "instruction_sha256": canonical_hash(instruction),
            "required_tables": sorted({str(value).casefold() for value in row.get("expected_tables") or []}),
            "gold": {
                "answer_type": gold["answer_type"],
                "value": gold.get("value"),
                "verification_sql": sql,
                "abs_tol": 1e-3,
                "rel_tol": 1e-5,
            },
            "source_manifest": source_manifest.name,
            "system_prompt": str(row.get("system_prompt") or "").strip() or SYSTEM_PROMPT,
            "system_prompt_source": "source" if str(row.get("system_prompt") or "").strip() else "fallback",
        }
        candidate["verifier_id"] = f"{candidate['environment_id']}:{task_id}"
        valid.append(candidate)
    return valid, rejected


def _stable_order(candidate: dict[str, Any], seed: str) -> str:
    return hashlib.sha256(f"{seed}:{candidate['verifier_id']}".encode()).hexdigest()


def select_split(
    candidates: list[dict[str, Any]],
    count: int,
    seed: str,
    blocked_task_ids: set[str],
    blocked_instruction_hashes: set[str],
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in candidates
        if row["task_id"] not in blocked_task_ids
        and row["instruction_sha256"] not in blocked_instruction_hashes
    ]
    eligible.sort(key=lambda row: _stable_order(row, seed))
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} leakage-free candidates for requested {count}")
    selected = eligible[:count]
    blocked_task_ids.update(row["task_id"] for row in selected)
    blocked_instruction_hashes.update(row["instruction_sha256"] for row in selected)
    return selected


def build_training_record(verifier: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    ground_truth = {
        "verifier_id": verifier["verifier_id"],
        "task_id": verifier["task_id"],
        "environment_id": verifier["environment_id"],
        "answer_type": verifier["gold"]["answer_type"],
        "expected_value_json": json.dumps(
            verifier["gold"]["value"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "verification_sql": verifier["gold"]["verification_sql"],
        "required_tables": verifier["required_tables"],
        "abs_tol": verifier["gold"]["abs_tol"],
        "rel_tol": verifier["gold"]["rel_tol"],
    }
    tools_kwargs = {
        name: {"create_kwargs": {"environment_id": verifier["environment_id"]}}
        for name in TOOL_NAMES
    }
    return {
        "data_source": "llin_pi_dwh_v2",
        "agent_name": "pi_agent",
        "prompt": [
            {"role": "system", "content": verifier.get("system_prompt") or SYSTEM_PROMPT},
            {"role": "user", "content": verifier["instruction"]},
        ],
        "ability": "full_pi_dwh",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": index,
            "split": split,
            "verifier_id": verifier["verifier_id"],
            "environment_id": verifier["environment_id"],
            "instruction_sha256": verifier["instruction_sha256"],
            "need_tools_kwargs": True,
            "tool_selection": TOOL_NAMES,
            "tools_kwargs": tools_kwargs,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--sandbox-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=160)
    parser.add_argument("--val-size", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--seed", default="llin-pi-formal-v2-20260803")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {
        "train": args.train_manifest,
        "val": args.val_manifest,
        "test": args.test_manifest,
    }
    validated: dict[str, list[dict[str, Any]]] = {}
    rejected: dict[str, dict[str, int]] = {}
    for split, path in sources.items():
        values, reasons = validate_candidates(read_jsonl(path), args.sandbox_root, path)
        validated[split] = values
        rejected[split] = dict(reasons)

    blocked_ids: set[str] = set()
    blocked_hashes: set[str] = set()
    selected: dict[str, list[dict[str, Any]]] = {}
    # Evaluation sets are reserved first; training cannot reuse their task identity.
    selected["test"] = select_split(
        validated["test"], args.test_size, args.seed + ":test", blocked_ids, blocked_hashes
    )
    selected["val"] = select_split(
        validated["val"], args.val_size, args.seed + ":val", blocked_ids, blocked_hashes
    )
    selected["train"] = select_split(
        validated["train"], args.train_size, args.seed + ":train", blocked_ids, blocked_hashes
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_verifiers: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    global_index = 0
    from datasets import Dataset

    for split in ("train", "val", "test"):
        records = []
        for verifier in selected[split]:
            records.append(build_training_record(verifier, split, global_index))
            global_index += 1
        Dataset.from_list(records).to_parquet(str(args.output_dir / f"pi_formal_{split}.parquet"))
        all_records.extend(records)
        all_verifiers.extend({**verifier, "split": split} for verifier in selected[split])

    # A deterministic union is used by the frozen-model baseline.  Keeping the
    # original split label in extra_info lets the report recover train/val/test
    # metrics without weakening the three physically isolated split files.
    Dataset.from_list(all_records).to_parquet(str(args.output_dir / "pi_formal_all.parquet"))

    with (args.output_dir / "pi_formal_verifier_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for verifier in all_verifiers:
            handle.write(json.dumps(verifier, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "contract": "llin-full-pi-dwh-v2",
        "seed": args.seed,
        "system_prompt_sha256": canonical_hash(SYSTEM_PROMPT),
        "tool_names": TOOL_NAMES,
        "validated_candidates": {key: len(value) for key, value in validated.items()},
        "rejected": rejected,
        "selected": {key: len(value) for key, value in selected.items()},
        "environments": {key: sorted({row["environment_id"] for row in value}) for key, value in selected.items()},
        "answer_types": {
            key: dict(Counter(row["gold"]["answer_type"] for row in value))
            for key, value in selected.items()
        },
        "task_id_overlap": 0,
        "instruction_hash_overlap": 0,
    }
    (args.output_dir / "pi_formal_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
