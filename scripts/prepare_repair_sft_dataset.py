#!/usr/bin/env python3
"""Build a mechanically verified, train-only repair SFT and replay dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shlex
import sqlite3
from typing import Any, Iterable
from urllib.parse import quote

import pandas as pd

from llin_verl.boss_pi_contract import canonical_json, contract_hashes, load_boss_pi_contract
from scripts.audit_formal_instruction_gold_alignment import classify
from scripts.prepare_pi_formal_dataset import gold_supported_by_rows


ALLOWED_SEMANTIC_WARNINGS = frozenset({"latest_instruction_without_temporal_sql"})
READ_ONLY_SQL_RE = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    from datasets import Dataset

    return Dataset.from_parquet(str(path)).to_list()


def task_id(row: dict[str, Any]) -> str:
    truth = (row.get("reward_model") or {}).get("ground_truth") or {}
    return str(truth.get("task_id") or "")


def semantic_warnings(review: dict[str, Any]) -> list[str]:
    warnings = list(classify(review))
    if review.get("source_instruction_in_current_task_definition") is False:
        warnings.append("later_task_definition_drift")
    return list(dict.fromkeys(warnings))


def eligible_candidates(
    train_rows: Iterable[dict[str, Any]], review_rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    review_by_task = {str(row.get("task_id") or ""): row for row in review_rows}
    candidates: list[dict[str, Any]] = []
    for row in train_rows:
        current_task_id = task_id(row)
        review = review_by_task.get(current_task_id)
        if not current_task_id or review is None:
            continue
        warnings = semantic_warnings(review)
        if review.get("split") != "train":
            continue
        if review.get("approved_for_grpo") is not True or review.get("review_status") != "approved":
            continue
        if review.get("source_instruction_in_current_task_definition") is not True:
            continue
        if not set(warnings).issubset(ALLOWED_SEMANTIC_WARNINGS):
            continue
        truth = (row.get("reward_model") or {}).get("ground_truth") or {}
        candidates.append(
            {
                "row": row,
                "review": review,
                "task_id": current_task_id,
                "warnings": warnings,
                "answer_type": str(truth.get("answer_type") or "missing"),
                "task_family": str(truth.get("task_family") or "missing"),
            }
        )
    return candidates


def select_diverse_candidates(
    candidates: Iterable[dict[str, Any]], row_count: int, seed: str
) -> list[dict[str, Any]]:
    pool = list(candidates)
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if len(pool) < row_count:
        raise ValueError(f"only {len(pool)} eligible candidates for requested {row_count}")

    selected: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    while len(selected) < row_count:
        candidate = min(
            pool,
            key=lambda item: (
                len(item["warnings"]),
                answer_type_counts[item["answer_type"]],
                family_counts[item["task_family"]],
                hashlib.sha256(f"{seed}:{item['task_id']}".encode()).hexdigest(),
            ),
        )
        pool.remove(candidate)
        selected.append(candidate)
        family_counts[candidate["task_family"]] += 1
        answer_type_counts[candidate["answer_type"]] += 1
    return selected


def execute_sql_with_columns(database: Path, sql: str, max_rows: int = 10_000) -> tuple[list[str], list[tuple]]:
    if not READ_ONLY_SQL_RE.match(sql or ""):
        raise ValueError("only SELECT/WITH repair SQL is allowed")
    resolved = database.resolve(strict=True)
    uri = f"file:{quote(resolved.as_posix(), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        connection.execute("PRAGMA query_only=ON")
        cursor = connection.execute(sql)
        columns = [str(item[0]) for item in cursor.description or []]
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError("repair SQL evidence exceeds max_rows")
        return columns, rows
    finally:
        connection.close()


def format_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def format_final_answer(answer_type: str, expected: Any) -> str:
    if answer_type == "numeric":
        return f"查询得到的目标数值是 **{format_scalar(expected)}**。"
    if answer_type != "table" or not isinstance(expected, list):
        raise ValueError(f"unsupported repair answer type/value: {answer_type!r}")
    if not expected:
        return "查询结果为空表。"
    label_key = "category" if any("category" in item for item in expected) else "date"
    label_title = "类别" if label_key == "category" else "日期"
    lines = [f"查询结果如下（{len(expected)} 行）：", "", f"| {label_title} | 数值 |", "|---|---:|"]
    for item in expected:
        lines.append(f"| {format_scalar(item.get(label_key, ''))} | {format_scalar(item.get('value'))} |")
    return "\n".join(lines)


def build_sft_row(
    candidate: dict[str, Any], boss_contract: dict[str, Any], database: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = candidate["row"]
    prompt = source.get("prompt") or []
    if [message.get("role") for message in prompt] != ["system", "user"]:
        raise ValueError(f"{candidate['task_id']}: source prompt is not system,user")
    if prompt[0].get("content") != boss_contract["system_prompt"]:
        raise ValueError(f"{candidate['task_id']}: source system prompt differs from boss contract")

    truth = (source.get("reward_model") or {}).get("ground_truth") or {}
    sql = str(truth.get("verification_sql") or "")
    expected = json.loads(str(truth.get("expected_value_json") or "null"))
    answer_type = str(truth.get("answer_type") or "")
    columns, sql_rows = execute_sql_with_columns(database, sql)
    if not sql_rows:
        raise ValueError(f"{candidate['task_id']}: repair SQL returned no rows")
    if not gold_supported_by_rows({"answer_type": answer_type, "value": expected}, sql_rows):
        raise ValueError(f"{candidate['task_id']}: expected value is not supported by repair SQL")

    call_id = f"call_repair_{candidate['task_id'].removeprefix('task_')}"
    command = f"sqlite3 -json /workspace/logistics.sqlite {shlex.quote(sql)}"
    tool_output = json.dumps(
        [dict(zip(columns, values)) for values in sql_rows],
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    messages = [
        dict(prompt[0]),
        dict(prompt[1]),
        {
            "role": "assistant",
            "content": "先用一条只读 SQL 取得完成任务所需的直接证据。",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": command},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": tool_output},
        {"role": "assistant", "content": format_final_answer(answer_type, expected)},
    ]
    sft_row = {
        "sample_id": f"repair-sft-{candidate['task_id']}",
        "task_id": candidate["task_id"],
        "messages": messages,
        "tools": boss_contract["tools"],
        "enable_thinking": False,
        "tool_argument_storage": "mapping_for_qwen36_chat_template",
        "purpose": "train236_mechanically_verified_repair",
        "source_split": "train",
        "source_task_family": candidate["task_family"],
        "semantic_warnings": candidate["warnings"],
    }
    evidence = {
        "task_id": candidate["task_id"],
        "answer_type": answer_type,
        "task_family": candidate["task_family"],
        "semantic_warnings": candidate["warnings"],
        "sql_rows": len(sql_rows),
        "sql_columns": len(columns),
        "source_prompt_sha256": sha256_value(prompt),
        "gold_sha256": sha256_value(
            {
                "answer_type": answer_type,
                "expected": expected,
                "verification_sql": sql,
            }
        ),
        "tool_output_sha256": hashlib.sha256(tool_output.encode("utf-8")).hexdigest(),
    }
    return sft_row, evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--sandbox-root", type=Path, required=True)
    parser.add_argument("--boss-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--seed", default="repair-sft-v1")
    parser.add_argument("--quiet", action="store_true", help="write the contract without printing it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = load_parquet_rows(args.train_file)
    val_rows = load_parquet_rows(args.val_file)
    test_rows = load_parquet_rows(args.test_file)
    review_rows = read_jsonl(args.review_queue)
    boss_contract = load_boss_pi_contract(args.boss_contract)

    candidates = eligible_candidates(train_rows, review_rows)
    selected = select_diverse_candidates(candidates, args.rows, args.seed)
    selected_ids = {item["task_id"] for item in selected}
    heldout_ids = {task_id(row) for row in [*val_rows, *test_rows]}
    overlap = sorted(selected_ids & heldout_ids)
    if overlap:
        raise ValueError(f"repair selection leaks into val/test: {overlap}")

    sft_rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    databases: dict[str, Path] = {}
    for candidate in selected:
        truth = (candidate["row"].get("reward_model") or {}).get("ground_truth") or {}
        environment_id = str(truth.get("environment_id") or "")
        database = args.sandbox_root / environment_id / "logistics.sqlite"
        databases[environment_id] = database
        sft_row, evidence_row = build_sft_row(candidate, boss_contract, database)
        sft_rows.append(sft_row)
        evidence.append(evidence_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = args.output_dir / "repair_sft_train.parquet"
    replay_path = args.output_dir / "repair_sft_replay.parquet"
    pd.DataFrame(sft_rows).to_parquet(sft_path, index=False)
    replay_rows = [item["row"] for item in selected]
    pd.DataFrame(replay_rows).to_parquet(replay_path, index=False)

    hashes = contract_hashes(boss_contract)
    report = {
        "contract": "train236-repair-sft-dataset-v1",
        "rows": len(sft_rows),
        "source_split": "train",
        "heldout_overlap": 0,
        "eligible_candidates": len(candidates),
        "selection_seed": args.seed,
        "allowed_semantic_warnings": sorted(ALLOWED_SEMANTIC_WARNINGS),
        "selected_task_ids": [item["task_id"] for item in selected],
        "answer_types": dict(Counter(item["answer_type"] for item in selected)),
        "task_families": dict(Counter(item["task_family"] for item in selected)),
        "semantic_warning_rows": sum(bool(item["warnings"]) for item in selected),
        "all_rows_approved": True,
        "all_rows_current_task_definition": True,
        "all_sql_read_only_executable_nonempty": True,
        "all_expected_values_match_sql": True,
        "boss_contract_hashes": hashes,
        "source_sha256": {
            "train": sha256_file(args.train_file),
            "val": sha256_file(args.val_file),
            "test": sha256_file(args.test_file),
            "review_queue": sha256_file(args.review_queue),
            "databases": {
                environment_id: sha256_file(path)
                for environment_id, path in sorted(databases.items())
            },
        },
        "outputs": {
            "sft_file": sft_path.name,
            "sft_sha256": sha256_file(sft_path),
            "replay_file": replay_path.name,
            "replay_sha256": sha256_file(replay_path),
        },
        "evidence": evidence,
        "promotion_allowed": False,
        "interpretation": (
            "This dataset is a train-only overfit and pipeline gate. It must not be reported as held-out accuracy. "
            "Promotion requires exact post-SFT replay plus a separate held-out canary."
        ),
    }
    report_path = args.output_dir / "contract.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
