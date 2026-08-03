#!/usr/bin/env python3
"""Independent quality gate for the formal PI train/val/test Parquet set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from llin_verl.pi_reward import execute_readonly_sql
from scripts.prepare_pi_formal_dataset import SYSTEM_PROMPT, TOOL_NAMES, canonical_hash, gold_supported_by_rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_rows(
    split_rows: dict[str, list[dict[str, Any]]],
    sandbox_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    all_ids: dict[str, set[str]] = {}
    all_hashes: dict[str, set[str]] = {}
    detail: dict[str, Any] = {}
    global_indexes: set[int] = set()
    global_verifiers: set[str] = set()
    for split, rows in split_rows.items():
        ids: set[str] = set()
        hashes: set[str] = set()
        answer_types: Counter[str] = Counter()
        environments: set[str] = set()
        for position, row in enumerate(rows):
            prompt = row.get("prompt") or []
            extra = row.get("extra_info") or {}
            truth = (row.get("reward_model") or {}).get("ground_truth") or {}
            verifier_id = str(truth.get("verifier_id") or "")
            task_id = str(truth.get("task_id") or "")
            instruction_hash = str(extra.get("instruction_sha256") or "")
            index = int(extra.get("index", -1))
            if row.get("agent_name") != "pi_agent":
                errors.append(f"{split}[{position}]: wrong agent_name")
            if [message.get("role") for message in prompt] != ["system", "user"]:
                errors.append(f"{split}[{position}]: prompt roles are not system,user")
            elif prompt[0].get("content") != SYSTEM_PROMPT:
                errors.append(f"{split}[{position}]: system prompt mismatch")
            if extra.get("split") != split:
                errors.append(f"{split}[{position}]: split metadata mismatch")
            if extra.get("tool_selection") != TOOL_NAMES or set((extra.get("tools_kwargs") or {})) != set(TOOL_NAMES):
                errors.append(f"{split}[{position}]: full PI tool contract missing")
            if verifier_id in global_verifiers:
                errors.append(f"duplicate verifier_id: {verifier_id}")
            global_verifiers.add(verifier_id)
            if index in global_indexes:
                errors.append(f"duplicate global index: {index}")
            global_indexes.add(index)
            ids.add(task_id)
            hashes.add(instruction_hash)
            environments.add(str(truth.get("environment_id") or ""))
            answer_type = str(truth.get("answer_type") or "")
            answer_types[answer_type] += 1
            try:
                expected = json.loads(str(truth["expected_value_json"]))
                database = (sandbox_root / str(truth["environment_id"]) / "logistics.sqlite").resolve(strict=True)
                sql_rows = execute_readonly_sql(database, str(truth["verification_sql"]))
                if not gold_supported_by_rows({"answer_type": answer_type, "value": expected}, sql_rows):
                    errors.append(f"{verifier_id}: gold no longer matches verifier SQL")
            except Exception as exc:
                errors.append(f"{verifier_id}: verifier failure {type(exc).__name__}: {exc}")
        all_ids[split] = ids
        all_hashes[split] = hashes
        detail[split] = {
            "rows": len(rows),
            "unique_task_ids": len(ids),
            "unique_instruction_hashes": len(hashes),
            "answer_types": dict(answer_types),
            "environments": sorted(environments),
        }
    splits = sorted(split_rows)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            if overlap := all_ids[left] & all_ids[right]:
                errors.append(f"task_id leakage {left}/{right}: {len(overlap)}")
            if overlap := all_hashes[left] & all_hashes[right]:
                errors.append(f"instruction leakage {left}/{right}: {len(overlap)}")
    return errors, detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sandbox-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--tool-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=160)
    parser.add_argument("--expected-val", type=int, default=20)
    parser.add_argument("--expected-test", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import Dataset

    paths = {split: args.data_dir / f"pi_formal_{split}.parquet" for split in ("train", "val", "test")}
    split_rows = {split: Dataset.from_parquet(str(path)).to_list() for split, path in paths.items()}
    errors, detail = audit_rows(split_rows, args.sandbox_root)
    expected = {"train": args.expected_train, "val": args.expected_val, "test": args.expected_test}
    for split, count in expected.items():
        if len(split_rows[split]) != count:
            errors.append(f"{split}: expected {count} rows, got {len(split_rows[split])}")

    prompt_tokens: dict[str, Any] = {}
    if args.tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True)
        tools = None
        if args.tool_config:
            import yaml

            config = yaml.safe_load(args.tool_config.read_text(encoding="utf-8"))
            tools = [entry["tool_schema"] for entry in config["tools"]]
        for split, rows in split_rows.items():
            values = []
            for row in rows:
                rendered = tokenizer.apply_chat_template(
                    row["prompt"],
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                values.append(len(tokenizer.encode(rendered, add_special_tokens=False)))
            prompt_tokens[split] = {
                "min": min(values),
                "mean": sum(values) / len(values),
                "max": max(values),
            }
            if max(values) > 4096:
                errors.append(f"{split}: prompt exceeds 4096 tokens")

    report = {
        "contract": "llin-full-pi-dwh-v2",
        "passed": not errors,
        "errors": errors,
        "detail": detail,
        "prompt_tokens_with_tool_schema": prompt_tokens,
        "system_prompt_sha256": canonical_hash(SYSTEM_PROMPT),
        "file_sha256": {split: file_sha256(path) for split, path in paths.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
