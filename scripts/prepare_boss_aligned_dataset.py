#!/usr/bin/env python3
"""Build GRPO prompts and SFT references from boss-owned PI sources only.

The builder deliberately has no prompt, system, tool, gold, or SQL fallback.
Conversation rows are joined to their original task manifest by the exact first
user instruction.  A task enters GRPO only after an explicit alignment-review
record approves the exact instruction and exact gold hash.  Everything else is
exported as an SFT/reference row and a review-queue item, never silently used as
online reward supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llin_verl.boss_pi_contract import contract_hashes, load_boss_pi_contract, sha256_text
from llin_verl.pi_reward import execute_readonly_sql
from scripts.prepare_pi_formal_dataset import gold_supported_by_rows


CONTRACT_NAME = "boss-pi-aligned-grpo-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceSpec:
    split: str
    label: str
    conversations: Path
    manifest: Path


def parse_source_values(conversations: list[str], manifests: list[str]) -> list[SourceSpec]:
    def parse(values: list[str], name: str) -> dict[tuple[str, str], Path]:
        result: dict[tuple[str, str], Path] = {}
        for raw in values:
            if "=" not in raw:
                raise ValueError(f"{name} must use SPLIT:LABEL=PATH: {raw!r}")
            identity, path = raw.split("=", 1)
            if ":" not in identity:
                raise ValueError(f"{name} must use SPLIT:LABEL=PATH: {raw!r}")
            split, label = identity.split(":", 1)
            if split not in {"train", "val", "test"} or not label:
                raise ValueError(f"invalid source identity: {identity!r}")
            key = (split, label)
            if key in result:
                raise ValueError(f"duplicate {name}: {identity}")
            result[key] = Path(path)
        return result

    conversation_map = parse(conversations, "--conversation")
    manifest_map = parse(manifests, "--manifest")
    if conversation_map.keys() != manifest_map.keys():
        raise ValueError("conversation and manifest source identities must match exactly")
    return [
        SourceSpec(split, label, conversation_map[(split, label)], manifest_map[(split, label)])
        for split, label in sorted(conversation_map)
    ]


def parse_pilot_sizes(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--pilot-size must use SPLIT=COUNT: {raw!r}")
        split, count = raw.split("=", 1)
        if split not in {"train", "val", "test"} or int(count) < 0:
            raise ValueError(f"invalid pilot size: {raw!r}")
        result[split] = int(count)
    return result


def first_user(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            value = message["content"].strip()
            if value:
                return value
    raise ValueError("conversation has no non-empty user instruction")


def instruction_belongs_to_task(instruction: str, row: dict[str, Any], guidance_prefix: str) -> bool:
    visible = instruction
    if guidance_prefix and visible.startswith(guidance_prefix):
        visible = visible[len(guidance_prefix) :]
    allowed = []
    for field in ("natural_language_instruction", "instruction"):
        if isinstance(row.get(field), str) and row[field].strip():
            allowed.append(row[field])
    for variant in row.get("instruction_variants") or []:
        if isinstance(variant, dict) and isinstance(variant.get("text"), str):
            allowed.append(variant["text"])
    return normalize_text(visible) in {normalize_text(value) for value in allowed}


def canonicalize_conversation(
    row: dict[str, Any],
    system_prompt: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("conversation is missing messages")
    copied = [dict(message) for message in messages if isinstance(message, dict)]
    if not copied:
        raise ValueError("conversation has no valid messages")
    # Boss fix_sft_data_system.py defines this exact canonicalization.  This is
    # source-owned data, not the old project fallback.
    if copied[0].get("role") == "system":
        copied[0] = {**copied[0], "content": system_prompt}
    else:
        copied.insert(0, {"role": "system", "content": system_prompt})
    return {
        **{key: value for key, value in row.items() if key not in {"messages", "tools"}},
        "messages": copied,
        "tools": tools,
    }


def review_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("source_label") or ""), str(row.get("task_id") or ""))
        if not all(key):
            raise ValueError("alignment review requires source_label and task_id")
        if key in result:
            raise ValueError(f"duplicate alignment review: {key}")
        result[key] = row
    return result


def review_approves(
    review: dict[str, Any] | None,
    instruction_sha256: str,
    gold_sha256: str,
) -> tuple[bool, str]:
    if review is None:
        return False, "missing_alignment_review"
    if review.get("approved_for_grpo") is not True:
        return False, "alignment_not_approved"
    if review.get("instruction_sha256") != instruction_sha256:
        return False, "review_instruction_hash_mismatch"
    if review.get("gold_sha256") != gold_sha256:
        return False, "review_gold_hash_mismatch"
    if not str(review.get("reviewer") or "").strip() or not str(review.get("reviewed_at") or "").strip():
        return False, "review_identity_missing"
    return True, "approved"


def build_grpo_record(candidate: dict[str, Any], system_prompt: str, tool_names: list[str], index: int) -> dict[str, Any]:
    gold = candidate["gold"]
    ground_truth = {
        "verifier_id": candidate["verifier_id"],
        "task_id": candidate["task_id"],
        "environment_id": candidate["environment_id"],
        "answer_type": gold["answer_type"],
        "expected_value_json": canonical_json(gold["value"]),
        "verification_sql": gold["verification_sql"],
        "required_tables": candidate["required_tables"],
        "must_use_fields": candidate["must_use_fields"],
        "task_family": "dwh",
        "reward_contract": "boss-primary-70-strict-evidence-30-v1",
        "abs_tol": 1e-3,
        "rel_tol": 1e-5,
    }
    return {
        "data_source": "boss_pi_aligned_v1",
        "agent_name": "pi_agent",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": candidate["instruction"]},
        ],
        "ability": "boss_pi_dwh",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": index,
            "split": candidate["split"],
            "source_label": candidate["source_label"],
            "verifier_id": candidate["verifier_id"],
            "environment_id": candidate["environment_id"],
            "instruction_sha256": candidate["instruction_sha256"],
            "gold_sha256": candidate["gold_sha256"],
            "alignment_reviewed": True,
            "response_messages_in_grpo_input": 0,
            "need_tools_kwargs": True,
            "tool_selection": tool_names,
            "tools_kwargs": {
                name: {"create_kwargs": {"environment_id": candidate["environment_id"]}}
                for name in tool_names
            },
        },
    }


def collect_source(
    spec: SourceSpec,
    sandbox_root: Path,
    system_prompt: str,
    tools: list[dict[str, Any]],
    reviews: dict[tuple[str, str], dict[str, Any]],
    guidance_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    conversations = [canonicalize_conversation(row, system_prompt, tools) for row in read_jsonl(spec.conversations)]
    by_instruction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for conversation in conversations:
        by_instruction[normalize_text(first_user(conversation["messages"]))].append(conversation)
        if str(conversation.get("task_id") or "").strip():
            by_task_id[str(conversation["task_id"]).strip()].append(conversation)

    candidates: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for manifest_row in read_jsonl(spec.manifest):
        if manifest_row.get("type") != "dwh":
            rejected["not_dwh_sft_only"] += 1
            continue
        task_id = str(manifest_row.get("task_id") or "").strip()
        manifest_instruction = str(manifest_row.get("instruction") or manifest_row.get("natural_language_instruction") or "").strip()
        matches = by_task_id.get(task_id, []) if task_id in by_task_id else by_instruction.get(normalize_text(manifest_instruction), [])
        if not task_id:
            rejected["missing_identity"] += 1
            continue
        if len(matches) != 1:
            rejected["conversation_match_missing" if not matches else "conversation_match_ambiguous"] += 1
            continue
        instruction = first_user(matches[0]["messages"])
        instruction_in_current_task = instruction_belongs_to_task(
            instruction,
            manifest_row,
            guidance_prefix,
        ) if task_id in by_task_id else True
        gold = manifest_row.get("gold_answer")
        if not isinstance(gold, dict) or gold.get("answer_type") not in {"numeric", "table"}:
            rejected["unsupported_or_missing_source_gold"] += 1
            continue
        sql = gold.get("verification_sql")
        if not isinstance(sql, str) or not sql.strip():
            rejected["missing_source_verification_sql"] += 1
            continue
        version = str(manifest_row.get("v") or spec.label)
        database = sandbox_root / "sft" / version / "logistics.sqlite"
        try:
            rows = execute_readonly_sql(database.resolve(strict=True), sql)
        except Exception:
            rejected["source_verification_sql_failed"] += 1
            continue
        if not rows or not gold_supported_by_rows(gold, rows):
            rejected["source_gold_result_mismatch"] += 1
            continue

        instruction_sha256 = canonical_hash(instruction)
        gold_value = {
            "answer_type": gold["answer_type"],
            "value": gold.get("value"),
            "verification_sql": sql,
        }
        gold_sha256 = canonical_hash(gold_value)
        approved, reason = review_approves(reviews.get((spec.label, task_id)), instruction_sha256, gold_sha256)
        reviewed_split = str((reviews.get((spec.label, task_id)) or {}).get("split") or spec.split)
        if reviewed_split not in {"train", "val", "test"}:
            approved, reason = False, "review_split_invalid"
        queue_row = {
            "source_label": spec.label,
            "split": reviewed_split,
            "task_id": task_id,
            "instruction": instruction,
            "instruction_sha256": instruction_sha256,
            "gold": gold_value,
            "gold_sha256": gold_sha256,
            "source_join_method": "task_id" if task_id in by_task_id else "exact_instruction",
            "source_instruction_in_current_task_definition": instruction_in_current_task,
            "approved_for_grpo": approved,
            "review_status": reason,
        }
        review_queue.append(queue_row)
        if not approved:
            rejected[reason] += 1
            continue
        candidate = {
            **queue_row,
            "source_label": spec.label,
            "source_manifest": spec.manifest.name,
            "source_conversation": spec.conversations.name,
            "environment_id": f"sft/{version}",
            "verifier_id": f"sft/{version}:{task_id}",
            "required_tables": sorted({str(value).casefold() for value in manifest_row.get("expected_tables") or []}),
            "must_use_fields": sorted(
                {
                    str(value).casefold()
                    for value in (manifest_row.get("verification_criteria") or {}).get("must_use_fields") or []
                }
            ),
        }
        candidates.append(candidate)
    return candidates, conversations, review_queue, rejected


def stable_select(rows: list[dict[str, Any]], count: int, seed: str) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"only {len(rows)} approved rows for pilot size {count}")
    return sorted(rows, key=lambda row: sha256_text(f"{seed}:{row['verifier_id']}"))[:count]


def build_dataset(
    specs: list[SourceSpec],
    sandbox_root: Path,
    reviews: dict[tuple[str, str], dict[str, Any]],
    pilot_sizes: dict[str, int],
    seed: str,
    contract: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tool_names = [item["function"]["name"] for item in contract["tools"]]
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sft_rows: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    rejected: dict[str, dict[str, int]] = {}
    source_files = []
    for spec in specs:
        candidates, conversations, queue, reasons = collect_source(
            spec,
            sandbox_root,
            contract["system_prompt"],
            contract["tools"],
            reviews,
            str(contract["runtime"].get("guidance_prefix") or ""),
        )
        for candidate in candidates:
            selected[candidate["split"]].append(candidate)
        sft_rows.extend(
            {
                "source_label": spec.label,
                "split": spec.split,
                "purpose": "sft_and_behavior_reference_only",
                **row,
            }
            for row in conversations
        )
        review_queue.extend(queue)
        rejected[f"{spec.split}:{spec.label}"] = dict(reasons)
        source_files.extend(
            [
                {"kind": "conversation", "split": spec.split, "label": spec.label, "path": str(spec.conversations), "sha256": file_sha256(spec.conversations)},
                {"kind": "manifest", "split": spec.split, "label": spec.label, "path": str(spec.manifest), "sha256": file_sha256(spec.manifest)},
            ]
        )

    mode = "pilot" if pilot_sizes else "full"
    if pilot_sizes:
        for split, count in pilot_sizes.items():
            selected[split] = stable_select(selected[split], count, seed + ":" + split)

    instruction_gold: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for split, rows in selected.items():
        for row in rows:
            instruction_gold[row["instruction_sha256"]][row["gold_sha256"]].add(
                f"{split}:{row['task_id']}"
            )
    conflicting_instruction_gold = {
        instruction_hash: {
            gold_hash: sorted(task_ids) for gold_hash, task_ids in sorted(gold_tasks.items())
        }
        for instruction_hash, gold_tasks in sorted(instruction_gold.items())
        if len(gold_tasks) > 1
    }
    if conflicting_instruction_gold:
        raise ValueError(
            "approved tasks contain identical instructions with conflicting gold labels: "
            + canonical_json(conflicting_instruction_gold)
        )

    task_ids: dict[str, set[str]] = {
        split: {row["task_id"] for row in rows} for split, rows in selected.items()
    }
    instruction_hashes: dict[str, set[str]] = {
        split: {row["instruction_sha256"] for row in rows} for split, rows in selected.items()
    }
    overlaps = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps.append(
            {
                "splits": [left, right],
                "task_ids": len(task_ids.get(left, set()) & task_ids.get(right, set())),
                "instruction_hashes": len(instruction_hashes.get(left, set()) & instruction_hashes.get(right, set())),
            }
        )
    if any(item["task_ids"] or item["instruction_hashes"] for item in overlaps):
        raise ValueError(f"boss-aligned split leakage: {overlaps}")

    records: dict[str, list[dict[str, Any]]] = {}
    index = 0
    for split in ("train", "val", "test"):
        records[split] = []
        for candidate in selected.get(split, []):
            records[split].append(build_grpo_record(candidate, contract["system_prompt"], tool_names, index))
            index += 1
    report = {
        "contract": CONTRACT_NAME,
        "mode": mode,
        "seed": seed,
        "source_files": source_files,
        "boss_contract": {**contract_hashes(contract), "source": contract["source"]},
        "selected": {split: len(records[split]) for split in ("train", "val", "test")},
        "sft_reference_rows": len(sft_rows),
        "alignment_review_rows": len(review_queue),
        "rejected": rejected,
        "invariants": {
            "uses_all_approved_by_default": mode == "full",
            "project_system_fallback_count": 0,
            "project_tool_schema_fallback_count": 0,
            "generated_instruction_count": 0,
            "generated_gold_or_sql_count": 0,
            "conflicting_instruction_gold_count": 0,
            "unreviewed_grpo_count": 0,
            "assistant_or_tool_messages_in_grpo_input": 0,
            "source_responses_exported_only_for_sft_and_regression": True,
            "split_overlap": overlaps,
        },
        "runtime_security_delta": contract["runtime"]["known_security_delta"],
    }
    return records, sft_rows, review_queue, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation", action="append", default=[], help="SPLIT:LABEL=PATH")
    parser.add_argument("--manifest", action="append", default=[], help="SPLIT:LABEL=PATH")
    parser.add_argument("--alignment-review", type=Path, required=True)
    parser.add_argument("--sandbox-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-size", action="append", default=[], help="SPLIT=COUNT; any use marks output pilot")
    parser.add_argument("--seed", default="boss-pi-aligned-v1-20260804")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    specs = parse_source_values(args.conversation, args.manifest)
    if not specs:
        raise ValueError("at least one boss conversation/manifest source is required")
    contract = load_boss_pi_contract()
    reviews = review_index(read_jsonl(args.alignment_review))
    records, sft_rows, review_queue, report = build_dataset(
        specs,
        args.sandbox_root,
        reviews,
        parse_pilot_sizes(args.pilot_size),
        args.seed,
        contract,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    Dataset = None
    if any(records[split] for split in ("train", "val", "test")):
        from datasets import Dataset as _Dataset

        Dataset = _Dataset
    for split in ("train", "val", "test"):
        if records[split]:
            path = args.output_dir / f"boss_pi_{split}.parquet"
            assert Dataset is not None
            Dataset.from_list(records[split]).to_parquet(str(path))
            artifacts[path.name] = {"sha256": file_sha256(path), "rows": len(records[split]), "purpose": "grpo"}
    sft_path = args.output_dir / "boss_pi_sft_reference.jsonl"
    review_path = args.output_dir / "boss_pi_alignment_review_queue.jsonl"
    write_jsonl(sft_path, sft_rows)
    write_jsonl(review_path, review_queue)
    artifacts[sft_path.name] = {"sha256": file_sha256(sft_path), "rows": len(sft_rows), "purpose": "sft_and_regression"}
    artifacts[review_path.name] = {"sha256": file_sha256(review_path), "rows": len(review_queue), "purpose": "human_alignment_review"}
    report["artifacts"] = artifacts
    report_path = args.output_dir / "boss_alignment_contract.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
