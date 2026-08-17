#!/usr/bin/env python3
"""Freeze an exact, private GRPO candidate pool from reviewed source Parquets.

The output remains sensitive: it contains prompts and hidden verifier material.
Only the companion safe summary is suitable for logs or Git.  The merger is
fail-closed on source counts, selectors, prompt/gold completeness, explicit
training permission, and cross-source task identity collisions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CONTRACT = "llin-grpo-candidate-pool-freeze-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"selector row {line_number} is not an object")
            rows.append(value)
    return rows


@dataclass(frozen=True)
class SourceSpec:
    label: str
    path: Path
    expected_rows: int
    selector_path: Path | None = None
    selector_field: str = "instruction_sha256"


def parse_source(value: str) -> SourceSpec:
    """Parse LABEL::PATH::COUNT[::SELECTOR_JSONL[::SELECTOR_FIELD]]."""

    parts = value.split("::")
    if len(parts) not in {3, 4, 5}:
        raise argparse.ArgumentTypeError(
            "source must be LABEL::PATH::COUNT[::SELECTOR_JSONL[::SELECTOR_FIELD]]"
        )
    label, raw_path, raw_count = parts[:3]
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("source label and path must be non-empty")
    try:
        expected_rows = int(raw_count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("source count must be an integer") from error
    if expected_rows <= 0:
        raise argparse.ArgumentTypeError("source count must be positive")
    selector_path = Path(parts[3]) if len(parts) >= 4 and parts[3] else None
    selector_field = parts[4] if len(parts) == 5 and parts[4] else "instruction_sha256"
    return SourceSpec(label, Path(raw_path), expected_rows, selector_path, selector_field)


def last_user_message(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        return ""
    users = [
        str(message.get("content") or "").strip()
        for message in prompt
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    return users[-1] if users else ""


def instruction_identity(row: dict[str, Any]) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        identity = str(extra.get("instruction_sha256") or "").strip()
        if identity:
            return identity
    identity = str(row.get("instruction_sha256") or "").strip()
    if identity:
        return identity
    instruction = last_user_message(row)
    if not instruction:
        raise ValueError("candidate row has no user prompt or instruction hash")
    return text_sha256(instruction)


def verifier_material(row: dict[str, Any]) -> Any:
    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict):
        return None
    ground_truth = reward_model.get("ground_truth")
    if isinstance(ground_truth, dict):
        return ground_truth if any(value not in (None, "", [], {}) for value in ground_truth.values()) else None
    return ground_truth if ground_truth not in (None, "", [], {}) else None


def explicitly_enabled(row: dict[str, Any], field: str) -> bool:
    if row.get(field) is True:
        return True
    extra = row.get("extra_info")
    return isinstance(extra, dict) and extra.get(field) is True


def selected_rows(spec: SourceSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = spec.path.resolve(strict=True)
    rows = pq.read_table(resolved).to_pylist()
    input_rows = len(rows)
    selector_sha256: str | None = None
    if spec.selector_path is not None:
        selector_resolved = spec.selector_path.resolve(strict=True)
        selector_rows = read_jsonl(selector_resolved)
        selector_values = [str(row.get(spec.selector_field) or "").strip() for row in selector_rows]
        if any(not value for value in selector_values):
            raise ValueError(f"{spec.label}: selector contains an empty identity")
        if len(selector_values) != len(set(selector_values)):
            raise ValueError(f"{spec.label}: selector identities are not unique")
        selectors = set(selector_values)
        by_identity: dict[str, dict[str, Any]] = {}
        for row in rows:
            identity = instruction_identity(row)
            if identity in by_identity:
                raise ValueError(f"{spec.label}: source instruction identities are not unique")
            by_identity[identity] = row
        missing = selectors - set(by_identity)
        if missing:
            raise ValueError(f"{spec.label}: {len(missing)} selector identities are absent")
        rows = [by_identity[value] for value in selector_values]
        selector_sha256 = file_sha256(selector_resolved)
    if len(rows) != spec.expected_rows:
        raise ValueError(
            f"{spec.label}: expected {spec.expected_rows} rows, found {len(rows)}"
        )
    return rows, {
        "input_rows": input_rows,
        "selected_rows": len(rows),
        "source_sha256": file_sha256(resolved),
        "selector_sha256": selector_sha256,
    }


def difficulty(row: dict[str, Any]) -> str:
    extra = row.get("extra_info")
    candidates = []
    if isinstance(extra, dict):
        candidates.extend(
            extra.get(key) for key in ("difficulty_level", "difficulty_band", "level")
        )
    candidates.extend(row.get(key) for key in ("difficulty_level", "difficulty_band", "level"))
    for value in candidates:
        if value not in (None, ""):
            return str(value)
    return "unknown"


def answer_type(row: dict[str, Any]) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, dict):
            value = str(ground_truth.get("answer_type") or "").strip()
            if value:
                return value
    return "unknown"


def freeze_pool(
    specs: list[SourceSpec],
    *,
    output_path: Path,
    safe_summary_path: Path,
    expected_total: int,
) -> dict[str, Any]:
    if not specs:
        raise ValueError("at least one source is required")
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError("source labels must be unique")

    frozen_rows: list[dict[str, Any]] = []
    source_summaries: dict[str, dict[str, Any]] = {}
    seen_identities: dict[str, str] = {}
    difficulty_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    for spec in specs:
        rows, source_summary = selected_rows(spec)
        source_summaries[spec.label] = source_summary
        for row in rows:
            if not last_user_message(row):
                raise ValueError(f"{spec.label}: candidate row has no non-empty user prompt")
            if verifier_material(row) is None:
                raise ValueError(f"{spec.label}: candidate row has no hidden verifier material")
            if explicitly_enabled(row, "training_allowed"):
                raise ValueError(f"{spec.label}: candidate row is already training-enabled")
            if explicitly_enabled(row, "promotion_allowed"):
                raise ValueError(f"{spec.label}: candidate row is already promotion-enabled")
            identity = instruction_identity(row)
            if identity in seen_identities:
                raise ValueError(
                    f"cross-source instruction collision: {seen_identities[identity]} and {spec.label}"
                )
            seen_identities[identity] = spec.label
            copied = dict(row)
            extra = dict(copied.get("extra_info") or {})
            extra.update(
                {
                    "candidate_pool_source": spec.label,
                    "candidate_pool_contract": CONTRACT,
                    "candidate_pool_frozen": True,
                    "training_allowed": False,
                    "promotion_allowed": False,
                }
            )
            copied["extra_info"] = extra
            if "training_allowed" in copied:
                copied["training_allowed"] = False
            if "promotion_allowed" in copied:
                copied["promotion_allowed"] = False
            frozen_rows.append(copied)
            difficulty_counts[difficulty(copied)] += 1
            answer_type_counts[answer_type(copied)] += 1

    if len(frozen_rows) != expected_total:
        raise ValueError(f"expected {expected_total} total rows, found {len(frozen_rows)}")
    if len(seen_identities) != expected_total:
        raise ValueError("global instruction identities are not unique")

    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(frozen_rows), temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(output_path)
    os.chmod(output_path, 0o600)
    output_sha256 = file_sha256(output_path)
    persisted = pq.read_table(output_path)
    if persisted.num_rows != expected_total:
        raise ValueError("persisted output row count changed after write")
    persisted_rows = persisted.to_pylist()
    persisted_source_counts = Counter(
        str((row.get("extra_info") or {}).get("candidate_pool_source") or "")
        for row in persisted_rows
    )
    expected_source_counts = Counter(
        {spec.label: spec.expected_rows for spec in specs}
    )
    if persisted_source_counts != expected_source_counts:
        raise ValueError("persisted output source counts changed after write")
    persisted_identities = set()
    for row in persisted_rows:
        if not last_user_message(row) or verifier_material(row) is None:
            raise ValueError("persisted output lost prompt or verifier material")
        if explicitly_enabled(row, "training_allowed") or explicitly_enabled(
            row, "promotion_allowed"
        ):
            raise ValueError("persisted output unexpectedly enables training or promotion")
        persisted_identities.add(instruction_identity(row))
    if len(persisted_identities) != expected_total:
        raise ValueError("persisted output instruction identities are not unique")

    summary = {
        "contract": CONTRACT,
        "candidate_rows": expected_total,
        "unique_instruction_identities": len(seen_identities),
        "source_counts": dict(expected_source_counts),
        "source_artifacts": source_summaries,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "answer_type_counts": dict(sorted(answer_type_counts.items())),
        "output_sha256": output_sha256,
        "sensitive_artifact_permissions": "0600",
        "training_allowed": False,
        "promotion_allowed": False,
        "requires_semantic_review_before_training": True,
        "contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths": False,
    }
    safe_summary_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    safe_temporary = safe_summary_path.with_name(
        safe_summary_path.name + f".tmp.{os.getpid()}"
    )
    safe_temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    safe_temporary.replace(safe_summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-summary", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    args = parser.parse_args()
    result = freeze_pool(
        args.source,
        output_path=args.output,
        safe_summary_path=args.safe_summary,
        expected_total=args.expected_total,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
