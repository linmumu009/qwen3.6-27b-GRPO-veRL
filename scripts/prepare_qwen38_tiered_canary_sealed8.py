from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.outcome_gated_contract import evidence_binding_hash


APPROVED43_SHA256 = "d86b53d906806b150d43a508dce9b0dd6d05105c07e03961e8e7bf9439ccd944"
RAW100_SHA256 = "c0befda32166340bf68e6b948a1e8fcc6f8f0887d7a5f38a4e6b1051b8f9f7af"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _instruction_id(row: dict[str, Any]) -> str:
    return str((row.get("extra_info") or {}).get("instruction_sha256") or "")


def _answer_type(row: dict[str, Any]) -> str:
    return str(((row.get("reward_model") or {}).get("ground_truth") or {}).get("answer_type") or "").casefold()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(
    approved43: Path,
    raw100: Path,
    output: Path,
    safe_summary: Path,
    tasks: Path | None = None,
    database_root: str = "/pi_sandbox",
) -> dict[str, Any]:
    if file_sha256(approved43) != APPROVED43_SHA256:
        raise ValueError("approved43 parquet hash mismatch")
    if file_sha256(raw100) != RAW100_SHA256:
        raise ValueError("raw100 parquet hash mismatch")

    approved_rows = pq.read_table(approved43).to_pylist()
    raw_rows = pq.read_table(raw100).to_pylist()
    task_rows = _read_jsonl(tasks) if tasks is not None else []
    approved_ids = {_instruction_id(row) for row in approved_rows}
    raw_ids = [_instruction_id(row) for row in raw_rows]
    if len(approved_rows) != 43 or len(approved_ids) != 43 or "" in approved_ids:
        raise ValueError("approved43 must contain 43 unique instruction identities")
    if len(raw_rows) != 100 or len(set(raw_ids)) != 100 or "" in raw_ids:
        raise ValueError("raw100 must contain 100 unique instruction identities")
    if not approved_ids <= set(raw_ids):
        raise ValueError("approved43 is not a subset of raw100")

    selected: list[dict[str, Any]] = []
    for kind in ("numeric", "table"):
        candidates = sorted(
            (row for row in raw_rows if _instruction_id(row) not in approved_ids and _answer_type(row) == kind),
            key=_instruction_id,
        )
        if len(candidates) < 4:
            raise ValueError(f"not enough disjoint {kind} sealed candidates")
        for row in candidates[:4]:
            copied = json.loads(json.dumps(row, ensure_ascii=False))
            extra = dict(copied.get("extra_info") or {})
            extra.update(
                {
                    "training_allowed": False,
                    "sealed_evaluation_only": True,
                    "approved43_authorization": False,
                    "pi_reward_database_root": str(database_root),
                }
            )
            copied["extra_info"] = extra
            if task_rows:
                source_index = int(extra.get("global_index", -1))
                if not 0 <= source_index < len(task_rows):
                    raise ValueError("sealed task global_index is outside frozen tasks file")
                task = task_rows[source_index]
                truth = (copied.get("reward_model") or {}).get("ground_truth") or {}
                criteria = task.get("verification_criteria") or {}
                truth["evidence_plan"] = task.get("evidence_plan") or {}
                truth["required_tables"] = task.get("expected_tables") or truth.get(
                    "required_tables", []
                )
                truth["must_use_fields"] = criteria.get("must_use_fields") or truth.get(
                    "must_use_fields", []
                )
                binding = evidence_binding_hash(truth)
                truth["process_evidence_binding_sha256"] = binding
                extra["process_evidence_binding_sha256"] = binding
            selected.append(copied)

    selected.sort(key=lambda row: (_answer_type(row), _instruction_id(row)))
    selected_ids = [_instruction_id(row) for row in selected]
    if len(selected) != 8 or len(set(selected_ids)) != 8 or approved_ids & set(selected_ids):
        raise AssertionError("sealed8 identity/disjointness invariant failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    safe_summary.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected), output)
    os.chmod(output, 0o600)
    identity_set_sha256 = hashlib.sha256("\n".join(sorted(selected_ids)).encode("ascii")).hexdigest()
    summary = {
        "schema_version": "qwen38-tiered-canary-sealed8-v1",
        "raw100_sha256": RAW100_SHA256,
        "approved43_sha256": APPROVED43_SHA256,
        "rows": 8,
        "unique_instruction_identities": 8,
        "answer_type_counts": {kind: sum(_answer_type(row) == kind for row in selected) for kind in ("numeric", "table")},
        "approved43_overlap": 0,
        "training_allowed_true": 0,
        "identity_set_sha256": identity_set_sha256,
        "sealed_parquet_sha256": file_sha256(output),
        "training_use_allowed": False,
        "process_binding_enriched": bool(task_rows),
        "run_local_database_root_configured": bool(database_root),
    }
    safe_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(safe_summary, 0o600)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic disjoint sealed8 for the Qwen3.8 tiered canary")
    parser.add_argument("--approved43", type=Path, required=True)
    parser.add_argument("--raw100", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-summary", type=Path, required=True)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--database-root", default="/pi_sandbox")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.approved43,
                args.raw100,
                args.output,
                args.safe_summary,
                args.tasks,
                args.database_root,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
