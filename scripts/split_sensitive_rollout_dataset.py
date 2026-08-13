#!/usr/bin/env python3
"""Split a sensitive rollout Parquet into two disjoint server-only arms."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(row: dict) -> str:
    return str(row["extra_info"]["verifier_id"])


def write_private(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def split_dataset(source: Path, output_dir: Path, *, expected_rows: int) -> dict:
    rows = pq.read_table(source).to_pylist()
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(rows)}")
    identities = [identity(row) for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("source verifier identities are not unique")

    arms = {"m05": rows[0::2], "m06": rows[1::2]}
    outputs = {}
    union = set()
    for name, arm_rows in arms.items():
        arm_ids = {identity(row) for row in arm_rows}
        if union & arm_ids:
            raise ValueError("split arms overlap")
        union |= arm_ids
        path = output_dir / f"boss_multisandbox_dwh_{name}.sensitive.parquet"
        write_private(path, arm_rows)
        outputs[name] = {
            "rows": len(arm_rows),
            "sha256": sha256(path),
            "versions": dict(
                sorted(Counter(str(row["extra_info"]["source_version"]) for row in arm_rows).items())
            ),
        }
    if union != set(identities):
        raise ValueError("split union does not equal source")

    manifest = {
        "contract": "boss-multisandbox-dwh-dual-server-split-v1",
        "source_rows": len(rows),
        "source_sha256": sha256(source),
        "arms": outputs,
        "disjoint": True,
        "complete_union": True,
        "samples_per_task": 8,
        "contains_prompts_gold_sql_task_ids_or_server_paths": False,
        "training_allowed": False,
        "rollout_screening_allowed": True,
    }
    manifest_path = output_dir / "dual_server_split_safe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=281)
    args = parser.parse_args()
    print(
        json.dumps(
            split_dataset(args.source, args.output_dir, expected_rows=args.expected_rows),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
