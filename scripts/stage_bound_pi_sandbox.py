#!/usr/bin/env python3
"""Stage only task-bound PI environments into a private run-local sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pyarrow.parquet as pq

from llin_verl.pi_tool_contract import ENVIRONMENT_PATTERN


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_modes(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)


def _rows(paths: list[Path]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in paths:
        values.extend(pq.read_table(path).to_pylist())
    return values


def stage(
    datasets: list[Path],
    source_root: Path,
    output_root: Path,
    safe_summary: Path,
) -> dict[str, Any]:
    rows = _rows(datasets)
    source = source_root.resolve(strict=True)
    environments: dict[str, set[str]] = {}
    for row in rows:
        truth = ((row.get("reward_model") or {}).get("ground_truth") or {})
        extra = row.get("extra_info") or {}
        environment = str(truth.get("environment_id") or extra.get("environment_id") or "")
        identity = str(extra.get("instruction_sha256") or "")
        if not ENVIRONMENT_PATTERN.fullmatch(environment):
            raise ValueError("dataset contains a missing or invalid environment identity")
        if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity.casefold()):
            raise ValueError("dataset contains a missing or invalid instruction identity")
        environments.setdefault(environment, set()).add(identity.casefold())

    temporary = output_root.with_name(output_root.name + ".staging")
    if output_root.exists() or temporary.exists():
        raise FileExistsError("private PI sandbox staging target already exists")
    temporary.mkdir(parents=True, mode=0o700)
    manifest: list[dict[str, str]] = []
    try:
        for environment in sorted(environments):
            source_environment = (source / environment).resolve(strict=True)
            source_environment.relative_to(source)
            database = source_environment / "logistics.sqlite"
            if not database.is_file() or database.stat().st_size <= 0:
                raise FileNotFoundError("bound environment is missing logistics.sqlite")
            if any(path.is_symlink() for path in source_environment.rglob("*")):
                raise ValueError("bound environment contains a symlink")
            destination = (temporary / environment).resolve()
            destination.relative_to(temporary.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copytree(source_environment, destination, copy_function=shutil.copy2)
            manifest.append(
                {
                    "environment_identity_sha256": hashlib.sha256(
                        environment.encode("utf-8")
                    ).hexdigest(),
                    "database_sha256": _file_sha256(database),
                }
            )
        _private_modes(temporary)
        temporary.rename(output_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    compound = hashlib.sha256(
        "\n".join(
            f"{item['environment_identity_sha256']} {item['database_sha256']}"
            for item in manifest
        ).encode("ascii")
    ).hexdigest()
    result = {
        "schema_version": "run-local-bound-pi-sandbox-v1",
        "dataset_rows": len(rows),
        "unique_instruction_identities": len(
            {identity for values in environments.values() for identity in values}
        ),
        "unique_environments": len(environments),
        "database_files": len(manifest),
        "database_manifest_compound_sha256": compound,
        "source_membership": "exact_dataset_environment_union",
        "private_mode": "directories0700_files0600",
        "sensitive_values_emitted": False,
    }
    safe_summary.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    safe_summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(safe_summary, 0o600)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--safe-summary", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            stage(args.dataset, args.source_root, args.output_root, args.safe_summary),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
