#!/usr/bin/env python3
"""Fail closed when a PI-Agent dataset cannot see its runtime sandbox."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sqlite3
import sys

import pyarrow.parquet as pq


REQUIRED_RUNTIME_FILES = ("logistics.sqlite", "schema_dictionary.md")
REQUIRED_RUNTIME_DIRECTORIES = ("documents",)


def _environment_ids(dataset: Path) -> set[str]:
    if not dataset.is_file():
        raise FileNotFoundError("PI runtime preflight dataset is missing")
    rows = pq.read_table(dataset, columns=["extra_info"]).to_pylist()
    environment_ids: set[str] = set()
    for row in rows:
        extra_info = row.get("extra_info") or {}
        environment_id = extra_info.get("environment_id")
        if not isinstance(environment_id, str) or not environment_id.strip():
            raise ValueError("PI runtime preflight found a row without environment_id")
        relative = PurePosixPath(environment_id)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("PI runtime preflight found an unsafe environment_id")
        environment_ids.add(environment_id)
    if not rows or not environment_ids:
        raise ValueError("PI runtime preflight found no runtime environments")
    return environment_ids


def _database_has_relations(path: Path) -> bool:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        count = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchone()[0]
    finally:
        connection.close()
    return int(count) > 0


def validate_dataset_runtime_environments(dataset: Path, sandbox_root: Path) -> dict:
    """Verify every dataset environment under the exact tool-visible root.

    The returned payload is intentionally aggregate-only so it is safe to put
    in the standalone contract.  Environment identifiers and database content
    are never emitted.
    """

    root = sandbox_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError("PI runtime preflight sandbox root is not a directory")
    environment_ids = _environment_ids(dataset)
    missing_or_invalid = 0
    for environment_id in environment_ids:
        relative = PurePosixPath(environment_id)
        try:
            environment = root.joinpath(*relative.parts).resolve(strict=True)
            environment.relative_to(root)
            valid = environment.is_dir()
            valid = valid and all(
                (environment / name).is_file() and (environment / name).stat().st_size > 0
                for name in REQUIRED_RUNTIME_FILES
            )
            valid = valid and all(
                (environment / name).is_dir() for name in REQUIRED_RUNTIME_DIRECTORIES
            )
            valid = valid and _database_has_relations(environment / "logistics.sqlite")
        except (FileNotFoundError, NotADirectoryError, OSError, sqlite3.Error, ValueError):
            valid = False
        if not valid:
            missing_or_invalid += 1
    if missing_or_invalid:
        raise FileNotFoundError(
            "PI runtime preflight failed: "
            f"missing_or_invalid={missing_or_invalid}, environments={len(environment_ids)}"
        )
    return {
        "contract": "pi-agent-runtime-visibility-preflight-v1",
        "valid": True,
        "environment_count": len(environment_ids),
        "required_files": list(REQUIRED_RUNTIME_FILES),
        "required_directories": list(REQUIRED_RUNTIME_DIRECTORIES),
        "sqlite_opened_read_only": True,
        "sqlite_has_relations": True,
        "environment_ids_emitted": False,
    }


def validate_reward_entrypoint(module_path: Path, function_name: str) -> dict:
    """Import the exact reward module and require a callable entry point."""

    path = module_path.resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError("PI runtime preflight reward module is not a file")
    if not function_name.isidentifier():
        raise ValueError("PI runtime preflight reward function name is invalid")
    spec = importlib.util.spec_from_file_location("llin_pi_runtime_reward_preflight", path)
    if spec is None or spec.loader is None:
        raise ImportError("PI runtime preflight could not load the reward module")
    module = importlib.util.module_from_spec(spec)
    project_root = str(path.parent.parent)
    sys.path.insert(0, project_root)
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(project_root)
        except ValueError:
            pass
    reward_fn = getattr(module, function_name, None)
    if not callable(reward_fn):
        raise AttributeError(
            f"PI runtime preflight reward function is missing or not callable: {function_name}"
        )
    return {
        "reward_module_imported": True,
        "reward_function": function_name,
        "reward_function_callable": True,
        "reward_module_path_emitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sandbox-root", type=Path, required=True)
    parser.add_argument("--reward-path", type=Path)
    parser.add_argument("--reward-function")
    args = parser.parse_args()
    if (args.reward_path is None) != (args.reward_function is None):
        parser.error("--reward-path and --reward-function must be provided together")
    result = validate_dataset_runtime_environments(args.dataset, args.sandbox_root)
    if args.reward_path is not None:
        result.update(validate_reward_entrypoint(args.reward_path, args.reward_function))
        result["contract"] = "pi-agent-runtime-and-reward-preflight-v2"
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
