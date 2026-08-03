#!/usr/bin/env python3
"""Verify that formal PI train/val files are identical on both Ray roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_manifest(paths: list[str]) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        item: dict[str, object] = {"exists": path.is_file()}
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            item.update({"size": path.stat().st_size, "sha256": digest.hexdigest()})
        manifest[raw_path] = item
    return manifest


def check_manifests(manifests: dict[str, dict[str, dict[str, object]]]) -> None:
    roles = sorted(manifests)
    if len(roles) < 2:
        raise ValueError(f"expected at least two Ray roles, got: {roles}")
    reference_role = roles[0]
    reference = manifests[reference_role]
    for role, manifest in manifests.items():
        for path, item in manifest.items():
            if not item.get("exists"):
                raise FileNotFoundError(f"{role} cannot read formal PI data: {path}")
        if manifest != reference:
            raise ValueError(
                f"formal PI data mismatch between {reference_role} and {role}: "
                f"{reference} != {manifest}"
            )


def run_check(paths: list[str], ray_address: str) -> dict[str, dict[str, dict[str, object]]]:
    import ray

    ray.init(address=ray_address, ignore_reinit_error=True)
    remote_manifest = ray.remote(file_manifest)
    futures = {
        role: remote_manifest.options(resources={resource: 0.001}, num_cpus=0.01).remote(paths)
        for role, resource in (
            ("trainer", "llin_trainer"),
            ("rollout", "llin_rollout"),
        )
    }
    manifests = {role: ray.get(future) for role, future in futures.items()}
    check_manifests(manifests)
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--ray-address", default="192.168.202.5:26379")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_check([args.train_file, args.val_file], args.ray_address)
    print(json.dumps({"status": "passed", "roles": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
