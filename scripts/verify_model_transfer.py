#!/usr/bin/env python3
"""Build or verify an exact file-level SHA256 manifest for an HF model tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CONTRACT = "llin-hf-model-file-transfer-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(model_dir: Path) -> list[dict[str, object]]:
    root = model_dir.resolve(strict=True)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        raise ValueError("model tree is empty")
    return files


def build(model_dir: Path, manifest_path: Path) -> dict[str, object]:
    files = inventory(model_dir)
    payload = {
        "contract": CONTRACT,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "contains_server_paths": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return payload


def verify(model_dir: Path, manifest_path: Path) -> dict[str, object]:
    expected = json.loads(manifest_path.resolve(strict=True).read_text(encoding="utf-8"))
    if expected.get("contract") != CONTRACT:
        raise ValueError("transfer manifest contract mismatch")
    observed = inventory(model_dir)
    if observed != expected.get("files"):
        raise ValueError("transferred model file inventory or SHA256 mismatch")
    return {
        "contract": CONTRACT,
        "status": "passed",
        "file_count": len(observed),
        "total_bytes": sum(int(item["bytes"]) for item in observed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify"):
        child = sub.add_parser(command)
        child.add_argument("--model-dir", type=Path, required=True)
        child.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.model_dir, args.manifest) if args.command == "build" else verify(args.model_dir, args.manifest)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
