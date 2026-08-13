#!/usr/bin/env python3
"""Copy a node-local artifact back through Ray's object store."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def validate_payload(payload: bytes, expected_jsonl_rows: int | None = None) -> dict[str, Any]:
    if not payload:
        raise ValueError("source artifact is empty")
    result: dict[str, Any] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if expected_jsonl_rows is not None:
        lines = [line for line in payload.splitlines() if line.strip()]
        if len(lines) != expected_jsonl_rows:
            raise ValueError(
                f"expected {expected_jsonl_rows} JSONL rows, found {len(lines)}"
            )
        for index, line in enumerate(lines, start=1):
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {index}: {exc}") from exc
        result["jsonl_rows"] = len(lines)
    return result


def write_atomic(path: Path, payload: bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.ray-copy.tmp")
    temporary.write_bytes(payload)
    if mode is not None:
        temporary.chmod(mode)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource", default="llin_rollout")
    parser.add_argument("--ray-address", default="192.168.202.5:26379")
    parser.add_argument("--expected-jsonl-rows", type=int)
    parser.add_argument(
        "--mode",
        type=lambda value: int(value, 8),
        help="optional octal output mode, for example 0600",
    )
    args = parser.parse_args()

    import ray

    @ray.remote(num_cpus=0, resources={args.resource: 0.0001})
    def read_node_local_file(source: str) -> bytes:
        return Path(source).read_bytes()

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    payload = ray.get(read_node_local_file.remote(str(args.source)))
    summary = validate_payload(payload, args.expected_jsonl_rows)
    write_atomic(args.output, payload, args.mode)
    summary.update(
        {
            "source": str(args.source),
            "output": str(args.output),
            "resource": args.resource,
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
