#!/usr/bin/env python3
"""Fail closed unless the isolated Qwen3.8 smoke Ray cluster is exactly 16+16."""

from __future__ import annotations

import argparse
import json


def safe_node_rows(nodes: list[dict]) -> list[dict[str, object]]:
    rows = []
    for node in nodes:
        if not node.get("Alive"):
            continue
        resources = node.get("Resources", {})
        rows.append(
            {
                "ip": str(node.get("NodeManagerAddress")),
                "npu": int(float(resources.get("NPU", 0))),
                "trainer": int(float(resources.get("llin_trainer", 0))),
                "rollout": int(float(resources.get("llin_rollout", 0))),
            }
        )
    return sorted(rows, key=lambda row: str(row["ip"]))


def validate_rows(rows: list[dict[str, object]]) -> None:
    expected = [
        {"ip": "192.168.202.4", "npu": 16, "trainer": 0, "rollout": 1},
        {"ip": "192.168.202.5", "npu": 16, "trainer": 1, "rollout": 0},
    ]
    if rows != expected:
        raise ValueError(f"isolated Ray topology mismatch: {rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="192.168.202.5:36379")
    args = parser.parse_args()

    import ray

    ray.init(address=args.address, ignore_reinit_error=True)
    rows = safe_node_rows(ray.nodes())
    validate_rows(rows)
    print(json.dumps({"contract": "llin-qwen38-smoke-ray-v1", "nodes": rows}))


if __name__ == "__main__":
    main()
