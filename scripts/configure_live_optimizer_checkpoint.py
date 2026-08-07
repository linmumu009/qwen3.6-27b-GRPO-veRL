#!/usr/bin/env python3
"""Inspect or update checkpoint contents on a live veRL Ray worker group."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


TARGET_CONTENTS = ["model", "optimizer", "extra"]


def configure_worker_checkpoint_contents(worker: Any, apply: bool = False) -> dict[str, Any]:
    """Inspect every engine in one colocated WorkerDict and optionally enable optimizer saves."""
    engines: list[dict[str, Any]] = []
    inspected_roles: list[dict[str, Any]] = []
    for role, role_worker in getattr(worker, "worker_dict", {}).items():
        candidates = [role_worker]
        candidates.extend(
            value
            for value in vars(role_worker).values()
            if value is not None and hasattr(value, "__dict__")
        )
        manager = None
        manager_owner = None
        for candidate in candidates:
            engine = getattr(candidate, "engine", None)
            possible = getattr(engine, "checkpoint_mananager", None)
            if possible is not None:
                manager = possible
                manager_owner = type(candidate).__name__
                break
        if manager is None:
            inspected_roles.append(
                {
                    "role": str(role),
                    "worker_type": type(role_worker).__name__,
                    "attributes": sorted(vars(role_worker)),
                }
            )
            continue
        before = list(manager.checkpoint_save_contents)
        if apply:
            manager.checkpoint_save_contents = list(TARGET_CONTENTS)
        after = list(manager.checkpoint_save_contents)
        engines.append(
            {
                "role": str(role),
                "manager_owner": manager_owner,
                "before": before,
                "after": after,
            }
        )
    return {
        "rank": int(os.environ.get("RANK", "-1")),
        "worker_group_prefix": os.environ.get("WG_PREFIX"),
        "engines": engines,
        "inspected_roles": inspected_roles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="192.168.202.5:26379")
    parser.add_argument("--expected-workers", type=int, default=16)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import ray

    ray.init(address=args.ray_address)
    named = ray.util.list_named_actors(all_namespaces=True)
    worker_infos = sorted(
        (
            info
            for info in named
            if "WorkerDict" in str(info.get("name", ""))
        ),
        key=lambda info: (str(info.get("namespace", "")), str(info.get("name", ""))),
    )
    if len(worker_infos) != args.expected_workers:
        raise RuntimeError(
            f"expected {args.expected_workers} live WorkerDict actors, found {len(worker_infos)}: {worker_infos}"
        )

    handles = [
        ray.get_actor(name=info["name"], namespace=info.get("namespace"))
        for info in worker_infos
    ]
    results = ray.get(
        [
            handle.execute_with_func_generator.remote(
                configure_worker_checkpoint_contents,
                args.apply,
            )
            for handle in handles
        ]
    )
    results = sorted(results, key=lambda item: item["rank"])
    failures = [
        item
        for item in results
        if len(item["engines"]) != 1
        or (args.apply and item["engines"][0]["after"] != TARGET_CONTENTS)
    ]
    payload = {
        "applied": args.apply,
        "expected_workers": args.expected_workers,
        "worker_count": len(results),
        "target_contents": TARGET_CONTENTS,
        "valid": not failures,
        "failures": failures,
        "workers": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
