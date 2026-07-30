#!/usr/bin/env python3
"""Verify that Ray can place training and rollout work on the intended hosts."""

from __future__ import annotations

import json
import socket

import ray


@ray.remote(num_cpus=0, resources={"llin_trainer": 0.0001})
def trainer_location() -> dict[str, str]:
    return {"role": "trainer", "host": socket.gethostname(), "ip": ray.util.get_node_ip_address()}


@ray.remote(num_cpus=0, resources={"llin_rollout": 0.0001})
def rollout_location() -> dict[str, str]:
    return {"role": "rollout", "host": socket.gethostname(), "ip": ray.util.get_node_ip_address()}


def main() -> None:
    ray.init(address="auto")
    locations = ray.get([trainer_location.remote(), rollout_location.remote()])
    print(json.dumps(locations, ensure_ascii=False))
    expected = {"trainer": "192.168.202.5", "rollout": "192.168.202.4"}
    for location in locations:
        if location["ip"] != expected[location["role"]]:
            raise SystemExit(f"role placement mismatch: {location}")


if __name__ == "__main__":
    main()
