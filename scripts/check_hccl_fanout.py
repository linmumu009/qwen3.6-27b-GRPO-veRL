#!/usr/bin/env python3
"""Validate the 1 trainer -> 8 rollout HCCL topology used by TP=8."""

from __future__ import annotations

import datetime
import json
import os
import socket

import ray
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


MASTER_ADDR = "192.168.202.5"
MASTER_PORT = 28410
WORLD_SIZE = 9


def run_rank(rank: int) -> dict[str, object]:
    dist.init_process_group(
        "hccl",
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=datetime.timedelta(minutes=5),
    )
    torch.npu.set_device(0)
    value = torch.tensor(float(rank + 1), device="npu")
    dist.all_reduce(value)
    result = {
        "rank": rank,
        "host": socket.gethostname(),
        "ip": ray.util.get_node_ip_address(),
        "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        "all_reduce": value.item(),
    }
    dist.destroy_process_group()
    return result


trainer_rank = ray.remote(
    num_cpus=1,
    resources={"NPU": 1, "llin_trainer": 0.0001},
)(run_rank)
rollout_rank = ray.remote(
    num_cpus=1,
    resources={"NPU": 1, "llin_rollout": 0.0001},
)(run_rank)


def main() -> None:
    ray.init(address=os.getenv("RAY_ADDRESS", "192.168.202.5:26379"))
    refs = [trainer_rank.remote(0)]
    refs.extend(rollout_rank.remote(rank) for rank in range(1, WORLD_SIZE))
    results = sorted(ray.get(refs), key=lambda item: int(item["rank"]))
    expected = sum(range(1, WORLD_SIZE + 1))
    if any(item["all_reduce"] != expected for item in results):
        raise RuntimeError(f"unexpected all-reduce result; expected {expected}: {results}")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
