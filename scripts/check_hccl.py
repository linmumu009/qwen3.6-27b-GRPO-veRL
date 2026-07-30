#!/usr/bin/env python3
"""Verify a two-host HCCL all-reduce on the pinned Ray resources."""

from __future__ import annotations

import datetime
import json
import os
import socket

import ray


@ray.remote(num_cpus=1, resources={"NPU": 1, "llin_trainer": 0.0001})
def trainer_rank(master_address: str) -> dict:
    return run_rank(rank=0, master_address=master_address)


@ray.remote(num_cpus=1, resources={"NPU": 1, "llin_rollout": 0.0001})
def rollout_rank(master_address: str) -> dict:
    return run_rank(rank=1, master_address=master_address)


def run_rank(rank: int, master_address: str) -> dict:
    import torch
    import torch.distributed as dist
    import torch_npu

    del torch_npu
    torch.npu.set_device(0)
    dist.init_process_group(
        backend="hccl",
        init_method=master_address,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(minutes=5),
    )
    value = torch.tensor([rank + 1.0], device="npu")
    dist.all_reduce(value)
    torch.npu.synchronize()
    result = {
        "rank": rank,
        "host": socket.gethostname(),
        "ip": ray.util.get_node_ip_address(),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "all_reduce": float(value.cpu().item()),
    }
    dist.destroy_process_group()
    return result


def main() -> None:
    ray.init(address="auto")
    address = "tcp://192.168.202.5:28400"
    results = ray.get([trainer_rank.remote(address), rollout_rank.remote(address)])
    if any(result["all_reduce"] != 3.0 for result in results):
        raise SystemExit(f"HCCL all-reduce mismatch: {results}")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
