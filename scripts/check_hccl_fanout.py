#!/usr/bin/env python3
"""Validate one trainer rank broadcasting to all rollout HCCL ranks."""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import socket

import ray
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


MASTER_ADDR = "192.168.202.5"
MASTER_PORT = int(os.getenv("MASTER_PORT", "28410"))
STATELESS_MASTER_PORT = int(os.getenv("STATELESS_MASTER_PORT", "28411"))
ROLLOUT_RANKS = int(os.getenv("ROLLOUT_RANKS", "16"))
WORLD_SIZE = 1 + ROLLOUT_RANKS
STATELESS_BYTES = int(os.getenv("STATELESS_BYTES", str(1024 * 1024)))


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
    broadcast_value = torch.tensor(-1.0, device="npu")
    if rank == 0:
        broadcast_value.fill_(123.5)
    dist.broadcast(broadcast_value, src=0)
    dist.destroy_process_group()

    # Exercise the exact vLLM Ascend stateless PyHCCL path used by veRL's
    # HCCLCheckpointEngine, not only torch.distributed ProcessGroupHCCL.
    import vllm

    if not hasattr(vllm, "__version__"):
        vllm.__version__ = importlib.metadata.version("vllm")
    from verl.utils.distributed import stateless_init_process_group

    pyhccl = stateless_init_process_group(
        MASTER_ADDR,
        STATELESS_MASTER_PORT,
        rank,
        WORLD_SIZE,
        torch.npu.current_device(),
    )
    signal = torch.tensor([1], dtype=torch.int8, device="npu")
    pyhccl.all_reduce(signal)
    stateless_bucket = torch.zeros(STATELESS_BYTES, dtype=torch.uint8, device="npu")
    if rank == 0:
        stateless_bucket.fill_(37)
    pyhccl.broadcast(stateless_bucket, src=0)
    result = {
        "rank": rank,
        "host": socket.gethostname(),
        "ip": ray.util.get_node_ip_address(),
        "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        "all_reduce": value.item(),
        "broadcast": broadcast_value.item(),
        "stateless_broadcast_first": stateless_bucket[0].item(),
        "stateless_broadcast_last": stateless_bucket[-1].item(),
    }
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
    if any(item["broadcast"] != 123.5 for item in results):
        raise RuntimeError(f"unexpected ProcessGroupHCCL broadcast result: {results}")
    if any(
        item["stateless_broadcast_first"] != 37 or item["stateless_broadcast_last"] != 37
        for item in results
    ):
        raise RuntimeError(f"unexpected stateless PyHCCL broadcast result: {results}")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
