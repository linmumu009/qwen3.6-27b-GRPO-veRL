#!/usr/bin/env python3
"""Patch veRL's vLLM weight-sync IPC rank mapping for Ascend DP replicas."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """\
    tp_size = max(int(getattr(parallel_config, "tensor_parallel_size", 1) or 1), 1)
    dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
    dp_local_size = int(getattr(parallel_config, "data_parallel_size_local", 1) or 1)
    if dp_size <= 1 and dp_local_size <= 1:
        return worker_local_rank
"""

NEW = """\
    tp_size = max(int(getattr(parallel_config, "tensor_parallel_size", 1) or 1), 1)

    # vLLM's multiprocessing DP backend narrows each EngineCore child to one
    # TP-sized device slice and resets its ParallelConfig DP fields to 1/0.
    # On Ascend that made every DP replica reuse IPC ranks 0..TP-1, while the
    # checkpoint senders use node-local ranks 0..(TP*DP)-1.  The runtime
    # visible-device slice preserves the missing DP offset (for example,
    # 0..7 for DP0 and 8..15 for DP1), so use it before the generic DP fields.
    ascend_visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if ascend_visible:
        visible_ranks = [item.strip() for item in ascend_visible.split(",") if item.strip()]
        if len(visible_ranks) >= tp_size and all(item.isdigit() for item in visible_ranks):
            return int(visible_ranks[worker_local_rank % len(visible_ranks)])

    dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
    dp_local_size = int(getattr(parallel_config, "data_parallel_size_local", 1) or 1)
    if dp_size <= 1 and dp_local_size <= 1:
        return worker_local_rank
"""


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        return "already-patched"
    if OLD not in text:
        raise RuntimeError(f"expected veRL rank-mapping block not found in {path}")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/verl/verl/workers/rollout/vllm_rollout/utils.py",
    )
    args = parser.parse_args()
    target = Path(args.target)
    print(f"{patch(target)}: {target}")


if __name__ == "__main__":
    main()
