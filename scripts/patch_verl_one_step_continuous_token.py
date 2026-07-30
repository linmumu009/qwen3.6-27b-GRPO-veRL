#!/usr/bin/env python3
"""Enable Continuous Token for text-only One-Step-Off-Policy datasets."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """\
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = create_resource_pool_manager(config, role_worker_mapping.keys())
"""

NEW = """\
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        # Some text-only Qwen checkpoints expose a vision-capable AutoProcessor.
        # AgentLoop disables Continuous Token whenever a processor is present,
        # even when this run explicitly excludes multimodal inputs.  Select the
        # tokenizer path for that safe, text-only combination.
        continuous_token = config.data.get("continuous_token", {})
        if continuous_token.get("enable", False) and not config.data.get("return_multi_modal_inputs", True):
            processor = None

        resource_pool_manager = create_resource_pool_manager(config, role_worker_mapping.keys())
"""


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        return "already-patched"
    if OLD not in text:
        raise RuntimeError(f"expected One-Step-Off-Policy processor block not found in {path}")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/verl/verl/experimental/one_step_off_policy/main_ppo.py",
    )
    args = parser.parse_args()
    target = Path(args.target)
    print(f"{patch(target)}: {target}")


if __name__ == "__main__":
    main()
