#!/usr/bin/env python3
"""Keep text-only fully-async runs on veRL's Continuous Token path."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """\
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
"""

NEW = """\
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        # Continuous Token is tokenizer-based. Qwen text checkpoints may still
        # expose a processor, so discard it only for an explicitly text-only run.
        continuous_token = config.data.get("continuous_token", {})
        if continuous_token.get("enable", False) and not config.data.get("return_multi_modal_inputs", True):
            processor = None

        self.components["tokenizer"] = tokenizer
"""


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        return "already-patched"
    if OLD not in text:
        raise RuntimeError(f"expected fully-async processor block not found in {path}")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/verl/verl/experimental/fully_async_policy/fully_async_main.py",
    )
    args = parser.parse_args()
    target = Path(args.target)
    print(f"{patch(target)}: {target}")


if __name__ == "__main__":
    main()
