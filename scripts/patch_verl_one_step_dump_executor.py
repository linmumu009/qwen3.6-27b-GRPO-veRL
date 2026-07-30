#!/usr/bin/env python3
"""Initialize veRL's rollout dump executor in the One-Step-Off-Policy trainer."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """\
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== SeparateRayPPOTrainer config ====================
"""

NEW = """\
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # The base PPO trainer creates this executor in its constructor, but the
        # separate One-Step-Off-Policy trainer intentionally bypasses that
        # constructor.  Rollout dumping still calls the inherited helper, so
        # initialize its executor here as well.
        self._init_dump_executor()

        # ==================== SeparateRayPPOTrainer config ====================
"""


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        return "already-patched"
    if OLD not in text:
        raise RuntimeError(f"expected One-Step-Off-Policy initializer block not found in {path}")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/verl/verl/experimental/one_step_off_policy/ray_trainer.py",
    )
    args = parser.parse_args()
    target = Path(args.target)
    print(f"{patch(target)}: {target}")


if __name__ == "__main__":
    main()
