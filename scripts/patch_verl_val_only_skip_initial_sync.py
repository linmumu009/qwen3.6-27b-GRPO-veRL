#!/usr/bin/env python3
"""Skip redundant actor-to-rollout weight broadcast for frozen validation."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_VAL_ONLY_SKIP_INITIAL_SYNC"


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already-patched"

    old = """\
        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self._fit_update_weights()
"""
    new = """\
        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        # LLIN_VAL_ONLY_SKIP_INITIAL_SYNC: a frozen baseline with resume
        # disabled has the same immutable MODEL_PATH on actor and rollout.
        # vLLM already loaded those safetensors, so a 1->16 broadcast is
        # redundant and can deadlock on the asymmetric Ascend communicator.
        val_only_base_model = (
            self.config.trainer.get("val_only", False)
            and self.config.trainer.get("resume_mode", "disable") == "disable"
        )
        if val_only_base_model:
            print("[LLIN_VAL_ONLY] skip initial actor-to-rollout weight sync")
        else:
            self._fit_update_weights()
"""
    if old not in text:
        raise RuntimeError(f"expected val-only sync anchor not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "patched"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"{patch(args.target)}: {args.target}")


if __name__ == "__main__":
    main()
