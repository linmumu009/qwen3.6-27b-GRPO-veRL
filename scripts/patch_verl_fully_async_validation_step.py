#!/usr/bin/env python3
"""Make fully-async validation artifacts use the trainer policy step."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


TRAINER_MARKER = "LLIN_FULLY_ASYNC_VALIDATION_STEP"
ROLLOUTER_MARKER = "LLIN_FULLY_ASYNC_VALIDATION_STEP"


def patch_trainer(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if TRAINER_MARKER in text:
        return "already-patched"

    old = "val_metrics = await self.rollouter.do_validate.remote()"
    count = text.count(old)
    if count != 2:
        raise RuntimeError(f"expected two validation RPC anchors in {path}, found {count}")
    new_call = "val_metrics = await self.rollouter.do_validate.remote(self.current_param_version)"
    text = text.replace(old, new_call)
    match = re.search(rf"(?m)^([ \t]+){re.escape(new_call)}$", text)
    if match is None:
        raise RuntimeError(f"expected indented validation RPC not found in {path}")
    indent = match.group(1)
    first_indented_call = f"{indent}{new_call}"
    marker_block = (
        f"{indent}# LLIN_FULLY_ASYNC_VALIDATION_STEP: pass the trainer policy version;\n"
        f"{indent}# the rollouter's own global_steps is a different data counter.\n"
        f"{first_indented_call}"
    )
    path.write_text(text.replace(first_indented_call, marker_block, 1), encoding="utf-8")
    return "patched"


def patch_rollouter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if ROLLOUTER_MARKER in text:
        return "already-patched"

    old = '''\
    def do_validate(self):
        """Run validation and return metrics"""
        timing_raw = {}
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = self._validate()
        return timing_raw | val_metrics
'''
    new = '''\
    def do_validate(self, validation_step: int | None = None):
        """Run validation and name artifacts with the trainer policy version."""
        timing_raw = {}
        original_global_steps = self.global_steps
        try:
            # LLIN_FULLY_ASYNC_VALIDATION_STEP: inherited validation dumping uses
            # self.global_steps, but this actor normally stores a rollout-data
            # counter there. Temporarily expose the trainer's policy step.
            if validation_step is not None:
                self.global_steps = int(validation_step)
            with marked_timer("rollouter/validate_time", timing_raw, color="green"):
                val_metrics: dict = self._validate()
        finally:
            self.global_steps = original_global_steps
        return timing_raw | val_metrics
'''
    if old not in text:
        raise RuntimeError(f"expected rollouter validation anchor not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer",
        default="/verl/verl/experimental/fully_async_policy/fully_async_trainer.py",
    )
    parser.add_argument(
        "--rollouter",
        default="/verl/verl/experimental/fully_async_policy/fully_async_rollouter.py",
    )
    args = parser.parse_args()
    print(f"{patch_trainer(Path(args.trainer))}: {args.trainer}")
    print(f"{patch_rollouter(Path(args.rollouter))}: {args.rollouter}")


if __name__ == "__main__":
    main()
