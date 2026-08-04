#!/usr/bin/env python3
"""Keep HF initialization when dist-checkpoint saving has no load path."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_DIST_CHECKPOINT_INIT_FALLBACK"

OLD = """\
        if self.engine_config.use_dist_checkpointing:
            load_mcore_dist_weights(
                module, self.engine_config.dist_checkpointing_path, is_value_model=self.is_value_model
            )
        else:
"""

NEW = """\
        # LLIN_DIST_CHECKPOINT_INIT_FALLBACK: use_dist_checkpointing selects
        # the model save format too. A fresh run has no dist checkpoint path,
        # so initialize from the configured HF model and still save Megatron
        # shards later. A real resume path continues to load dist weights.
        if self.engine_config.use_dist_checkpointing and self.engine_config.dist_checkpointing_path:
            load_mcore_dist_weights(
                module, self.engine_config.dist_checkpointing_path, is_value_model=self.is_value_model
            )
        else:
"""


def patch(target: Path) -> str:
    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        return "already-patched"
    if OLD not in source:
        raise RuntimeError(f"Refusing to patch unexpected veRL source: {target}")
    target.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("/verl/verl/workers/engine/megatron/transformer_impl.py"),
    )
    args = parser.parse_args()
    print(f"{patch(args.target)}: {args.target}")


if __name__ == "__main__":
    main()
