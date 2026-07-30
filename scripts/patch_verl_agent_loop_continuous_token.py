#!/usr/bin/env python3
"""Keep text-only AgentLoop workers on the Continuous Token path."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """\
        continuous_token_config = self.data_config.continuous_token
        if continuous_token_config.enable and self.processor is None:
"""

NEW = """\
        continuous_token_config = self.data_config.continuous_token
        # Ray AgentLoop workers reconstruct ModelConfig and may reload a
        # vision-capable processor even when the run is explicitly text-only.
        # Continuous Token is tokenizer-based, so discard that unused processor
        # for this safe configuration.
        if continuous_token_config.enable and not self.data_config.get("return_multi_modal_inputs", True):
            self.processor = None
        if continuous_token_config.enable and self.processor is None:
"""


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        return "already-patched"
    if OLD not in text:
        raise RuntimeError(f"expected AgentLoop Continuous Token block not found in {path}")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/verl/verl/experimental/agent_loop/agent_loop.py",
    )
    args = parser.parse_args()
    target = Path(args.target)
    print(f"{patch(target)}: {target}")


if __name__ == "__main__":
    main()
