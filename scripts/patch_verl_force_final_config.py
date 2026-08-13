#!/usr/bin/env python3
"""Add LLIN opt-in fields to veRL's strict MultiTurnConfig dataclass."""

from __future__ import annotations

import argparse
from pathlib import Path


FIELDS = (
    "force_final_after_assistant_turns: int = 0",
    "force_final_reserve_response_tokens: int = 0",
    "force_final_max_response_tokens: int = 0",
    "force_final_max_retries: int = 0",
    "agent_timeout_seconds: float = 0.0",
)


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if all(field in text for field in FIELDS):
        return "already-patched"
    if FIELDS[0] not in text:
        needle = "    num_repeat_rollouts: Optional[int] = None\n"
        if needle not in text:
            raise RuntimeError(f"expected MultiTurnConfig anchor not found in {path}")
        replacement = (
            needle
            + "    # LLIN_FORCE_FINAL: opt-in PI sentinel controls; zero keeps upstream behavior.\n"
            + "    "
            + "\n    ".join(FIELDS)
            + "\n"
        )
        text = text.replace(needle, replacement, 1)
    else:
        needle = "    force_final_reserve_response_tokens: int = 0\n"
        if needle not in text:
            raise RuntimeError(f"expected force-final upgrade anchor not found in {path}")
        missing = [field for field in FIELDS[2:] if field not in text]
        text = text.replace(needle, needle + "".join(f"    {field}\n" for field in missing), 1)
    path.write_text(text, encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="/verl/verl/workers/config/rollout.py")
    args = parser.parse_args()
    target = Path(args.target)
    print(f"{patch(target)}: {target}")


if __name__ == "__main__":
    main()
