#!/usr/bin/env python3
"""Allow val-only runs initialized from a dist checkpoint to sync into vLLM."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_VAL_ONLY_FORCE_DIST_SYNC"


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already-patched"
    anchor = '            and self.config.trainer.get("resume_mode", "disable") == "disable"\n'
    if anchor not in text:
        raise RuntimeError(f"expected val-only skip-sync condition not found in {path}")
    if "import os\n" not in text:
        future = "from __future__ import annotations\n"
        if future in text:
            text = text.replace(future, future + "\nimport os\n", 1)
        else:
            text = "import os\n" + text
    replacement = anchor + '            and os.environ.get("LLIN_VAL_ONLY_FORCE_DIST_SYNC") != "1"  # LLIN_VAL_ONLY_FORCE_DIST_SYNC\n'
    path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(f"{patch(args.target)}: {args.target}")


if __name__ == "__main__":
    main()
