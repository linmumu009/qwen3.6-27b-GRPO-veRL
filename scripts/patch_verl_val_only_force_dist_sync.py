#!/usr/bin/env python3
"""Allow val-only dist checkpoints to sync into vLLM via trainer config.

Ray task actors do not reliably inherit ad-hoc environment variables exported
by the submitting shell.  The force flag therefore lives in the serialized
trainer config, which is available inside ``OneStepTaskRunner``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_VAL_ONLY_FORCE_DIST_SYNC"
CONFIG_CONDITION = (
    '            and not self.config.trainer.get("val_only_force_dist_sync", False)'
    "  # LLIN_VAL_ONLY_FORCE_DIST_SYNC\n"
)


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if CONFIG_CONDITION in text:
        return "already-patched"
    environment_condition = (
        '            and os.environ.get("LLIN_VAL_ONLY_FORCE_DIST_SYNC") != "1"'
        "  # LLIN_VAL_ONLY_FORCE_DIST_SYNC\n"
    )
    if environment_condition in text:
        path.write_text(text.replace(environment_condition, CONFIG_CONDITION, 1), encoding="utf-8")
        return "migrated-env-to-config"
    anchor = '            and self.config.trainer.get("resume_mode", "disable") == "disable"\n'
    if anchor not in text:
        raise RuntimeError(f"expected val-only skip-sync condition not found in {path}")
    replacement = anchor + CONFIG_CONDITION
    path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(f"{patch(args.target)}: {args.target}")


if __name__ == "__main__":
    main()
