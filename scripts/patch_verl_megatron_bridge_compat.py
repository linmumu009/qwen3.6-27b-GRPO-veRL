#!/usr/bin/env python3
"""Patch veRL's Bridge import to use the local Ascend compatibility helpers."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """try:
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.utils.train_utils import LinearForLastLayer, freeze_moe_router, make_value_model
except ImportError:
    # `pip install verl[mcore]` or
    print("Megatron-Bridge package not found. Please install Megatron-Bridge with `pip install megatron-bridge`")
    raise
"""

NEW = """try:
    from megatron.bridge import AutoBridge
    try:
        from megatron.bridge.training.utils.train_utils import LinearForLastLayer, freeze_moe_router, make_value_model
    except ImportError:
        from llin_verl.megatron_bridge_compat import LinearForLastLayer, freeze_moe_router, make_value_model
except ImportError:
    # `pip install verl[mcore]` or
    print("Megatron-Bridge package not found. Please install Megatron-Bridge with `pip install megatron-bridge`")
    raise
"""

PEFT_OLD = """        if provider is not None:
            from megatron.bridge.peft.utils import create_peft_hook, load_peft_adapter_checkpoint
            from megatron.bridge.training.utils.config_utils import create_ddp_config
"""

PEFT_NEW_V1 = """        if provider is not None:
            from megatron.bridge.training.utils.config_utils import create_ddp_config
"""

PEFT_NEW = """        if provider is not None:
            try:
                from megatron.bridge.training.utils.config_utils import create_ddp_config
            except ImportError:
                from llin_verl.megatron_bridge_compat import create_ddp_config
"""

PEFT_HOOK_OLD = """            if peft_cls is not None:
                from verl.utils.megatron_peft_utils import print_adapter_info
"""

PEFT_HOOK_NEW = """            if peft_cls is not None:
                from megatron.bridge.peft.utils import create_peft_hook, load_peft_adapter_checkpoint
                from verl.utils.megatron_peft_utils import print_adapter_info
"""


def patch(target: Path) -> str:
    source = target.read_text(encoding="utf-8")
    if NEW in source:
        return "already-patched"
    if OLD not in source:
        raise RuntimeError(f"Refusing to patch unexpected veRL source: {target}")
    target.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    return "patched"


def patch_peft_import(target: Path) -> str:
    source = target.read_text(encoding="utf-8")
    if PEFT_NEW in source and PEFT_HOOK_NEW in source:
        return "already-patched"
    if PEFT_NEW_V1 in source and PEFT_HOOK_NEW in source:
        target.write_text(source.replace(PEFT_NEW_V1, PEFT_NEW, 1), encoding="utf-8")
        return "patched"
    if PEFT_OLD not in source or PEFT_HOOK_OLD not in source:
        raise RuntimeError(f"Refusing to patch unexpected veRL PEFT source: {target}")
    source = source.replace(PEFT_OLD, PEFT_NEW, 1)
    source = source.replace(PEFT_HOOK_OLD, PEFT_HOOK_NEW, 1)
    target.write_text(source, encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("/verl/verl/models/mcore/bridge.py"),
    )
    args = parser.parse_args()
    print(f"{patch(args.target)}: {args.target}")
    peft_target = args.target.parents[2] / "utils" / "megatron_utils.py"
    print(f"{patch_peft_import(peft_target)}: {peft_target}")


if __name__ == "__main__":
    main()
