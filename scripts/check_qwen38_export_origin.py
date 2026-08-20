#!/usr/bin/env python3
"""Fail closed unless an HF export came from the expected Qwen3.8 policy step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTRACT = "llin-qwen38-hf-export-origin-gate-v1"


def check(model: Path, expected_policy_step: int) -> dict[str, Any]:
    model = model.resolve(strict=True)
    manifest_path = model / "llin_export_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    actor_checkpoint = str(payload.get("actor_checkpoint") or "")
    marker = f"global_step_{expected_policy_step}"
    checks = {
        "export_verification_valid": (payload.get("verification") or {}).get("valid") is True,
        "actor_checkpoint_step_exact": marker in actor_checkpoint,
        "config_present": (model / "config.json").is_file(),
        "weight_index_present": (model / "model.safetensors.index.json").is_file(),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("HF export origin gate failed: " + ", ".join(failed))
    return {
        "contract": CONTRACT,
        "valid": True,
        "expected_policy_step": expected_policy_step,
        "checks": checks,
        "contains_model_path_actor_checkpoint_or_tensor_hashes": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-policy-step", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.model, args.expected_policy_step)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
