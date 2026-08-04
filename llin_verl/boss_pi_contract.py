"""Authoritative PI system/tool contract copied from the boss source tree.

The values are intentionally loaded from a checked-in JSON snapshot.  Dataset
builders and launch gates use the same hashes, so a project fallback cannot be
silently substituted for the original PI contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "configs" / "boss_pi_contract.json"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_boss_pi_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract") != "boss-pi-qwen36-v1":
        raise ValueError(f"unexpected boss PI contract: {contract.get('contract')!r}")
    system = contract.get("system_prompt")
    tools = contract.get("tools")
    if not isinstance(system, str) or not system.strip():
        raise ValueError("boss PI contract is missing system_prompt")
    if not isinstance(tools, list) or [item.get("function", {}).get("name") for item in tools] != [
        "bash",
        "read",
        "edit",
        "write",
    ]:
        raise ValueError("boss PI contract tool order/schema is invalid")
    expected = contract.get("integrity", {})
    if expected.get("system_prompt_sha256") != sha256_text(system):
        raise ValueError("boss PI system prompt hash mismatch")
    if expected.get("tools_sha256") != sha256_text(canonical_json(tools)):
        raise ValueError("boss PI tool schema hash mismatch")
    return contract


def contract_hashes(contract: dict[str, Any]) -> dict[str, str]:
    return {
        "system_prompt_sha256": sha256_text(contract["system_prompt"]),
        "tools_sha256": sha256_text(canonical_json(contract["tools"])),
        "contract_sha256": sha256_text(canonical_json(contract)),
    }
