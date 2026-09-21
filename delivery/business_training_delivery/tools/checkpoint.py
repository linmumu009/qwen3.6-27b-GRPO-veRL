"""Structural checkpoint gate shared by the launcher and local tests."""
import json
from pathlib import Path


def checkpoint_ready(directory):
    manifests = list(directory.rglob("ckpt_contents.json"))
    if not manifests:
        return False
    for manifest in manifests:
        value = json.loads(manifest.read_text())
        entry = value.get("contents", {}).get("model", {})
        if entry.get("format") != "megatron_dist_checkpoint":
            return False
        relative = Path(entry.get("path", ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            return False
        weights = manifest.parent / relative
        shards = list(weights.glob("*.distcp"))
        if not (weights / ".metadata").is_file() or not shards or any(p.stat().st_size == 0 for p in shards):
            return False
    return True
