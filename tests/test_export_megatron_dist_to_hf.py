from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_megatron_dist_to_hf import (
    _base_fallback_keys,
    resolve_model_checkpoint,
    validate_export_paths,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    actor = tmp_path / "run" / "actor"
    model = actor / "model" / "dist_ckpt"
    model.mkdir(parents=True)
    (model / "metadata.json").write_text("{}", encoding="utf-8")
    (model / "__0_0.distcp").write_bytes(b"weights")
    _write_json(
        actor / "ckpt_contents.json",
        {"contents": {"model": {"format": "megatron_dist_checkpoint", "path": "model/dist_ckpt"}}},
    )
    base = tmp_path / "base"
    _write_json(base / "config.json", {})
    _write_json(base / "model.safetensors.index.json", {"weight_map": {}})
    return actor, base


def test_resolves_manifest_model_path(tmp_path: Path) -> None:
    actor, _ = _checkpoint(tmp_path)
    assert resolve_model_checkpoint(actor) == actor / "model" / "dist_ckpt"


def test_rejects_output_inside_resume_checkpoint(tmp_path: Path) -> None:
    actor, base = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="must not be inside"):
        validate_export_paths(actor, base, actor / "hf")


def test_rejects_existing_output(tmp_path: Path) -> None:
    actor, base = _checkpoint(tmp_path)
    output = tmp_path / "export"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validate_export_paths(actor, base, output)


def test_rejects_non_dist_checkpoint_manifest(tmp_path: Path) -> None:
    actor, _ = _checkpoint(tmp_path)
    _write_json(
        actor / "ckpt_contents.json",
        {"contents": {"model": {"format": "huggingface", "path": "model/huggingface"}}},
    )
    with pytest.raises(ValueError, match="expected megatron_dist_checkpoint"):
        resolve_model_checkpoint(actor)


def test_allows_exactly_15_frozen_mtp_fallback_tensors(tmp_path: Path) -> None:
    actor, base = _checkpoint(tmp_path)
    model = resolve_model_checkpoint(actor)
    _write_json(
        base / "model.safetensors.index.json",
        {"weight_map": {**{f"mtp.tensor_{i}": "model.safetensors" for i in range(15)}, "model.x": "x"}},
    )
    assert len(_base_fallback_keys(base, model)) == 15


def test_rejects_base_mtp_fallback_when_checkpoint_contains_mtp(tmp_path: Path) -> None:
    actor, base = _checkpoint(tmp_path)
    model = resolve_model_checkpoint(actor)
    (model / "metadata.json").write_text('"language_model.mtp.layers.0"', encoding="utf-8")
    _write_json(
        base / "model.safetensors.index.json",
        {"weight_map": {f"mtp.tensor_{i}": "model.safetensors" for i in range(15)}},
    )
    with pytest.raises(RuntimeError, match="fallback would be unsafe"):
        _base_fallback_keys(base, model)
