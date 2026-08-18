from pathlib import Path

import pytest

from scripts.verify_model_transfer import build, verify


def test_file_manifest_detects_transfer_corruption(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "weights").write_bytes(b"weights")
    manifest = tmp_path / "transfer.safe.json"
    built = build(model, manifest)
    assert verify(model, manifest)["total_bytes"] == built["total_bytes"]
    (model / "weights").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify(model, manifest)
