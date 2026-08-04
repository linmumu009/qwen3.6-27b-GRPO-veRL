import json
from pathlib import Path

from scripts.verify_checkpoint_integrity import verify_checkpoint


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _manifest(path: Path, model_format: str, model_path: str) -> None:
    _write_json(
        path / "actor" / "ckpt_contents.json",
        {
            "global_step": 5,
            "contents": {"model": {"format": model_format, "path": model_path}},
        },
    )


def test_hf_checkpoint_requires_all_base_tensor_keys(tmp_path: Path):
    base = tmp_path / "base"
    checkpoint = tmp_path / "checkpoint"
    model = checkpoint / "actor" / "model" / "huggingface"
    _write_json(
        base / "model.safetensors.index.json",
        {"weight_map": {"layer.0": "base-1", "layer.1": "base-2"}},
    )
    _write_json(
        model / "model.safetensors.index.json",
        {"weight_map": {"layer.0": "model-00001.safetensors"}},
    )
    (model / "model-00001.safetensors").write_bytes(b"weights")
    _manifest(checkpoint, "huggingface", "model/huggingface")

    result = verify_checkpoint(checkpoint, base)

    assert not result["valid"]
    assert result["missing_tensor_count"] == 1
    assert result["missing_tensor_examples"] == ["layer.1"]


def test_hf_checkpoint_accepts_exact_nonempty_index(tmp_path: Path):
    base = tmp_path / "base"
    checkpoint = tmp_path / "checkpoint"
    model = checkpoint / "actor" / "model" / "huggingface"
    weight_map = {"layer.0": "model-00001.safetensors", "layer.1": "model-00002.safetensors"}
    _write_json(base / "model.safetensors.index.json", {"weight_map": weight_map})
    _write_json(model / "model.safetensors.index.json", {"weight_map": weight_map})
    (model / "model-00001.safetensors").write_bytes(b"first")
    (model / "model-00002.safetensors").write_bytes(b"second")
    _manifest(checkpoint, "huggingface", "model/huggingface")

    result = verify_checkpoint(checkpoint, base)

    assert result["valid"]
    assert result["checkpoint_tensor_count"] == 2
    assert result["referenced_shard_count"] == 2


def test_megatron_dist_checkpoint_requires_metadata_and_shards(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    model = checkpoint / "actor" / "model" / "dist_ckpt"
    model.mkdir(parents=True)
    _manifest(checkpoint, "megatron_dist_checkpoint", "model/dist_ckpt")

    incomplete = verify_checkpoint(checkpoint)
    assert not incomplete["valid"]

    (model / ".metadata").write_bytes(b"metadata")
    (model / "__0_0.distcp").write_bytes(b"weights")
    complete = verify_checkpoint(checkpoint)

    assert complete["valid"]
    assert complete["shard_count"] == 1
