from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_qwen38_model_compat import validate_static_compatibility
from scripts.check_qwen38_smoke_ray_cluster import validate_rows


ROOT = Path(__file__).resolve().parents[1]


def write_model(path: Path, *, hidden_size: int = 5120, extra_key: bool = False) -> None:
    path.mkdir()
    text_config = {
        "model_type": "qwen3_5_text",
        "hidden_size": hidden_size,
        "intermediate_size": 17408,
        "num_hidden_layers": 4,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "mtp_num_hidden_layers": 1,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    }
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "language_model_only": False,
                "text_config": text_config,
            }
        ),
        encoding="utf-8",
    )
    keys = {"model.language_model.layers.0.weight": "part-1.safetensors"}
    if extra_key:
        keys["unexpected.weight"] = "part-1.safetensors"
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1234}, "weight_map": keys}),
        encoding="utf-8",
    )


def test_static_gate_accepts_shape_and_tensor_key_compatible_models(tmp_path: Path) -> None:
    reference = tmp_path / "qwen36"
    candidate = tmp_path / "qwen38"
    write_model(reference)
    write_model(candidate)

    summary = validate_static_compatibility(reference, candidate)

    assert summary["tensor_key_set_equal"] is True
    assert summary["hf_tensor_keys"] == 1
    assert summary["training_checkpoint_reuse_allowed"] is False
    assert summary["initialization"] == "candidate_hf_weights"


@pytest.mark.parametrize("change", ["shape", "keys"])
def test_static_gate_rejects_incompatible_candidate(tmp_path: Path, change: str) -> None:
    reference = tmp_path / "qwen36"
    candidate = tmp_path / "qwen38"
    write_model(reference)
    write_model(
        candidate,
        hidden_size=4096 if change == "shape" else 5120,
        extra_key=change == "keys",
    )

    with pytest.raises(ValueError):
        validate_static_compatibility(reference, candidate)


def test_qwen38_smoke_is_one_step_and_never_resumes_qwen36_checkpoint() -> None:
    script = (ROOT / "scripts" / "run_pi_qwen38_megatron_smoke.sh").read_text(
        encoding="utf-8"
    )
    assert "initialization=qwen38_hf_base" in script
    assert "qwen36_step120_checkpoint_reused=false" in script
    assert "TRAIN_TP=4 TRAIN_PP=2 TRAIN_CP=2 TRAIN_NPUS=16" in script
    assert "ROLLOUT_TP=8 ROLLOUT_NPUS=16" in script
    assert "TOTAL_TRAINING_STEPS=1 SAVE_FREQ=-1" in script
    assert "OPTIMIZER_CPU_OFFLOAD=false ENGINE_OPTIMIZER_OFFLOAD=false" in script
    assert "192.168.202.5:36379" in script
    assert "SOURCE_CHECKPOINT" not in script


def test_qwen38_smoke_ray_cluster_uses_isolated_ports() -> None:
    head = (ROOT / "scripts" / "start_ray_qwen38_smoke_m05.sh").read_text(
        encoding="utf-8"
    )
    worker = (ROOT / "scripts" / "start_ray_qwen38_smoke_m06.sh").read_text(
        encoding="utf-8"
    )
    assert "ray stop" not in head
    assert "ray stop" not in worker
    assert "36379" in head and "36379" in worker
    assert "37000" in head and "37999" in head
    assert "38000" in worker and "38999" in worker
    assert "/tmp/q38-ray-m05" in head


def test_qwen38_smoke_ray_gate_requires_exact_physical_16_plus_16() -> None:
    valid = [
        {"ip": "192.168.202.4", "npu": 16, "trainer": 0, "rollout": 1},
        {"ip": "192.168.202.5", "npu": 16, "trainer": 1, "rollout": 0},
    ]
    validate_rows(valid)
    with pytest.raises(ValueError, match="topology mismatch"):
        validate_rows(valid + [valid[0]])
