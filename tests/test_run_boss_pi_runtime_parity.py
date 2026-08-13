import json

import pytest

from scripts.run_boss_pi_runtime_parity import (
    validate_generation_config,
    validate_pi_model_metadata,
)


def write_models(tmp_path, *, context_window=49_152, max_tokens=8_192):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "local": {
                        "models": [
                            {
                                "id": "Qwen3.6-27B",
                                "contextWindow": context_window,
                                "maxTokens": max_tokens,
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_pi_model_metadata_accepts_exact_runtime_contract(tmp_path):
    result = validate_pi_model_metadata(
        write_models(tmp_path),
        "Qwen3.6-27B",
        49_152,
        8_192,
    )

    assert result == {
        "served_model": "Qwen3.6-27B",
        "context_window": 49_152,
        "max_tokens_per_request": 8_192,
    }


def test_pi_model_metadata_rejects_client_server_context_mismatch(tmp_path):
    with pytest.raises(ValueError, match="contextWindow mismatch"):
        validate_pi_model_metadata(
            write_models(tmp_path, context_window=262_144),
            "Qwen3.6-27B",
            49_152,
            8_192,
        )


def test_pi_generation_config_requires_stochastic_matched_defaults(tmp_path):
    path = tmp_path / "generation_config.json"
    path.write_text(
        json.dumps(
            {
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
            }
        ),
        encoding="utf-8",
    )

    result = validate_generation_config(path, 1.0, 0.95, 20)

    assert result["temperature"] == 1.0
    assert result["do_sample"] is True
    assert result["source"].startswith("server_generation_config")


def test_pi_generation_config_rejects_greedy_default(tmp_path):
    path = tmp_path / "generation_config.json"
    path.write_text(
        json.dumps(
            {
                "do_sample": False,
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="generation_config mismatch"):
        validate_generation_config(path, 1.0, 0.95, 20)
