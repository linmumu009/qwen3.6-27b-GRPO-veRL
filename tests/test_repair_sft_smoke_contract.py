from pathlib import Path

import pandas as pd

from scripts.prepare_repair_sft_smoke_dataset import build_rows


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_rows_include_tool_plan_result_and_final_answer():
    row = build_rows(1)[0]

    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert row["messages"][2]["tool_calls"][0]["function"]["name"] == "query_sql"
    assert row["messages"][4]["content"] == "The answer is 5."
    assert row["tools"][0]["function"]["parameters"]["required"] == ["query"]


def test_smoke_rows_round_trip_through_parquet(tmp_path: Path):
    path = tmp_path / "smoke.parquet"
    pd.DataFrame(build_rows(2)).to_parquet(path, index=False)

    restored = pd.read_parquet(path).to_dict(orient="records")

    assert len(restored) == 2
    assert len(restored[0]["messages"]) == 5
    assert restored[1]["sample_id"] == "repair-sft-smoke-0001"


def test_launcher_uses_official_sft_with_model_only_initialization():
    script = (ROOT / "scripts" / "run_repair_sft_megatron_smoke.sh").read_text(encoding="utf-8")

    assert "-m verl.trainer.sft_trainer" in script
    assert "engine.use_dist_checkpointing=true" in script
    assert "engine.dist_checkpointing_path=${SOURCE_MODEL_DIST_CKPT}" in script
    assert "trainer.resume_mode=disable" in script
    assert "Qwen36AssistantMaskSFTDataset" in script
    assert "Megatron-Bridge-de93536e/src" in script
    assert "checkpoint.save_contents=[extra]" in script
    assert "trainer.total_training_steps=1" in script
    assert "data.pad_mode=no_padding" in script


def test_qwen36_custom_dataset_uses_exact_template_and_assistant_mask():
    source = (ROOT / "scripts" / "qwen36_assistant_mask_sft_dataset.py").read_text(encoding="utf-8")

    assert "class Qwen36AssistantMaskSFTDataset" in source
    assert "apply_chat_template(" in source
    assert "add_generation_prompt=False" in source
    assert "expected_assistant_turns" in source
    assert "sample {item} has no assistant loss tokens" in source
