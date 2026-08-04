import json
import sqlite3
from pathlib import Path

import pytest

from llin_verl.boss_pi_contract import contract_hashes, load_boss_pi_contract
from scripts.check_boss_alignment_contract import validate_alignment_contract
from scripts.prepare_boss_aligned_dataset import (
    CONTRACT_NAME,
    SourceSpec,
    build_dataset,
    canonical_hash,
    canonicalize_conversation,
    file_sha256,
    parse_pilot_sizes,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def make_source(tmp_path: Path) -> tuple[SourceSpec, Path, dict]:
    conversation_path = tmp_path / "conversation.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    sandbox_root = tmp_path / "sandbox"
    database = sandbox_root / "sft" / "v1" / "logistics.sqlite"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database)
    connection.execute("create table metric(value real)")
    connection.execute("insert into metric values (20.5)")
    connection.commit()
    connection.close()

    write_jsonl(
        conversation_path,
        [
            {
                "messages": [
                    {"role": "system", "content": "旧的物流分析师 fallback"},
                    {"role": "user", "content": "老板的原始问题"},
                    {"role": "assistant", "content": "历史回答"},
                ]
            }
        ],
    )
    gold = {
        "answer_type": "numeric",
        "value": 20.5,
        "verification_sql": "SELECT SUM(value) FROM metric",
    }
    write_jsonl(
        manifest_path,
        [
            {
                "task_id": "task_1",
                "v": "v1",
                "type": "dwh",
                "instruction": "老板的原始问题",
                "expected_tables": ["metric"],
                "gold_answer": gold,
            }
        ],
    )
    return SourceSpec("train", "v1", conversation_path, manifest_path), sandbox_root, gold


def approval(gold: dict) -> dict:
    return {
        "source_label": "v1",
        "task_id": "task_1",
        "instruction_sha256": canonical_hash("老板的原始问题"),
        "gold_sha256": canonical_hash(gold),
        "approved_for_grpo": True,
        "reviewer": "reviewer-a",
        "reviewed_at": "2026-08-04",
        "split": "train",
    }


def test_boss_contract_replaces_project_fallback_with_source_owned_system_and_tools():
    contract = load_boss_pi_contract()
    row = canonicalize_conversation(
        {"messages": [{"role": "system", "content": "项目 fallback"}, {"role": "user", "content": "问题"}]},
        contract["system_prompt"],
        contract["tools"],
    )

    assert row["messages"][0]["content"] == contract["system_prompt"]
    assert row["tools"] == contract["tools"]
    assert [item["function"]["name"] for item in row["tools"]] == ["bash", "read", "edit", "write"]


def test_unreviewed_source_is_sft_only_and_never_enters_grpo(tmp_path: Path):
    spec, sandbox_root, _ = make_source(tmp_path)
    records, sft_rows, queue, report = build_dataset(
        [spec], sandbox_root, {}, {}, "seed", load_boss_pi_contract()
    )

    assert records["train"] == []
    assert len(sft_rows) == 1
    assert sft_rows[0]["messages"][-1]["role"] == "assistant"
    assert queue[0]["review_status"] == "missing_alignment_review"
    assert report["invariants"]["unreviewed_grpo_count"] == 0


def test_reviewed_source_uses_all_approved_and_grpo_input_has_no_response(tmp_path: Path):
    spec, sandbox_root, gold = make_source(tmp_path)
    records, sft_rows, _, report = build_dataset(
        [spec],
        sandbox_root,
        {("v1", "task_1"): approval(gold)},
        {},
        "seed",
        load_boss_pi_contract(),
    )

    assert report["mode"] == "full"
    assert report["invariants"]["uses_all_approved_by_default"] is True
    assert len(records["train"]) == 1
    assert [message["role"] for message in records["train"][0]["prompt"]] == ["system", "user"]
    assert records["train"][0]["extra_info"]["response_messages_in_grpo_input"] == 0
    assert sft_rows[0]["messages"][-1]["content"] == "历史回答"


def test_task_id_join_keeps_historical_instruction_for_review_even_if_current_variants_changed(tmp_path: Path):
    spec, sandbox_root, gold = make_source(tmp_path)
    conversation = json.loads(spec.conversations.read_text(encoding="utf-8"))
    conversation["task_id"] = "task_1"
    conversation["messages"][1]["content"] = "历史运行时真实问题"
    write_jsonl(spec.conversations, [conversation])
    review = approval(gold)
    review["instruction_sha256"] = canonical_hash("历史运行时真实问题")

    records, _, queue, _ = build_dataset(
        [spec],
        sandbox_root,
        {("v1", "task_1"): review},
        {},
        "seed",
        load_boss_pi_contract(),
    )

    assert len(records["train"]) == 1
    assert records["train"][0]["prompt"][1]["content"] == "历史运行时真实问题"
    assert queue[0]["source_join_method"] == "task_id"
    assert queue[0]["source_instruction_in_current_task_definition"] is False


def test_pilot_is_explicit_and_cannot_be_mislabeled_full(tmp_path: Path):
    spec, sandbox_root, gold = make_source(tmp_path)
    _, _, _, report = build_dataset(
        [spec],
        sandbox_root,
        {("v1", "task_1"): approval(gold)},
        parse_pilot_sizes(["train=1"]),
        "seed",
        load_boss_pi_contract(),
    )
    assert report["mode"] == "pilot"
    assert report["invariants"]["uses_all_approved_by_default"] is False


def test_formal_gate_rejects_pilot_and_accepts_integrity_checked_full_data(tmp_path: Path):
    train = tmp_path / "boss_pi_train.parquet"
    val = tmp_path / "boss_pi_val.parquet"
    train.write_bytes(b"train")
    val.write_bytes(b"val")
    contract = load_boss_pi_contract()
    report = {
        "contract": CONTRACT_NAME,
        "mode": "full",
        "boss_contract": contract_hashes(contract),
        "invariants": {
            "uses_all_approved_by_default": True,
            "source_responses_exported_only_for_sft_and_regression": True,
            "split_overlap": [],
            "project_system_fallback_count": 0,
            "project_tool_schema_fallback_count": 0,
            "generated_instruction_count": 0,
            "generated_gold_or_sql_count": 0,
            "unreviewed_grpo_count": 0,
            "assistant_or_tool_messages_in_grpo_input": 0,
        },
        "artifacts": {
            train.name: {"sha256": file_sha256(train), "purpose": "grpo"},
            val.name: {"sha256": file_sha256(val), "purpose": "grpo"},
        },
    }
    (tmp_path / "boss_alignment_contract.json").write_text(json.dumps(report), encoding="utf-8")

    validate_alignment_contract(tmp_path)
    report["mode"] = "pilot"
    (tmp_path / "boss_alignment_contract.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="refuses pilot"):
        validate_alignment_contract(tmp_path)
