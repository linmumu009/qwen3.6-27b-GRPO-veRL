import json
import sqlite3
from pathlib import Path

from llin_verl.boss_reward_shadow import (
    boss_task_to_ground_truth,
    compute_shadow_score,
    openai_messages_to_pi_events,
)


def make_database(root: Path) -> None:
    database = root / "sft" / "v1" / "logistics.sqlite"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database)
    connection.execute("create table metric(value real)")
    connection.executemany("insert into metric values (?)", [(600.0,), (21.62,)])
    connection.commit()
    connection.close()


def dwh_task(value: float = 621.62) -> dict:
    return {
        "task_id": "task_1",
        "type": "dwh",
        "v": "v1",
        "expected_tables": ["metric"],
        "gold_answer": {
            "answer_type": "numeric",
            "value": value,
            "verification_sql": "SELECT SUM(value) FROM metric",
        },
    }


def bash_event(sql: str, ok: bool = True) -> dict:
    return {
        "name": "bash",
        "arguments": {"command": f'sqlite3 /workspace/logistics.sqlite "{sql}"'},
        "ok": ok,
        "response_preview": "621.62",
    }


def test_dwh_strict_shadow_reward_requires_answer_sql_and_verified_gold(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))
    truth = boss_task_to_ground_truth(dwh_task())

    result = compute_shadow_score(
        "shadow",
        "最终结果为 621.62。",
        truth,
        {"pi_tool_events": [bash_event("SELECT SUM(value) FROM metric")]},
    )

    assert result["score"] == 1.0
    assert result["acc"] == 1.0
    assert result["gold_sql_verified"] == 1.0
    assert result["online_eligible"] == 1.0
    assert result["deployment_status"] == "shadow_only"


def test_dwh_number_collision_without_sql_is_capped_at_point_15(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))
    truth = boss_task_to_ground_truth(dwh_task())

    result = compute_shadow_score(
        "shadow",
        "可能的结果包括 1、2、621.62、999。",
        truth,
        {"pi_tool_events": [bash_event("SELECT value FROM metric LIMIT 1")]},
    )

    assert result["final_answer_correct"] == 1.0
    assert result["sql_evidence_correct"] == 0.0
    assert result["score"] == 0.15
    assert result["acc"] == 0.0


def test_dwh_gold_sql_mismatch_is_not_online_eligible(tmp_path, monkeypatch):
    make_database(tmp_path)
    monkeypatch.setenv("PI_AGENT_SANDBOX_LOWER", str(tmp_path))
    truth = boss_task_to_ground_truth(dwh_task(999.0))

    result = compute_shadow_score(
        "shadow",
        "最终结果为 999。",
        truth,
        {"pi_tool_events": [bash_event("SELECT SUM(value) FROM metric")]},
    )

    assert result["gold_sql_verified"] == 0.0
    assert result["online_eligible"] == 0.0
    assert result["score"] == 0.0


def kb_task(*, subtype: str = "single_doc_lookup", gold: str | None = None) -> dict:
    return {
        "task_id": "KT-1",
        "type": "kb",
        "v": "v1",
        "subtype": subtype,
        "gold_answer": gold,
        "source_documents": [] if subtype == "unanswerable" else ["doc_003"],
    }


def test_kb_answer_mention_is_not_document_access_evidence():
    truth = boss_task_to_ground_truth(kb_task(gold="根据《理赔政策》，赔付比例为 93.3%。"))
    result = compute_shadow_score(
        "shadow",
        "根据 doc_003，赔付比例为 93.3%。",
        truth,
        {"pi_tool_events": [{"name": "bash", "arguments": {"command": "ls /workspace"}, "ok": True}]},
    )

    assert result["gold_numbers_ok"] == 1.0
    assert result["source_documents_ok"] == 0.0
    assert result["online_eligible"] == 0.0
    assert result["score"] == 0.15


def test_kb_find_discovers_document_but_does_not_count_as_content_access():
    truth = boss_task_to_ground_truth(kb_task(gold="根据《理赔政策》，赔付比例为 93.3%。"))
    event = {
        "name": "bash",
        "arguments": {"command": "find /workspace -name '*理赔*'"},
        "ok": True,
        "response_preview": "/workspace/docs/doc_003.md",
    }
    result = compute_shadow_score(
        "shadow",
        "赔付比例为 93.3%。",
        truth,
        {"pi_tool_events": [event]},
    )

    assert result["source_documents_ok"] == 0.0
    assert result["score"] == 0.15


def test_kb_successful_tool_evidence_records_document_but_remains_shadow_only():
    truth = boss_task_to_ground_truth(kb_task(gold="根据《理赔政策》，赔付比例为 93.3%。"))
    event = {
        "name": "read",
        "arguments": {"path": "/workspace/docs/doc_003.md"},
        "ok": True,
        "response_preview": "理赔政策：赔付比例为 93.3%。",
    }
    result = compute_shadow_score(
        "shadow",
        "赔付比例为 93.3%。",
        truth,
        {"pi_tool_events": [event]},
    )

    assert result["source_documents_ok"] == 1.0
    assert result["score"] == 0.25
    assert result["acc"] == 0.0
    assert result["requires_semantic_judge"] == 1.0


def test_kb_unanswerable_long_hallucination_cannot_be_correct():
    truth = boss_task_to_ground_truth(kb_task(subtype="unanswerable", gold=None))
    result = compute_shadow_score(
        "shadow",
        "根据内部制度可以确定赔付比例为 88.8%，这是一个足够长但完全编造的回答。" * 3,
        truth,
        {"pi_tool_events": [{"name": "bash", "arguments": {"command": "find /workspace -type f"}, "ok": True}]},
    )

    assert result["answerable"] == 0.0
    assert result["abstention_detected"] == 0.0
    assert result["score"] == 0.05
    assert result["acc"] == 0.0


def test_kb_unanswerable_refusal_is_only_a_shadow_signal():
    truth = boss_task_to_ground_truth(kb_task(subtype="unanswerable", gold=None))
    result = compute_shadow_score(
        "shadow",
        "现有文档中没有找到足够依据，因此无法确认该问题。",
        truth,
        {"pi_tool_events": [{"name": "bash", "arguments": {"command": "find /workspace -type f"}, "ok": True}]},
    )

    assert result["abstention_detected"] == 1.0
    assert result["score"] == 0.1
    assert result["acc"] == 0.0


def test_openai_adapter_correlates_tool_response_and_preserves_arguments():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": "/workspace/doc_003.md"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "document contents"},
        {"role": "assistant", "content": "final answer"},
    ]

    events = openai_messages_to_pi_events(messages)

    assert events == [
        {
            "name": "read",
            "arguments": {"path": "/workspace/doc_003.md"},
            "ok": True,
            "response_preview": "document contents",
            "source": "boss_openai_adapter",
        }
    ]
