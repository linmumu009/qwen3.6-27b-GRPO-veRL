import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from llin_verl.boss_pi_contract import load_boss_pi_contract
from scripts.apply_qwen38_grpo_candidate_review import apply_review
import scripts.audit_qwen38_grpo_candidates as audit
from scripts.audit_open_multisandbox_dwh import canonical_hash


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    database = sqlite3.connect(source / "logistics.sqlite")
    database.execute("CREATE TABLE fact_shipments(category TEXT, value INTEGER)")
    database.executemany(
        "INSERT INTO fact_shipments VALUES (?, ?)",
        [("A", 10), ("B", 8)],
    )
    database.commit()
    database.close()

    instruction = "请按业务结果从高到低列出前5项，口径包含2025年1月、签收、区域和运单量。"
    sql = "SELECT category, value FROM fact_shipments ORDER BY value DESC LIMIT 5"
    gold = [{"category": "A", "value": 10}, {"category": "B", "value": 8}]
    task = {
        "task_id": "sensitive-task-id",
        "task_type": "shipment_count",
        "source_sandbox_version": "20260628_v15",
        "difficulty_level": 2,
        "natural_language_instruction": instruction,
        "semantic_anchors": ["2025年1月", "签收", "区域", "运单量", "5"],
        "semantic_contract": {
            "family": "shipment_count",
            "metric": "shipment_count",
            "group_dimension": "region",
            "required_anchors": ["2025年1月", "签收", "区域", "运单量", "5"],
            "explanation_is_open_ended": False,
            "verified_deliverable": "top_five_category_value_table",
        },
        "query_plan": {
            "family": "shipment_count",
            "metric_key": "shipment_count",
            "group_field": "region",
            "anchors": ["2025年1月", "签收", "区域", "运单量", "5"],
            "output_shape": "category_value_table",
        },
        "expected_tables": ["fact_shipments"],
        "gold_answer": {
            "answer_type": "table",
            "value": gold,
            "verification_sql": sql,
        },
        "validation": {"result_sha256": canonical_hash(gold)},
        "instruction_generation": {"semantic_validation_passed": True},
        "_qa_status": "passed",
        "training_allowed": False,
    }
    (source / "dwh_tasks.jsonl").write_text(
        json.dumps(task, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    contract = load_boss_pi_contract()
    row = {
        "data_source": "llin_open_multisandbox_dwh_step120_rollout_v1",
        "agent_name": "pi_agent",
        "prompt": [
            {"role": "system", "content": contract["system_prompt"]},
            {
                "role": "user",
                "content": contract["runtime"]["guidance_prefix"] + instruction,
            },
        ],
        "ability": "boss_pi_dwh",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answer_type": "table",
                "expected_value_json": json.dumps(
                    gold, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "verification_sql": sql,
                "required_tables": ["fact_shipments"],
                "task_family": "open_plan_first_dwh",
                "reward_contract": "pure-final-outcome-screening-v1",
            },
        },
        "extra_info": {
            "source_version": "20260628_v15",
            "difficulty_level": 2,
            "instruction_sha256": canonical_hash(instruction),
            "gold_sha256": canonical_hash(gold),
            "correct_count": 1,
            "completed_count": 2,
            "timeout_count": 0,
            "mechanical_screen_passed": True,
            "api_semantic_validation_passed": True,
            "explicit_semantic_reviewed": False,
            "training_allowed": False,
            "promotion_allowed": False,
            "response_messages_in_grpo_input": 0,
        },
    }
    candidate_dir = tmp_path / "v15"
    candidate_dir.mkdir()
    candidate = candidate_dir / "grpo_variance_candidates.sensitive.parquet"
    pq.write_table(pa.Table.from_pylist([row]), candidate)
    selector = tmp_path / "selector.json"
    audit.extract_selectors([candidate], selector, host_label="m05")
    return source, candidate, selector


def _passing_judgments(tasks, config, *, pass_index, max_retries):
    del config, pass_index, max_retries
    return [
        {
            "instruction_unambiguously_entails_plan": True,
            "plan_fully_answers_instruction": True,
            "final_answer_contract_is_clear": True,
            "reason_codes": [],
            "confidence": "high",
        }
        for _ in tasks
    ]


def test_extract_and_audit_approve_only_after_two_semantic_passes(tmp_path, monkeypatch):
    source, _, selector = _fixture(tmp_path)
    api_config = tmp_path / "api.json"
    api_config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(audit, "_load_api_config", lambda path: {})
    monkeypatch.setattr(audit, "semantic_judge_batch", _passing_judgments)

    summary = audit.audit_candidates(
        [selector],
        {"20260628_v15": source},
        tmp_path / "review",
        api_config_path=api_config,
        local_deterministic_semantic_review=False,
        expected_count=1,
        batch_size=1,
        max_retries=1,
        abs_tol=0.0,
    )

    assert summary["candidate_count"] == 1
    assert summary["approved_candidates"] == 1
    assert summary["rejected_candidates"] == 0
    assert summary["needs_manual_review"] == 0
    assert summary["training_allowed"] is False
    decisions = json.loads(
        (tmp_path / "review" / "candidate_review_decisions.sensitive.json").read_text(
            encoding="utf-8"
        )
    )
    decision = decisions["decisions"][0]
    assert decision["decision"] == "approved_candidate"
    assert all(decision["deterministic_checks"].values())
    assert len(decision["semantic_judge_passes"]) == 2


def test_audit_without_semantic_judge_stays_pending(tmp_path):
    source, _, selector = _fixture(tmp_path)
    summary = audit.audit_candidates(
        [selector],
        {"20260628_v15": source},
        tmp_path / "review",
        api_config_path=None,
        local_deterministic_semantic_review=False,
        expected_count=1,
        batch_size=1,
        max_retries=1,
        abs_tol=0.0,
    )
    assert summary["approved_candidates"] == 0
    assert summary["needs_manual_review"] == 1


def test_apply_review_writes_only_approved_rows_with_training_disabled(tmp_path, monkeypatch):
    source, candidate, selector = _fixture(tmp_path)
    api_config = tmp_path / "api.json"
    api_config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(audit, "_load_api_config", lambda path: {})
    monkeypatch.setattr(audit, "semantic_judge_batch", _passing_judgments)
    audit.audit_candidates(
        [selector],
        {"20260628_v15": source},
        tmp_path / "review",
        api_config_path=api_config,
        local_deterministic_semantic_review=False,
        expected_count=1,
        batch_size=1,
        max_retries=1,
        abs_tol=0.0,
    )

    summary = apply_review(
        [candidate],
        tmp_path / "review" / "candidate_review_decisions.sensitive.json",
        tmp_path / "applied",
        host_label="m05",
    )
    assert summary["reviewed_candidates"] == 1
    assert summary["approved_candidates"] == 1
    assert summary["training_allowed"] is False
    rows = pq.read_table(
        tmp_path / "applied" / "semantic_approved_candidates.sensitive.parquet"
    ).to_pylist()
    assert len(rows) == 1
    assert rows[0]["extra_info"]["explicit_semantic_reviewed"] is True
    assert rows[0]["extra_info"]["training_allowed"] is False
    assert rows[0]["extra_info"]["promotion_allowed"] is False


def test_local_semantic_review_accepts_equivalent_open_explanation_wording():
    anchors = [
        "2025年1月",
        "电子",
        "区域",
        "全流程改善优先",
        "20%",
        "35%",
        "45%",
        "3",
        "5",
    ]
    instruction = (
        "请判断2025年1月电子业务中最该关注的区域，采用全流程改善优先口径；"
        "综合分按运单量20%、派送35%、全流程45%，先折算成百分制，少于3票不参与，"
        "从高到低列出前5项和综合分，并解释判断。"
    )
    sql = """WITH base AS (
SELECT region AS category FROM fact_waybill WHERE month = '2025-01' AND goods = '电子'
), normalized AS (
SELECT region, volume, volume_index, delivery_index, process_index FROM base WHERE volume >= 3
)
SELECT region AS category,
ROUND(0.20 * volume_index + 0.35 * delivery_index + 0.45 * process_index, 2) AS value
FROM normalized ORDER BY value DESC, category ASC LIMIT 5"""
    task = {
        "task_type": "management_prioritization",
        "difficulty_level": 5,
        "natural_language_instruction": instruction,
        "semantic_anchors": anchors,
        "semantic_contract": {
            "family": "management_prioritization",
            "metric": "attention_score",
            "group_dimension": "region",
            "required_anchors": anchors,
            "explanation_is_open_ended": True,
            "verified_deliverable": "top_five_category_value_table",
        },
        "query_plan": {
            "family": "management_prioritization",
            "metric_key": "attention_score",
            "group_field": "region",
            "anchors": anchors,
            "output_shape": "category_value_table",
        },
        "gold_answer": {"verification_sql": sql},
        "expected_tables": ["fact_waybill"],
        "expected_operations": [
            "filter",
            "aggregate",
            "join",
            "group_by",
            "order_by",
            "top_k",
        ],
    }
    judgments = audit.local_semantic_review(task)
    assert len(judgments) == 2
    assert all(not row["reason_codes"] for row in judgments)
    assert all(row["instruction_unambiguously_entails_plan"] for row in judgments)
