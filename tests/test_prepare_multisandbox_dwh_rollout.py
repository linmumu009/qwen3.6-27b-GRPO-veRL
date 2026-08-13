import json
from pathlib import Path

import pyarrow.parquet as pq

from llin_verl.boss_pi_contract import load_boss_pi_contract
from scripts.prepare_multisandbox_dwh_rollout import prepare
from scripts.profile_boss_sandbox_catalog import canonical_hash


def candidate(version, task_id, instruction, task_type, answer_type="numeric", **updates):
    gold = {
        "answer_type": answer_type,
        "value": 12.5 if answer_type == "numeric" else [{"category": "华东", "value": 12.5}],
        "verification_sql": "SELECT SUM(value) FROM fact_metric",
    }
    row = {
        "version": version,
        "task_id": task_id,
        "environment_id": f"sft/{version}",
        "task_type": task_type,
        "task_category": "answerable",
        "instruction": instruction,
        "instruction_sha256": canonical_hash(instruction),
        "gold": gold,
        "gold_sha256": canonical_hash(gold),
        "expected_tables": ["fact_metric"],
        "globally_unique_instruction": True,
        "mechanical_sql_verified": True,
        "semantic_review_flags": [],
    }
    row.update(updates)
    return row


def test_prepare_builds_strict_full_and_stratified_probe(tmp_path: Path):
    rows = [
        candidate("20260628_v20", "same", "数值聚合", "aggregate_query"),
        candidate("20260628_v21", "same", "单指标", "single_metric_query"),
        candidate("20260628_v22", "c", "数值比较", "comparison_analysis"),
        candidate("20260628_v23", "d", "表格聚合", "aggregate_query", "table"),
        candidate("20260628_v24", "e", "表格单指标", "single_metric_query", "table"),
        candidate("20260628_v25", "f", "表格比较", "comparison_analysis", "table"),
        candidate("20260628_v15", "old", "旧v15", "aggregate_query"),
        candidate(
            "20260628_v26",
            "trend",
            "趋势题",
            "trend_analysis",
        ),
    ]
    source = tmp_path / "candidates.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = prepare(source, tmp_path / "out", expected_rows=6, probe_tasks=6, seed="s")

    assert manifest["strict_rows"] == 6
    assert manifest["probe_rows"] == 6
    assert manifest["task_types"] == {
        "aggregate_query": 2,
        "comparison_analysis": 2,
        "single_metric_query": 2,
    }
    full = pq.read_table(tmp_path / "out" / "boss_multisandbox_dwh_281.sensitive.parquet").to_pylist()
    assert len(full) == 6
    assert len({row["extra_info"]["verifier_id"] for row in full}) == 6
    contract = load_boss_pi_contract()
    guidance = contract["runtime"]["guidance_prefix"]
    assert all(row["prompt"][0]["content"] == contract["system_prompt"] for row in full)
    assert all(row["prompt"][1]["content"].startswith(guidance) for row in full)
    assert all(row["extra_info"]["training_allowed"] is False for row in full)
    assert all(row["extra_info"]["tool_selection"] == ["bash", "read", "write", "edit"] for row in full)
    assert manifest["contains_prompts_gold_sql_task_ids_or_server_paths"] is False


def test_prepare_fails_closed_when_profiler_contract_changes(tmp_path: Path):
    row = candidate("20260628_v20", "a", "题目", "aggregate_query")
    row["semantic_review_flags"] = ["warning"]
    source = tmp_path / "candidates.jsonl"
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        prepare(source, tmp_path / "out", expected_rows=1, probe_tasks=1, seed="s")
    except ValueError as exc:
        assert "profiler contract" in str(exc)
    else:
        raise AssertionError("expected the strict builder to fail closed")
