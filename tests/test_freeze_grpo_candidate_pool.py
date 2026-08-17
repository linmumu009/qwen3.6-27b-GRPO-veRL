from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.freeze_grpo_candidate_pool import SourceSpec, freeze_pool, text_sha256


def _row(instruction: str, *, training_allowed: bool = False) -> dict:
    return {
        "data_source": "test",
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": instruction},
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answer_type": "numeric",
                "expected_value_json": "1",
                "verification_sql": "SELECT 1",
            },
        },
        "extra_info": {
            "instruction_sha256": text_sha256(instruction),
            "difficulty_level": 3,
            "training_allowed": training_allowed,
        },
    }


def _parquet(path: Path, rows: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_freeze_exact_pool_with_selector_and_safe_summary(tmp_path: Path):
    first = _parquet(tmp_path / "first.parquet", [_row("one"), _row("two")])
    full = _parquet(tmp_path / "full.parquet", [_row("three"), _row("four")])
    selector = tmp_path / "selector.jsonl"
    selector.write_text(
        json.dumps({"instruction_sha256": text_sha256("four")}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "pool.sensitive.parquet"
    safe = tmp_path / "pool.safe.json"

    summary = freeze_pool(
        [
            SourceSpec("reviewed", first, 2),
            SourceSpec("selected", full, 1, selector),
        ],
        output_path=output,
        safe_summary_path=safe,
        expected_total=3,
    )

    rows = pq.read_table(output).to_pylist()
    assert len(rows) == 3
    assert [row["extra_info"]["candidate_pool_source"] for row in rows] == [
        "reviewed",
        "reviewed",
        "selected",
    ]
    assert all(row["extra_info"]["training_allowed"] is False for row in rows)
    assert summary["source_counts"] == {"reviewed": 2, "selected": 1}
    assert summary["unique_instruction_identities"] == 3
    assert summary["difficulty_counts"] == {"3": 3}
    assert summary["contains_prompts_gold_sql_task_ids_tool_outputs_or_server_paths"] is False
    assert json.loads(safe.read_text(encoding="utf-8"))["output_sha256"]


def test_freeze_rejects_cross_source_instruction_collision(tmp_path: Path):
    first = _parquet(tmp_path / "first.parquet", [_row("same")])
    second = _parquet(tmp_path / "second.parquet", [_row("same")])

    with pytest.raises(ValueError, match="cross-source instruction collision"):
        freeze_pool(
            [SourceSpec("first", first, 1), SourceSpec("second", second, 1)],
            output_path=tmp_path / "pool.parquet",
            safe_summary_path=tmp_path / "safe.json",
            expected_total=2,
        )


def test_freeze_rejects_enabled_or_incomplete_rows(tmp_path: Path):
    enabled = _parquet(tmp_path / "enabled.parquet", [_row("enabled", training_allowed=True)])
    with pytest.raises(ValueError, match="already training-enabled"):
        freeze_pool(
            [SourceSpec("enabled", enabled, 1)],
            output_path=tmp_path / "pool.parquet",
            safe_summary_path=tmp_path / "safe.json",
            expected_total=1,
        )

    incomplete_row = _row("missing gold")
    incomplete_row["reward_model"]["ground_truth"] = None
    incomplete = _parquet(tmp_path / "incomplete.parquet", [incomplete_row])
    with pytest.raises(ValueError, match="no hidden verifier material"):
        freeze_pool(
            [SourceSpec("incomplete", incomplete, 1)],
            output_path=tmp_path / "pool2.parquet",
            safe_summary_path=tmp_path / "safe2.json",
            expected_total=1,
        )


def test_freeze_rejects_wrong_source_count_or_missing_selector(tmp_path: Path):
    source = _parquet(tmp_path / "source.parquet", [_row("one")])
    with pytest.raises(ValueError, match="expected 2 rows"):
        freeze_pool(
            [SourceSpec("source", source, 2)],
            output_path=tmp_path / "pool.parquet",
            safe_summary_path=tmp_path / "safe.json",
            expected_total=2,
        )

    selector = tmp_path / "selector.jsonl"
    selector.write_text(
        json.dumps({"instruction_sha256": text_sha256("absent")}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selector identities are absent"):
        freeze_pool(
            [SourceSpec("source", source, 1, selector)],
            output_path=tmp_path / "pool2.parquet",
            safe_summary_path=tmp_path / "safe2.json",
            expected_total=1,
        )
