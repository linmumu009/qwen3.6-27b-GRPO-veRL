import json
from pathlib import Path

import pytest

from scripts.migrate_profiler_candidate_gold_hash import migrate
from scripts.profile_boss_sandbox_catalog import canonical_hash


def legacy_row():
    instruction = "计算总量"
    gold = {
        "answer_type": "numeric",
        "value": 3,
        "verification_sql": "SELECT 3",
    }
    return {
        "instruction": instruction,
        "instruction_sha256": canonical_hash(instruction),
        "gold": gold,
        "gold_sha256": canonical_hash({**gold, "legacy_metadata": True}),
    }


def test_migrate_repairs_exact_exported_gold_hash(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    source.write_text(json.dumps(legacy_row(), ensure_ascii=False) + "\n", encoding="utf-8")

    summary = migrate(source, output, expected_rows=1, expected_repairs=1)
    repaired = json.loads(output.read_text(encoding="utf-8"))

    assert summary["repaired_rows"] == 1
    assert repaired["gold_sha256"] == canonical_hash(repaired["gold"])


def test_migrate_fails_closed_on_unexpected_gold_shape(tmp_path: Path):
    row = legacy_row()
    row["gold"]["extra"] = "not exported by the v1 profiler"
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected exported gold shape"):
        migrate(source, tmp_path / "output.jsonl", expected_rows=1, expected_repairs=1)
