import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_disjoint_short_protocol_summary_is_safe_and_fail_closed() -> None:
    path = ROOT / "docs" / "disjoint_step120_short_protocol_audit_20260812_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    payload = path.read_text(encoding="utf-8")

    assert summary["scope"]["tasks"] == 64
    assert summary["native"]["rows_with_recognized_readonly_sqlite"] == 1
    assert summary["step120"]["rows_with_recognized_readonly_sqlite"] == 4
    assert summary["native"]["duplicate_bash_calls"] == 83
    assert summary["step120"]["duplicate_bash_calls"] == 83
    gate = summary["pair_gate"]
    assert gate["observed_pairs"] == 1
    assert gate["minimum_pairs"] == 48
    assert gate["gate_passed"] is False
    assert gate["optimizer_initialized"] is False
    assert gate["training_allowed"] is False
    assert gate["promotion_allowed"] is False
    assert "/workspace/" not in payload
    assert "SELECT " not in payload
    assert "task_000" not in payload
