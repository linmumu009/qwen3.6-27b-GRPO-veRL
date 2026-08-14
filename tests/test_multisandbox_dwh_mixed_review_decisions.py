import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "configs" / "multisandbox_dwh_mixed_review_decisions_20260814.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def test_mixed_review_decisions_are_anonymous_consistent_and_fail_closed():
    payload = json.loads(DECISIONS.read_text(encoding="utf-8"))
    decisions = payload["decisions"]
    assert decisions
    assert len({row["instruction_sha256"] for row in decisions}) == len(decisions)

    approved = []
    rejected = []
    for row in decisions:
        assert SHA256_RE.fullmatch(row["instruction_sha256"])
        assert SHA256_RE.fullmatch(row["gold_sha256"])
        assert 1 <= row["correct_count"] <= 7
        checks = [
            row["instruction_unambiguously_entails_gold"],
            row["verification_sql_fully_answers_instruction"],
            row["expected_value_supported_by_query_result"],
            row["final_outcome_routing_trustworthy"],
        ]
        if row["decision"] == "approved_candidate":
            assert all(checks)
            approved.append(row)
        else:
            assert row["decision"] == "rejected"
            assert not all(checks)
            rejected.append(row)

    summary = payload["summary"]
    assert summary["reviewed"] == len(decisions)
    assert summary["approved_candidates"] == len(approved)
    assert summary["rejected"] == len(rejected)
    assert payload["training_allowed"] is False
    assert payload["promotion_allowed"] is False
    assert payload[
        "contains_task_ids_prompts_sql_gold_values_final_answers_or_tool_outputs"
    ] is False

    serialized = DECISIONS.read_text(encoding="utf-8")
    for forbidden in (
        '"task_id"',
        '"prompt"',
        '"verification_sql"',
        '"expected_value_json"',
        '"final_answer"',
        '"tool_output"',
    ):
        assert forbidden not in serialized
