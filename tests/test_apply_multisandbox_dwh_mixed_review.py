import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.apply_multisandbox_dwh_mixed_review import apply_decisions


def test_apply_decisions_keeps_only_semantically_approved_rows_fail_closed(tmp_path: Path):
    mixed = tmp_path / "mixed.parquet"
    rows = [
        {
            "prompt": [{"role": "user", "content": f"q{index}"}],
            "reward_model": {"ground_truth": {"expected_value_json": str(index)}},
            "extra_info": {
                "instruction_sha256": f"i{index}",
                "gold_sha256": f"g{index}",
                "explicit_semantic_reviewed": False,
                "training_allowed": False,
            },
        }
        for index in range(2)
    ]
    pq.write_table(pa.Table.from_pylist(rows), mixed)
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "arm": "m06",
                        "instruction_sha256": "i0",
                        "gold_sha256": "g0",
                        "correct_count": 3,
                        "decision": "approved_candidate",
                        "instruction_unambiguously_entails_gold": True,
                        "verification_sql_fully_answers_instruction": True,
                        "expected_value_supported_by_query_result": True,
                        "final_outcome_routing_trustworthy": True,
                    },
                    {
                        "arm": "m06",
                        "instruction_sha256": "i1",
                        "gold_sha256": "g1",
                        "correct_count": 2,
                        "decision": "rejected",
                        "instruction_unambiguously_entails_gold": False,
                        "verification_sql_fully_answers_instruction": False,
                        "expected_value_supported_by_query_result": True,
                        "final_outcome_routing_trustworthy": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = apply_decisions(
        mixed,
        decisions,
        tmp_path / "output",
        arm="m06",
        require_all_reviewed=True,
    )

    assert summary["reviewed"] == 2
    assert summary["approved_candidates"] == 1
    assert summary["rejected"] == 1
    assert summary["unreviewed"] == 0
    assert summary["training_allowed"] is False
    approved = pq.read_table(
        tmp_path / "output" / "semantic_approved_candidates.sensitive.parquet"
    ).to_pylist()
    assert len(approved) == 1
    assert approved[0]["extra_info"]["instruction_sha256"] == "i0"
    assert approved[0]["extra_info"]["explicit_semantic_reviewed"] is True
    assert approved[0]["extra_info"]["training_allowed"] is False
