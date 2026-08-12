import json
from pathlib import Path

import pandas as pd

from llin_verl.boss_pi_contract import canonical_json
from scripts.analyze_semantic_plan_sufficiency_gate import decide
from scripts.check_semantic_plan_sufficiency_gate import check_dataset
from scripts.prepare_semantic_plan_gate_outputs import prepare as prepare_outputs
from scripts.prepare_semantic_plan_sufficiency_gate import (
    ARMS,
    HINT_PREFIX,
    full_semantic_plan,
    operator_plan,
    sha256_value,
)


ROOT = Path(__file__).resolve().parents[1]


def _database(path: Path) -> None:
    import sqlite3

    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE orders (
            carrier_id TEXT,
            amount REAL,
            created_at TEXT
        );
        CREATE TABLE carriers (
            carrier_id TEXT,
            carrier_name TEXT
        );
        INSERT INTO orders VALUES ('c1', 10, '2026-01-02');
        INSERT INTO carriers VALUES ('c1', 'alpha');
        """
    )
    connection.close()


def test_plans_separate_operators_from_schema_and_literals(tmp_path: Path):
    database = tmp_path / "gate.sqlite"
    _database(database)
    sql = (
        "SELECT c.carrier_name, SUM(o.amount) FROM orders o "
        "JOIN carriers c ON o.carrier_id = c.carrier_id "
        "WHERE DATE(o.created_at) >= '2026-01-01' "
        "GROUP BY c.carrier_name ORDER BY SUM(o.amount) DESC LIMIT 3"
    )

    operator = operator_plan(sql)
    encoded_operator = canonical_json(operator)
    assert operator["aggregation"]["calls"] == [{"function": "SUM", "distinct": False}]
    assert operator["aggregation"]["grouping_required"] is True
    assert operator["filter"]["comparison_operators"] == [">="]
    assert operator["temporal"]["operators"] == ["DATE"]
    assert "orders" not in encoded_operator
    assert "carrier_name" not in encoded_operator
    assert "2026-01-01" not in encoded_operator

    full = full_semantic_plan(sql, database)
    grounding = full["grounding"]
    assert grounding["tables"] == ["carriers", "orders"]
    assert grounding["columns_by_role"]["projection"] == [
        "carriers.carrier_name",
        "orders.amount",
    ]
    assert grounding["columns_by_role"]["filter"] == ["orders.created_at"]
    assert grounding["equality_join_edges"] == [
        ["carriers.carrier_id", "orders.carrier_id"]
    ]
    assert "2026-01-01" not in canonical_json(full)

    no_aliases = full_semantic_plan(
        "SELECT SUM(amount) FROM orders JOIN carriers "
        "ON orders.carrier_id = carriers.carrier_id",
        database,
    )
    assert no_aliases["grounding"]["tables"] == ["carriers", "orders"]


def _gate_files(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    evidence = []
    base = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "error-call",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": "sqlite3 /workspace/logistics.sqlite 'SELECT amount FROM orders'"
                        },
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "error-call", "content": "10"},
    ]
    plans = {
        "control": None,
        "operator_oracle": {
            "scope": "operator_only",
            "aggregation": {"calls": [{"function": "SUM", "distinct": False}]},
        },
        "full_plan_oracle": {
            "scope": "full_semantic_plan",
            "aggregation": {"calls": [{"function": "SUM", "distinct": False}]},
            "grounding": {
                "tables": ["orders"],
                "columns_by_role": {"projection": ["orders.amount"]},
                "join_count": 0,
                "equality_join_edges": [],
            },
        },
    }
    instruction = "Return exactly one bash tool call with one read-only SELECT, then stop."
    for index in range(16):
        task_id = f"task_{index:06d}"
        for arm in ARMS:
            gate_id = f"{task_id}::{arm}"
            hint = {
                "role": "user",
                "content": HINT_PREFIX
                + json.dumps(
                    {"instruction": instruction, "semantic_plan": plans[arm]},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            prompt = json.loads(json.dumps(base)) + [hint]
            rows.append(
                {
                    "prompt": prompt,
                    "data_source": "semantic_plan_sufficiency_gate",
                    "semantic_plan_gate_id": gate_id,
                    "semantic_plan_gate_arm": arm,
                    "semantic_plan_gate_source_task_id": task_id,
                    "extra_info": {
                        "tool_selection": ["bash"],
                        "semantic_plan_gate_id": gate_id,
                    },
                    "reward_model": {
                        "ground_truth": {
                            "task_id": task_id,
                            "semantic_plan_gate_id": gate_id,
                            "semantic_plan_gate_arm": arm,
                            "semantic_plan_gate_source_task_id": task_id,
                            "semantic_plan_gate_aggregation_critical": index < 9,
                        }
                    },
                }
            )
            evidence.append(
                {
                    "gate_id": gate_id,
                    "task_id": task_id,
                    "arm": arm,
                    "aggregation_critical": index < 9,
                    "base_prompt_sha256": sha256_value(prompt[:4]),
                    "hint_sha256": sha256_value(hint),
                    "error_query_sha256": sha256_value("SELECT amount FROM orders"),
                    "correction_query_sha256": sha256_value("SELECT SUM(amount) FROM orders"),
                    "semantic_difference_labels": ["aggregation_grouping"],
                }
            )
    data_file = tmp_path / "gate.parquet"
    pd.DataFrame(rows).to_parquet(data_file, index=False)
    contract_file = tmp_path / "contract.json"
    contract_file.write_text(
        json.dumps(
            {
                "contract": "semantic-plan-sufficiency-gate-dataset-v1",
                "rows": 48,
                "output_sha256": __import__("hashlib").sha256(data_file.read_bytes()).hexdigest(),
                "evidence": evidence,
            }
        ),
        encoding="utf-8",
    )
    return data_file, contract_file


def test_cpu_gate_checks_48_row_balance_and_plan_boundaries(tmp_path: Path):
    data_file, contract_file = _gate_files(tmp_path)
    result = check_dataset(data_file, contract_file)

    assert result["rows"] == 48
    assert result["tasks"] == 16
    assert result["rows_per_arm"] == {arm: 16 for arm in sorted(ARMS)}
    assert result["aggregation_critical_tasks"] == 9
    assert result["all_arms_share_identical_first_error_state"] is True


def test_one_turn_output_adapter_audits_parallel_calls_without_executing_them(tmp_path: Path):
    validation = tmp_path / "validation.jsonl"
    output = tmp_path / "generated.jsonl"
    rows = []
    for index in range(16):
        task_id = f"task_{index:06d}"
        for arm in ARMS:
            output_text = (
                "<tool_call>\n<function=bash>\n<parameter=command>\n"
                "sqlite3 /workspace/logistics.sqlite 'SELECT 1'\n"
                "</parameter>\n</function>\n</tool_call>"
            )
            if index == 0 and arm == "operator_oracle":
                output_text += (
                    "\n<tool_call>\n<function=read>\n<parameter=path>\n"
                    "/workspace/schema.txt\n</parameter>\n</function>\n</tool_call>"
                )
            rows.append(
                {
                    "gts": {
                        "semantic_plan_gate_id": f"{task_id}::{arm}",
                        "semantic_plan_gate_arm": arm,
                        "semantic_plan_gate_source_task_id": task_id,
                    },
                    "output": output_text,
                }
            )
    validation.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = prepare_outputs(validation, output)

    assert summary["rows"] == 48
    assert summary["contract"] == "semantic-plan-sufficiency-gate-output-adapter-v2"
    assert summary["generated_tool_calls"] == 49
    assert summary["generated_bash_tool_calls"] == 48
    assert summary["tool_call_name_counts"] == {"bash": 48, "read": 1}
    assert summary["tool_call_count_histogram"] == {"1": 47, "2": 1}
    assert summary["rows_with_exactly_one_bash_call"] == 47
    assert summary["rows_with_multiple_tool_calls"] == 1
    assert summary["max_tool_calls_per_row"] == 2
    assert summary["generated_tool_responses"] == 0
    adapted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    parallel = next(row for row in adapted if row["gate_id"].endswith("::operator_oracle"))
    assert parallel["output_protocol"] == {
        "tool_call_count": 2,
        "bash_tool_call_count": 1,
        "tool_call_names": ["bash", "read"],
        "exactly_one_bash_call": False,
    }


def test_decision_prefers_operator_then_full_then_realization():
    def rows(operator_success: int, full_success: int, control_success: int = 2):
        result = []
        for index in range(16):
            task_id = f"task_{index:06d}"
            for arm, count in (
                ("control", control_success),
                ("operator_oracle", operator_success),
                ("full_plan_oracle", full_success),
            ):
                result.append(
                    {
                        "task_id": task_id,
                        "arm": arm,
                        "aggregation_critical": index < 9,
                        "verified_or_equivalent": index < count,
                    }
                )
        return result

    assert decide(rows(4, 8))["selected_training_target"] == (
        "semantic_plan_selection_or_contrast_supervision"
    )
    assert decide(rows(3, 8))["selected_training_target"] == (
        "schema_grounding_and_compositional_plan_training"
    )
    assert decide(rows(3, 7))["selected_training_target"] == (
        "plan_to_sql_realization_and_recovery"
    )


def test_launcher_is_forward_only_one_generation_and_saves_no_checkpoint():
    script = (ROOT / "scripts" / "run_semantic_plan_sufficiency_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "tasks=16" in script
    assert "rows=48" in script
    assert "MAX_ASSISTANT_TURNS=1" in script
    assert "MAX_USER_TURNS=1" in script
    assert "generated_tool_execution=false" in script
    assert "run_pi_frozen_baseline.sh" in script
    assert "trainer.val_only=True" in script
    assert "trainer.save_freq=-1" in script
    assert "optimizer_initialized=false" in script
    assert "checkpoint_saved=false" in script
