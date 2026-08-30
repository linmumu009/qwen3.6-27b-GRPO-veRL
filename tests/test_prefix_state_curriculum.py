from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3

import pytest

from llin_verl.outcome_gated_contract import evidence_binding_hash
from llin_verl.prefix_state_curriculum import (
    RESET_MODE,
    adapt_pi_prefix_messages,
    prefix_group_base,
    prefix_group_key,
    require_same_prefix_group,
    stable_json_sha256,
    validate_ready_state,
    validate_runtime_prefix,
    validate_suffix_response_mask,
)
from llin_verl.tiered_query_cost_reward import compute_tiered_query_cost_reward
from scripts.patch_verl_grpo_strict_variance_gate import patch_trainer
from scripts.prepare_prefix_state_curriculum_runtime import _runtime_row


ROOT = Path(__file__).resolve().parents[1]


def pi_messages() -> list[dict]:
    return [
        {"role": "system", "content": "system contract"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {
                    "type": "toolCall",
                    "id": "call-1",
                    "name": "bash",
                    "arguments": {"command": "sqlite3 logistics.sqlite 'SELECT 1'"},
                },
            ],
        },
        {
            "role": "toolResult",
            "toolCallId": "call-1",
            "toolName": "bash",
            "content": [{"type": "text", "text": "1"}],
            "isError": False,
        },
    ]


def state(*, ready: bool = True) -> dict:
    messages = pi_messages()
    return {
        "prefix_state_id": "state-1",
        "task_id": "task-1",
        "source_trajectory_id": "trajectory-1",
        "split": "train",
        "stage": "stage-01",
        "prefix_messages": json.dumps(messages),
        "prefix_message_count": len(messages),
        "generated_suffix_start_message_index": len(messages),
        "response_mask_unit": "message_index_requires_runtime_token_boundary_adapter",
        "remaining_assistant_decisions": 2,
        "remaining_tool_rounds": 1,
        "environment_id": "sft/v1",
        "database_sha256": "db-sha",
        "workspace_identity_sha256": "workspace-sha",
        "workspace_reset_contract": json.dumps(
            {
                "database_mount": "read_only_sha256_bound",
                "mode": RESET_MODE,
                "mutable_prefix_requires_snapshot": True,
                "snapshot_available": False,
            }
        ),
        "reward_contract": "tiered-query-cost-trajectory-shadow-v1",
        "reward_scope": "generated_suffix_only",
        "final_correctness_scope": "combined_prefix_plus_suffix",
        "prefix_participates_in_gradient": False,
        "prefix_counts_toward_process_or_efficiency_reward": False,
        "inherited_evidence": True,
        "prefix_query_attempt_count": 1,
        "prefix_tool_response_tokens": 12,
        "training_ready": ready,
        "quarantine_reasons": json.dumps([] if ready else ["mutable_state"]),
    }


def truth_row() -> dict:
    truth = {
        "task_id": "task-1",
        "environment_id": "sft/v1",
        "answer_type": "numeric",
        "expected_value_json": "30.0",
        "verification_sql": "SELECT SUM(value) FROM fact_metric",
        "required_tables": ["fact_metric"],
        "must_use_fields": ["value"],
        "abs_tol": 1e-3,
        "rel_tol": 1e-5,
    }
    return {
        "task_id": "task-1",
        "database_sha256": "db-sha",
        "gold_sql_model_visible": False,
        "dataset_record": {
            "data_source": "dwh",
            "ability": "dwh",
            "prompt": [{"role": "user", "content": "replaced"}],
            "reward_model": {"style": "rule", "ground_truth": truth},
            "extra_info": {
                "environment_id": "sft/v1",
                "instruction_sha256": "a" * 64,
                "training_allowed": False,
                "tool_selection": ["bash"],
            },
        },
    }


def test_native_roles_and_tool_call_ids_survive_without_flattening() -> None:
    adapted = adapt_pi_prefix_messages(pi_messages())
    assert [message["role"] for message in adapted] == ["system", "user", "assistant", "tool"]
    assert adapted[2]["tool_calls"][0]["id"] == "call-1"
    assert adapted[3]["tool_call_id"] == "call-1"
    assert isinstance(adapted[2]["tool_calls"][0]["function"]["arguments"], dict)
    assert all("thinking" not in json.dumps(message) for message in adapted)


def test_future_or_partial_tool_state_fails_closed() -> None:
    hidden = pi_messages()
    hidden[2]["content"].insert(0, {"type": "thinking", "thinking": "secret"})
    with pytest.raises(ValueError, match="hidden or unsupported"):
        adapt_pi_prefix_messages(hidden)
    partial = pi_messages()[:-1]
    with pytest.raises(ValueError, match="incomplete tool round"):
        adapt_pi_prefix_messages(partial)


def test_quarantine_and_unrestorable_mutable_state_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-ready"):
        validate_ready_state(state(ready=False))
    mutable = state()
    mutable["prefix_mutable_side_effects"] = True
    with pytest.raises(ValueError, match="no restorable snapshot"):
        validate_ready_state(mutable)


def test_runtime_row_keeps_truth_hidden_and_declares_suffix_only_boundary() -> None:
    runtime = _runtime_row(state(), truth_row(), "/run/private/pi_sandbox")
    extra = runtime["extra_info"]
    assert runtime["prompt"][2]["tool_calls"][0]["id"] == "call-1"
    assert extra["generated_suffix_start_message_index"] == len(runtime["prompt"])
    assert extra["response_mask_scope"] == "generated_suffix_assistant_tokens_only"
    assert extra["prefix_counts_toward_process_or_efficiency_reward"] is False
    assert extra["teacher_suffix_model_visible"] is False
    assert "expected_value_json" not in json.dumps(runtime["prompt"])
    validate_runtime_prefix(extra, runtime["prompt"])
    changed = copy.deepcopy(runtime["prompt"])
    changed[-1]["content"] = "tampered"
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        validate_runtime_prefix(extra, changed)


def test_suffix_response_mask_excludes_prompt_and_generated_tool_tokens() -> None:
    validate_suffix_response_mask([1, 2, 3], [4, 5, 6, 7], [1, 1, 0, 1])
    with pytest.raises(ValueError, match="lengths differ"):
        validate_suffix_response_mask([1], [2, 3], [1])
    with pytest.raises(ValueError, match="binary"):
        validate_suffix_response_mask([1], [2], [2])


def test_exact_group_identity_includes_task_prefix_and_policy() -> None:
    assert prefix_group_base("task", "state") == "task::state"
    assert prefix_group_key("task", "state", 3) == "task::state::policy-3"
    assert require_same_prefix_group(["task"] * 8, ["state"] * 8, [3] * 8) == (
        "task::state::policy-3"
    )
    with pytest.raises(ValueError, match="different prefix"):
        require_same_prefix_group(["task"] * 8, ["a"] * 7 + ["b"], [3] * 8)


def test_prefix_evidence_and_cost_do_not_enter_frozen_reward(tmp_path: Path) -> None:
    database = tmp_path / "sft" / "v1" / "logistics.sqlite"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE fact_metric(value REAL)")
        connection.executemany("INSERT INTO fact_metric VALUES (?)", [(10.0,), (20.0,)])
    truth = truth_row()["dataset_record"]["reward_model"]["ground_truth"]
    truth["process_evidence_binding_sha256"] = evidence_binding_hash(truth)
    base = {
        "pi_reward_database_root": str(tmp_path),
        "request_id": "request-1",
        "pi_trajectory_request_id": "request-1",
        "pi_trajectory_environment_id": "sft/v1",
        "pi_environment_id": "sft/v1",
        "pi_tool_log_present": True,
        "pi_tool_protocol_complete": True,
        "trajectory_timeout": False,
        "runtime_error": False,
        "instruction_sha256": "a" * 64,
        # Audit-only inherited values must never be read by the reward.
        "prefix_query_attempt_count_audit_only": 99,
        "prefix_tool_response_tokens_audit_only": 99999,
    }
    guess = compute_tiered_query_cost_reward("dwh", "Final result: 30", truth, {**base, "pi_tool_events": []})
    assert guess["reward"] == 0
    assert guess["guess_correct_blocked"] == 1
    assert guess["query_attempt_count"] == 0
    assert guess["tool_response_tokens"] == 0

    event = {
        "name": "bash",
        "arguments": {"command": "sqlite3 logistics.sqlite 'SELECT SUM(value) FROM fact_metric'"},
        "sql_statements": ["SELECT SUM(value) FROM fact_metric"],
        "ok": True,
        "response_preview": "30",
        "response_token_count": 20,
        "observed_tool_response": True,
        "call_parse_valid": True,
        "source": "runtime_structured_pi_workspace",
        "command_origin": "model",
        "workspace_request_id": "request-1",
        "environment_id": "sft/v1",
    }
    grounded = compute_tiered_query_cost_reward(
        "dwh",
        "Final result: 30",
        truth,
        {
            **base,
            "pi_tool_events": [event],
            "pi_workspace_request_id": "request-1",
            "pi_workspace_released": True,
        },
    )
    assert grounded["success"] == 1
    assert grounded["reward"] >= 0.8
    assert grounded["query_attempt_count"] == 1
    assert grounded["tool_response_tokens"] == 20


def test_trainer_patch_uses_exact_prefix_policy_group_key(tmp_path: Path) -> None:
    source = ROOT / "reference" / "verl" / "verl" / "experimental" / "separation" / "ray_trainer.py"
    target = tmp_path / "ray_trainer.py"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    assert patch_trainer(target) == "patched"
    text = target.read_text(encoding="utf-8")
    assert "LLIN_PREFIX_POLICY_GROUP_KEY_V6" in text
    assert "prefix_group_key(str(task_id), str(state_id), policy_version)" in text
    assert 'getattr(self, "current_param_version", self.global_steps - 1)' in text
    compile(text, str(target), "exec")
