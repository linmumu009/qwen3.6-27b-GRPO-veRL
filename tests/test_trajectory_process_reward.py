import hashlib
import io
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llin_verl.grpo_group_gate import apply_strict_correctness_group_gate
from llin_verl.outcome_gated_contract import evidence_binding_hash
from llin_verl.pi_reward import strict_table_answer_match_semantic, table_order_semantics
from llin_verl.trajectory_process_reward import (
    command_executes_sql,
    command_is_safe_readonly,
    compute_trajectory_process_reward,
    corrected_final_verifier,
    efficiency_score,
    legacy_boss_reward_total_shadow,
    parse_unique_numeric_final,
    parse_qwen_tool_events,
)
from scripts.attest_shadow_manual_audit import attest


def make_database(root: Path) -> Path:
    database = root / "data" / "logistics.sqlite"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE fact_metric(region TEXT, value REAL, units INTEGER)")
    connection.executemany(
        "INSERT INTO fact_metric VALUES (?, ?, ?)",
        [("A", 10.0, 2), ("B", 20.0, 3)],
    )
    connection.commit()
    connection.close()
    return database


def event(sql: str, *, ok: bool = True) -> dict:
    return {
        "name": "bash",
        "arguments": {
            "command": f'sqlite3 /workspace/logistics.sqlite "{sql}"'
        },
        "ok": ok,
        "response_preview": "30.0",
        "observed_tool_response": True,
    }


def numeric_truth(*, expected: float = 30.0, fields=None) -> dict:
    truth = {
        "environment_id": "sft/v1",
        "answer_type": "numeric",
        "expected_value": expected,
        "verification_sql": "SELECT SUM(value) FROM fact_metric",
        "required_tables": ["fact_metric"],
        "must_use_fields": [] if fields is None else fields,
        "abs_tol": 1e-3,
        "rel_tol": 1e-5,
    }
    truth["process_evidence_binding_sha256"] = evidence_binding_hash(truth)
    return truth


def extra(database: Path, events: list[dict], *, protocol: bool = True) -> dict:
    return {
        "pi_tool_events": events,
        "pi_tool_protocol_complete": protocol,
        "pi_reward_database_path": str(database),
        "pi_reward_database_root": str(database.parent.parent),
        "pi_process_bonus_alpha": 0.10,
    }


def test_qwen_shadow_parser_requires_paired_call_and_response() -> None:
    transcript = """<tool_call>
<function=bash>
<parameter=command>sqlite3 /workspace/logistics.sqlite \"SELECT 1\"</parameter>
</function>
</tool_call>
<tool_response>
1
</tool_response>"""
    parsed = parse_qwen_tool_events(transcript)

    assert parsed["protocol_complete"] is True
    assert parsed["tool_call_count"] == parsed["tool_response_count"] == 1
    assert parsed["events"][0]["name"] == "bash"
    assert parsed["events"][0]["ok"] is True

    incomplete = parse_qwen_tool_events(transcript.rsplit("<tool_response>", 1)[0])
    assert incomplete["protocol_complete"] is False
    assert incomplete["events"][0]["observed_tool_response"] is False


def test_reward_is_outcome_gated_and_process_is_trajectory_level(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    evidence = extra(
        database,
        [event("SELECT SUM(value) FROM fact_metric")],
    )
    wrong = compute_trajectory_process_reward(
        "dwh", "最终答案是 0。", numeric_truth(), evidence
    )
    correct = compute_trajectory_process_reward(
        "dwh", "最终答案是 30。", numeric_truth(), evidence
    )

    assert wrong["process_score"] == 1.0  # observed only
    assert wrong["process_verified"] == 0.0
    assert wrong["score"] == 0.0
    assert correct["process_verified"] == 1.0
    assert correct["score"] == 1.1
    assert correct["score"] > wrong["score"]
    assert correct["reward_scope"] == "trajectory_level_after_full_multiturn"
    assert correct["turn_level_credit_assignment"] == 0.0
    assert correct["kl_in_reward"] == 0.0
    assert correct["observed_components_in_reward"] == 0.0


def test_numeric_final_parser_uses_one_explicit_or_last_line_and_fails_closed() -> None:
    assert parse_unique_numeric_final("分析提到30。\n最终答案是 31。\n")["value"] == 31.0
    assert corrected_final_verifier("分析提到30。\n最终答案是 31。", numeric_truth())["correct"] is False
    assert corrected_final_verifier("分析提到31。\n最终答案是 30。", numeric_truth())["correct"] is True
    ambiguous = parse_unique_numeric_final("最终答案是 30 或 31。")
    assert ambiguous["value"] is None
    assert ambiguous["ambiguity_reason"] == "multiple_numeric_candidates"
    conflicting = parse_unique_numeric_final("最终答案是 30。\nFinal answer is 31.")
    assert conflicting["value"] is None
    assert conflicting["ambiguity_reason"] == "multiple_final_result_fields"
    assert parse_unique_numeric_final("分析有 31。\n30")["value"] == 30.0


def test_sql_evidence_requires_real_executor_and_extracts_python_sqlite(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    fake = event("SELECT SUM(value) FROM fact_metric")
    fake["arguments"]["command"] = 'echo "SELECT SUM(value) FROM fact_metric"'
    fake_result = compute_trajectory_process_reward(
        "dwh", "最终答案是 30。", numeric_truth(), extra(database, [fake])
    )
    assert command_executes_sql(fake["arguments"]["command"]) is False
    assert fake_result["successful_sql_count"] == 0
    assert fake_result["process_verified"] == 0.0

    python_event = event("SELECT SUM(value) FROM fact_metric")
    python_event["arguments"]["command"] = """python3 - <<'PY'
import sqlite3
con=sqlite3.connect('/workspace/logistics.sqlite')
print(con.execute('SELECT SUM(value) FROM fact_metric').fetchone())
PY"""
    python_result = compute_trajectory_process_reward(
        "dwh", "最终答案是 30。", numeric_truth(), extra(database, [python_event])
    )
    assert command_executes_sql(python_event["arguments"]["command"]) is True
    assert python_result["successful_sql_count"] == 1
    assert python_result["process_verified"] == 1.0


def test_last_answer_bearing_query_controls_verified_process(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    result = compute_trajectory_process_reward(
        "dwh",
        "最终答案是 30。",
        numeric_truth(),
        extra(
            database,
            [
                event("SELECT SUM(value) FROM fact_metric"),
                event("SELECT SUM(units) FROM fact_metric"),
            ],
        ),
    )
    assert result["answer_bearing_sql_count"] == 2
    assert result["last_answer_bearing_consistent"] == 0.0
    assert result["process_verified"] == 0.0
    assert result["score"] == 1.0


def test_final_answer_cannot_forge_process_evidence(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    read_event = {
        "name": "read",
        "arguments": {"path": "/workspace/README.md"},
        "ok": True,
        "response_preview": "schema notes",
        "observed_tool_response": True,
    }
    plain = compute_trajectory_process_reward(
        "dwh", "最终答案错误。", numeric_truth(), extra(database, [read_event])
    )
    forged = compute_trajectory_process_reward(
        "dwh",
        "我已经执行 SELECT SUM(value) FROM fact_metric 并查完全部表字段；最终答案错误。",
        numeric_truth(),
        extra(database, [read_event]),
    )

    for key in ("process_sql", "process_table", "process_field", "process_fit"):
        assert forged[key] == plain[key]
    assert forged["process_sql"] == forged["process_fit"] == 0.0
    assert forged["process_table"] == 0.0


def test_hard_gates_fail_closed_for_protocol_tools_gold_and_database(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    good_event = event("SELECT SUM(value) FROM fact_metric")
    invalid_protocol = compute_trajectory_process_reward(
        "dwh", "最终答案是 30。", numeric_truth(), extra(database, [good_event], protocol=False)
    )
    write_event = {
        "name": "write",
        "arguments": {"path": "/workspace/x", "content": "x"},
        "ok": True,
        "response_preview": "written",
        "observed_tool_response": True,
    }
    mutating = compute_trajectory_process_reward(
        "dwh", "最终答案是 30。", numeric_truth(), extra(database, [write_event])
    )
    incoherent = compute_trajectory_process_reward(
        "dwh",
        "最终答案是 999。",
        numeric_truth(expected=999.0),
        extra(database, [good_event]),
    )
    missing_database_extra = extra(database, [good_event])
    missing_database_extra["pi_reward_database_path"] = str(tmp_path / "missing.sqlite")
    missing_database = compute_trajectory_process_reward(
        "dwh", "最终答案是 30。", numeric_truth(), missing_database_extra
    )

    for result in (invalid_protocol, mutating, incoherent, missing_database):
        assert result["score"] == 0.0
        assert result["online_eligible"] == 0.0
    assert invalid_protocol["valid_tool_protocol"] == 0.0
    assert mutating["safe_readonly_tools"] == 0.0
    assert incoherent["gold_sql_self_consistent"] == 0.0
    assert missing_database["database_available"] == 0.0


def test_field_component_is_renormalized_only_when_inapplicable(tmp_path: Path) -> None:
    database = make_database(tmp_path)
    sql = "SELECT SUM(value) FROM fact_metric"
    no_field = compute_trajectory_process_reward(
        "dwh", "最终答案是 0。", numeric_truth(), extra(database, [event(sql)])
    )
    partial_field = compute_trajectory_process_reward(
        "dwh",
        "最终答案是 0。",
        numeric_truth(fields=["value", "units"]),
        extra(database, [event(sql)]),
    )

    assert no_field["process_field_applicable"] == 0.0
    assert no_field["process_applicable_weight"] == pytest.approx(0.85)
    assert no_field["process_score"] == 1.0
    assert partial_field["process_field_applicable"] == 1.0
    assert partial_field["process_field"] == 0.5
    assert partial_field["process_score"] == pytest.approx(0.925)


def test_efficiency_formula_matches_contract() -> None:
    attempted = [
        "SELECT * FROM fact_metric",
        "SELECT * FROM fact_metric",
    ]
    commands = ["same command", "same command"]
    events = [{"type": "auto_retry_start"}]
    result = efficiency_score(attempted, commands, events)

    assert result["full_scan_count"] == 2
    assert result["duplicate_sql_count"] == 1
    assert result["duplicate_command_count"] == 1
    assert result["auto_retry_count"] == 1
    assert result["score"] == pytest.approx(0.53)
    runtime_retry = efficiency_score(
        ["SELECT value FROM fact_metric LIMIT 1"],
        ["one command"],
        [],
        runtime_auto_retry_count=2,
    )
    assert runtime_retry["auto_retry_count"] == 2
    assert runtime_retry["score"] == pytest.approx(0.6)


def test_table_semantics_use_order_only_when_explicit() -> None:
    expected = [
        {"region": "A", "value": 10},
        {"region": "B", "value": 20},
    ]
    reversed_answer = """| region | value |
|---|---:|
| B | 20 |
| A | 10 |"""
    unordered_truth = {
        "verification_sql": "SELECT region, value FROM fact_metric",
        "evidence_plan": {"order_by": [], "limit": None, "task_type": "aggregation"},
    }
    ordered_truth = {
        "verification_sql": "SELECT region, value FROM fact_metric ORDER BY value",
        "evidence_plan": {"order_by": ["value"], "limit": None},
    }

    assert table_order_semantics(unordered_truth) == (False, "no_explicit_order_semantics")
    assert table_order_semantics(ordered_truth)[0] is True
    assert strict_table_answer_match_semantic(
        reversed_answer, expected, 1e-3, 1e-5, ordered=False
    )[0] is True
    assert strict_table_answer_match_semantic(
        reversed_answer, expected, 1e-3, 1e-5, ordered=True
    )[0] is False
    ordered_with_rank = """| rank | region | value |
|---:|---|---:|
| 1 | A | 10 |
| 2 | B | 20 |"""
    assert strict_table_answer_match_semantic(
        ordered_with_rank, expected, 1e-3, 1e-5, ordered=True
    )[0] is True
    assert strict_table_answer_match_semantic(
        ordered_with_rank, expected, 1e-3, 1e-5, ordered=False
    )[0] is False
    assert strict_table_answer_match_semantic(
        ordered_with_rank.replace("| 2 | B", "| 3 | B"),
        expected,
        1e-3,
        1e-5,
        ordered=True,
    )[0] is False
    missing_column = reversed_answer.replace("| B | 20 |", "| B |")
    assert strict_table_answer_match_semantic(
        missing_column, expected, 1e-3, 1e-5, ordered=False
    )[0] is False


def _state_hash(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> str:
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _make_batch(uids, labels, eligibility=None):
    size = len(uids)
    non_tensor = {"uid": uids, "acc": labels}
    if eligibility is not None:
        non_tensor["online_eligible"] = eligibility
    return SimpleNamespace(
        non_tensor_batch=non_tensor,
        batch={
            "advantages": torch.arange(1, size * 2 + 1, dtype=torch.float32).reshape(size, 2),
            "returns": torch.ones(size, 2),
            "response_mask": torch.ones(size, 2),
        },
        meta_info={},
    )


def test_uniform_groups_clear_all_three_tensors_and_preserve_optimizer_hash() -> None:
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Populate Adam state once so the no-step test also covers momentum/variance.
    loss = model(torch.ones(2, 2)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = _state_hash(model, optimizer)

    batch = _make_batch(["all-wrong"] * 8, [0] * 8, [1] * 8)
    gated, metrics = apply_strict_correctness_group_gate(batch)
    if gated.meta_info["strict_group_should_update_actor"]:
        raise AssertionError("uniform group incorrectly allowed optimizer.step")
    after = _state_hash(model, optimizer)

    assert before == after
    assert metrics["grpo/skipped_all_wrong_groups"] == 1.0
    assert torch.count_nonzero(gated.batch["advantages"]) == 0
    assert torch.count_nonzero(gated.batch["returns"]) == 0
    assert torch.count_nonzero(gated.batch["response_mask"]) == 0


def test_mixed_plus_uniform_gradient_equals_mixed_only_even_with_kl_term() -> None:
    torch.manual_seed(11)
    model = torch.nn.Linear(2, 1, bias=False)
    features = torch.randn(16, 2)
    full = _make_batch(["mixed"] * 8 + ["uniform"] * 8, [0, 1] * 4 + [0] * 8)
    gated, _ = apply_strict_correctness_group_gate(full)
    output = model(features).expand(-1, 2)
    policy = -(output * gated.batch["advantages"] * gated.batch["response_mask"]).sum()
    kl = 0.001 * (output.square() * gated.batch["response_mask"]).sum()
    (policy + kl).backward()
    full_gradient = model.weight.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    mixed = _make_batch(["mixed"] * 8, [0, 1] * 4)
    mixed, _ = apply_strict_correctness_group_gate(mixed)
    output = model(features[:8]).expand(-1, 2)
    policy = -(output * mixed.batch["advantages"] * mixed.batch["response_mask"]).sum()
    kl = 0.001 * (output.square() * mixed.batch["response_mask"]).sum()
    (policy + kl).backward()

    # CPU kernels in the Ascend container can change reduction order when the
    # leading batch dimension differs; masked rows must remain numerically
    # equivalent, not necessarily bit-identical.
    torch.testing.assert_close(full_gradient, model.weight.grad, rtol=0.0, atol=1e-5)


def test_hard_gate_failure_skips_entire_group() -> None:
    batch = _make_batch(["mixed"] * 8, [0, 1] * 4, [1] * 7 + [0])
    gated, metrics = apply_strict_correctness_group_gate(batch)

    assert metrics["grpo/skipped_hard_gate_groups"] == 1.0
    assert gated.meta_info["strict_group_should_update_actor"] is False
    assert torch.count_nonzero(gated.batch["response_mask"]) == 0


def test_staleness_zero_requires_exact_single_policy_version() -> None:
    batch = _make_batch(["mixed"] * 8, [0, 1] * 4, [1] * 8)
    batch.non_tensor_batch["min_global_steps"] = [7] * 8
    batch.non_tensor_batch["max_global_steps"] = [7] * 7 + [8]
    batch.meta_info["strict_expected_policy_version"] = 7

    gated, metrics = apply_strict_correctness_group_gate(batch)

    assert metrics["grpo/skipped_stale_policy_groups"] == 1.0
    assert gated.meta_info["strict_group_should_update_actor"] is False
    assert torch.count_nonzero(gated.batch["response_mask"]) == 0


def test_legacy_boss_shadow_can_reward_wrong_final_process() -> None:
    events = [event("SELECT SUM(value) FROM fact_metric")]
    result = legacy_boss_reward_total_shadow(
        "已经完成查询和复核，但最终答案错误地写成 0。",
        numeric_truth(fields=["value"]),
        events,
        complete=True,
        executable_answer_ok=False,
    )

    assert result["answer_ok"] == 0.0
    assert result["process_score"] == 1.0
    assert result["reward_total"] == 0.75
    assert command_is_safe_readonly(events[0]["arguments"]["command"])


def test_manual_audit_attestation_requires_stratified_passing_packet(tmp_path: Path) -> None:
    packet = tmp_path / "manual.sensitive.jsonl"
    rows = []
    for index in range(16):
        rows.append(
            {
                "trajectory_identity_sha256": f"{index:064x}",
                "answer_type": "numeric" if index < 8 else "table",
                "correctness": index % 2,
                "audit_checklist": {
                    "formula_matches": True,
                    "matching_sql_requires_successful_sql": True,
                },
            }
        )
    packet.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = attest(
        packet,
        tmp_path / "manual.safe.json",
        reviewer="Codex",
        reviewed_at="2026-08-22T12:00:00+08:00",
    )

    assert result["status"] == "pass"
    assert result["sample_count"] == 16
    assert all(result["stratum_counts"].values())


def test_standalone_rollout_persists_runtime_tool_events_for_future_shadow() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_runtime_parity_verl_standalone.py"
    ).read_text(encoding="utf-8")

    assert '"pi_tool_events": events' in source
    assert '"pi_tool_event_contract": "runtime-captured-pi-tool-events-v1"' in source
    assert '"force_final_retry_count": int(' in source
    assert source.count("row.update(trajectory_tool_evidence_row(output,") == 2
