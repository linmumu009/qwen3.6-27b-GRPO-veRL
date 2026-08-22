#!/usr/bin/env python3
"""Generate and execute task-bound mutation tests for all approved43 tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from llin_verl.grounded_trajectory_reward import (
    REWARD_CONTRACT,
    capture_deterministic_composition,
    compute_grounded_trajectory_reward,
)
from llin_verl.outcome_gated_contract import evidence_binding_hash
from llin_verl.pi_reward import _normalize_full_expected_table, execute_readonly_sql, table_order_semantics


CONTRACT = "qwen38-approved43-grounded-verifier-casepack-v2"
APPROVED_SHA256 = "d86b53d906806b150d43a508dce9b0dd6d05105c07e03961e8e7bf9439ccd944"
MANIFEST_SHA256 = "1426bc09a3dbaf4709fd89227790603afb7a2bf11beeba80946057d490e0f424"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_private_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _truth(approved_row: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(approved_row["reward_model"]["ground_truth"], ensure_ascii=False))
    criteria = task.get("verification_criteria") or {}
    value["evidence_plan"] = task.get("evidence_plan") or {}
    value["required_tables"] = task.get("expected_tables") or value.get("required_tables", [])
    value["must_use_fields"] = criteria.get("must_use_fields") or value.get("must_use_fields", [])
    value["process_evidence_binding_sha256"] = evidence_binding_hash(value)
    return value


def _format_table(rows: list[list[Any]]) -> str:
    width = len(rows[0]) if rows else 1
    header = [f"c{index + 1}" for index in range(width)]
    lines = ["|" + "|".join(header) + "|", "|" + "|".join(["---"] * width) + "|"]
    lines.extend("|" + "|".join(str(value) for value in row) + "|" for row in rows)
    return "\n".join(lines)


def _expected(truth: dict[str, Any]) -> Any:
    if "expected_value_json" in truth:
        return json.loads(str(truth["expected_value_json"]))
    return truth.get("expected_value")


def _final(truth: dict[str, Any], *, rows: list[list[Any]] | None = None, wrong: bool = False) -> str:
    if str(truth.get("answer_type") or "") == "numeric":
        expected = float(_expected(truth))
        if wrong:
            expected += max(1.0, abs(expected) * 0.01, float(truth.get("abs_tol", 1e-3)) * 10)
        return f"最终答案是 {expected:.12g}。"
    table = rows if rows is not None else _normalize_full_expected_table(_expected(truth))
    return _format_table([list(row) for row in (table or [])])


def _event(sql: str, rows: list[tuple[Any, ...]], *, observed: bool = True, valid: bool = True) -> dict[str, Any]:
    if '"""' in sql:
        raise ValueError("casepack SQL contains unsupported triple-quote delimiter")
    command = (
        "python3 -c 'import sqlite3; "
        f'print(sqlite3.connect("/workspace/logistics.sqlite").execute("""{sql}""").fetchall())'
        "'"
    )
    event = {
        "name": "bash",
        "arguments": {"command": command},
        "ok": True,
        "response_preview": json.dumps([list(row) for row in rows], ensure_ascii=False, default=str),
        "response_truncated": False,
        "observed_tool_response": observed,
        "call_parse_valid": valid,
        "source": "runtime_structured_pi_workspace",
    }
    if not observed:
        event.pop("response_preview")
    return event


_WHERE_RE = re.compile(
    r"\bwhere\b(.*?)(?=\bgroup\s+by\b|\bhaving\b|\border\s+by\b|\blimit\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_JOIN_RE = re.compile(
    r"(\bjoin\s+[^\s,(]+(?:\s+(?:as\s+)?[A-Za-z_][A-Za-z0-9_$]*)?\s+on\s+)"
    r"(.*?)(?=\bjoin\b|\bwhere\b|\bgroup\s+by\b|\bhaving\b|\border\s+by\b|\blimit\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_GROUP_RE = re.compile(
    r"(\bgroup\s+by\b)(.*?)(?=\bhaving\b|\border\s+by\b|\blimit\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_ORDER_RE = re.compile(r"(\border\s+by\b)(.*?)(?=\blimit\b|$)", re.IGNORECASE | re.DOTALL)
_LIMIT_RE = re.compile(r"(\blimit\s+)(\d+)", re.IGNORECASE)
_DATE_LITERAL_RE = re.compile(r"'\d{4}-\d{2}-\d{2}(?:[ T][^']*)?'")
_STRING_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")
_NUMERIC_LITERAL_RE = re.compile(r"(?<![A-Za-z0-9_$])\d+(?:\.\d+)?")


def _where_span(sql: str) -> tuple[int, int] | None:
    match = _WHERE_RE.search(sql)
    return (match.start(1), match.end(1)) if match else None


def _mutate_time_filter(sql: str) -> str | None:
    span = _where_span(sql)
    if not span:
        return None
    start, end = span
    clause = sql[start:end]
    match = _DATE_LITERAL_RE.search(clause)
    if not match:
        return None
    replacement = "'1900-01-01'" if match.group(0) != "'1900-01-01'" else "'2999-12-31'"
    return sql[: start + match.start()] + replacement + sql[start + match.end() :]


def _mutate_filter_value(sql: str) -> str | None:
    span = _where_span(sql)
    if not span:
        return None
    start, end = span
    clause = sql[start:end]
    string_match = _STRING_LITERAL_RE.search(clause)
    if string_match:
        replacement = "'__grounded_wrong_filter__'"
        return sql[: start + string_match.start()] + replacement + sql[start + string_match.end() :]
    number_match = _NUMERIC_LITERAL_RE.search(clause)
    if number_match:
        replacement = str(float(number_match.group(0)) + 9973.0)
        return sql[: start + number_match.start()] + replacement + sql[start + number_match.end() :]
    return None


def _mutate_join_condition(sql: str) -> str | None:
    match = _JOIN_RE.search(sql)
    if not match:
        return None
    return sql[: match.start(2)] + " 1 = 0 " + sql[match.end(2) :]


def _mutate_group_granularity(sql: str) -> str | None:
    match = _GROUP_RE.search(sql)
    if not match:
        return None
    return sql[: match.start(2)] + " '__grounded_wrong_group__' " + sql[match.end(2) :]


def _mutate_order(sql: str) -> str | None:
    match = _ORDER_RE.search(sql)
    if not match:
        return None
    clause = match.group(2)
    if re.search(r"\bdesc\b", clause, re.IGNORECASE):
        changed = re.sub(r"\bdesc\b", "ASC", clause, count=1, flags=re.IGNORECASE)
    elif re.search(r"\basc\b", clause, re.IGNORECASE):
        changed = re.sub(r"\basc\b", "DESC", clause, count=1, flags=re.IGNORECASE)
    else:
        changed = clause.rstrip() + " DESC "
    return sql[: match.start(2)] + changed + sql[match.end(2) :]


def _mutate_topn(sql: str) -> str | None:
    match = _LIMIT_RE.search(sql)
    if not match:
        return None
    value = int(match.group(2))
    replacement = str(value + 1 if value == 0 else max(0, value - 1))
    return sql[: match.start(2)] + replacement + sql[match.end(2) :]


def _mutate_unit_ratio(sql: str, answer_type: str) -> str | None:
    if re.search(r"\b100(?:\.0+)?\b", sql):
        return re.sub(r"\b100(?:\.0+)?\b", "1.0", sql, count=1)
    if answer_type == "numeric":
        return f"SELECT (SELECT * FROM ({sql.rstrip(';')})) * 1000.0"
    return None


def _extra(database: Path, events: list[dict[str, Any]], *, protocol: bool = True) -> dict[str, Any]:
    return {
        "pi_tool_events": events,
        "pi_tool_log_present": True,
        "pi_tool_protocol_complete": protocol,
        "pi_reward_database_path": str(database),
        "pi_reward_database_root": str(database.parent.parent),
        "trajectory_timeout": False,
        "runtime_error": False,
        "pi_tool_event_source": "runtime_structured_pi_workspace",
    }


def _run_case(
    *,
    task_position: int,
    name: str,
    expected_state: str,
    final: str,
    events: list[dict[str, Any]],
    truth: dict[str, Any],
    database: Path,
    applicable: bool = True,
) -> dict[str, Any]:
    if not applicable:
        return {
            "anonymous_task_position": task_position,
            "case": name,
            "applicable": False,
            "expected_state": "NOT_APPLICABLE",
            "passed": True,
        }
    result = compute_grounded_trajectory_reward("dwh", final, truth, _extra(database, events))
    return {
        "anonymous_task_position": task_position,
        "case": name,
        "applicable": True,
        "expected_state": expected_state,
        "observed_state": result["judge_state"],
        "judge_reason": result["judge_reason"],
        "evidence_route": result["evidence_route"],
        "passed": result["judge_state"] == expected_state,
        "private_final": final,
        "private_events": events,
    }


def _run_adversarial_case(
    *,
    task_position: int,
    name: str,
    sql: str | None,
    final: str,
    truth: dict[str, Any],
    database: Path,
) -> dict[str, Any]:
    if not sql:
        return _run_case(
            task_position=task_position,
            name=name,
            expected_state="NOT_APPLICABLE",
            final=final,
            events=[],
            truth=truth,
            database=database,
            applicable=False,
        )
    try:
        rows = execute_readonly_sql(database, sql, query_timeout_seconds=5.0)
    except (OSError, ValueError, sqlite3.Error):
        return _run_case(
            task_position=task_position,
            name=name,
            expected_state="NOT_APPLICABLE",
            final=final,
            events=[],
            truth=truth,
            database=database,
            applicable=False,
        )
    result = compute_grounded_trajectory_reward(
        "dwh", final, truth, _extra(database, [_event(sql, rows)])
    )
    return {
        "anonymous_task_position": task_position,
        "case": name,
        "applicable": True,
        "expected_state": "FAIL_OR_UNKNOWN",
        "observed_state": result["judge_state"],
        "judge_reason": result["judge_reason"],
        "evidence_route": result["evidence_route"],
        "passed": result["judge_state"] != "PASS",
        "private_final": final,
        "private_events": [_event(sql, rows)],
        "adversarial_wrong_sql": True,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if file_sha256(args.approved43) != APPROVED_SHA256:
        raise ValueError("approved43 Parquet hash mismatch")
    if file_sha256(args.manifest) != MANIFEST_SHA256:
        raise ValueError("approved43 manifest hash mismatch")
    approved = pq.read_table(args.approved43).to_pylist()
    manifest = read_jsonl(args.manifest)
    tasks = read_jsonl(args.tasks)
    if len(approved) != 43 or len(manifest) != 43:
        raise ValueError("approved package is not exact 43")
    manifest_ids = {str(row["instruction_sha256"]) for row in manifest}
    approved_ids = {str(row["extra_info"]["instruction_sha256"]) for row in approved}
    if len(approved_ids) != 43 or approved_ids != manifest_ids:
        raise ValueError("approved Parquet and manifest identities differ")

    cases: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for position, approved_row in enumerate(approved):
        source_index = int(approved_row["extra_info"]["global_index"])
        task = tasks[source_index]
        truth = _truth(approved_row, task)
        gold_sql = str(truth["verification_sql"])
        gold_rows = execute_readonly_sql(args.database, gold_sql, query_timeout_seconds=5.0)
        answer_type = str(truth.get("answer_type") or "")
        normalized = _normalize_full_expected_table(_expected(truth)) if answer_type == "table" else None
        correct_final = _final(truth)
        wrong_final = _final(truth, wrong=True) if answer_type == "numeric" else _final(
            truth, rows=[list(row) for row in (normalized or [])[:-1]]
        )
        wrappers = [
            f"SELECT * FROM ({gold_sql.rstrip(';')}) AS equivalent_q",
            f"WITH equivalent_q AS ({gold_sql.rstrip(';')}) SELECT * FROM equivalent_q",
        ]
        direct_event = _event(gold_sql, gold_rows)
        task_cases = [
            _run_case(task_position=position, name="gold_direct_query", expected_state="PASS", final=correct_final, events=[direct_event], truth=truth, database=args.database),
            *[
                _run_case(
                    task_position=position,
                    name=f"equivalent_sql_variant_{index + 1}",
                    expected_state="PASS",
                    final=correct_final,
                    events=[_event(sql, execute_readonly_sql(args.database, sql, query_timeout_seconds=5.0))],
                    truth=truth,
                    database=args.database,
                )
                for index, sql in enumerate(wrappers)
            ],
            _run_case(task_position=position, name="correct_final_without_tool", expected_state="FAIL", final=correct_final, events=[], truth=truth, database=args.database),
        ]
        if answer_type == "numeric":
            delta = max(1.0, abs(float(_expected(truth))) * 0.01)
            wrong_sql = f"SELECT (SELECT * FROM ({gold_sql.rstrip(';')})) + {delta:.12g}"
        else:
            keep = max(0, len(gold_rows) - 1)
            wrong_sql = f"SELECT * FROM ({gold_sql.rstrip(';')}) AS wrong_q LIMIT {keep}"
        wrong_rows = execute_readonly_sql(args.database, wrong_sql, query_timeout_seconds=5.0)
        task_cases.extend([
            _run_case(task_position=position, name="wrong_sql_guess_correct", expected_state="FAIL", final=correct_final, events=[_event(wrong_sql, wrong_rows)], truth=truth, database=args.database),
            _run_case(task_position=position, name="correct_sql_wrong_final", expected_state="FAIL", final=wrong_final, events=[direct_event], truth=truth, database=args.database),
        ])
        unrelated_sql = "SELECT COUNT(*) FROM sqlite_master"
        task_cases.append(_run_case(
            task_position=position,
            name="correct_evidence_then_unrelated_1x1",
            expected_state="PASS",
            final=correct_final,
            events=[direct_event, _event(unrelated_sql, execute_readonly_sql(args.database, unrelated_sql, query_timeout_seconds=5.0))],
            truth=truth,
            database=args.database,
        ))

        if answer_type == "numeric":
            part_a = gold_sql.rstrip(";")
            part_b = "SELECT 0"
            expression = "a + b"
            composed_output: Any = float(_expected(truth))
            selector = "fetchone()[0]"
        else:
            part_a = gold_sql.rstrip(";")
            part_b = "SELECT 0 WHERE 0"
            expression = "a + b"
            composed_output = [list(row) for row in gold_rows]
            selector = "fetchall()"
        command = (
            "python3 - <<'PY'\nimport sqlite3\n"
            "con = sqlite3.connect('/workspace/logistics.sqlite')\n"
            f"a = con.execute({part_a!r}).{selector}\n"
            f"b = con.execute({part_b!r}).{selector}\n"
            f"result = {expression}\nprint(result)\nPY"
        )
        response = str(composed_output)
        trace = capture_deterministic_composition(command, response)
        composed_event = {
            "name": "bash",
            "arguments": {"command": command},
            "ok": True,
            "response_preview": response,
            "response_truncated": False,
            "observed_tool_response": True,
            "call_parse_valid": True,
            "source": "runtime_structured_pi_workspace",
            "composition_trace": trace,
        }
        task_cases.append(_run_case(
            task_position=position,
            name="multiquery_python_deterministic_composition",
            expected_state="PASS",
            final=correct_final,
            events=[composed_event],
            truth=truth,
            database=args.database,
        ))

        missing = dict(direct_event)
        missing["observed_tool_response"] = False
        missing.pop("response_preview", None)
        task_cases.append(_run_case(
            task_position=position,
            name="missing_tool_response",
            expected_state="UNKNOWN",
            final=correct_final,
            events=[missing],
            truth=truth,
            database=args.database,
        ))
        unsafe = dict(direct_event)
        unsafe["arguments"] = {"command": "rm /workspace/logistics.sqlite"}
        malformed = dict(direct_event)
        malformed["call_parse_valid"] = False
        task_cases.extend([
            _run_case(task_position=position, name="unsafe_model_behavior", expected_state="FAIL", final=correct_final, events=[unsafe], truth=truth, database=args.database),
            _run_case(task_position=position, name="malformed_model_behavior", expected_state="FAIL", final=correct_final, events=[malformed], truth=truth, database=args.database),
        ])

        adversarial = {
            "adversarial_delete_time_filter": _mutate_time_filter(gold_sql),
            "adversarial_change_filter_value": _mutate_filter_value(gold_sql),
            "adversarial_omit_join_condition": _mutate_join_condition(gold_sql),
            "adversarial_change_aggregation_granularity": _mutate_group_granularity(gold_sql),
            "adversarial_wrong_order": _mutate_order(gold_sql),
            "adversarial_wrong_topn": _mutate_topn(gold_sql),
            "adversarial_wrong_unit_or_ratio": _mutate_unit_ratio(gold_sql, answer_type),
        }
        for name, mutated_sql in adversarial.items():
            task_cases.append(
                _run_adversarial_case(
                    task_position=position,
                    name=name,
                    sql=mutated_sql,
                    final=correct_final,
                    truth=truth,
                    database=args.database,
                )
            )

        if answer_type == "table" and normalized:
            rows = [list(row) for row in normalized]
            ordered, _ = table_order_semantics(truth)
            mutations = {
                "table_reordered": (list(reversed(rows)), "FAIL" if ordered and len(rows) > 1 else "PASS"),
                "table_missing_row": (rows[:-1], "FAIL"),
                "table_missing_column": ([row[:-1] for row in rows], "FAIL"),
                "table_duplicate_row": (rows + [rows[0]], "FAIL"),
            }
            for name, (mutated, expected_state) in mutations.items():
                task_cases.append(_run_case(
                    task_position=position,
                    name=name,
                    expected_state=expected_state,
                    final=_final(truth, rows=mutated),
                    events=[direct_event],
                    truth=truth,
                    database=args.database,
                ))
        else:
            for name in ("table_reordered", "table_missing_row", "table_missing_column", "table_duplicate_row"):
                task_cases.append(_run_case(
                    task_position=position,
                    name=name,
                    expected_state="NOT_APPLICABLE",
                    final=correct_final,
                    events=[direct_event],
                    truth=truth,
                    database=args.database,
                    applicable=False,
                ))
        cases.extend(task_cases)
        applicable = [row for row in task_cases if row["applicable"]]
        task_summaries.append({
            "anonymous_task_position": position,
            "answer_type": answer_type,
            "applicable_cases": len(applicable),
            "passed_cases": sum(bool(row["passed"]) for row in applicable),
            "all_pass": all(bool(row["passed"]) for row in applicable),
        })

    failures = [row for row in cases if row["applicable"] and not row["passed"]]
    adversarial_rows = [row for row in cases if row.get("adversarial_wrong_sql")]
    adversarial_wrong_pass = [row for row in adversarial_rows if row.get("observed_state") == "PASS"]
    safe = {
        "contract": CONTRACT,
        "reward_contract": REWARD_CONTRACT,
        "training_status": "paused_cpu_only_no_model_no_rollout_no_optimizer_no_npu",
        "approved_tasks": len(approved),
        "tasks_with_casepacks": len(task_summaries),
        "all_43_tasks_executed": len(task_summaries) == 43,
        "case_rows": len(cases),
        "applicable_case_rows": sum(bool(row["applicable"]) for row in cases),
        "passed_case_rows": sum(bool(row["passed"]) for row in cases if row["applicable"]),
        "failed_case_rows": len(failures),
        "adversarial_wrong_sql_applicable_rows": len(adversarial_rows),
        "adversarial_wrong_sql_pass_count": len(adversarial_wrong_pass),
        "zero_wrong_semantic_mutations_pass": not adversarial_wrong_pass,
        "tasks_all_pass": sum(bool(row["all_pass"]) for row in task_summaries),
        "anonymous_task_results": task_summaries,
        "required_case_families": sorted({row["case"] for row in cases}),
        "table_task_count": sum(row["answer_type"] == "table" for row in task_summaries),
        "numeric_task_count": sum(row["answer_type"] == "numeric" for row in task_summaries),
        "approved43_parquet_sha256": APPROVED_SHA256,
        "approved43_manifest_sha256": MANIFEST_SHA256,
        "database_sha256": file_sha256(args.database),
        "private_casepack_mode": "0600",
        "formal_training_allowed": False,
        "status": "pass" if not failures and not adversarial_wrong_pass and len(task_summaries) == 43 else "fail",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_private_jsonl(args.output_dir / "private" / "verifier_cases.sensitive.jsonl", cases)
    (args.output_dir / "safe_summary.json").write_text(
        json.dumps(safe, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved43", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
