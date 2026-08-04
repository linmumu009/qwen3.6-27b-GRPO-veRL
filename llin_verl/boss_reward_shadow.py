"""Shadow reward aligned with the boss KB/DWH evaluation contracts.

This module is deliberately not wired into the training launchers.  It ports
only the deterministic, auditable parts of the boss evaluators and makes the
remaining semantic gaps explicit.  In particular, KB samples never become
online-reward eligible merely because an answer is long or repeats a document
identifier.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from llin_verl.pi_reward import (
    contains_expected_number,
    execute_readonly_sql,
    extract_final_assistant_answer,
    extract_numbers,
    extract_selects,
    final_answer_correct,
    rows_equal,
)
from llin_verl.pi_tool_contract import command_is_safe, extract_table_names


ALLOWED_TOOLS = {"bash", "read", "write", "edit"}
_DOC_TITLE_RE = re.compile(r"《([^》]+)》")
_VERSION_STATUS_RE = re.compile(r"(?:版本|状态)[:：]\s*([^\s，。；,;.）)]+)")
_ABSTENTION_RE = re.compile(
    r"(?:无法|不能|未能|没有).{0,16}(?:确认|判断|找到|检索到|提供|回答)|"
    r"(?:信息|材料|文档|依据|证据).{0,12}(?:不足|缺失|不存在)|"
    r"无法从现有",
    re.IGNORECASE,
)
_TOOL_ERROR_RE = re.compile(
    r"(?:traceback|permissionerror|filenotfounderror|command timed out|"
    r"no such file|not found|error:|exception:)",
    re.IGNORECASE,
)
_DOCUMENT_CONTENT_COMMAND_RE = re.compile(
    r"(?:^|[;&|()\s])(?:cat|grep|rg|sed|awk|head|tail|less|more)\b",
    re.IGNORECASE,
)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_content_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return _content_text(value[key])
    return ""


def final_answer_from_openai(messages: list[dict[str, Any]]) -> str:
    """Return the last terminal assistant answer from an OpenAI trajectory."""
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        if message.get("tool_calls"):
            return ""
        return _content_text(message.get("content")).strip()
    return ""


def openai_messages_to_pi_events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt boss OpenAI messages to the audited ``pi_tool_events`` contract."""
    responses: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id"):
            responses[str(message["tool_call_id"])] = _content_text(message.get("content"))

    events: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            name = str(function.get("name") or call.get("name") or "")
            raw_arguments = function.get("arguments", call.get("arguments", {}))
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": raw_arguments}
            else:
                arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
            response = responses.get(str(call.get("id") or ""), "")
            events.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "ok": bool(response) and not _TOOL_ERROR_RE.search(response),
                    "response_preview": response[:4000],
                    "source": "boss_openai_adapter",
                }
            )
    return events


def _event_commands(events: Iterable[dict[str, Any]]) -> list[str]:
    return [
        str((event.get("arguments") or {}).get("command"))
        for event in events
        if event.get("name") == "bash"
        and isinstance((event.get("arguments") or {}).get("command"), str)
    ]


def _document_tokens(value: str) -> set[str]:
    folded = str(value or "").strip().casefold()
    if not folded:
        return set()
    path = Path(folded)
    return {folded, path.name, path.stem}


def successful_document_access(
    events: Iterable[dict[str, Any]],
    source_documents: Iterable[str],
) -> tuple[set[str], set[str]]:
    """Return required and accessed documents from successful tool evidence.

    The final answer is intentionally excluded: mentioning ``doc_003`` is not
    evidence that the agent read it.
    """
    required = {str(value).strip() for value in source_documents if str(value).strip()}
    accessed: set[str] = set()
    for event in events:
        if not event.get("ok"):
            continue
        name = str(event.get("name") or "")
        arguments = event.get("arguments") or {}
        command = str(arguments.get("command") or "") if isinstance(arguments, dict) else ""
        # Listing or finding a filename is discovery, not evidence that the
        # document content was read.  Count the native read tool or a successful
        # shell content command only.
        if name != "read" and not (
            name == "bash" and _DOCUMENT_CONTENT_COMMAND_RE.search(command)
        ):
            continue
        blob = (
            _json_text(arguments)
            + "\n"
            + str(event.get("response_preview") or "")
        ).casefold()
        for document in required:
            if any(token and token in blob for token in _document_tokens(document)):
                accessed.add(document)
    return required, accessed


def _contains_value(
    rows: list[tuple[Any, ...]],
    value: Any,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    for row in rows:
        for actual in row:
            if isinstance(value, (int, float)) and isinstance(actual, (int, float)):
                if math.isclose(float(actual), float(value), abs_tol=abs_tol, rel_tol=rel_tol):
                    return True
            elif str(value).strip().casefold() == str(actual).strip().casefold():
                return True
    return False


def gold_supported_by_rows(
    answer_type: str,
    expected_value: Any,
    rows: list[tuple[Any, ...]],
    abs_tol: float,
    rel_tol: float,
) -> bool:
    if answer_type == "numeric":
        return isinstance(expected_value, (int, float)) and _contains_value(
            rows, expected_value, abs_tol, rel_tol
        )
    if answer_type != "table" or not isinstance(expected_value, list) or not expected_value:
        return False
    for item in expected_value:
        if not isinstance(item, dict) or not _contains_value(
            rows, item.get("value"), abs_tol, rel_tol
        ):
            return False
        label = item.get("category", item.get("date"))
        if label is not None and not _contains_value(rows, label, 0.0, 0.0):
            return False
    return True


def _expected_value(ground_truth: dict[str, Any]) -> Any:
    value = ground_truth.get("expected_value")
    if "expected_value_json" in ground_truth:
        try:
            value = json.loads(str(ground_truth["expected_value_json"]))
        except json.JSONDecodeError:
            return None
    return value


def boss_task_to_ground_truth(task: dict[str, Any]) -> dict[str, Any]:
    """Losslessly adapt a boss task row to the shadow reward contract."""
    task_id = str(task.get("task_id") or "")
    family = str(task.get("type") or "").casefold()
    version = str(task.get("v") or "")
    base = {
        "task_id": task_id,
        "task_family": family,
        "environment_id": f"sft/{version}" if version else "",
        "subtype": task.get("subtype") or task.get("task_type"),
    }
    if family == "dwh":
        gold = task.get("gold_answer")
        if not isinstance(gold, dict):
            return {**base, "supported": False, "unsupported_reason": "non_structured_dwh_gold"}
        answer_type = str(gold.get("answer_type") or "")
        sql = gold.get("verification_sql")
        supported = answer_type in {"numeric", "table"} and isinstance(sql, str) and bool(sql.strip())
        return {
            **base,
            "supported": supported,
            "unsupported_reason": "" if supported else "dwh_requires_semantic_judge",
            "answer_type": answer_type,
            "expected_value_json": _json_text(gold.get("value")),
            "verification_sql": sql or "",
            "required_tables": task.get("expected_tables") or [],
            "must_use_fields": (task.get("verification_criteria") or {}).get("must_use_fields") or [],
            "abs_tol": 1e-3,
            "rel_tol": 1e-5,
        }
    if family == "kb":
        gold = task.get("gold_answer")
        subtype = str(task.get("subtype") or task.get("task_type") or "")
        answerable = subtype != "unanswerable" and gold is not None
        return {
            **base,
            "supported": True,
            "answerable": answerable,
            "gold_answer": "" if gold is None else str(gold),
            "source_documents": task.get("source_documents") or [],
        }
    return {**base, "supported": False, "unsupported_reason": "unsupported_task_family"}


def _database_path(root: str | Path, environment_id: str) -> Path:
    sandbox = Path(root).resolve(strict=True)
    database = (sandbox / environment_id / "logistics.sqlite").resolve(strict=True)
    database.relative_to(sandbox)
    if not database.is_file():
        raise FileNotFoundError(database)
    return database


def _common_evidence(
    solution_str: str,
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    events = extra_info.get("pi_tool_events") or []
    if not isinstance(events, list):
        events = []
    commands = _event_commands(events)
    valid_protocol = bool(events) and all(event.get("name") in ALLOWED_TOOLS for event in events)
    safe = all(command_is_safe(command) for command in commands)
    answer = extract_final_assistant_answer(solution_str)
    return {
        "events": events,
        "commands": commands,
        "answer": answer,
        "has_final_answer": bool(answer),
        "valid_tool_protocol": valid_protocol,
        "safe": safe,
    }


def _score_dwh(
    ground_truth: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any]:
    if not ground_truth.get("supported"):
        return {
            "score": 0.0,
            "acc": 0.0,
            "online_eligible": 0.0,
            "requires_semantic_judge": 1.0,
            "unsupported_reason": ground_truth.get("unsupported_reason", "unsupported_dwh"),
        }
    answer_type = str(ground_truth.get("answer_type") or "")
    expected_value = _expected_value(ground_truth)
    abs_tol = float(ground_truth.get("abs_tol", 1e-3))
    rel_tol = float(ground_truth.get("rel_tol", 1e-5))
    answer_ok = final_answer_correct(
        common["answer"], answer_type, expected_value, abs_tol, rel_tol
    )
    commands = common["commands"]
    required_tables = {str(value).casefold() for value in ground_truth.get("required_tables", [])}
    used_tables = {table for command in commands for table in extract_table_names(command)}
    tables_ok = required_tables.issubset(used_tables) if required_tables else True
    successful_bash = any(
        event.get("name") == "bash" and event.get("ok") for event in common["events"]
    )

    sql_evidence = False
    gold_verified = False
    verifier_error = ""
    try:
        database = _database_path(
            os.environ.get("PI_AGENT_SANDBOX_LOWER", "/pi_sandbox"),
            str(ground_truth.get("environment_id") or ""),
        )
        gold_rows = execute_readonly_sql(database, str(ground_truth.get("verification_sql") or ""))
        gold_verified = bool(gold_rows) and gold_supported_by_rows(
            answer_type, expected_value, gold_rows, abs_tol, rel_tol
        )
        if gold_verified:
            for sql in extract_selects(commands):
                try:
                    if rows_equal(
                        execute_readonly_sql(database, sql),
                        gold_rows,
                        abs_tol,
                        rel_tol,
                    ):
                        sql_evidence = True
                        break
                except (ValueError, OSError, sqlite3.Error):
                    continue
    except (ValueError, OSError, sqlite3.Error) as exc:
        verifier_error = f"{type(exc).__name__}: {exc}"

    eligible = bool(gold_verified)
    strict = bool(
        eligible
        and common["safe"]
        and common["valid_tool_protocol"]
        and common["has_final_answer"]
        and successful_bash
        and answer_ok
        and sql_evidence
        and tables_ok
    )
    if not common["safe"] or not common["valid_tool_protocol"] or not eligible:
        score = 0.0
    elif strict:
        score = 1.0
    elif answer_ok and sql_evidence:
        score = 0.85
    elif sql_evidence:
        score = 0.20
    elif answer_ok:
        score = 0.15
    elif common["has_final_answer"]:
        score = 0.05
    else:
        score = 0.0
    return {
        "score": score,
        "acc": float(strict),
        "online_eligible": float(eligible),
        "requires_semantic_judge": 0.0,
        "final_answer_correct": float(answer_ok),
        "sql_evidence_correct": float(sql_evidence),
        "gold_sql_verified": float(gold_verified),
        "required_table_used": float(tables_ok),
        "successful_bash": float(successful_bash),
        "verifier_error": verifier_error,
    }


def _score_kb(
    ground_truth: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any]:
    source_documents = [str(value) for value in ground_truth.get("source_documents", [])]
    required_docs, accessed_docs = successful_document_access(common["events"], source_documents)
    docs_ok = bool(required_docs) and required_docs.issubset(accessed_docs)
    answer = common["answer"]
    gold = str(ground_truth.get("gold_answer") or "")
    gold_numbers = list(dict.fromkeys(extract_numbers(gold)))
    numbers_ok = bool(gold_numbers) and all(
        contains_expected_number(answer, expected) for expected in gold_numbers
    )
    anchors = _DOC_TITLE_RE.findall(gold) + _VERSION_STATUS_RE.findall(gold)
    anchor_ok = bool(anchors) and all(anchor.casefold() in answer.casefold() for anchor in anchors)
    abstains = bool(_ABSTENTION_RE.search(answer))
    answerable = bool(ground_truth.get("answerable"))

    # KB remains shadow-only until a calibrated semantic judge is available.
    # The deterministic signals are intentionally capped at 0.25.
    if not common["safe"] or not common["valid_tool_protocol"]:
        score = 0.0
    elif answerable:
        content_signal = numbers_ok if gold_numbers else anchor_ok
        score = (
            0.05 * float(common["has_final_answer"])
            + 0.10 * float(docs_ok)
            + 0.10 * float(content_signal)
        )
    else:
        score = 0.05 * float(common["has_final_answer"]) + 0.05 * float(abstains)
    return {
        "score": round(score, 6),
        "acc": 0.0,
        "online_eligible": 0.0,
        "requires_semantic_judge": 1.0,
        "answerable": float(answerable),
        "source_documents_required": sorted(required_docs),
        "source_documents_accessed": sorted(accessed_docs),
        "source_documents_ok": float(docs_ok),
        "gold_number_count": len(gold_numbers),
        "gold_numbers_ok": float(numbers_ok),
        "gold_anchors_ok": float(anchor_ok),
        "abstention_detected": float(abstains),
        "unsupported_reason": "kb_semantic_judge_not_calibrated",
    }


def compute_shadow_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Compute the non-production candidate reward for boss KB/DWH tasks."""
    del data_source
    common = _common_evidence(solution_str, extra_info)
    family = str(ground_truth.get("task_family") or "").casefold()
    if family == "dwh":
        result = _score_dwh(ground_truth, common)
    elif family == "kb":
        result = _score_kb(ground_truth, common)
    else:
        result = {
            "score": 0.0,
            "acc": 0.0,
            "online_eligible": 0.0,
            "requires_semantic_judge": 1.0,
            "unsupported_reason": "unsupported_task_family",
        }
    return {
        **result,
        "task_id": ground_truth.get("task_id"),
        "task_family": family,
        "safe": float(common["safe"]),
        "valid_tool_protocol": float(common["valid_tool_protocol"]),
        "has_final_answer": float(common["has_final_answer"]),
        "deployment_status": "shadow_only",
    }
