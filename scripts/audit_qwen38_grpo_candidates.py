#!/usr/bin/env python3
"""Fail-closed semantic and evidence audit for Qwen3.8 GRPO candidates.

The workflow deliberately separates candidate extraction from central review:

* ``extract`` runs beside a sensitive candidate Parquet and writes hash-only
  selectors plus structural routing facts.
* ``audit`` runs beside the source sandboxes.  It joins selectors to source
  tasks, replays verification SQL against read-only SQLite databases, and can
  require two independent semantic-judge passes through the private chat API.

No prompt, task ID, SQL, gold value, database row, credential, or server path is
written to the safe summary.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from typing import Any, Iterable, Sequence

from llin_verl.boss_pi_contract import canonical_json, load_boss_pi_contract
from scripts.audit_open_multisandbox_dwh import canonical_hash


CONTRACT = "llin-qwen38-grpo-candidate-semantic-audit-v1"
SELECTOR_CONTRACT = "llin-qwen38-grpo-candidate-hash-selector-v1"
TECHNICAL_RE = re.compile(
    r"SQL|SQLite|数据库|数据仓库|表名|字段名|category|value|SELECT|JOIN|GROUP\s+BY|HAVING|CTE|窗口函数",
    re.IGNORECASE,
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
READ_ONLY_SQL_RE = re.compile(r"^\s*(?:WITH\b|SELECT\b)", re.IGNORECASE)
FORBIDDEN_SQL_RE = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|REPLACE|DROP|ALTER|CREATE|ATTACH|DETACH|VACUUM|PRAGMA)\b",
    re.IGNORECASE,
)
LEVEL_FAMILIES = {
    1: "grouped_ranking",
    2: "period_comparison",
    3: "process_diagnosis",
    4: "baseline_attribution",
    5: "management_prioritization",
}


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_private_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _required_candidate_checks(row: dict[str, Any]) -> dict[str, bool]:
    extra = row.get("extra_info") or {}
    reward = row.get("reward_model") or {}
    truth = reward.get("ground_truth") or {}
    prompt = row.get("prompt") or []
    system_messages = [item for item in prompt if item.get("role") == "system"]
    user_messages = [item for item in prompt if item.get("role") == "user"]
    try:
        parsed_expected = json.loads(str(truth.get("expected_value_json") or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed_expected = None
    expected_hash_matches = (
        parsed_expected is not None
        and canonical_hash(parsed_expected) == str(extra.get("gold_sha256") or "")
    )
    return {
        "data_source_exact": row.get("data_source")
        == "llin_open_multisandbox_dwh_step120_rollout_v1",
        "agent_exact": row.get("agent_name") == "pi_agent",
        "ability_exact": row.get("ability") == "boss_pi_dwh",
        "single_system_and_user_message": len(system_messages) == 1 and len(user_messages) == 1,
        "rule_reward": reward.get("style") == "rule",
        "table_answer": truth.get("answer_type") == "table",
        "reward_contract_exact": truth.get("reward_contract")
        == "pure-final-outcome-screening-v1",
        "task_family_exact": truth.get("task_family") == "open_plan_first_dwh",
        "expected_value_hash_matches": expected_hash_matches,
        "mechanical_screen_passed": extra.get("mechanical_screen_passed") is True,
        "api_generation_validation_passed": extra.get("api_semantic_validation_passed") is True,
        "not_preapproved": extra.get("explicit_semantic_reviewed") is False,
        "training_disabled": extra.get("training_allowed") is False,
        "promotion_disabled": extra.get("promotion_allowed") is False,
        "no_response_messages_in_input": int(extra.get("response_messages_in_grpo_input", -1)) == 0,
    }


def extract_selectors(
    candidate_paths: Sequence[Path],
    output: Path,
    *,
    host_label: str,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    selectors: list[dict[str, Any]] = []
    for path in candidate_paths:
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            extra = row["extra_info"]
            truth = row["reward_model"]["ground_truth"]
            prompt = row["prompt"]
            system = next((item["content"] for item in prompt if item["role"] == "system"), "")
            user = next((item["content"] for item in prompt if item["role"] == "user"), "")
            checks = _required_candidate_checks(row)
            correct_count = int(extra.get("correct_count", extra.get("adaptive_correct_count", 0)))
            completed_count = int(extra.get("completed_count", extra.get("adaptive_completed_count", 0)))
            timeout_count = int(extra.get("timeout_count", extra.get("adaptive_timeout_count", 0)))
            selectors.append(
                {
                    "host": host_label,
                    "batch": path.parent.name,
                    "source_version": str(extra["source_version"]),
                    "difficulty_level": int(extra["difficulty_level"]),
                    "instruction_sha256": str(extra["instruction_sha256"]),
                    "gold_sha256": str(extra["gold_sha256"]),
                    "verification_sql_sha256": text_sha256(str(truth["verification_sql"])),
                    "system_content_sha256": text_sha256(str(system)),
                    "user_content_sha256": text_sha256(str(user)),
                    "required_tables_sha256": canonical_hash(sorted(truth.get("required_tables") or [])),
                    "correct_count": correct_count,
                    "completed_count": completed_count,
                    "timeout_count": timeout_count,
                    "reward_variance_observed": 0 < correct_count < completed_count,
                    "candidate_checks": checks,
                }
            )
    keys = [(row["instruction_sha256"], row["gold_sha256"]) for row in selectors]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate candidate identity within selector export")
    if any(not SHA256_RE.fullmatch(value) for key in keys for value in key):
        raise ValueError("candidate selector contains an invalid SHA-256 identity")
    if any(not all(row["candidate_checks"].values()) for row in selectors):
        raise ValueError("candidate structural contract failed during selector export")
    if any(not row["reward_variance_observed"] for row in selectors):
        raise ValueError("selector export contains a non-mixed candidate")
    payload = {
        "contract": SELECTOR_CONTRACT,
        "host": host_label,
        "candidate_count": len(selectors),
        "selectors": selectors,
        "contains_prompts_sql_gold_values_task_ids_or_tool_outputs": False,
        "training_allowed": False,
    }
    write_private_json(output, payload)
    return {
        "contract": SELECTOR_CONTRACT,
        "host": host_label,
        "candidate_count": len(selectors),
        "version_counts": dict(sorted(Counter(row["batch"] for row in selectors).items())),
        "difficulty_counts": dict(
            sorted(Counter(str(row["difficulty_level"]) for row in selectors).items())
        ),
        "training_allowed": False,
    }


def _normalize_rows(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [{"category": str(row["category"]), "value": row["value"]} for row in rows]


def _tables_match(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]], *, abs_tol: float
) -> bool:
    if len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected, strict=True):
        if str(left.get("category")) != str(right.get("category")):
            return False
        try:
            if not math.isclose(
                float(left.get("value")),
                float(right.get("value")),
                rel_tol=0.0,
                abs_tol=abs_tol,
            ):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _deterministic_review(
    selector: dict[str, Any],
    task: dict[str, Any],
    connection: sqlite3.Connection,
    *,
    system_prompt: str,
    guidance_prefix: str,
    abs_tol: float,
) -> tuple[dict[str, bool], list[str]]:
    instruction = str(task["natural_language_instruction"])
    gold = task["gold_answer"]
    sql = str(gold["verification_sql"])
    expected = gold["value"]
    anchors = [str(value) for value in task["semantic_anchors"]]
    required_tables = sorted(str(value).casefold() for value in task["expected_tables"])
    safe_sql = bool(READ_ONLY_SQL_RE.search(sql)) and not FORBIDDEN_SQL_RE.search(sql)
    replay: list[dict[str, Any]] = []
    replay_ok = False
    if safe_sql:
        replay = _normalize_rows(connection.execute(sql).fetchall())
        replay_ok = _tables_match(replay, expected, abs_tol=abs_tol)
    checks = {
        "source_training_disabled": task.get("training_allowed") is False,
        "source_qa_passed": task.get("_qa_status") == "passed",
        "source_generation_semantic_validation_passed":
            (task.get("instruction_generation") or {}).get("semantic_validation_passed") is True,
        "source_version_matches": str(task["source_sandbox_version"])
        == str(selector["source_version"]),
        "difficulty_matches": int(task["difficulty_level"])
        == int(selector["difficulty_level"]),
        "instruction_hash_matches": canonical_hash(instruction)
        == str(selector["instruction_sha256"]),
        "gold_hash_matches": canonical_hash(expected) == str(selector["gold_sha256"]),
        "verification_sql_hash_matches": text_sha256(sql)
        == str(selector["verification_sql_sha256"]),
        "system_prompt_hash_matches": text_sha256(system_prompt)
        == str(selector["system_content_sha256"]),
        "user_prompt_routing_matches": text_sha256(guidance_prefix + instruction)
        == str(selector["user_content_sha256"]),
        "required_tables_match": canonical_hash(required_tables)
        == str(selector["required_tables_sha256"]),
        "candidate_structure_passed": all(selector["candidate_checks"].values()),
        "reward_variance_observed": selector["reward_variance_observed"] is True,
        "instruction_has_no_technical_leak": TECHNICAL_RE.search(instruction) is None,
        "all_semantic_anchors_present": all(anchor in instruction for anchor in anchors),
        "top_five_explicit": re.search(r"前\s*5", instruction) is not None,
        "descending_order_explicit": "高到低" in instruction,
        "query_plan_present": bool(task.get("query_plan")),
        "semantic_contract_present": bool(task.get("semantic_contract")),
        "verified_deliverable_exact":
            (task.get("semantic_contract") or {}).get("verified_deliverable")
            == "top_five_category_value_table",
        "verification_sql_read_only": safe_sql,
        "verification_sql_replay_matches_gold": replay_ok,
        "result_hash_matches": replay_ok
        and canonical_hash(replay) == str(task["validation"]["result_sha256"]),
        "result_shape_matches": replay_ok and 2 <= len(replay) <= 5,
        "gold_answer_type_table": gold.get("answer_type") == "table",
        "canonical_expected_value_serializes": canonical_json(expected)
        == canonical_json(replay) if replay_ok else False,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return checks, reasons


def semantic_review_payload(task: dict[str, Any], review_id: str) -> dict[str, Any]:
    return {
        "review_id": review_id,
        "instruction": str(task["natural_language_instruction"]),
        "semantic_anchors": list(task["semantic_anchors"]),
        "semantic_contract": task["semantic_contract"],
        "query_plan": task["query_plan"],
        "required_output": "按结果从高到低列出前5项业务名称和对应数值",
    }


def build_review_prompt(tasks: Sequence[dict[str, Any]], *, pass_index: int) -> str:
    perspective = (
        "逐字核对题意是否完整、无歧义地覆盖冻结口径"
        if pass_index == 1
        else "采用反证法主动寻找漏条件、错口径、隐含歧义和结果路由风险"
    )
    payload = [semantic_review_payload(task, f"r{index:03d}") for index, task in enumerate(tasks)]
    return f"""你是独立的数据任务语义审核员。本轮审查方式：{perspective}。

你只能依据给出的业务题面、semantic_anchors、semantic_contract、query_plan 和 required_output 审核，不能自行补充未写明的条件。

逐题判定三个字段：
1. instruction_unambiguously_entails_plan：普通员工只读题面，是否会唯一得到query_plan中的指标、分组、筛选、排序、Top5、门槛、权重和时间口径。
2. plan_fully_answers_instruction：query_plan是否覆盖题面全部硬性要求，没有少算、多算或答非所问。
3. final_answer_contract_is_clear：题面是否清楚要求最终输出按数值从高到低的前5项名称+数值表；开放解释只能是附加内容，不能改变可核验答案。

任何不确定、需要猜测或仅“大致一致”都判false。reason_codes只允许使用短英文枚举；没有问题时必须为空数组。confidence只允许high/medium/low。只输出严格JSON数组，不要Markdown或解释。

返回格式：
[{"review_id":"r000","instruction_unambiguously_entails_plan":true,"plan_fully_answers_instruction":true,"final_answer_contract_is_clear":true,"reason_codes":[],"confidence":"high"}]

待审核数据：
""" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _numeric_tokens(text: str) -> set[str]:
    return {str(int(value)) for value in re.findall(r"(?<![A-Za-z])\d+(?![A-Za-z])", text)}


def _sql_business_literals(sql: str) -> list[str]:
    return [value.replace("''", "'") for value in re.findall(r"'((?:''|[^'])*)'", sql)]


def _date_literal_to_chinese(value: str) -> str | None:
    match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if not match:
        return None
    return f"{int(match.group(1))}年{int(match.group(2))}月"


def local_semantic_review(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Run two complementary, server-local semantic checks.

    Pass 1 follows instruction -> frozen semantic contract. Pass 2 follows the
    verification SQL -> business literals/aggregation/output contract and then
    back to the instruction.  Both are deterministic and expose reason codes.
    """

    instruction = str(task["natural_language_instruction"])
    plan = task["query_plan"]
    contract = task["semantic_contract"]
    sql = str(task["gold_answer"]["verification_sql"])
    level = int(task["difficulty_level"])
    anchors = [str(value) for value in task["semantic_anchors"]]

    pass1_checks = {
        "family_consistent": task["task_type"]
        == LEVEL_FAMILIES[level]
        == plan.get("family")
        == contract.get("family"),
        "metric_consistent": plan.get("metric_key") == contract.get("metric"),
        "group_dimension_consistent": plan.get("group_field")
        == contract.get("group_dimension"),
        "anchors_consistent": anchors
        == [str(value) for value in plan.get("anchors") or []]
        == [str(value) for value in contract.get("required_anchors") or []],
        "all_anchors_explicit": all(anchor in instruction for anchor in anchors),
        "no_unanchored_numbers": _numeric_tokens(instruction)
        <= _numeric_tokens(" ".join(anchors)),
        "output_shape_exact": plan.get("output_shape") == "category_value_table",
        "deliverable_exact": contract.get("verified_deliverable")
        == "top_five_category_value_table",
        "descending_top_five_explicit": "高到低" in instruction
        and re.search(r"前\s*5", instruction) is not None,
        "difficulty_family_language_matches": (
            (level == 1 and "对应结果" in instruction)
            or (level == 2 and "环比增幅" in instruction and "百分比" in instruction)
            or (level == 3 and "全流程" in instruction and "占" in instruction)
            or (level == 4 and "整体平均" in instruction and "基线" in instruction)
            or (
                level == 5
                and "综合" in instruction
                and "百分制" in instruction
                and len(re.findall(r"\d+%", instruction)) == 3
            )
        ),
        "open_explanation_routing_matches": bool(contract.get("explanation_is_open_ended"))
        == (
            level == 5
            and any(
                term in instruction
                for term in ("说明", "解释", "判断", "理由", "原因", "看法", "依据")
            )
        ),
    }
    pass1_reasons = [name for name, passed in pass1_checks.items() if not passed]

    literals_supported = True
    for literal in _sql_business_literals(sql):
        if literal in {"%Y-%m"}:
            continue
        rendered_date = _date_literal_to_chinese(literal)
        if rendered_date:
            literals_supported = literals_supported and rendered_date in instruction
        else:
            literals_supported = literals_supported and literal in instruction
    expected_tables = [str(value).casefold() for value in task.get("expected_tables") or []]
    metric = str(plan.get("metric_key") or "")
    metric_sql_patterns = {
        "process_avg": r"AVG\(p\.duration_hours\s*\+\s*d\.duration_hours\)",
        "delivery_avg": r"AVG\(d\.duration_hours\s*\+\s*0\.10\s*\*\s*p\.duration_hours\)",
        "handoff_gap": r"AVG\(ABS\(d\.duration_hours\s*-\s*p\.duration_hours\)\)",
        "late_stage_share": r"AVG\(d\.duration_hours\s*/\s*\(p\.duration_hours\s*\+\s*d\.duration_hours\)\)",
        "shipment_count_change_rate": r"COUNT\(\*\).*current_value\s*-\s*previous_value",
        "weight_sum_change_rate": r"SUM\(w\.cargo_weight_kg\).*current_value\s*-\s*previous_value",
        "delivery_avg_change_rate": r"AVG\(d\.duration_hours.*current_value\s*-\s*previous_value",
        "process_avg_change_rate": r"AVG\(p\.duration_hours\s*\+\s*d\.duration_hours\).*current_value\s*-\s*previous_value",
        "process_share": r"phase_avg\s*/\s*total_avg",
        "process_gap_rate": r"group_avg\s*-\s*o\.overall_avg",
        "delivery_gap_rate": r"group_avg\s*-\s*o\.overall_avg",
        "transit_gap_rate": r"group_avg\s*-\s*o\.overall_avg",
        "sorting_gap_rate": r"group_avg\s*-\s*o\.overall_avg",
        "attention_score": r"volume_index\s*\+.*delivery_index\s*\+.*process_index",
    }
    metric_pattern = metric_sql_patterns.get(metric)
    pass2_checks = {
        "sql_business_literals_are_in_instruction": literals_supported,
        "sql_group_dimension_matches_plan": re.search(
            rf"\b{re.escape(str(plan.get('group_field') or ''))}\b", sql
        )
        is not None,
        "sql_metric_formula_matches_plan": metric_pattern is not None
        and re.search(metric_pattern, sql, re.IGNORECASE | re.DOTALL) is not None,
        "sql_expected_tables_match": all(
            re.search(rf"\b{re.escape(table)}\b", sql, re.IGNORECASE)
            for table in expected_tables
        ),
        "sql_minimum_sample_gate_matches": re.search(
            r"(?:COUNT\(\*\)|sample_count|previous_count|current_count|volume)\s*>=\s*3",
            sql,
            re.IGNORECASE,
        )
        is not None,
        "sql_descending_top_five_matches": re.search(
            r"ORDER\s+BY\s+value\s+DESC\s*,\s*category\s+ASC\s+LIMIT\s+5",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        is not None,
        "sql_category_value_projection_matches": re.search(r"\bAS\s+category\b", sql, re.IGNORECASE)
        is not None
        and re.search(r"\bAS\s+value\b", sql, re.IGNORECASE) is not None,
        "expected_operations_complete": {
            "filter",
            "aggregate",
            "join",
            "group_by",
            "order_by",
            "top_k",
        }
        <= set(task.get("expected_operations") or []),
        "sql_read_only": bool(READ_ONLY_SQL_RE.search(sql))
        and not FORBIDDEN_SQL_RE.search(sql),
    }
    if level == 5:
        weights = [int(value) for value in re.findall(r"(\d+)%", instruction)]
        pass2_checks["weights_sum_to_100"] = len(weights) == 3 and sum(weights) == 100
        pass2_checks["weights_match_sql"] = all(
            f"{value / 100:.2f}" in sql for value in weights
        )
    pass2_reasons = [name for name, passed in pass2_checks.items() if not passed]

    return [
        {
            "instruction_unambiguously_entails_plan": not pass1_reasons,
            "plan_fully_answers_instruction": not pass1_reasons,
            "final_answer_contract_is_clear": pass1_checks[
                "descending_top_five_explicit"
            ]
            and pass1_checks["deliverable_exact"],
            "reason_codes": pass1_reasons,
            "confidence": "high" if not pass1_reasons else "low",
            "method": "instruction_to_contract",
        },
        {
            "instruction_unambiguously_entails_plan": not pass2_reasons,
            "plan_fully_answers_instruction": not pass2_reasons,
            "final_answer_contract_is_clear": pass2_checks[
                "sql_descending_top_five_matches"
            ]
            and pass2_checks["sql_category_value_projection_matches"],
            "reason_codes": pass2_reasons,
            "confidence": "high" if not pass2_reasons else "low",
            "method": "sql_to_instruction_adversarial",
        },
    ]


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end < start:
        raise ValueError("semantic judge response does not contain a JSON array")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("semantic judge response must be an array")
    return value


def _load_api_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    api_key = config.get("api_key") or os.environ.get(config.get("api_key_env", "CHAT_API_KEY"))
    if not api_key:
        raise ValueError("private chat API key is not configured")
    if not config.get("base_url") or not config.get("default_model"):
        raise ValueError("private chat API base_url/default_model is missing")
    return {**config, "_resolved_api_key": api_key}


def _chat_completion(prompt: str, config: dict[str, Any], *, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=config["_resolved_api_key"],
        base_url=config["base_url"],
        max_retries=0,
    )
    try:
        response = client.chat.completions.create(
            model=config["default_model"],
            messages=[
                {
                    "role": "system",
                    "content": "你是严格、保守的数据任务语义审核员。证据不足即拒绝，不负责改写题目。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
            timeout=float(config.get("timeout_seconds", 120)),
        )
    finally:
        client.close()
    content = response.choices[0].message.content
    if not content:
        raise ValueError("private chat API returned empty content")
    return str(content)


def semantic_judge_batch(
    tasks: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    pass_index: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    expected_ids = {f"r{index:03d}" for index in range(len(tasks))}
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = _chat_completion(
                build_review_prompt(tasks, pass_index=pass_index),
                config,
                max_tokens=min(8192, 800 + len(tasks) * 500),
            )
            rows = _extract_json_array(raw)
            by_id: dict[str, dict[str, Any]] = {}
            for row in rows:
                review_id = str(row.get("review_id") or "")
                if review_id in by_id:
                    raise ValueError("duplicate review_id from semantic judge")
                checks = [
                    row.get("instruction_unambiguously_entails_plan"),
                    row.get("plan_fully_answers_instruction"),
                    row.get("final_answer_contract_is_clear"),
                ]
                if any(type(value) is not bool for value in checks):
                    raise ValueError("semantic judge returned non-boolean checks")
                if row.get("confidence") not in {"high", "medium", "low"}:
                    raise ValueError("semantic judge returned invalid confidence")
                if not isinstance(row.get("reason_codes"), list):
                    raise ValueError("semantic judge returned invalid reason_codes")
                by_id[review_id] = {
                    "instruction_unambiguously_entails_plan": checks[0],
                    "plan_fully_answers_instruction": checks[1],
                    "final_answer_contract_is_clear": checks[2],
                    "reason_codes": [str(value)[:80] for value in row["reason_codes"][:8]],
                    "confidence": row["confidence"],
                }
            if set(by_id) != expected_ids:
                raise ValueError("semantic judge response IDs do not match the batch")
            return [by_id[f"r{index:03d}"] for index in range(len(tasks))]
        except Exception as exc:  # retry only a bounded number of times
            last_error = exc
            if attempt < max_retries:
                time.sleep(float(config.get("retry_delay_seconds", 2.0)) * attempt)
    raise RuntimeError(
        f"semantic judge pass {pass_index} failed after {max_retries} attempts: "
        f"{type(last_error).__name__}"
    ) from last_error


def _load_selector_payloads(paths: Sequence[Path]) -> list[dict[str, Any]]:
    selectors: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("contract") != SELECTOR_CONTRACT:
            raise ValueError(f"unsupported selector contract: {path}")
        if int(payload.get("candidate_count", -1)) != len(payload.get("selectors") or []):
            raise ValueError(f"selector count mismatch: {path}")
        selectors.extend(payload["selectors"])
    keys = [(row["instruction_sha256"], row["gold_sha256"]) for row in selectors]
    if len(keys) != len(set(keys)):
        raise ValueError("cross-host or cross-sandbox candidate identity collision")
    return selectors


def audit_candidates(
    selector_paths: Sequence[Path],
    source_dirs: dict[str, Path],
    output_dir: Path,
    *,
    api_config_path: Path | None,
    local_deterministic_semantic_review: bool,
    expected_count: int,
    batch_size: int,
    max_retries: int,
    abs_tol: float,
) -> dict[str, Any]:
    if api_config_path is not None and local_deterministic_semantic_review:
        raise ValueError("choose either private API review or local deterministic semantic review")
    selectors = _load_selector_payloads(selector_paths)
    if len(selectors) != expected_count:
        raise ValueError(f"expected {expected_count} selectors, got {len(selectors)}")
    boss_contract = load_boss_pi_contract()
    system_prompt = str(boss_contract["system_prompt"])
    guidance_prefix = str(boss_contract["runtime"]["guidance_prefix"])

    tasks_by_version: dict[str, dict[str, dict[str, Any]]] = {}
    connections: dict[str, sqlite3.Connection] = {}
    for version, source_dir in source_dirs.items():
        tasks = read_jsonl(source_dir / "dwh_tasks.jsonl")
        by_hash: dict[str, dict[str, Any]] = {}
        for task in tasks:
            identity = canonical_hash(task["natural_language_instruction"])
            if identity in by_hash:
                raise ValueError(f"duplicate source instruction identity in {version}")
            by_hash[identity] = task
        tasks_by_version[version] = by_hash
        connection = sqlite3.connect(
            f"file:{(source_dir / 'logistics.sqlite').as_posix()}?mode=ro", uri=True
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connections[version] = connection

    detailed: list[dict[str, Any]] = []
    selected_tasks: list[dict[str, Any]] = []
    try:
        for selector in selectors:
            version = str(selector["source_version"])
            if version not in tasks_by_version:
                raise ValueError(f"missing source sandbox mapping for version {version}")
            task = tasks_by_version[version].get(str(selector["instruction_sha256"]))
            if task is None:
                raise ValueError("candidate selector does not join to a source task")
            checks, reasons = _deterministic_review(
                selector,
                task,
                connections[version],
                system_prompt=system_prompt,
                guidance_prefix=guidance_prefix,
                abs_tol=abs_tol,
            )
            detailed.append(
                {
                    "host": selector["host"],
                    "batch": selector["batch"],
                    "source_version": version,
                    "difficulty_level": int(selector["difficulty_level"]),
                    "instruction_sha256": selector["instruction_sha256"],
                    "gold_sha256": selector["gold_sha256"],
                    "correct_count": int(selector["correct_count"]),
                    "completed_count": int(selector["completed_count"]),
                    "timeout_count": int(selector["timeout_count"]),
                    "deterministic_checks": checks,
                    "deterministic_reason_codes": reasons,
                    "semantic_judge_passes": [],
                }
            )
            selected_tasks.append(task)
    finally:
        for connection in connections.values():
            connection.close()

    if api_config_path is not None:
        config = _load_api_config(api_config_path)
        for start in range(0, len(selected_tasks), batch_size):
            batch = selected_tasks[start : start + batch_size]
            for pass_index in (1, 2):
                judgments = semantic_judge_batch(
                    batch,
                    config,
                    pass_index=pass_index,
                    max_retries=max_retries,
                )
                for offset, judgment in enumerate(judgments):
                    detailed[start + offset]["semantic_judge_passes"].append(judgment)
    elif local_deterministic_semantic_review:
        for row, task in zip(detailed, selected_tasks, strict=True):
            row["semantic_judge_passes"] = local_semantic_review(task)

    for row in detailed:
        deterministic_passed = all(row["deterministic_checks"].values())
        judgments = row["semantic_judge_passes"]
        semantic_complete = len(judgments) == 2
        semantic_passed = semantic_complete and all(
            judgment["instruction_unambiguously_entails_plan"]
            and judgment["plan_fully_answers_instruction"]
            and judgment["final_answer_contract_is_clear"]
            and judgment["confidence"] in {"high", "medium"}
            and not judgment["reason_codes"]
            for judgment in judgments
        )
        semantic_agreement = semantic_complete and all(
            judgments[0][field] == judgments[1][field]
            for field in (
                "instruction_unambiguously_entails_plan",
                "plan_fully_answers_instruction",
                "final_answer_contract_is_clear",
            )
        )
        if not deterministic_passed:
            decision = "rejected"
        elif semantic_passed and semantic_agreement:
            decision = "approved_candidate"
        else:
            decision = "needs_manual_review"
        row.update(
            {
                "decision": decision,
                "instruction_unambiguously_entails_gold": semantic_passed,
                "verification_sql_fully_answers_instruction": deterministic_passed
                and semantic_passed,
                "expected_value_supported_by_query_result": bool(
                    row["deterministic_checks"]["verification_sql_replay_matches_gold"]
                    and row["deterministic_checks"]["result_hash_matches"]
                ),
                "final_outcome_routing_trustworthy": bool(
                    row["deterministic_checks"]["user_prompt_routing_matches"]
                    and row["deterministic_checks"]["candidate_structure_passed"]
                    and semantic_passed
                ),
                "explicit_semantic_reviewed": decision in {"approved_candidate", "rejected"},
                "training_allowed": False,
                "promotion_allowed": False,
            }
        )

    decision_counts = Counter(row["decision"] for row in detailed)
    by_batch: dict[str, dict[str, int]] = {}
    for batch in sorted({row["batch"] for row in detailed}):
        subset = [row for row in detailed if row["batch"] == batch]
        by_batch[batch] = {
            "candidates": len(subset),
            "approved": sum(row["decision"] == "approved_candidate" for row in subset),
            "rejected": sum(row["decision"] == "rejected" for row in subset),
            "needs_manual_review": sum(row["decision"] == "needs_manual_review" for row in subset),
        }
    detailed_payload = {
        "contract": CONTRACT,
        "expected_candidate_count": expected_count,
        "candidate_count": len(detailed),
        "semantic_judge_required": api_config_path is not None,
        "semantic_review_method": (
            "private_api_two_pass"
            if api_config_path is not None
            else "server_local_deterministic_two_pass"
            if local_deterministic_semantic_review
            else "not_run"
        ),
        "semantic_judge_passes_required": 2,
        "decisions": detailed,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_private_json(output_dir / "candidate_review_decisions.sensitive.json", detailed_payload)
    approved_selectors = [
        {
            "host": row["host"],
            "batch": row["batch"],
            "instruction_sha256": row["instruction_sha256"],
            "gold_sha256": row["gold_sha256"],
            "decision": row["decision"],
        }
        for row in detailed
        if row["decision"] == "approved_candidate"
    ]
    write_private_json(
        output_dir / "approved_candidate_selectors.sensitive.json",
        {
            "contract": CONTRACT,
            "selectors": approved_selectors,
            "training_allowed": False,
            "promotion_allowed": False,
        },
    )
    safe_summary = {
        "contract": CONTRACT,
        "candidate_count": len(detailed),
        "reviewed_with_deterministic_evidence": len(detailed),
        "reviewed_with_two_semantic_judge_passes": sum(
            len(row["semantic_judge_passes"]) == 2 for row in detailed
        ),
        "approved_candidates": decision_counts["approved_candidate"],
        "rejected_candidates": decision_counts["rejected"],
        "needs_manual_review": decision_counts["needs_manual_review"],
        "all_candidates_accounted_for": sum(decision_counts.values()) == len(detailed),
        "by_batch": by_batch,
        "difficulty_counts": dict(
            sorted(Counter(str(row["difficulty_level"]) for row in detailed).items())
        ),
        "deterministic_gate": {
            "identity_prompt_routing_sql_gold_and_schema_checked": True,
            "verification_sql_replayed_read_only": True,
            "gold_result_hash_reconciled": True,
        },
        "semantic_gate": {
            "method": (
                "private_api_two_pass"
                if api_config_path is not None
                else "server_local_deterministic_two_pass"
                if local_deterministic_semantic_review
                else "not_run"
            ),
            "passes_required": 2,
            "unanimous_check_vector_required": True,
            "low_confidence_rejected_from_auto_approval": True,
            "empty_reason_codes_required": True,
        },
        "explicit_semantic_review_completed_for_approved_rows": True,
        "training_allowed": False,
        "promotion_allowed": False,
        "contains_prompts_sql_gold_values_task_ids_hashes_server_paths_or_tool_outputs": False,
    }
    (output_dir / "safe_summary.json").write_text(
        json.dumps(safe_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return safe_summary


def parse_source_specs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        version, separator, raw_path = value.partition("=")
        if not separator or not version or not raw_path:
            raise ValueError("source specs must be VERSION=PATH")
        if version in result:
            raise ValueError(f"duplicate source version mapping: {version}")
        result[version] = Path(raw_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--candidate", action="append", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--host-label", required=True)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--selector", action="append", type=Path, required=True)
    audit_parser.add_argument("--source", action="append", required=True)
    audit_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser.add_argument("--api-config", type=Path)
    audit_parser.add_argument("--local-deterministic-semantic-review", action="store_true")
    audit_parser.add_argument("--expected-count", type=int, default=70)
    audit_parser.add_argument("--batch-size", type=int, default=7)
    audit_parser.add_argument("--max-retries", type=int, default=3)
    audit_parser.add_argument("--abs-tol", type=float, default=0.011)

    args = parser.parse_args(argv)
    if args.command == "extract":
        summary = extract_selectors(args.candidate, args.output, host_label=args.host_label)
    else:
        if not 1 <= args.batch_size <= 12:
            raise ValueError("batch-size must be between 1 and 12")
        summary = audit_candidates(
            args.selector,
            parse_source_specs(args.source),
            args.output_dir,
            api_config_path=args.api_config,
            local_deterministic_semantic_review=args.local_deterministic_semantic_review,
            expected_count=args.expected_count,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            abs_tol=args.abs_tol,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
