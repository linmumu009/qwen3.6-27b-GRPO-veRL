#!/usr/bin/env python3
"""Rewrite plan-first DWH instructions with the boss OpenAI-compatible API.

The API only receives business-language semantic facts.  SQL, gold answers,
database rows, and credentials never enter the prompt or output logs.  Every
rewrite is validated against its originating QueryPlan before it can replace
the deterministic draft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

CONTRACT = "llin-plan-first-dwh-api-natural-v1"
DEFAULT_VERSION = "20260814_llin_dwh_planfirst_api_v1"
TECHNICAL_INSTRUCTION_RE = re.compile(
    r"SQL|SQLite|数据库|数据仓库|表名|字段名|category|value|SELECT|JOIN|GROUP\s+BY|HAVING",
    re.IGNORECASE,
)

ROLE_LABELS = {
    "company_owner": "公司老板或高管，关心结论，表达简洁",
    "finance": "财务人员，关心费用口径，但不会写 SQL",
    "data_analyst": "业务分析人员，表达清楚，但不在问题里说表名或字段名",
    "operations": "运营人员，关心业务执行和履约",
    "warehouse_manager": "仓储负责人，关心仓库与货量",
    "regional_manager": "区域负责人，关心区域差异",
    "procurement": "采购或承运商管理人员，关心合作方表现",
    "customer_service": "客服人员，关心客户体验和时效",
    "planning": "计划人员，关心资源和运力安排",
    "sales": "销售人员，关心客户和区域业务",
    "general_employee": "普通公司员工，没有数据或 SQL 背景",
}

METRIC_PATTERNS = {
    "shipment_count": re.compile(r"多少.{0,5}(?:票|单)|运单数量|单量"),
    "customer_count": re.compile(r"(?:多少|几).{0,5}(?:位|个)?.{0,3}客户|不同客户|客户数量"),
    "freight_sum": re.compile(r"(?:总运费|运费.{0,6}(?:总额|合计|一共|总共))"),
    "weight_sum": re.compile(r"(?:总重量|合计.{0,5}(?:千克|公斤)|一共.{0,5}(?:千克|公斤))"),
    "delivery_avg": re.compile(r"平均.{0,8}(?:配送|送达).{0,8}(?:时间|时长|小时|多久|耗时)"),
    "freight_avg": re.compile(r"平均.{0,5}(?:每票|单票).{0,3}运费|平均运费"),
    "delay_rate": re.compile(r"延误率"),
}

GROUP_PATTERNS = {
    "服务等级": re.compile(
        r"(?:各|不同|每个).{0,3}(?:配送服务|服务类型|服务等级)"
    ),
    "仓库名称": re.compile(r"各仓库|不同仓库|每个仓库|各仓"),
    "区域名称": re.compile(r"各区域|不同区域|每个区域|各地"),
    "承运商名称": re.compile(r"各承运商|不同承运商|每个承运商|各家承运商"),
}

FEEDBACK_GUIDANCE = {
    "length_out_of_range": "句子长度不合格，请控制在24到360个汉字之间。",
    "metric_not_explicit": "统计指标没有说清楚，请直接写出 must_preserve.metric 的指标名称。",
    "month_missing_or_changed": "月份被遗漏或改错，请逐字保留 conditions 中的2025年具体月份。",
    "signed_status_missing": "遗漏了已经签收这一状态，请明确写出签收条件。",
    "not_cancelled_condition_missing": "遗漏了未取消这一否定条件，请明确写出没有取消或排除已取消。",
    "comparison_dimension_missing": "没有说清按什么对象逐项比较，请明确写出 comparison_dimension。",
    "descending_order_missing": "遗漏排序方向，请明确写出从高到低。",
    "ascending_order_missing": "遗漏排序方向，请明确写出从低到高。",
    "top_n_missing": "遗漏前N名，请按 must_preserve.top_n 明确写出前几名。",
    "minimum_sample_rule_missing": "遗漏最低样本要求，请明确写出至少有3票符合条件。",
    "warehouse_type_not_explicit": "仓库类型口径不够明确，请写成“发货仓类型为某类”，不要写得像只限定了一个具体仓库。",
}


def _feedback_guidance(reason: str) -> str:
    if reason.startswith("filter_value_missing:"):
        value = reason.split(":", 1)[1]
        return f"遗漏筛选值“{value}”，请在新问题中逐字保留这个业务值。"
    if reason.startswith("technical_term:"):
        value = reason.split(":", 1)[1]
        return f"出现了禁止的技术词“{value}”，请改成普通员工能听懂的业务说法。"
    return FEEDBACK_GUIDANCE.get(reason, f"请修正校验问题：{reason}。")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _quoted_values(descriptions: Sequence[str]) -> list[str]:
    values: list[str] = []
    for description in descriptions:
        values.extend(re.findall(r"“([^”]+)”", description))
    return values


def semantic_payload(task: dict[str, Any]) -> dict[str, Any]:
    plan = task["query_plan"]
    descriptions = [str(item["description"]) for item in plan["filters"]]
    payload: dict[str, Any] = {
        "task_id": str(task["task_id"]),
        "target_speaker": ROLE_LABELS[str(task["instruction_role"])],
        "draft": str(task["natural_language_instruction"]),
        "must_preserve": {
            "metric": str(plan["metric_description"]),
            "conditions": descriptions,
            "result_shape": "一个数字" if not plan.get("group_sql") else "逐项比较并列出名称和结果",
        },
    }
    if plan.get("group_description"):
        payload["must_preserve"]["comparison_dimension"] = str(plan["group_description"])
    if plan.get("order_direction"):
        payload["must_preserve"]["order"] = (
            "从高到低" if plan["order_direction"] == "DESC" else "从低到高"
        )
    if plan.get("limit") is not None:
        payload["must_preserve"]["top_n"] = int(plan["limit"])
    if plan.get("having_description"):
        payload["must_preserve"]["minimum_sample_rule"] = str(plan["having_description"])
    feedback = task.get("_rewrite_feedback")
    if feedback:
        payload["previous_attempt_rejected_because"] = list(feedback)
        payload["required_corrections_for_retry"] = [
            _feedback_guidance(str(reason)) for reason in feedback
        ]
    return payload


def build_prompt(tasks: Sequence[dict[str, Any]]) -> str:
    payloads = [semantic_payload(task) for task in tasks]
    return """请把下面每条明确的数据需求改写成自然的中文问题。

使用者来自同一家公司的不同岗位，包括老板、财务、业务分析、运营、仓储、区域、采购、客服、计划、销售和普通员工。请按照每条给定的 target_speaker 调整语气，让它像真人临时向内部数据助手提问，而不是统一模板。

硬性要求：
1. must_preserve 中的指标、日期、筛选条件、比较维度、排序方向、前 N 名和最低样本要求必须全部保留，不能增加新条件。
2. 不要出现 SQL、SQLite、数据库、数据仓库、表名、字段名、category、value、SELECT、JOIN、GROUP BY、HAVING。
3. 不要解释怎么算，不要给答案，不要提“根据上述要求”。
4. 语言应当让没有 SQL 背景的普通员工也能读懂；每条用 1 到 3 句话。
5. 每个 task_id 只返回一条 instruction。
6. 如果存在 required_corrections_for_retry，上一版已经被校验器拒绝；新版本必须逐条修正这些问题，同时仍完整保留 must_preserve。

只输出严格 JSON 数组，不要 Markdown 代码块：
[{"task_id":"...","instruction":"..."}]

待改写数据：
""" + json.dumps(payloads, ensure_ascii=False, separators=(",", ":"))


def _extract_json(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start < 0 or end < start:
        raise ValueError("API response does not contain a JSON array")
    return json.loads(cleaned[start : end + 1])


def parse_response(text: str, tasks: Sequence[dict[str, Any]]) -> dict[str, str]:
    value = _extract_json(text)
    if not isinstance(value, list):
        raise ValueError("API response must be an array")
    expected_ids = {str(task["task_id"]) for task in tasks}
    parsed: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "")
        instruction = str(item.get("instruction") or "").strip()
        if task_id in expected_ids and instruction:
            parsed[task_id] = instruction
    if set(parsed) != expected_ids:
        missing = sorted(expected_ids - set(parsed))
        raise ValueError(f"API response is missing task IDs: {missing}")
    return parsed


def validate_rewrite(task: dict[str, Any], instruction: str) -> list[str]:
    reasons: list[str] = []
    if len(instruction) < 24 or len(instruction) > 360:
        reasons.append("length_out_of_range")
    match = TECHNICAL_INSTRUCTION_RE.search(instruction)
    if match:
        reasons.append(f"technical_term:{match.group(0)}")

    plan = task["query_plan"]
    metric_key = str(plan["metric_key"])
    if not METRIC_PATTERNS[metric_key].search(instruction):
        reasons.append("metric_not_explicit")

    descriptions = [str(item["description"]) for item in plan["filters"]]
    month_match = next((re.search(r"2025\s*年\s*(\d{1,2})\s*月", item) for item in descriptions if "2025" in item), None)
    if month_match and not re.search(
        rf"2025\s*年\s*{int(month_match.group(1))}\s*月", instruction
    ):
        reasons.append("month_missing_or_changed")
    for value in _quoted_values(descriptions):
        if value not in instruction:
            reasons.append(f"filter_value_missing:{value}")
    if any(str(item.get("sql") or "").startswith("w.warehouse_type =") for item in plan["filters"]):
        if not re.search(r"(?:发货仓|仓库?)类型", instruction):
            reasons.append("warehouse_type_not_explicit")
    if any("已经签收" in item for item in descriptions) and "签收" not in instruction:
        reasons.append("signed_status_missing")
    if any("没有取消" in item for item in descriptions):
        if "取消" not in instruction or not re.search(r"不|未|没有|排除", instruction):
            reasons.append("not_cancelled_condition_missing")

    group_description = plan.get("group_description")
    if group_description and not GROUP_PATTERNS[str(group_description)].search(instruction):
        reasons.append("comparison_dimension_missing")
    direction = plan.get("order_direction")
    if direction == "DESC" and "高到低" not in instruction:
        reasons.append("descending_order_missing")
    if direction == "ASC" and "低到高" not in instruction:
        reasons.append("ascending_order_missing")
    limit = plan.get("limit")
    if limit is not None and not re.search(rf"前\s*{int(limit)}\s*(?:名|个|家|项)?", instruction):
        reasons.append("top_n_missing")
    if plan.get("having_description") and not re.search(r"至少(?:有)?\s*3\s*票", instruction):
        reasons.append("minimum_sample_rule_missing")

    return reasons


def _load_api_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    api_key = config.get("api_key") or os.environ.get(config.get("api_key_env", "CHAT_API_KEY"))
    if not api_key:
        raise ValueError("boss chat API key is not configured")
    if not config.get("base_url") or not config.get("default_model"):
        raise ValueError("boss chat API base_url/default_model is missing")
    config["_resolved_api_key"] = api_key
    return config


def _call_api(tasks: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, str]:
    messages = [
        {
            "role": "system",
            "content": "你是企业内部数据需求编辑，只负责把已冻结的业务口径改写成自然中文，绝不改变任何统计语义。",
        },
        {"role": "user", "content": build_prompt(tasks)},
    ]
    max_tokens = min(8192, 900 + len(tasks) * 550)
    timeout = float(config.get("timeout_seconds", 60))
    request_payload = {
        "model": config["default_model"],
        "messages": messages,
        "max_tokens": max_tokens,
    }
    socks_proxy = str(config.get("_socks_proxy") or "")
    if socks_proxy:
        url = str(config["base_url"]).rstrip("/") + "/chat/completions"
        escape = lambda value: str(value).replace("\\", "\\\\").replace('"', '\\"')
        curl_config = (
            f'url = "{escape(url)}"\n'
            f'proxy = "{escape(socks_proxy)}"\n'
            f'header = "Authorization: Bearer {escape(config["_resolved_api_key"])}"\n'
            'header = "Content-Type: application/json"\n'
            "silent\nshow-error\nfail-with-body\nrequest = POST\n"
        ).encode("utf-8")
        body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        with tempfile.TemporaryFile() as config_handle, tempfile.TemporaryFile() as body_handle:
            config_handle.write(curl_config)
            config_handle.flush()
            config_handle.seek(0)
            body_handle.write(body)
            body_handle.flush()
            body_handle.seek(0)
            completed = subprocess.run(
                [
                    "curl",
                    "--config",
                    f"/proc/self/fd/{config_handle.fileno()}",
                    "--data-binary",
                    f"@/proc/self/fd/{body_handle.fileno()}",
                    "--max-time",
                    str(int(timeout)),
                ],
                pass_fds=(config_handle.fileno(), body_handle.fileno()),
                capture_output=True,
                timeout=timeout + 10,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"proxied curl failed with exit code {completed.returncode}")
        payload = json.loads(completed.stdout.decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
    else:
        try:
            from openai import OpenAI
        except ImportError:
            from urllib.request import Request, urlopen

            body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
            request = Request(
                str(config["base_url"]).rstrip("/") + "/chat/completions",
                data=body,
                headers={
                    "Authorization": "Bearer " + str(config["_resolved_api_key"]),
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
        else:
            client = OpenAI(
                api_key=config["_resolved_api_key"],
                base_url=config["base_url"],
                max_retries=0,
            )
            try:
                response = client.chat.completions.create(
                    model=config["default_model"],
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            finally:
                client.close()
            content = response.choices[0].message.content
    if not content:
        raise ValueError("boss chat API returned empty content")
    return parse_response(str(content), tasks)


def rewrite_batch(
    tasks: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    max_retries: int,
) -> dict[str, str]:
    last_reasons: dict[str, list[str]] = {}
    attempt_tasks = list(tasks)
    for attempt in range(1, max_retries + 1):
        try:
            rewrites = _call_api(attempt_tasks, config)
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(f"API batch failed after {max_retries} attempts: {type(exc).__name__}") from exc
            time.sleep(float(config.get("retry_delay_seconds", 2.0)) * attempt)
            continue
        last_reasons = {
            task["task_id"]: validate_rewrite(task, rewrites[task["task_id"]])
            for task in tasks
        }
        if not any(last_reasons.values()):
            return rewrites
        attempt_tasks = [
            {
                **task,
                "_rewrite_feedback": last_reasons.get(str(task["task_id"]), []),
            }
            for task in tasks
        ]
        if attempt < max_retries:
            time.sleep(float(config.get("retry_delay_seconds", 2.0)) * attempt)
    compact = {task_id: reasons for task_id, reasons in last_reasons.items() if reasons}
    raise ValueError(f"API rewrites failed semantic validation: {compact}")


def rewrite_sandbox(
    source_dir: Path,
    output_root: Path,
    config_path: Path,
    *,
    version: str,
    batch_size: int,
    max_workers: int,
    max_retries: int,
    socks_proxy: str | None = None,
    resume_incomplete: bool = False,
    limit: int | None = None,
    contract: str = CONTRACT,
    instruction_style: str = "boss_api_mixed_company_roles_natural_language_v1",
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", version):
        raise ValueError("invalid output version")
    output_dir = output_root / version
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    tasks = _read_jsonl(source_dir / "dwh_tasks.jsonl")
    if limit is not None:
        if limit <= 0 or limit > len(tasks):
            raise ValueError("limit must be within source task count")
        tasks = tasks[:limit]
    config = _load_api_config(config_path)
    if socks_proxy:
        config["_socks_proxy"] = socks_proxy

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    if incomplete.exists() and not resume_incomplete:
        raise FileExistsError(f"incomplete output already exists: {incomplete}")
    if not incomplete.exists():
        shutil.copytree(source_dir, incomplete)
    existing_rows = _read_jsonl(incomplete / "dwh_tasks.jsonl") if resume_incomplete else []
    source_ids = {str(task["task_id"]) for task in tasks}
    source_by_id = {str(task["task_id"]): task for task in tasks}
    rewritten_by_id: dict[str, dict[str, Any]] = {
        str(task["task_id"]): task
        for task in existing_rows
        if str(task.get("task_id") or "") in source_ids
        and (task.get("instruction_generation") or {}).get("semantic_validation_passed") is True
        and task.get("query_plan") == source_by_id[str(task["task_id"])].get("query_plan")
        and task.get("gold_answer") == source_by_id[str(task["task_id"])].get("gold_answer")
        and not validate_rewrite(
            source_by_id[str(task["task_id"])],
            str(task.get("natural_language_instruction") or ""),
        )
    }
    try:
        pending_tasks = [
            task for task in tasks if str(task["task_id"]) not in rewritten_by_id
        ]
        batches = [
            (start, pending_tasks[start : start + batch_size])
            for start in range(0, len(pending_tasks), batch_size)
        ]
        failed_batches: list[tuple[int, str]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(rewrite_batch, batch, config, max_retries=max_retries): (start, batch)
                for start, batch in batches
            }
            for future in as_completed(futures):
                batch_start, batch = futures[future]
                try:
                    results = future.result()
                except Exception as exc:
                    failed_batches.append((batch_start, type(exc).__name__))
                    continue
                for task in batch:
                    updated = dict(task)
                    source_instruction = str(task["natural_language_instruction"])
                    final_instruction = results[str(task["task_id"])]
                    updated["natural_language_instruction"] = final_instruction
                    updated["instruction_variants"] = [final_instruction]
                    updated["instruction_style"] = instruction_style
                    updated["instruction_generation"] = {
                        "method": "boss_openai_compatible_chat_api",
                        "source_instruction_sha256": _source_hash(source_instruction),
                        "semantic_validation_passed": True,
                        "attempt_policy": "fail_closed",
                        "requested_max_workers": max_workers,
                    }
                    updated["generation_contract"] = contract
                    rewritten_by_id[str(task["task_id"])] = updated
                ordered_partial = [
                    rewritten_by_id[str(task["task_id"])]
                    for task in tasks
                    if str(task["task_id"]) in rewritten_by_id
                ]
                _write_jsonl(incomplete / "dwh_tasks.jsonl", ordered_partial)
                print(f"validated_api_rewrites={len(rewritten_by_id)}/{len(tasks)}", flush=True)

        if failed_batches:
            failed_task_count = sum(
                len(pending_tasks[start : start + batch_size])
                for start, _error in failed_batches
            )
            raise RuntimeError(
                f"{failed_task_count} task rewrites failed this pass; "
                "rerun with --resume-incomplete and lower concurrency"
            )

        rewritten = [rewritten_by_id[str(task["task_id"])] for task in tasks]

        if limit is not None:
            # A limited run is an API smoke artifact, never a complete sandbox.
            summary_name = "api_smoke_summary.json"
            summary = {
                "contract": contract,
                "task_count": len(rewritten),
                "api_used": True,
                "semantic_validation_passed": len(rewritten),
                "training_allowed": False,
            }
            (incomplete / summary_name).write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        else:
            if len(rewritten) != len(tasks):
                raise ValueError(f"expected {len(tasks)} API rewrites, got {len(rewritten)}")
            summary_path = incomplete / "generation_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update(
                {
                    "contract": contract,
                    "environment_id": f"sft/{version}",
                    "external_api_used": True,
                    "instruction_style": instruction_style,
                    "api_rewrite_rows": len(rewritten),
                    "api_semantic_validation_passed_rows": len(rewritten),
                    "training_allowed": False,
                }
            )
            summary["files"]["dwh_tasks.jsonl"] = _sha256(incomplete / "dwh_tasks.jsonl")
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        incomplete.replace(output_dir)
    except Exception:
        # Keep the private .incomplete directory for diagnosis/resume evidence;
        # it is never a valid sft/<version> environment.
        raise
    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sandbox", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, required=True)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--socks-proxy")
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.batch_size > 12:
        raise ValueError("batch-size must be between 1 and 12")
    if args.max_workers <= 0 or args.max_workers > 32:
        raise ValueError("max-workers must be between 1 and 32")
    output = rewrite_sandbox(
        args.source_sandbox,
        args.output_root,
        args.api_config,
        version=args.version,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        socks_proxy=args.socks_proxy,
        resume_incomplete=args.resume_incomplete,
        limit=args.limit,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
