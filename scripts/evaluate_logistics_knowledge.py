"""Evaluate logistics multiple-choice knowledge through an OpenAI-compatible API.

Benchmark prompts, gold answers, and raw model outputs are written only to the
private JSONL.  The safe JSON contains hashes, dimensions, correctness flags,
and aggregate metrics suitable for version control and paired comparison.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SC_KNOWLEDGE_FILES = (
    "SC-bench-main/data/multiple_choices_clean_final_clean.jsonl",
    "SC-bench-main/data/single_choices_clean_final_clean.jsonl",
    "SC-bench-main/data/true_false_clean_final_clean.jsonl",
)
PROMPT_VERSION = "logistics-mcq-zero-based-json-v2"


@dataclass(frozen=True)
class EvalItem:
    item_hash: str
    dataset: str
    source_id: str
    category: str
    question_type: str
    question: str
    options: tuple[str, ...]
    expected: tuple[int, ...]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def _make_item(
    *,
    dataset: str,
    source_id: str,
    category: str,
    question_type: str,
    question: str,
    options: Iterable[str],
    expected: Iterable[int],
) -> EvalItem:
    normalized_options = tuple(str(value).strip() for value in options)
    normalized_expected = tuple(sorted({int(value) for value in expected}))
    if not question.strip():
        raise ValueError(f"{dataset}:{source_id} has an empty question")
    if len(normalized_options) < 1:
        raise ValueError(f"{dataset}:{source_id} has unsupported option count {len(normalized_options)}")
    allowed = set(range(len(normalized_options)))
    if not normalized_expected or not set(normalized_expected) <= allowed:
        raise ValueError(f"{dataset}:{source_id} has invalid expected labels {normalized_expected}")
    fingerprint = {
        "dataset": dataset,
        "question_type": question_type,
        "question": question.strip(),
        "options": normalized_options,
        "expected": normalized_expected,
    }
    return EvalItem(
        item_hash=_canonical_hash(fingerprint),
        dataset=dataset,
        source_id=source_id,
        category=category.strip() or "unknown",
        question_type=question_type.strip() or "unknown",
        question=question.strip(),
        options=normalized_options,
        expected=normalized_expected,
    )


def load_logistika(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            source_id = str(row.get("question_id") or index)
            choices = json.loads(row.get("choices") or "[]")
            answer_indices = json.loads(row.get("answer") or "[]")
            expected = [int(value) for value in answer_indices]
            items.append(
                _make_item(
                    dataset="LogistikaBench",
                    source_id=source_id,
                    category=str(row.get("subject") or "unknown"),
                    question_type=str(row.get("question_type") or "multiple_choice"),
                    question=str(row.get("question") or ""),
                    options=(str(value) for value in choices),
                    expected=expected,
                )
            )
    return items


def _sc_expected(raw_options: list[Any], raw_answer: Any) -> tuple[list[str], list[int]]:
    options = [str(value.get("text", "")) if isinstance(value, dict) else str(value) for value in raw_options]
    keys = [str(value.get("key", index)) if isinstance(value, dict) else str(index) for index, value in enumerate(raw_options)]
    answers = raw_answer if isinstance(raw_answer, list) else [raw_answer]
    text_to_index = {option.strip().casefold(): index for index, option in enumerate(options)}
    key_to_index = {key.strip().casefold(): index for index, key in enumerate(keys)}
    expected: list[int] = []
    for value in answers:
        normalized = str(value).strip()
        folded = normalized.casefold()
        if folded in key_to_index:
            expected.append(key_to_index[folded])
        elif folded in text_to_index:
            expected.append(text_to_index[folded])
        else:
            raise ValueError(f"SC-bench answer {value!r} does not match an option")
    return options, expected


def load_sc_knowledge(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with zipfile.ZipFile(path) as archive:
        for member in SC_KNOWLEDGE_FILES:
            with archive.open(member) as handle:
                for line_index, raw_line in enumerate(handle, start=1):
                    if not raw_line.strip():
                        continue
                    outer = json.loads(raw_line.decode("utf-8"))
                    row = outer.get("output", outer)
                    raw_options = row.get("options") or []
                    if not raw_options and str(row.get("question_type") or "") == "true_or_false":
                        raw_options = ["true", "false"]
                    options, expected = _sc_expected(raw_options, row.get("answer") or [])
                    source_id = f"{Path(member).name}:{line_index}"
                    items.append(
                        _make_item(
                            dataset="SC-bench-knowledge",
                            source_id=source_id,
                            category=str(row.get("field") or "unknown"),
                            question_type=str(row.get("question_type") or Path(member).stem),
                            question=str(row.get("question") or ""),
                            options=options,
                            expected=expected,
                        )
                    )
    return items


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs without printing their values."""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_messages(item: EvalItem) -> list[dict[str, str]]:
    choices = "\n".join(f"[{index}] {value}" for index, value in enumerate(item.options))
    return [
        {
            "role": "system",
            "content": (
                "Answer the closed-book logistics question using only the listed zero-based option indices. "
                "Return exactly one JSON object in the form {\"answers\":[0]}. "
                "For questions with multiple correct choices, include every correct index. "
                "Do not include reasoning, markdown, or any other keys."
            ),
        },
        {"role": "user", "content": f"Question:\n{item.question}\n\nOptions:\n{choices}"},
    ]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_answers(text: str, choice_count: int) -> tuple[tuple[int, ...], bool]:
    value = _extract_json_object(text.strip())
    if value is None or set(value) != {"answers"}:
        return (), False
    raw_answers = value["answers"]
    if isinstance(raw_answers, str):
        raw_answers = [part for part in re.split(r"[\s,;/|]+", raw_answers) if part]
    if not isinstance(raw_answers, list):
        return (), False
    answers_list: list[int] = []
    for answer in raw_answers:
        if isinstance(answer, bool):
            return (), False
        if isinstance(answer, int):
            answers_list.append(answer)
        elif isinstance(answer, str) and answer.strip().isdigit():
            answers_list.append(int(answer.strip()))
        else:
            return (), False
    answers = tuple(sorted(set(answers_list)))
    allowed = set(range(choice_count))
    if not answers or not set(answers) <= allowed:
        return (), False
    return answers, True


def _usage_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage) if isinstance(usage, dict) else None


def _call_one(
    client: Any,
    model: str,
    item: EvalItem,
    max_tokens: int,
    retries: int,
    chat_template_disable_thinking: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            extra_body: dict[str, Any] = {"enable_thinking": False}
            if chat_template_disable_thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            response = client.chat.completions.create(
                model=model,
                messages=build_messages(item),
                temperature=0,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            message = response.choices[0].message
            prediction = message.content or ""
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
            parsed, parse_ok = parse_answers(prediction, len(item.options))
            return {
                "prompt_version": PROMPT_VERSION,
                "chat_template_disable_thinking": chat_template_disable_thinking,
                "item_hash": item.item_hash,
                "dataset": item.dataset,
                "source_id": item.source_id,
                "category": item.category,
                "question_type": item.question_type,
                "question": item.question,
                "options": list(item.options),
                "expected": list(item.expected),
                "prediction": prediction,
                "reasoning": reasoning,
                "parsed": list(parsed),
                "parse_ok": parse_ok,
                "correct": parse_ok and parsed == item.expected,
                "elapsed_sec": round(time.perf_counter() - started, 6),
                "usage": _usage_dict(response),
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - exercised against remote endpoints
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return {
        "prompt_version": PROMPT_VERSION,
        "chat_template_disable_thinking": chat_template_disable_thinking,
        "item_hash": item.item_hash,
        "dataset": item.dataset,
        "source_id": item.source_id,
        "category": item.category,
        "question_type": item.question_type,
        "question": item.question,
        "options": list(item.options),
        "expected": list(item.expected),
        "prediction": "",
        "reasoning": "",
        "parsed": [],
        "parse_ok": False,
        "correct": False,
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "usage": None,
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _group_metrics(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group_key in sorted(grouped):
        group = grouped[group_key]
        record = {key: value for key, value in zip(keys, group_key)}
        record.update(
            {
                "items": len(group),
                "correct": sum(bool(row["correct"]) for row in group),
                "accuracy": round(sum(bool(row["correct"]) for row in group) / len(group), 6),
                "parse_failures": sum(not bool(row["parse_ok"]) for row in group),
                "api_failures": sum(bool(row.get("api_error_type")) for row in group),
            }
        )
        output.append(record)
    return output


def safe_row(row: dict[str, Any]) -> dict[str, Any]:
    error = row.get("error")
    return {
        "item_hash": row["item_hash"],
        "dataset": row["dataset"],
        "category": row["category"],
        "question_type": row["question_type"],
        "choice_count": len(row["options"]),
        "expected_count": len(row["expected"]),
        "predicted_count": len(row["parsed"]),
        "parse_ok": bool(row["parse_ok"]),
        "correct": bool(row["correct"]),
        "api_error_type": str(error).split(":", 1)[0] if error else None,
        "elapsed_sec": row["elapsed_sec"],
    }


def build_safe_result(
    rows: list[dict[str, Any]],
    *,
    model: str,
    endpoint_label: str,
    concurrency: int,
    input_hashes: dict[str, str],
    elapsed_sec: float,
    chat_template_disable_thinking: bool = False,
) -> dict[str, Any]:
    safe_rows = [safe_row(row) for row in sorted(rows, key=lambda value: value["item_hash"])]
    successful_latencies = [float(row["elapsed_sec"]) for row in rows if not row.get("error")]
    category_groups = _group_metrics(safe_rows, ("dataset", "category"))
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "schema_version": 1,
        "evaluation_type": "closed_book_logistics_multiple_choice",
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "endpoint_label": endpoint_label,
        "request": {
            "temperature": 0,
            "max_output_tokens": 128,
            "enable_thinking": False,
            "chat_template_disable_thinking": chat_template_disable_thinking,
            "concurrency": concurrency,
        },
        "input_sha256": input_hashes,
        "items": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 6) if rows else 0.0,
        "macro_category_accuracy": round(statistics.fmean(group["accuracy"] for group in category_groups), 6)
        if category_groups
        else 0.0,
        "parse_failures": sum(not bool(row["parse_ok"]) for row in rows),
        "api_failures": sum(bool(row.get("error")) for row in rows),
        "degenerate_single_option_items": sum(len(row["options"]) == 1 for row in rows),
        "wall_elapsed_sec": round(elapsed_sec, 6),
        "latency_sec": {
            "p50": round(_percentile(successful_latencies, 0.50), 6),
            "p95": round(_percentile(successful_latencies, 0.95), 6),
        },
        "by_dataset": _group_metrics(safe_rows, ("dataset",)),
        "by_dataset_category": category_groups,
        "by_dataset_question_type": _group_metrics(safe_rows, ("dataset", "question_type")),
        "rows": safe_rows,
        "private_content_included": False,
    }


def write_private_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda value: value["item_hash"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    os.replace(temporary, path)


def load_private_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            row = json.loads(raw_line)
            rows[str(row["item_hash"])] = row
    return rows


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def mcnemar_exact_pvalue(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    cutoff = min(improved, regressed)
    log_sum = -math.inf
    for value in range(cutoff + 1):
        log_probability = (
            math.lgamma(discordant + 1)
            - math.lgamma(value + 1)
            - math.lgamma(discordant - value + 1)
            - discordant * math.log(2)
        )
        log_sum = _logaddexp(log_sum, log_probability)
    return min(1.0, 2 * math.exp(log_sum))


def _paired_metrics(base_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = {str(row["item_hash"]): row for row in base_rows}
    candidate = {str(row["item_hash"]): row for row in candidate_rows}
    if set(base) != set(candidate):
        missing_candidate = len(set(base) - set(candidate))
        missing_baseline = len(set(candidate) - set(base))
        raise ValueError(
            f"item hashes differ: {missing_candidate} missing from candidate, {missing_baseline} missing from baseline"
        )
    pairs = [(base[item_hash], candidate[item_hash]) for item_hash in sorted(base)]
    improved = sum(not bool(left["correct"]) and bool(right["correct"]) for left, right in pairs)
    regressed = sum(bool(left["correct"]) and not bool(right["correct"]) for left, right in pairs)
    both_correct = sum(bool(left["correct"]) and bool(right["correct"]) for left, right in pairs)
    both_wrong = len(pairs) - improved - regressed - both_correct
    base_correct = both_correct + regressed
    candidate_correct = both_correct + improved
    return {
        "items": len(pairs),
        "baseline_correct": base_correct,
        "baseline_accuracy": round(base_correct / len(pairs), 6) if pairs else 0.0,
        "candidate_correct": candidate_correct,
        "candidate_accuracy": round(candidate_correct / len(pairs), 6) if pairs else 0.0,
        "delta_accuracy_points": round(100 * (candidate_correct - base_correct) / len(pairs), 4) if pairs else 0.0,
        "improved_0_to_1": improved,
        "regressed_1_to_0": regressed,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_correct": improved - regressed,
        "mcnemar_exact_pvalue": round(mcnemar_exact_pvalue(improved, regressed), 12),
    }


def _paired_groups(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    group_keys = sorted({tuple(str(row[key]) for key in keys) for row in baseline_rows})
    output: list[dict[str, Any]] = []
    for group_key in group_keys:
        base_group = [row for row in baseline_rows if tuple(str(row[key]) for key in keys) == group_key]
        candidate_group = [row for row in candidate_rows if tuple(str(row[key]) for key in keys) == group_key]
        output.append(
            {
                **{key: value for key, value in zip(keys, group_key)},
                **_paired_metrics(base_group, candidate_group),
            }
        )
    return output


def compare_safe_results(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if baseline.get("prompt_version") != candidate.get("prompt_version"):
        raise ValueError("prompt versions differ")
    if baseline.get("input_sha256") != candidate.get("input_sha256"):
        raise ValueError("input hashes differ")
    overall = _paired_metrics(baseline["rows"], candidate["rows"])
    dataset_groups = _paired_groups(baseline["rows"], candidate["rows"], ("dataset",))
    dimensions = _paired_groups(baseline["rows"], candidate["rows"], ("dataset", "category"))
    question_type_groups = _paired_groups(
        baseline["rows"], candidate["rows"], ("dataset", "question_type")
    )
    choice_count_groups = _paired_groups(baseline["rows"], candidate["rows"], ("dataset", "choice_count"))
    return {
        "schema_version": 1,
        "comparison_type": "paired_closed_book_logistics_multiple_choice",
        "prompt_version": baseline["prompt_version"],
        "input_sha256": baseline["input_sha256"],
        "baseline_model": baseline["model"],
        "candidate_model": candidate["model"],
        "overall": overall,
        "by_dataset": dataset_groups,
        "macro_category_accuracy": {
            "baseline": round(statistics.fmean(row["baseline_accuracy"] for row in dimensions), 6),
            "candidate": round(statistics.fmean(row["candidate_accuracy"] for row in dimensions), 6),
            "delta_points": round(
                100
                * statistics.fmean(row["candidate_accuracy"] - row["baseline_accuracy"] for row in dimensions),
                4,
            ),
        }
        if dimensions
        else {"baseline": 0.0, "candidate": 0.0, "delta_points": 0.0},
        "by_dataset_category": dimensions,
        "by_dataset_question_type": question_type_groups,
        "by_dataset_choice_count": choice_count_groups,
        "private_content_included": False,
    }


def _evaluate(args: argparse.Namespace) -> int:
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if args.env_file:
        load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env) or os.environ.get("DASHSCOPE_API_KEY") or "EMPTY"
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")
    if not base_url:
        raise RuntimeError("missing --base-url or OpenAI-compatible base URL environment variable")

    items = load_logistika(args.logistika) + load_sc_knowledge(args.sc_zip)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        items = items[: args.limit]
    if len({item.item_hash for item in items}) != len(items):
        raise ValueError("duplicate item hashes detected")
    existing = load_private_rows(args.private_output) if args.resume else {}
    valid_hashes = {item.item_hash for item in items}
    completed = {
        item_hash: row
        for item_hash, row in existing.items()
        if item_hash in valid_hashes
        and not row.get("error")
        and row.get("prompt_version") == PROMPT_VERSION
        and bool(row.get("chat_template_disable_thinking")) == args.chat_template_disable_thinking
    }

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)
    pending = [item for item in items if item.item_hash not in completed]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                _call_one,
                client,
                args.model,
                item,
                args.max_output_tokens,
                args.retries,
                args.chat_template_disable_thinking,
            )
            for item in pending
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed[row["item_hash"]] = row
            if len(completed) % args.checkpoint_every == 0:
                write_private_rows(args.private_output, list(completed.values()))
    rows = [completed[item.item_hash] for item in items]
    write_private_rows(args.private_output, rows)

    input_hashes = {
        "logistika": sha256_bytes(args.logistika.read_bytes()),
        "sc_zip": sha256_bytes(args.sc_zip.read_bytes()),
    }
    result = build_safe_result(
        rows,
        model=args.model,
        endpoint_label=args.endpoint_label,
        concurrency=args.concurrency,
        input_hashes=input_hashes,
        elapsed_sec=time.perf_counter() - started,
        chat_template_disable_thinking=args.chat_template_disable_thinking,
    )
    result["request"]["max_output_tokens"] = args.max_output_tokens
    args.safe_output.parent.mkdir(parents=True, exist_ok=True)
    args.safe_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return 0 if not result["api_failures"] else 2


def _compare(args: argparse.Namespace) -> int:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare_safe_results(baseline, candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Run one model endpoint")
    evaluate.add_argument("--logistika", type=Path, required=True)
    evaluate.add_argument("--sc-zip", type=Path, required=True)
    evaluate.add_argument("--env-file", type=Path)
    evaluate.add_argument("--base-url")
    evaluate.add_argument("--api-key-env", default="OPENAI_API_KEY")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--endpoint-label", required=True)
    evaluate.add_argument("--concurrency", type=int, default=64)
    evaluate.add_argument("--max-output-tokens", type=int, default=128)
    evaluate.add_argument("--chat-template-disable-thinking", action="store_true")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--timeout", type=float, default=120.0)
    evaluate.add_argument("--retries", type=int, default=2)
    evaluate.add_argument("--checkpoint-every", type=int, default=64)
    evaluate.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    evaluate.add_argument("--private-output", type=Path, required=True)
    evaluate.add_argument("--safe-output", type=Path, required=True)
    evaluate.set_defaults(handler=_evaluate)

    compare = subparsers.add_parser("compare", help="Pair two safe evaluation outputs")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(handler=_compare)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
