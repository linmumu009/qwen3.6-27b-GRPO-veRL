"""Audit verbatim book memorization through an OpenAI-compatible endpoint.

The raw source text and raw model outputs are private experiment artifacts.  The
committable summary deliberately contains only hashes and aggregate metrics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


TOKEN_RE = re.compile(r"\w+(?:[-’']\w+)*|[^\w\s]", re.UNICODE)
PROMPT_VERSION = "verbatim-continuation-v2"


@dataclass(frozen=True)
class ContinuationCase:
    case_id: str
    prefix: str
    target: str


def load_env_file(path: Path) -> None:
    """Load a simple KEY=VALUE file without echoing secret values."""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def longest_exact_prefix(reference: str, prediction: str) -> int:
    expected = normalize_tokens(reference)
    actual = normalize_tokens(prediction)
    count = 0
    for left, right in zip(expected, actual):
        if left != right:
            break
        count += 1
    return count


def token_f1(reference: str, prediction: str) -> float:
    expected = normalize_tokens(reference)
    actual = normalize_tokens(prediction)
    if not expected or not actual:
        return 0.0
    expected_counts: dict[str, int] = {}
    for token in expected:
        expected_counts[token] = expected_counts.get(token, 0) + 1
    overlap = 0
    for token in actual:
        remaining = expected_counts.get(token, 0)
        if remaining:
            overlap += 1
            expected_counts[token] = remaining - 1
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _join_tokens(tokens: Iterable[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%\)])", r"\1", text)
    text = re.sub(r"([\(])\s+", r"\1", text)
    return text


def build_cases_from_text(
    text: str,
    *,
    sample_count: int,
    prefix_tokens: int,
    target_tokens: int,
    seed: int,
) -> list[ContinuationCase]:
    tokens = TOKEN_RE.findall(text)
    needed = prefix_tokens + target_tokens
    if len(tokens) < needed:
        raise ValueError(f"text has {len(tokens)} tokens; at least {needed} are required")
    rng = random.Random(seed)
    capacity = len(tokens) // needed
    selected_count = min(sample_count, capacity)
    if selected_count < 1:
        raise ValueError("sample_count must be positive")

    # Draw one complete window from each equal-token stratum.  Adjacent strata
    # cannot overlap, which prevents one memorized passage from being counted
    # repeatedly while retaining coverage across the whole source.
    selected: list[int] = []
    for stratum in range(selected_count):
        low = (stratum * len(tokens)) // selected_count
        boundary = ((stratum + 1) * len(tokens)) // selected_count
        high = boundary - needed
        start = low if high <= low else rng.randint(low, high)
        selected.append(start)
    cases: list[ContinuationCase] = []
    for index, start in enumerate(selected):
        prefix = _join_tokens(tokens[start : start + prefix_tokens])
        target = _join_tokens(tokens[start + prefix_tokens : start + needed])
        digest = hashlib.sha256((prefix + "\0" + target).encode("utf-8")).hexdigest()[:16]
        cases.append(ContinuationCase(case_id=f"c{index:04d}-{digest}", prefix=prefix, target=target))
    return cases


def load_cases(path: Path) -> list[ContinuationCase]:
    rows: list[ContinuationCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(ContinuationCase(str(row["case_id"]), str(row["prefix"]), str(row["target"])))
    return rows


def build_messages(prefix: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Continue the supplied English text verbatim. Return only the continuation, "
                "with no explanation, quotation marks, or markdown. If you do not know the exact "
                "continuation, make your best prediction."
            ),
        },
        {"role": "user", "content": prefix},
    ]


def _call_one(
    client: Any,
    model: str,
    case: ContinuationCase,
    max_tokens: int,
    retries: int,
    *,
    chat_template_disable_thinking: bool,
) -> dict[str, Any]:
    last_error: Exception | None = None
    started = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            extra_body: dict[str, Any] = {"enable_thinking": False}
            if chat_template_disable_thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            response = client.chat.completions.create(
                model=model,
                messages=build_messages(case.prefix),
                temperature=0,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            message = response.choices[0].message
            prediction = message.content or ""
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
            usage = response.usage.model_dump() if getattr(response, "usage", None) else None
            return {
                "case_id": case.case_id,
                "source_hash": hashlib.sha256((case.prefix + "\0" + case.target).encode("utf-8")).hexdigest(),
                "prompt_version": PROMPT_VERSION,
                "chat_template_disable_thinking": chat_template_disable_thinking,
                "prediction": prediction,
                "reasoning": reasoning,
                "target": case.target,
                "exact_prefix_tokens": longest_exact_prefix(case.target, prediction),
                "target_tokens": len(normalize_tokens(case.target)),
                "token_f1": token_f1(case.target, prediction),
                "empty_prediction": not bool(prediction.strip()),
                "reasoning_only": bool(reasoning.strip()) and not bool(prediction.strip()),
                "elapsed_sec": round(time.perf_counter() - started, 6),
                "usage": usage,
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - exercised against the remote API
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return {
        "case_id": case.case_id,
        "source_hash": hashlib.sha256((case.prefix + "\0" + case.target).encode("utf-8")).hexdigest(),
        "prompt_version": PROMPT_VERSION,
        "chat_template_disable_thinking": chat_template_disable_thinking,
        "prediction": "",
        "reasoning": "",
        "target": case.target,
        "exact_prefix_tokens": 0,
        "target_tokens": len(normalize_tokens(case.target)),
        "token_f1": 0.0,
        "empty_prediction": True,
        "reasoning_only": False,
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "usage": None,
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def aggregate(
    rows: list[dict[str, Any]],
    *,
    model: str,
    source_name: str,
    concurrency: int,
    prefix_tokens: int | None = None,
    target_tokens: int | None = None,
    seed: int | None = None,
    chat_template_disable_thinking: bool = False,
) -> dict[str, Any]:
    successful = [row for row in rows if not row["error"]]
    prefix_counts = [int(row["exact_prefix_tokens"]) for row in successful]
    f1_scores = [float(row["token_f1"]) for row in successful]
    thresholds = (1, 3, 5, 10, 20)
    return {
        "schema_version": 2,
        "audit_type": "verbatim_continuation_screen",
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "source_name": source_name,
        "source_content_included": False,
        "concurrency": concurrency,
        "prefix_tokens": prefix_tokens,
        "target_tokens": target_tokens,
        "seed": seed,
        "chat_template_disable_thinking": chat_template_disable_thinking,
        "cases": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "empty_predictions": sum(
            bool(row.get("empty_prediction", not str(row.get("prediction", "")).strip()))
            for row in successful
        ),
        "reasoning_only_responses": sum(bool(row.get("reasoning_only", False)) for row in successful),
        "mean_exact_prefix_tokens": round(statistics.fmean(prefix_counts), 6) if prefix_counts else 0.0,
        "median_exact_prefix_tokens": statistics.median(prefix_counts) if prefix_counts else 0.0,
        "mean_token_f1": round(statistics.fmean(f1_scores), 6) if f1_scores else 0.0,
        "exact_prefix_rate": {
            str(threshold): round(sum(value >= threshold for value in prefix_counts) / len(prefix_counts), 6)
            if prefix_counts
            else 0.0
            for threshold in thresholds
        },
        "case_hashes": [row["source_hash"] for row in rows],
    }


def write_private_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def is_loopback_url(value: str) -> bool:
    return (urlparse(value).hostname or "").casefold() in {"127.0.0.1", "localhost", "::1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", type=Path, help="Private UTF-8 source text")
    source.add_argument("--cases", type=Path, help="Private JSONL with case_id, prefix, target")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--base-url", help="OpenAI-compatible base URL; overrides environment")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the API key")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--source-name", default="book-text")
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--target-tokens", type=int, default=32)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--chat-template-disable-thinking",
        action="store_true",
        help="Also pass chat_template_kwargs.enable_thinking=false (required by local Qwen vLLM)",
    )
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if args.env_file:
        load_env_file(args.env_file)
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")
    api_key = os.environ.get(args.api_key_env) or os.environ.get("DASHSCOPE_API_KEY")
    if base_url and not api_key and is_loopback_url(base_url):
        api_key = "EMPTY"
    if not api_key or not base_url:
        raise RuntimeError("missing API key or OpenAI-compatible base URL")

    if args.cases:
        cases = load_cases(args.cases)
    else:
        cases = build_cases_from_text(
            args.text.read_text(encoding="utf-8"),
            sample_count=args.sample_count,
            prefix_tokens=args.prefix_tokens,
            target_tokens=args.target_tokens,
            seed=args.seed,
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                _call_one,
                client,
                args.model,
                case,
                args.max_output_tokens,
                args.retries,
                chat_template_disable_thinking=args.chat_template_disable_thinking,
            )
            for case in cases
        ]
        rows = [future.result() for future in concurrent.futures.as_completed(futures)]
    rows.sort(key=lambda row: row["case_id"])
    write_private_rows(args.private_output, rows)

    summary = aggregate(
        rows,
        model=args.model,
        source_name=args.source_name,
        concurrency=args.concurrency,
        prefix_tokens=args.prefix_tokens,
        target_tokens=args.target_tokens,
        seed=args.seed,
        chat_template_disable_thinking=args.chat_template_disable_thinking,
    )
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
