#!/usr/bin/env python3
"""Paraphrase frozen logistics exam stems with fail-closed semantic checks."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REWRITE_VERSION = "logistics-exam-stem-paraphrase-v1"
NUMBER_RE = re.compile(r"(?<!\w)[$£€¥]?\d[\d,.]*(?:\s?(?:%|[A-Za-z]{1,8}))?", re.IGNORECASE)
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9/-]{1,}\b")
QUOTED_RE = re.compile(r'["“]([^"”]+)["”]')
LOGIC_FAMILIES = {
    "negative": re.compile(r"\b(?:not|never|no|cannot|can't|incorrect|false|wrong|except|excluding|without)\b", re.I),
    "except": re.compile(r"\b(?:except|excluding|other than)\b", re.I),
    "least": re.compile(r"\b(?:least|minimum|fewest|lowest|smallest|worst)\b", re.I),
    "most": re.compile(r"\b(?:most|maximum|greatest|highest|best|primary|main|principal|predominant)\b", re.I),
    "before": re.compile(r"\b(?:before|earlier|prior)\b", re.I),
    "after": re.compile(r"\b(?:after|later|subsequent)\b", re.I),
}


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_space(value: str) -> str:
    return " ".join(str(value).split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def write_jsonl_private(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _extract_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response does not contain a JSON object")


def deterministic_validation(original: str, rewritten: str) -> list[str]:
    reasons: list[str] = []
    left = normalize_space(original)
    right = normalize_space(rewritten)
    if not right:
        return ["empty"]
    if left.casefold() == right.casefold():
        reasons.append("unchanged")
    ratio = len(right) / max(1, len(left))
    if ratio < 0.6 or ratio > 1.8:
        reasons.append("length_ratio")
    normalize_number = lambda value: re.sub(r"[\s,]", "", value).casefold()
    if Counter(normalize_number(value) for value in NUMBER_RE.findall(left)) != Counter(
        normalize_number(value) for value in NUMBER_RE.findall(right)
    ):
        reasons.append("numbers_changed")
    left_acronyms = Counter(ACRONYM_RE.findall(left))
    right_acronyms = Counter(ACRONYM_RE.findall(right))
    if any(value not in right_acronyms for value in left_acronyms):
        reasons.append("acronyms_changed")
    if any(value.casefold() not in right.casefold() for value in QUOTED_RE.findall(left)):
        reasons.append("quoted_terms_changed")
    for name, pattern in LOGIC_FAMILIES.items():
        left_has = bool(pattern.search(left))
        right_has = bool(pattern.search(right))
        if name == "except":
            right_has = right_has or bool(LOGIC_FAMILIES["negative"].search(right))
        if left_has and not right_has:
            reasons.append(f"logic_family_changed:{name}")
        if name == "negative" and right_has and not left_has:
            reasons.append("logic_family_changed:negative_added")
    if "answer" in right.casefold() and "answer" not in left.casefold():
        reasons.append("answer_meta_language_added")
    return reasons


def item_validation(row: dict[str, Any], rewritten: str) -> list[str]:
    reasons = deterministic_validation(normalize_space(row["question"]), rewritten)
    original = normalize_space(row["question"]).casefold()
    candidate = normalize_space(rewritten)
    if str(row.get("question_type")) != "true_or_false" and not candidate.endswith("?"):
        reasons.append("not_a_question")
    folded_candidate = candidate.casefold()
    for option in row.get("options", []):
        normalized_option = normalize_space(option).casefold().strip(".?!")
        if len(normalized_option) >= 8 and normalized_option not in original and normalized_option in folded_candidate:
            reasons.append("choice_text_added")
            break
    return reasons


def normalize_rewritten_form(row: dict[str, Any], rewritten: str) -> str:
    candidate = normalize_space(rewritten)
    if str(row.get("question_type")) != "true_or_false" and candidate and not candidate.endswith("?"):
        candidate = candidate.rstrip(".!；;。") + "?"
    return candidate


def build_rewrite_messages(row: dict[str, Any], feedback: list[str]) -> list[dict[str, str]]:
    expected = sorted(set(int(value) for value in row["expected"]))
    payload = {
        "original_question": normalize_space(row["question"]),
        "choices": [normalize_space(value) for value in row["options"]],
        "correct_answer_texts": [normalize_space(row["options"][index]) for index in expected],
        "must_preserve_verbatim": sorted(
            set(NUMBER_RE.findall(normalize_space(row["question"])))
            | set(ACRONYM_RE.findall(normalize_space(row["question"])))
            | set(QUOTED_RE.findall(normalize_space(row["question"])))
        ),
        "must_preserve_logic": sorted(
            name for name, pattern in LOGIC_FAMILIES.items() if pattern.search(normalize_space(row["question"]))
        ),
    }
    if feedback:
        payload["previous_attempt_rejected_because"] = feedback
    requested_form = (
        "Rewrite original_question as one declarative assertion with the same truth value."
        if str(row.get("question_type")) == "true_or_false"
        else "Rewrite original_question as one English question ending in a question mark."
    )
    return [
        {
            "role": "system",
            "content": (
                "You paraphrase only the stem of a logistics benchmark question. Preserve exactly its meaning, "
                "truth value, polarity, entities, quantities, units, conditions, scope, and which supplied choices "
                "are correct. Do not answer the question, mention the correct choice, add facts, remove constraints, "
                "or copy the original wording. Keep every number, unit, abbreviation, quoted technical term, negation, "
                "and least/most comparison. Keep the rewritten stem at roughly the same length. Return exactly one "
                "JSON object and no markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{requested_form} The choices and correct answer "
                "texts are context for semantic preservation only and must not be appended to the question. "
                "Return {\"rewritten_question\":\"...\"}.\n" + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]


def build_verify_messages(row: dict[str, Any], rewritten: str) -> list[dict[str, str]]:
    expected = sorted(set(int(value) for value in row["expected"]))
    payload = {
        "original_question": normalize_space(row["question"]),
        "rewritten_question": rewritten,
        "choices": [normalize_space(value) for value in row["options"]],
        "correct_answer_texts": [normalize_space(row["options"][index]) for index in expected],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a semantic-equivalence verifier for exam-question paraphrases. Accept ordinary wording and "
                "grammar changes when meaning and correct choices stay identical. Reject only a real change to truth "
                "value, negation, quantity, unit, entity, condition, scope, or correct choices."
            ),
        },
        {
            "role": "user",
            "content": (
                "Check the candidate. Return exactly one JSON object with boolean keys equivalent and "
                "same_correct_answer, plus an issues array of short strings.\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]


def verify_payload(value: dict[str, Any]) -> tuple[bool, list[str]]:
    required = {"equivalent", "same_correct_answer", "issues"}
    if not required <= set(value) or not isinstance(value.get("issues"), list):
        return False, ["invalid_verifier_schema"]
    issues = [normalize_space(item) for item in value["issues"] if normalize_space(item)]
    passed = all(value.get(key) is True for key in ("equivalent", "same_correct_answer"))
    if issues:
        passed = False
    return passed, issues


class ApiClient:
    def __init__(self, *, config: dict[str, Any], model: str, timeout: float):
        from openai import OpenAI

        api_key = config.get("api_key") or os.environ.get(str(config.get("api_key_env") or "CHAT_API_KEY"))
        if not api_key or not config.get("base_url"):
            raise ValueError("chat API key or base_url is missing")
        self.client = OpenAI(api_key=api_key, base_url=str(config["base_url"]), timeout=timeout, max_retries=0)
        self.model = model
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def call(self, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"enable_thinking": False},
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            with self._lock:
                self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        content = response.choices[0].message.content or ""
        return _extract_object(content)


def rewrite_one(row: dict[str, Any], api: ApiClient, *, max_retries: int) -> dict[str, Any]:
    item_hash = str(row["item_hash"])
    original = normalize_space(row["question"])
    feedback: list[str] = []
    last_error_type = "unknown"
    for attempt in range(1, max_retries + 1):
        try:
            rewrite_payload = api.call(build_rewrite_messages(row, feedback), max_tokens=512)
            rewritten = normalize_rewritten_form(row, rewrite_payload.get("rewritten_question") or "")
            deterministic_reasons = item_validation(row, rewritten)
            if deterministic_reasons:
                feedback = deterministic_reasons
                last_error_type = "deterministic_validation"
                continue
            verifier = api.call(build_verify_messages(row, rewritten), max_tokens=256)
            semantic_passed, verifier_issues = verify_payload(verifier)
            if not semantic_passed:
                feedback = ["semantic_verifier_rejected"] + verifier_issues[:4]
                last_error_type = "semantic_verification"
                continue
            return {
                "rewrite_version": REWRITE_VERSION,
                "item_hash": item_hash,
                "original_question_sha256": text_hash(original),
                "rewritten_question_sha256": text_hash(rewritten),
                "rewritten_question": rewritten,
                "deterministic_validation_passed": True,
                "semantic_validation_passed": True,
                "attempts": attempt,
            }
        except Exception as exc:  # pragma: no cover - exercised against remote API
            last_error_type = type(exc).__name__
            feedback = [f"request_failed:{last_error_type}"]
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    return {
        "rewrite_version": REWRITE_VERSION,
        "item_hash": item_hash,
        "original_question_sha256": text_hash(original),
        "rewritten_question_sha256": None,
        "rewritten_question": "",
        "deterministic_validation_passed": False,
        "semantic_validation_passed": False,
        "attempts": max_retries,
        "error_type": last_error_type,
        "rejection_reasons": feedback,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency < 1 or args.max_retries < 1:
        raise ValueError("concurrency and max_retries must be positive")
    source_rows = read_jsonl(args.source)
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(source_rows):
            raise ValueError("limit must be within the source row count")
        source_rows = source_rows[: args.limit]
    if len({str(row["item_hash"]) for row in source_rows}) != len(source_rows):
        raise ValueError("source item hashes are not unique")

    incomplete = args.private_output.with_suffix(args.private_output.suffix + ".incomplete")
    if args.private_output.exists() or args.safe_output.exists():
        raise FileExistsError("refusing to overwrite a complete rewrite artifact")
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and incomplete.exists():
        completed = {str(row["item_hash"]): row for row in read_jsonl(incomplete)}
    elif incomplete.exists():
        raise FileExistsError("incomplete artifact exists; use --resume")
    source_ids = {str(row["item_hash"]) for row in source_rows}
    if not set(completed) <= source_ids:
        raise ValueError("incomplete artifact contains unknown item hashes")

    config = json.loads(args.api_config.read_text(encoding="utf-8"))
    api = ApiClient(config=config, model=args.model, timeout=args.timeout)
    pending = [row for row in source_rows if str(row["item_hash"]) not in completed]
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_rows = {
            executor.submit(rewrite_one, row, api, max_retries=args.max_retries): row for row in pending
        }
        newly_completed = 0
        for future in concurrent.futures.as_completed(future_rows):
            result = future.result()
            if result.get("semantic_validation_passed") is True:
                completed[str(result["item_hash"])] = result
                newly_completed += 1
            else:
                failures.append(result)
            if newly_completed and newly_completed % 20 == 0:
                ordered_partial = [completed[str(row["item_hash"])] for row in source_rows if str(row["item_hash"]) in completed]
                write_jsonl_private(incomplete, ordered_partial)
                print(f"validated_rewrites={len(completed)}/{len(source_rows)}", flush=True)

    ordered = [completed[str(row["item_hash"])] for row in source_rows if str(row["item_hash"]) in completed]
    write_jsonl_private(incomplete, ordered)
    if failures or len(ordered) != len(source_rows):
        failure_types = Counter(str(row.get("error_type") or "validation") for row in failures)
        print(json.dumps({"validated": len(ordered), "expected": len(source_rows), "failure_types": failure_types}, default=dict))
        return 2

    os.replace(incomplete, args.private_output)
    os.chmod(args.private_output, 0o600)
    attempts = Counter(int(row["attempts"]) for row in ordered)
    summary = {
        "schema_version": 1,
        "rewrite_version": REWRITE_VERSION,
        "private_content_included": False,
        "model": args.model,
        "concurrency": args.concurrency,
        "source_items": len(source_rows),
        "validated_rewrites": len(ordered),
        "unchanged_questions": 0,
        "semantic_validation_passed": len(ordered),
        "attempt_distribution": {str(key): value for key, value in sorted(attempts.items())},
        "source_sha256": sha256_file(args.source),
        "private_output_sha256": sha256_file(args.private_output),
        "api_usage": {
            "prompt_tokens": api.prompt_tokens,
            "completion_tokens": api.completion_tokens,
        },
    }
    write_json(args.safe_output, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
