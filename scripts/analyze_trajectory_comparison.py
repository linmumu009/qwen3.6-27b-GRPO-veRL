#!/usr/bin/env python3
"""Compare historical PI trajectories with veRL rollout dumps.

The script is intentionally read-only. It can run on the server that owns the
original tar archive and prints JSON so the analysis can be captured without
copying trajectory text or database contents.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
import tarfile
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def iter_jsonl_bytes(lines: Iterable[bytes]) -> Iterable[dict[str, Any]]:
    for raw in lines:
        if raw.strip():
            yield json.loads(raw)


def first_original_record(
    archive: Path,
) -> tuple[str, int, dict[str, Any], dict[str, int], dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as handle:
        members = [member for member in handle.getmembers() if member.isfile()]
        if not members:
            raise ValueError(f"no files in archive: {archive}")
        member = members[0]
        stream = handle.extractfile(member)
        if stream is None:
            raise ValueError(f"cannot extract: {member.name}")
        rows = list(iter_jsonl_bytes(stream))
        if not rows:
            raise ValueError(f"no JSON rows in: {member.name}")
        type_counts = Counter(str(row.get("type", "<missing>")) for row in rows)
        first_by_type: dict[str, Any] = {}
        for row in rows:
            first_by_type.setdefault(str(row.get("type", "<missing>")), value_shape(row))
        return member.name, len(rows), rows[0], dict(type_counts), first_by_type


def first_rollout_record(directory: Path) -> tuple[str, int, dict[str, Any]]:
    paths = sorted(directory.glob("*.jsonl"), key=lambda path: int(path.stem))
    if not paths:
        raise ValueError(f"no rollout JSONL files in: {directory}")
    rows = [
        json.loads(line)
        for line in paths[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no JSON rows in: {paths[0]}")
    return paths[0].name, len(rows), rows[0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        return list(iter_jsonl_bytes(handle))


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def content_text(value: Any) -> str:
    """Extract human-readable text without duplicating nested message fields."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (content_text(item) for item in value)))
    if not isinstance(value, dict):
        return ""
    for key in ("text", "content", "value"):
        if key in value:
            return content_text(value[key])
    return ""


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "n": len(clean),
        "min": min(clean) if clean else None,
        "mean": mean(clean) if clean else None,
        "median": median(clean) if clean else None,
        "p90": quantile(clean, 0.90),
        "p95": quantile(clean, 0.95),
        "max": max(clean) if clean else None,
    }


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_scale * y_scale) if x_scale and y_scale else None


def count_tool_calls(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        for item in message.get("content") or []:
            if isinstance(item, dict) and str(item.get("type", "")).lower() in {
                "toolcall",
                "tool_call",
                "tool-use",
                "tool_use",
            }:
                total += 1
    return total


def profile_original_events(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agent_end = next((row for row in reversed(rows) if row.get("type") == "agent_end"), None)
    if not agent_end:
        return {"valid": False, "reason": "missing_agent_end"}
    messages = agent_end.get("messages") or []
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    user_messages = [message for message in messages if message.get("role") == "user"]
    timestamps = [
        int(message["timestamp"])
        for message in messages
        if isinstance(message.get("timestamp"), (int, float))
    ]
    assistant_usage_tokens = sum(
        int((message.get("usage") or {}).get("output") or 0) for message in assistant_messages
    )
    assistant_text = "\n".join(content_text(message.get("content") or []) for message in assistant_messages)
    models = sorted({str(message.get("model")) for message in assistant_messages if message.get("model")})
    return {
        "valid": True,
        "user_texts": [content_text(message.get("content") or []) for message in user_messages],
        "assistant_chars": len(assistant_text),
        "assistant_usage_tokens": assistant_usage_tokens,
        "assistant_turns": len(assistant_messages),
        "tool_calls": count_tool_calls(assistant_messages),
        "tool_results": sum(message.get("role") in {"tool", "toolResult"} for message in messages),
        "elapsed_s": (max(timestamps) - min(timestamps)) / 1000 if len(timestamps) >= 2 else None,
        "models": models,
    }


def load_original_profiles(archive: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if not member.isfile() or not member.name.endswith(".jsonl"):
                continue
            stream = handle.extractfile(member)
            if stream is None:
                continue
            profiles[Path(member.name).name] = profile_original_events(list(iter_jsonl_bytes(stream)))

    valid = [profile for profile in profiles.values() if profile.get("valid")]
    model_counts = Counter(model for profile in valid for model in profile.get("models", []))
    summary = {
        "files": len(profiles),
        "valid_trajectories": len(valid),
        "missing_agent_end": len(profiles) - len(valid),
        "assistant_usage_tokens": describe(profile["assistant_usage_tokens"] for profile in valid),
        "assistant_chars": describe(profile["assistant_chars"] for profile in valid),
        "assistant_turns": describe(profile["assistant_turns"] for profile in valid),
        "tool_calls": describe(profile["tool_calls"] for profile in valid),
        "elapsed_s": describe(profile["elapsed_s"] for profile in valid if profile["elapsed_s"] is not None),
        "models": dict(model_counts),
    }
    return profiles, summary


def profile_converted_record(record: dict[str, Any], tokenizer) -> dict[str, Any]:
    messages = record.get("messages") or []
    tools = record.get("tools") or []
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    user_messages = [message for message in messages if message.get("role") == "user"]
    assistant_texts = [content_text(message.get("content") or "") for message in assistant_messages]
    tool_calls = 0
    for message, text in zip(assistant_messages, assistant_texts, strict=True):
        tool_calls += len(message.get("tool_calls") or [])
        if not message.get("tool_calls"):
            tool_calls += len(re.findall(r"<tool_call(?:>|\s)", text))
    first_assistant = next(
        (index for index, message in enumerate(messages) if message.get("role") == "assistant"),
        len(messages),
    )
    prompt_messages = messages[:first_assistant]
    payload_parts = []
    for message in messages[first_assistant:]:
        role = str(message.get("role") or "")
        payload = content_text(message.get("content") or "")
        if message.get("tool_calls"):
            payload = "\n".join(
                filter(
                    None,
                    (
                        payload,
                        json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True),
                    ),
                )
            )
        payload_parts.append(f"{role}\n{payload}")
    trajectory_payload_tokens = token_count(tokenizer, "\n".join(payload_parts))
    template_error = None
    try:
        full_ids = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        trajectory_response_tokens = max(0, len(full_ids) - len(prompt_ids))
    except Exception as exc:
        template_error = f"{type(exc).__name__}: {exc}"
        trajectory_response_tokens = token_count(
            tokenizer,
            json.dumps(messages[first_assistant:], ensure_ascii=False, sort_keys=True),
        )
    return {
        "valid": True,
        "user_texts": [content_text(message.get("content") or "") for message in user_messages],
        "assistant_chars": sum(len(text) for text in assistant_texts),
        "assistant_tokens_reencoded": sum(token_count(tokenizer, text) for text in assistant_texts),
        "trajectory_response_tokens_reencoded": trajectory_response_tokens,
        "trajectory_payload_tokens_reencoded": trajectory_payload_tokens,
        "assistant_turns": len(assistant_messages),
        "tool_calls": tool_calls,
        "tool_results": sum(message.get("role") in {"tool", "toolResult"} for message in messages),
        "messages": len(messages),
        "template_error": template_error,
    }


def load_converted_profiles(
    paths: list[Path],
    tokenizer,
    wanted_keys: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    profiles: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        with path.open("rb") as handle:
            for record in iter_jsonl_bytes(handle):
                task_id = str(record.get("task_id") or "")
                key = (path.name, task_id)
                if task_id and key in wanted_keys:
                    profiles[key] = profile_converted_record(record, tokenizer)
    valid = list(profiles.values())
    return profiles, {
        "records": len(valid),
        "assistant_tokens_reencoded": describe(profile["assistant_tokens_reencoded"] for profile in valid),
        "trajectory_response_tokens_reencoded": describe(
            profile["trajectory_response_tokens_reencoded"] for profile in valid
        ),
        "trajectory_payload_tokens_reencoded": describe(
            profile["trajectory_payload_tokens_reencoded"] for profile in valid
        ),
        "assistant_chars": describe(profile["assistant_chars"] for profile in valid),
        "assistant_turns": describe(profile["assistant_turns"] for profile in valid),
        "tool_calls": describe(profile["tool_calls"] for profile in valid),
        "tool_results": describe(profile["tool_results"] for profile in valid),
        "template_errors": sum(profile["template_error"] is not None for profile in valid),
        "template_error_examples": sorted(
            {profile["template_error"] for profile in valid if profile["template_error"]}
        )[:3],
    }


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_rollouts(directory: Path, tokenizer) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl"), key=lambda item: int(item.stem)):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            output = str(row.get("output") or "")
            records.append(
                {
                    "step": int(row.get("step") or int(path.stem)),
                    "line": line_number,
                    "verifier_id": str((row.get("gts") or {}).get("verifier_id") or ""),
                    "output_chars": len(output),
                    "output_tokens_reencoded": token_count(tokenizer, output),
                    "tool_call_tags": len(re.findall(r"<tool_call(?:>|\s)", output)),
                    "score": float(row.get("score") or 0),
                }
            )
    return records


_STEP_METRIC_PATTERNS = {
    "response_mean": re.compile(r"response_length/mean:(?:np\.float64\()?([-+0-9.eE]+)"),
    "response_max": re.compile(r"response_length/max:(?:np\.float64\()?([-+0-9.eE]+)"),
    "response_min": re.compile(r"response_length/min:(?:np\.float64\()?([-+0-9.eE]+)"),
    "response_clip_ratio": re.compile(r"response_length/clip_ratio:(?:np\.float64\()?([-+0-9.eE]+)"),
    "aborted_ratio": re.compile(r"response/aborted_ratio:(?:np\.float64\()?([-+0-9.eE]+)"),
    "generate_min_s": re.compile(
        r"timing_s/agent_loop/generate_sequences/min:(?:np\.float64\()?([-+0-9.eE]+)"
    ),
    "generate_mean_s": re.compile(
        r"timing_s/agent_loop/generate_sequences/mean:(?:np\.float64\()?([-+0-9.eE]+)"
    ),
    "generate_max_s": re.compile(
        r"timing_s/agent_loop/generate_sequences/max:(?:np\.float64\()?([-+0-9.eE]+)"
    ),
    "slowest_response_tokens": re.compile(
        r"timing_s/agent_loop/slowest/response_length:(?:np\.float64\()?([-+0-9.eE]+)"
    ),
}


def parse_driver_steps(path: Path) -> list[dict[str, float]]:
    steps: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        step_match = re.search(r"step:(\d+)\s+-\s+training/global_step:", line)
        if not step_match:
            continue
        item: dict[str, float] = {"step": float(step_match.group(1))}
        for name, pattern in _STEP_METRIC_PATTERNS.items():
            match = pattern.search(line)
            if match:
                item[name] = float(match.group(1))
        if len(item) > 1:
            steps.append(item)
    return steps


def summarize_timeout_bounds(steps: list[dict[str, float]], thresholds: list[float]) -> list[dict[str, Any]]:
    """Report batch-level evidence; exact per-sample discard counts are not logged."""
    results = []
    for threshold in thresholds:
        affected = [step for step in steps if step.get("generate_max_s", 0) > threshold]
        results.append(
            {
                "timeout_s": threshold,
                "batches_with_at_least_one_timeout": len(affected),
                "batch_share": len(affected) / len(steps) if steps else None,
                "minimum_trajectories_affected": len(affected),
                "exact_trajectories_affected": None,
                "max_observed_s": max(
                    (step.get("generate_max_s", 0) for step in affected),
                    default=None,
                ),
            }
        )
    return results


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    if not args.prompt_jsonl or not args.verifier_manifest or not args.tokenizer_path:
        raise ValueError(
            "--prompt-jsonl, --verifier-manifest and --tokenizer-path are required outside --schema-only"
        )
    tokenizer = load_tokenizer(args.tokenizer_path)
    rollout_records = load_rollouts(args.rollout_dir, tokenizer)
    rollout_ids = {record["verifier_id"] for record in rollout_records}
    verifier_manifest_records = read_jsonl(args.verifier_manifest)
    verifier_manifest_ids = {
        str(record.get("verifier_id") or "")
        for record in verifier_manifest_records
        if record.get("verifier_id")
    }
    prompt_records = {
        str(record.get("verifier_id") or (record.get("metadata") or {}).get("verifier_id")): record
        for record in read_jsonl(args.prompt_jsonl)
    }
    original_profiles, original_summary = load_original_profiles(args.original_archive)
    wanted_converted_keys = set()
    for verifier_id in rollout_ids:
        record = prompt_records.get(verifier_id) or {}
        environment = ((record.get("metadata") or {}).get("environment") or {})
        wanted_converted_keys.add(
            (
                Path(str(environment.get("source_file") or "")).name,
                str(environment.get("task_id") or ""),
            )
        )
    converted_profiles, converted_summary = load_converted_profiles(
        args.converted_jsonl,
        tokenizer,
        wanted_converted_keys,
    )

    per_prompt = []
    for verifier_id in sorted(rollout_ids):
        prompt_record = prompt_records.get(verifier_id)
        if not prompt_record:
            per_prompt.append({"verifier_id": verifier_id, "matched": False, "reason": "missing_prompt_record"})
            continue
        metadata = prompt_record.get("metadata") or {}
        environment = metadata.get("environment") or {}
        source_file = Path(str(environment.get("source_file") or "")).name
        task_id = str(environment.get("task_id") or "")
        original = converted_profiles.get((source_file, task_id))
        if not original or not original.get("valid"):
            per_prompt.append(
                {
                    "verifier_id": verifier_id,
                    "source_file": source_file,
                    "matched": False,
                    "reason": "missing_valid_original_trajectory",
                }
            )
            continue
        expected_prompt = "\n".join(
            str(message.get("content") or "")
            for message in prompt_record.get("messages") or []
            if message.get("role") == "user"
        )
        original_prompts = [normalize_text(text) for text in original["user_texts"]]
        prompt_exact = normalize_text(expected_prompt) in original_prompts
        ours = [record for record in rollout_records if record["verifier_id"] == verifier_id]
        ours_tokens = [record["output_tokens_reencoded"] for record in ours]
        ours_chars = [record["output_chars"] for record in ours]
        original_tokens = int(original["trajectory_payload_tokens_reencoded"])
        per_prompt.append(
            {
                "verifier_id": verifier_id,
                "task_id": task_id,
                "source_file": source_file,
                "prompt_sha256": metadata.get("prompt_sha256"),
                "matched": prompt_exact,
                "original": {
                    key: original[key]
                    for key in (
                        "assistant_tokens_reencoded",
                        "trajectory_response_tokens_reencoded",
                        "trajectory_payload_tokens_reencoded",
                        "assistant_chars",
                        "assistant_turns",
                        "tool_calls",
                        "tool_results",
                        "messages",
                        "template_error",
                    )
                },
                "ours": {
                    "trajectories": len(ours),
                    "output_tokens_reencoded": describe(ours_tokens),
                    "output_chars": describe(ours_chars),
                    "tool_call_tags": describe(record["tool_call_tags"] for record in ours),
                    "score": describe(record["score"] for record in ours),
                    "token_ratio_to_original": describe(
                        token / original_tokens for token in ours_tokens if original_tokens > 0
                    ),
                },
            }
        )

    driver_steps = parse_driver_steps(args.driver_log) if args.driver_log else []
    matched = [item for item in per_prompt if item.get("matched")]
    all_reencoded = [record["output_tokens_reencoded"] for record in rollout_records]
    exact_step_tokens = []
    for step in driver_steps:
        if "response_mean" in step:
            exact_step_tokens.extend([step["response_mean"]] * 16)
    return {
        "comparison_scope": {
            "original_archive_files": original_summary["files"],
            "our_rollout_records": len(rollout_records),
            "our_rollout_steps": len({record["step"] for record in rollout_records}),
            "unique_rollout_prompts": len(rollout_ids),
            "exact_prompt_matches": len(matched),
            "rollout_prompts_in_prompt_file": len(rollout_ids & set(prompt_records)),
            "verifier_manifest_records": len(verifier_manifest_records),
            "rollout_verifiers_in_manifest": len(rollout_ids & verifier_manifest_ids),
            "comparison_note": (
                "Same-prompt results compare one historical reference trajectory with repeated "
                "new rollout samples for each matched prompt."
            ),
        },
        "original_population": original_summary,
        "converted_population_for_exact_sources": converted_summary,
        "our_population": {
            "output_tokens_reencoded": describe(all_reencoded),
            "output_chars": describe(record["output_chars"] for record in rollout_records),
            "tool_call_tags": describe(record["tool_call_tags"] for record in rollout_records),
            "score": describe(record["score"] for record in rollout_records),
            "logged_step_response_mean": describe(
                step["response_mean"] for step in driver_steps if "response_mean" in step
            ),
            "logged_step_response_min": describe(
                step["response_min"] for step in driver_steps if "response_min" in step
            ),
            "logged_step_response_max": describe(
                step["response_max"] for step in driver_steps if "response_max" in step
            ),
            "reencode_to_logged_mean_ratio": (
                mean(all_reencoded) / mean(exact_step_tokens)
                if all_reencoded and exact_step_tokens
                else None
            ),
            "logged_response_clip_ratio": describe(
                step["response_clip_ratio"] for step in driver_steps if "response_clip_ratio" in step
            ),
            "logged_aborted_ratio": describe(
                step["aborted_ratio"] for step in driver_steps if "aborted_ratio" in step
            ),
        },
        "same_prompt": per_prompt,
        "generation_time": {
            "steps": len(driver_steps),
            "per_batch_min_s": describe(
                step["generate_min_s"] for step in driver_steps if "generate_min_s" in step
            ),
            "per_batch_mean_s": describe(
                step["generate_mean_s"] for step in driver_steps if "generate_mean_s" in step
            ),
            "per_batch_max_s": describe(
                step["generate_max_s"] for step in driver_steps if "generate_max_s" in step
            ),
            "slowest_response_tokens": describe(
                step["slowest_response_tokens"]
                for step in driver_steps
                if "slowest_response_tokens" in step
            ),
            "slowest_time_vs_response_tokens_pearson": pearson_correlation(
                [
                    step["generate_max_s"]
                    for step in driver_steps
                    if "generate_max_s" in step and "slowest_response_tokens" in step
                ],
                [
                    step["slowest_response_tokens"]
                    for step in driver_steps
                    if "generate_max_s" in step and "slowest_response_tokens" in step
                ],
            ),
            "timeout_batch_bounds": summarize_timeout_bounds(driver_steps, args.timeout_s),
            "limitation": (
                "The driver retains min/mean/max per 16-trajectory batch, not all individual "
                "durations. Timeout counts are therefore lower bounds at batch level."
            ),
        },
    }


def value_shape(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        if isinstance(value, (list, dict, str)):
            return {"type": type(value).__name__, "len": len(value)}
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "len": len(value),
            "keys": {key: value_shape(item, depth + 1) for key, item in value.items()},
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "len": len(value),
            "first": value_shape(value[0], depth + 1) if value else None,
        }
    if isinstance(value, str):
        return {"type": "str", "len": len(value)}
    return {"type": type(value).__name__, "value": value}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-archive", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--prompt-jsonl", type=Path)
    parser.add_argument("--verifier-manifest", type=Path)
    parser.add_argument("--converted-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--driver-log", type=Path)
    parser.add_argument("--timeout-s", type=float, action="append", default=[180, 210, 240, 270, 300])
    parser.add_argument("--schema-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.schema_only:
        print(json.dumps(run_analysis(args), ensure_ascii=False, indent=2))
        return
    (
        original_name,
        original_rows,
        original,
        original_type_counts,
        original_shape_by_type,
    ) = first_original_record(args.original_archive)
    rollout_name, rollout_rows, rollout = first_rollout_record(args.rollout_dir)
    result = {
        "original": {
            "file": original_name,
            "rows_in_first_file": original_rows,
            "shape": value_shape(original),
            "event_type_counts": original_type_counts,
            "shape_by_event_type": original_shape_by_type,
        },
        "rollout": {
            "file": rollout_name,
            "rows_in_first_file": rollout_rows,
            "shape": value_shape(rollout),
        },
    }
    for label, path in (
        ("prompt_jsonl", args.prompt_jsonl),
        ("verifier_manifest", args.verifier_manifest),
    ):
        if path:
            with path.open("rb") as handle:
                first = next(iter_jsonl_bytes(handle))
            result[label] = {"shape": value_shape(first)}
    if args.converted_jsonl:
        result["converted_jsonl"] = []
        for path in args.converted_jsonl:
            with path.open("rb") as handle:
                first = next(iter_jsonl_bytes(handle))
            result["converted_jsonl"].append({"file": path.name, "shape": value_shape(first)})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
