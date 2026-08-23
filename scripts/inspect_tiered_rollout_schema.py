#!/usr/bin/env python3
"""Emit only schema and observability counts for private tiered rollout dumps."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from llin_verl.outcome_gated_contract import evidence_binding_hash
from llin_verl.grounded_trajectory_reward import _final_decision
from llin_verl.pi_reward import extract_answer_numbers, extract_final_assistant_answer


def _rows(directory: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            values.extend(json.loads(line) for line in handle if line.strip())
    return values


def _ground_truth(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.casefold())


def _answer_structure(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    pipe_widths = [len([cell for cell in line.split("|") if cell.strip()]) for line in lines if "|" in line]
    signature = {
        "lines": min(len(lines), 99),
        "pipe_lines": len(pipe_widths),
        "pipe_widths": pipe_widths[:16],
        "markdown_separators": sum(
            bool(line) and set(line.replace("|", "").replace(":", "").replace(" ", "")) <= {"-"}
            for line in lines
        ),
        "tab_lines": sum("\t" in line for line in lines),
        "comma_lines": sum("," in line or "，" in line for line in lines),
        "colon_lines": sum(":" in line or "：" in line for line in lines),
        "equals_lines": sum("=" in line for line in lines),
        "bullet_lines": sum(
            line.startswith(("- ", "* ", "+ "))
            or bool(re.match(r"^\s*\d+[.)、]\s*", line))
            for line in lines
        ),
        "code_fences": sum(line.startswith("```") for line in lines),
        "numeric_candidates": len(extract_answer_numbers(value)),
        "json_like": value.lstrip().startswith(("[", "{")),
    }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _redacted_structure(value: str) -> str:
    """Keep only presentation punctuation; replace all words/numbers."""

    lines: list[str] = []
    for raw in [line.strip() for line in value.splitlines() if line.strip()][-24:]:
        line = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", "◊", raw)
        line = re.sub(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?", "¤", line)
        line = re.sub(r"[A-Za-z_\u3400-\u9fff]+", "T", line)
        line = line.replace("¤", "N").replace("◊", "D")
        line = re.sub(r"T+", "T", line)
        lines.append(line[:240])
    return "\\n".join(lines)[:2400]


def _inspect_dataset(
    path: Path,
    roots: list[Path],
    rollout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    ground_truth_keys: Counter[str] = Counter()
    extra_keys: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    root_coverage: dict[str, Counter[str]] = {
        root.name or str(index): Counter() for index, root in enumerate(roots)
    }
    by_identity: dict[str, dict[str, Any]] = {}
    for row in rows:
        truth = ((row.get("reward_model") or {}).get("ground_truth") or {})
        extra = row.get("extra_info") or {}
        ground_truth_keys.update(truth.keys())
        extra_keys.update(extra.keys())
        answer_types[str(truth.get("answer_type") or "missing")] += 1
        identity = str(extra.get("instruction_sha256") or "")
        if identity:
            by_identity[identity] = truth
        environment = str(truth.get("environment_id") or "")
        for index, root in enumerate(roots):
            coverage = root_coverage[root.name or str(index)]
            if not environment:
                coverage["environment_missing"] += 1
                continue
            try:
                candidate = (root / environment / "logistics.sqlite").resolve(strict=True)
                candidate.relative_to(root.resolve(strict=True))
                coverage["resolved_readable"] += bool(candidate.is_file() and candidate.stat().st_size > 0)
            except (OSError, ValueError):
                coverage["resolve_failed"] += 1
    final_states: Counter[str] = Counter()
    unsupported_shapes: Counter[str] = Counter()
    unsupported_redacted: Counter[str] = Counter()
    missing_join = 0
    for row in rollout_rows:
        truth = by_identity.get(str(row.get("task_identity_sha256") or ""))
        if truth is None:
            missing_join += 1
            continue
        decision, details = _final_decision(str(row.get("output") or ""), truth)
        final_states[f"{decision.state.value}:{decision.reason}"] += 1
        if str(row.get("judge_reason") or "") == "unsupported_table_presentation":
            answer = extract_final_assistant_answer(str(row.get("output") or ""))
            unsupported_shapes[_answer_structure(answer)] += 1
            unsupported_redacted[_redacted_structure(answer)] += 1
    return {
        "rows": len(rows),
        "ground_truth_keys": dict(sorted(ground_truth_keys.items())),
        "extra_info_keys": dict(sorted(extra_keys.items())),
        "answer_type_counts": dict(sorted(answer_types.items())),
        "root_coverage": {
            key: dict(sorted(value.items())) for key, value in sorted(root_coverage.items())
        },
        "rollout_join_missing": missing_join,
        "current_final_decision_counts": dict(sorted(final_states.items())),
        "unsupported_table_structure_signatures": dict(sorted(unsupported_shapes.items())),
        "unsupported_table_redacted_presentations": dict(sorted(unsupported_redacted.items())),
    }


def inspect(
    directory: Path,
    *,
    database_root: Path | None = None,
    dataset: Path | None = None,
    candidate_roots: list[Path] | None = None,
) -> dict[str, Any]:
    rows = _rows(directory)
    keys = Counter(key for row in rows for key in row)
    event_keys: Counter[str] = Counter()
    token_errors: Counter[str] = Counter()
    identity = Counter()
    ground_truth_keys: Counter[str] = Counter()
    database = Counter()
    infrastructure_errors: Counter[str] = Counter()
    output_shapes: Counter[str] = Counter()
    ground_truth_types: Counter[str] = Counter()
    event_count = 0
    for row in rows:
        events = row.get("pi_tool_events")
        events = events if isinstance(events, list) else []
        event_count += len(events)
        request_id = str(row.get("pi_workspace_request_id") or "")
        environment_id = str(row.get("pi_environment_id") or "")
        identity["row_request_present"] += bool(request_id)
        identity["row_environment_present"] += bool(environment_id)
        identity["all_event_requests_match"] += bool(events) and all(
            str(event.get("workspace_request_id") or "") == request_id for event in events
        )
        identity["all_event_environments_match"] += bool(events) and all(
            str(event.get("environment_id") or "") == environment_id for event in events
        )
        identity["task_sha256_valid"] += _sha256(str(row.get("task_identity_sha256") or ""))
        identity["trajectory_sha256_valid"] += _sha256(
            str(row.get("trajectory_identity_sha256") or "")
        )
        ground_truth_types[type(row.get("gts")).__name__] += 1
        truth = _ground_truth(row.get("gts"))
        ground_truth_keys.update(truth.keys())
        database["ground_truth_parseable"] += bool(truth)
        environment = str(truth.get("environment_id") or "")
        database["environment_present"] += bool(environment)
        database["binding_valid"] += bool(
            truth.get("process_evidence_binding_sha256")
            and truth.get("process_evidence_binding_sha256") == evidence_binding_hash(truth)
        )
        if database_root is not None and environment:
            try:
                candidate = (database_root / environment / "logistics.sqlite").resolve(strict=True)
                candidate.relative_to(database_root.resolve(strict=True))
                database["resolved_readable"] += candidate.is_file() and candidate.stat().st_size > 0
            except (OSError, ValueError):
                database["resolve_failed"] += 1
        infrastructure_errors[str(row.get("infrastructure_error_type") or "none")] += 1
        output = str(row.get("output") or "")
        nonempty = [line.strip() for line in output.splitlines() if line.strip()]
        if not nonempty:
            output_shapes["empty"] += 1
        elif any("|" in line for line in nonempty):
            output_shapes["pipe_table_like"] += 1
        elif output.lstrip().startswith(("[", "{")):
            output_shapes["json_like"] += 1
        elif any("\t" in line for line in nonempty):
            output_shapes["tabular_delimited_like"] += 1
        elif len(nonempty) >= 2:
            output_shapes["multiline_other"] += 1
        else:
            output_shapes["singleline_other"] += 1
        for event in events:
            if not isinstance(event, dict):
                event_keys["<non-object>"] += 1
                continue
            event_keys.update(event)
            token = event.get("response_token_count")
            if isinstance(token, int) and not isinstance(token, bool) and token >= 0:
                token_errors["observable"] += 1
            else:
                token_errors[str(event.get("response_token_count_error") or "missing")] += 1
    report = {
        "rows": len(rows),
        "files": len(list(directory.glob("*.jsonl"))),
        "row_keys": dict(sorted(keys.items())),
        "event_count": event_count,
        "event_keys": dict(sorted(event_keys.items())),
        "token_observability": dict(sorted(token_errors.items())),
        "identity": dict(sorted(identity.items())),
        "ground_truth_keys": dict(sorted(ground_truth_keys.items())),
        "ground_truth_types": dict(sorted(ground_truth_types.items())),
        "database": dict(sorted(database.items())),
        "database_available_stored": dict(
            sorted(Counter(str(row.get("database_available")) for row in rows).items())
        ),
        "infrastructure_error_types": dict(sorted(infrastructure_errors.items())),
        "output_shapes": dict(sorted(output_shapes.items())),
        "sensitive_values_emitted": False,
    }
    if dataset is not None:
        report["dataset"] = _inspect_dataset(dataset, candidate_roots or [], rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--candidate-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            inspect(
                args.rollout_dir,
                database_root=args.database_root,
                dataset=args.dataset,
                candidate_roots=args.candidate_root,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
