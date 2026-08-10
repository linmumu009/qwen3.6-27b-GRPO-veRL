#!/usr/bin/env python3
"""Compare Step 100/120/200 on aligned val20 and boss-exact outputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import fmean
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llin_verl.pi_reward import (
    dense_final_answer_correctness,
    extract_final_assistant_answer,
)
from scripts.compare_boss_exact_evaluations import compare as compare_boss_exact


VALIDATION_METRICS = (
    "dense30_score",
    "dense_final_answer_correctness",
    "base_score",
    "boss_reward",
    "boss_answer_correct",
    "boss_process_score",
    "boss_efficiency_score",
    "evidence_reward",
    "final_answer_correct",
    "sql_evidence_correct",
    "acc",
    "required_table_used",
    "has_final_answer",
    "bash_command_count",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_id(row: dict[str, Any]) -> str:
    value = str((row.get("gts") or {}).get("task_id") or "")
    if not value:
        raise ValueError("validation row missing gts.task_id")
    return value


def index_unique(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = task_id(row)
        if key in output:
            raise ValueError(f"{label}: duplicate task_id={key}")
        output[key] = row
    return output


def _eligible(row: dict[str, Any]) -> bool:
    return bool(row.get("safe") and row.get("valid_tool_protocol") and row.get("gold_sql_verified"))


def _expected_value(row: dict[str, Any]) -> Any:
    value = (row.get("gts") or {}).get("expected_value_json")
    return json.loads(value) if isinstance(value, str) else value


def normalize_validation_row(row: dict[str, Any]) -> dict[str, Any]:
    gts = row.get("gts") or {}
    dense = dense_final_answer_correctness(
        extract_final_assistant_answer(str(row.get("output") or "")),
        str(gts["answer_type"]),
        _expected_value(row),
        float(gts["abs_tol"]),
        float(gts["rel_tol"]),
    )
    base = (
        0.7 * float(row.get("boss_reward") or 0.0)
        + 0.3 * float(row.get("evidence_reward") or 0.0)
        if _eligible(row)
        else 0.0
    )
    dense30 = 0.7 * base + 0.3 * dense if _eligible(row) else 0.0
    output = dict(row)
    output.update(
        {
            "dense_final_answer_correctness": float(dense),
            "base_score": float(base),
            "dense30_score": float(dense30),
        }
    )
    return output


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(deltas: list[float], seed: int = 120, draws: int = 20_000) -> list[float]:
    rng = random.Random(seed)
    n = len(deltas)
    samples = [fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(draws)]
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def exact_sign_test_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_metric(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    pairs = [(float(left[key].get(metric) or 0.0), float(right[key].get(metric) or 0.0)) for key in sorted(left)]
    deltas = [right_value - left_value for left_value, right_value in pairs]
    wins = sum(delta > 1e-12 for delta in deltas)
    losses = sum(delta < -1e-12 for delta in deltas)
    return {
        "left_mean": fmean(left_value for left_value, _ in pairs),
        "right_mean": fmean(right_value for _, right_value in pairs),
        "mean_delta": fmean(deltas),
        "paired_bootstrap_95pct_ci": bootstrap_mean_ci(deltas),
        "wins": wins,
        "losses": losses,
        "ties": len(deltas) - wins - losses,
        "exact_sign_test_p": exact_sign_test_p(wins, losses),
    }


def validation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str((row.get("gts") or {}).get("answer_type") or "unknown")].append(row)

    def means(group: list[dict[str, Any]]) -> dict[str, float]:
        return {metric: fmean(float(row.get(metric) or 0.0) for row in group) for metric in VALIDATION_METRICS}

    return {
        "rows": len(rows),
        "answer_types": dict(Counter(str((row.get("gts") or {}).get("answer_type")) for row in rows)),
        "means": means(rows),
        "counts": {
            metric: sum(float(row.get(metric) or 0.0) > 0 for row in rows)
            for metric in (
                "boss_answer_correct",
                "final_answer_correct",
                "sql_evidence_correct",
                "acc",
                "required_table_used",
                "has_final_answer",
            )
        },
        "by_answer_type": {
            answer_type: {"rows": len(group), "means": means(group)}
            for answer_type, group in sorted(by_type.items())
        },
    }


def validation_integrity(
    paths: dict[str, Path],
    indexes: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    labels = list(indexes)
    common_ids = set(indexes[labels[0]])
    for label in labels[1:]:
        common_ids &= set(indexes[label])
    task_sets_identical = all(set(indexes[label]) == set(indexes[labels[0]]) for label in labels[1:])
    gts_identical = 0
    prompt_identical = 0
    for key in sorted(common_ids):
        gts_blobs = [json.dumps(indexes[label][key].get("gts"), sort_keys=True, ensure_ascii=False) for label in labels]
        input_blobs = [str(indexes[label][key].get("input") or "") for label in labels]
        gts_identical += len(set(gts_blobs)) == 1
        prompt_identical += len(set(input_blobs)) == 1

    formula_checks: dict[str, Any] = {}
    for label, rows in indexes.items():
        legacy_mismatch = 0
        dense30_mismatch = 0
        dense_field_mismatch = 0
        for row in rows.values():
            normalized = normalize_validation_row(row)
            if label in {"step100", "step200"} and not math.isclose(
                float(row.get("score") or 0.0), normalized["base_score"], abs_tol=1e-6
            ):
                legacy_mismatch += 1
            if label == "step120":
                if not math.isclose(float(row.get("score") or 0.0), normalized["dense30_score"], abs_tol=1e-6):
                    dense30_mismatch += 1
                if not math.isclose(
                    float(row.get("dense_final_answer_correctness") or 0.0),
                    normalized["dense_final_answer_correctness"],
                    abs_tol=1e-9,
                ):
                    dense_field_mismatch += 1
        formula_checks[label] = {
            "legacy_formula_mismatches": legacy_mismatch,
            "dense30_formula_mismatches": dense30_mismatch,
            "dense_field_mismatches": dense_field_mismatch,
        }

    return {
        "files": {label: {"sha256": sha256(path), "rows": len(indexes[label])} for label, path in paths.items()},
        "task_sets_identical": task_sets_identical,
        "common_task_count": len(common_ids),
        "gts_identical_count": gts_identical,
        "prompt_identical_count": prompt_identical,
        "verifier_error_counts": {
            label: sum(bool(row.get("verifier_error")) for row in rows.values()) for label, rows in indexes.items()
        },
        "formula_checks": formula_checks,
    }


_GLOBAL_STEP_RE = re.compile(r"training/global_step:(\d+(?:\.\d+)?).*?critic/score/mean:([-+0-9.eE]+)")
_TRAIN_STAGE_RE = re.compile(
    r"\[LLIN_TRAIN_STAGE\] step=(\d+).*?queue_wait_s=([-+0-9.eE]+).*?update_actor_s=([-+0-9.eE]+).*?step_s=([-+0-9.eE]+)"
)
_VALIDATE_TIME_RE = re.compile(r"rollouter/validate_time:([-+0-9.eE]+)")
_SAVE_START_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO .*?model/dist_ckpt will save")
_SAVE_END_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO .*?Checkpoint save completed")


def parse_driver_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    score_rows = [
        {"global_step": int(float(match.group(1))), "score_mean": float(match.group(2))}
        for match in _GLOBAL_STEP_RE.finditer(text)
    ]
    stage_rows = [
        {
            "stage_step": int(match.group(1)),
            "queue_wait_s": float(match.group(2)),
            "update_actor_s": float(match.group(3)),
            "step_s": float(match.group(4)),
        }
        for match in _TRAIN_STAGE_RE.finditer(text)
    ]
    validate_times = [float(value) for value in _VALIDATE_TIME_RE.findall(text)]
    save_starts = [datetime.strptime(value, "%Y-%m-%d %H:%M:%S") for value in _SAVE_START_RE.findall(text)]
    save_ends = [datetime.strptime(value, "%Y-%m-%d %H:%M:%S") for value in _SAVE_END_RE.findall(text)]
    save_time_s = (max(save_ends) - min(save_starts)).total_seconds() if save_starts and save_ends else None
    return {
        "source_sha256": sha256(path),
        "score_metric_rows": score_rows,
        "score_metric_coverage": len(score_rows),
        "stage_rows": stage_rows,
        "stage_count": len(stage_rows),
        "score_mean": fmean(row["score_mean"] for row in score_rows),
        "score_first5_mean": fmean(row["score_mean"] for row in score_rows[:5]),
        "score_last5_mean": fmean(row["score_mean"] for row in score_rows[-5:]),
        "mean_queue_wait_s": fmean(row["queue_wait_s"] for row in stage_rows),
        "mean_update_actor_s": fmean(row["update_actor_s"] for row in stage_rows),
        "mean_step_s": fmean(row["step_s"] for row in stage_rows),
        "validation_time_s": validate_times[-1] if validate_times else None,
        "checkpoint_save_time_s": save_time_s,
    }


def boss_exact_with_uncertainty(
    left_path: Path,
    right_path: Path,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    result = compare_boss_exact(left_path, right_path, left_label, right_label)
    deltas = [float(row["delta"]) for row in result["paired_results"]]
    result["paired_reward"]["mean_delta_bootstrap_95pct_ci"] = bootstrap_mean_ci(deltas)
    result["paired_reward"]["exact_sign_test_p"] = exact_sign_test_p(
        int(result["paired_reward"]["wins"]), int(result["paired_reward"]["losses"])
    )
    return result


def portable_adapter_summary(path: Path) -> dict[str, Any]:
    """Keep the adapter audit while removing machine-specific absolute paths."""
    result = json.loads(path.read_text(encoding="utf-8"))
    for key in ("trajectory_output", "manifest_output"):
        value = result.get(key)
        if value:
            result[key] = f"evaluations/boss_exact_step120_20260810/{Path(str(value)).name}"
    return result


def analyze(
    validation_paths: dict[str, Path],
    boss_exact_paths: dict[str, Path],
    driver_log: Path,
    adapter_summary: Path,
) -> dict[str, Any]:
    raw = {label: read_jsonl(path) for label, path in validation_paths.items()}
    raw_indexes = {label: index_unique(rows, label) for label, rows in raw.items()}
    normalized = {label: [normalize_validation_row(row) for row in rows] for label, rows in raw.items()}
    normalized_indexes = {label: index_unique(rows, label) for label, rows in normalized.items()}

    comparisons = {}
    for right_label in ("step120", "step200"):
        left_label = "step100"
        comparisons[f"{left_label}_vs_{right_label}"] = {
            metric: paired_metric(normalized_indexes[left_label], normalized_indexes[right_label], metric)
            for metric in VALIDATION_METRICS
        }

    boss_exact = {
        "step100_vs_step120": boss_exact_with_uncertainty(
            boss_exact_paths["step100"], boss_exact_paths["step120"], "step100", "step120"
        ),
        "step120_vs_step200": boss_exact_with_uncertainty(
            boss_exact_paths["step120"], boss_exact_paths["step200"], "step120", "step200"
        ),
    }
    return {
        "analysis": "step120_dense_correctness_trial",
        "generated_at": "2026-08-10T16:30:00+08:00",
        "validation_integrity": validation_integrity(validation_paths, raw_indexes),
        "validation": {label: validation_summary(rows) for label, rows in normalized.items()},
        "paired_validation": comparisons,
        "boss_exact": boss_exact,
        "boss_exact_adapter": portable_adapter_summary(adapter_summary),
        "training_run": parse_driver_log(driver_log),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for step in (100, 120, 200):
        parser.add_argument(f"--step{step}-validation", type=Path, required=True)
        parser.add_argument(f"--step{step}-boss-exact", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--adapter-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        {f"step{step}": getattr(args, f"step{step}_validation") for step in (100, 120, 200)},
        {f"step{step}": getattr(args, f"step{step}_boss_exact") for step in (100, 120, 200)},
        args.driver_log,
        args.adapter_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
