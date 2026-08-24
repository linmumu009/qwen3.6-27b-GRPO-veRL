#!/usr/bin/env python3
"""Build the leakage-screened 80-prompt Step-120-open-source curriculum."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from llin_verl.opensource_reward import compute_score, extract_explicit_answer


DEFAULT_SEED = "step120-opensource-v1-20260824"
DEFAULT_QUOTAS = {"MATH": 48, "PHYBench": 16, "C-Eval-dev": 8, "GSM8K": 8}
CEVAL_STEM_SUBJECTS = {
    "advanced_mathematics",
    "college_chemistry",
    "college_physics",
    "college_programming",
    "computer_architecture",
    "computer_network",
    "discrete_mathematics",
    "electrical_engineer",
    "high_school_chemistry",
    "high_school_mathematics",
    "high_school_physics",
    "operating_system",
    "probability_and_statistics",
}
SYSTEM_PROMPTS = {
    "MATH": "Solve the problem carefully. Show the essential reasoning and put only the final answer inside \\boxed{...}.",
    "PHYBench": "Solve the physics problem carefully. Derive the result and put only the final expression inside \\boxed{...}.",
    "C-Eval-dev": "Choose the correct option. Briefly justify it and put the final option letter inside \\boxed{...}.",
    "GSM8K": "Solve the word problem step by step and put only the final numeric answer inside \\boxed{...}.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def canonical_prompt(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def prompt_fingerprint(value: str) -> str:
    value = canonical_prompt(value)
    value = re.sub(r"[^\w]+", "", value, flags=re.UNICODE)
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""


def _stable_key(seed: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def _source_rows(pattern: str) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for filename in sorted(glob.glob(pattern)):
        path = Path(filename)
        for index, row in enumerate(read_jsonl(path)):
            yield path, index, row


def _all_prompt_values(row: dict[str, Any]) -> Iterable[str]:
    for key in (
        "problem", "question", "content", "prompt", "input", "task_input",
        "en_informal", "zh_informal", "formal_statement_raw", "formal_statement",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            yield value


def heldout_fingerprints(raw_root: Path) -> tuple[set[str], dict[str, int]]:
    patterns = {
        "AIME24": ["AIME24/*__test.jsonl"],
        "AIME25": ["AIME25/*__test.jsonl"],
        "OlymMATH": ["OlymMATH/*__test.jsonl"],
        "AMO-Bench": ["AMO-Bench/*__test.jsonl"],
        "Omni-MATH": ["Omni-MATH/*__test.jsonl"],
        "ATLAS": ["ATLAS/*__val.jsonl", "ATLAS/*__test.jsonl"],
        "MATH-500": ["MATH-500/*__test.jsonl"],
    }
    fingerprints: set[str] = set()
    counts: dict[str, int] = {}
    for dataset, relative_patterns in patterns.items():
        dataset_values: set[str] = set()
        for relative in relative_patterns:
            for _, _, row in _source_rows(str(raw_root / relative)):
                dataset_values.update(
                    fp for fp in (prompt_fingerprint(value) for value in _all_prompt_values(row)) if fp
                )
        counts[dataset] = len(dataset_values)
        fingerprints.update(dataset_values)
    return fingerprints, counts


def _boxed_gold(solution: str) -> str:
    value, present = extract_explicit_answer(solution)
    if not present or not value:
        raise ValueError("MATH solution has no boxed final answer")
    return value


def _ceval_prompt(row: dict[str, Any]) -> str:
    question = str(row.get("question") or "").strip()
    options = [f"{letter}. {str(row.get(letter) or '').strip()}" for letter in "ABCD"]
    return question + "\n" + "\n".join(options)


def _candidate(
    dataset: str,
    task_id: str,
    prompt: str,
    answer: str,
    answer_type: str,
    source: Path,
    source_row: int,
    difficulty: str,
    subject: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "task_id": task_id,
        "prompt": prompt.strip(),
        "answer": answer.strip(),
        "answer_type": answer_type,
        "source_path": str(source),
        "source_row": source_row,
        "difficulty": difficulty,
        "subject": subject,
        "prompt_fingerprint": prompt_fingerprint(prompt),
    }


def collect_candidates(raw_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = {key: [] for key in DEFAULT_QUOTAS}
    audit: dict[str, Any] = {}

    math_path = raw_root / "MATH/default__train.jsonl"
    math_rows = read_jsonl(math_path)
    math500_exact = {
        canonical_prompt(value)
        for _, _, row in _source_rows(str(raw_root / "MATH-500/*__test.jsonl"))
        for value in _all_prompt_values(row)
    }
    first_exact = {canonical_prompt(str(row.get("problem") or "")) for row in math_rows[:7500]}
    tail_exact = {canonical_prompt(str(row.get("problem") or "")) for row in math_rows[7500:]}
    if (
        len(math_rows) != 12500
        or first_exact & math500_exact
        or len(tail_exact & math500_exact) != len(math500_exact)
    ):
        raise ValueError("MATH 12.5K no longer matches the verified 7.5K-train + 5K-test layout")
    math500_fingerprints = {
        prompt_fingerprint(value)
        for _, _, row in _source_rows(str(raw_root / "MATH-500/*__test.jsonl"))
        for value in _all_prompt_values(row)
    }
    first_fingerprints = {
        prompt_fingerprint(str(row.get("problem") or "")) for row in math_rows[:7500]
    }
    tail_fingerprints = {
        prompt_fingerprint(str(row.get("problem") or "")) for row in math_rows[7500:]
    }
    audit["math_partition"] = {
        "rows": len(math_rows),
        "train_prefix_rows": 7500,
        "heldout_tail_rows": 5000,
        "math500_exact_in_train_prefix": len(first_exact & math500_exact),
        "math500_exact_in_heldout_tail": len(tail_exact & math500_exact),
        "math500_conservative_fingerprint_in_train_prefix": len(
            first_fingerprints & math500_fingerprints
        ),
        "math500_conservative_fingerprint_in_heldout_tail": len(
            tail_fingerprints & math500_fingerprints
        ),
    }
    for index, row in enumerate(math_rows[:7500]):
        level = str(row.get("level") or "")
        if level not in {"Level 4", "Level 5"}:
            continue
        try:
            answer = _boxed_gold(str(row.get("solution") or ""))
        except ValueError:
            continue
        pools["MATH"].append(
            _candidate("MATH", f"math-train-{index}", str(row.get("problem") or ""), answer,
                       "math", math_path, index, level, str(row.get("type") or ""))
        )

    phy_path = raw_root / "PHYBench/default__train.jsonl"
    phy_rows = read_jsonl(phy_path)
    all_phy_ids = {str(row.get("id")) for row in phy_rows if row.get("id") is not None}
    seen_phy_ids: set[str] = set()
    seen_phy_prompts: set[str] = set()
    for index, row in enumerate(phy_rows):
        identity = str(row.get("id") or "")
        content = str(row.get("content") or "").strip()
        answer = str(row.get("answer") or "").strip()
        fingerprint = prompt_fingerprint(content)
        if not identity or identity in seen_phy_ids or not fingerprint or fingerprint in seen_phy_prompts or not answer:
            continue
        seen_phy_ids.add(identity)
        seen_phy_prompts.add(fingerprint)
        pools["PHYBench"].append(
            _candidate("PHYBench", f"phybench-{identity}", content, answer, "math", phy_path,
                       index, "challenge", str(row.get("tag") or ""))
        )
    audit["phybench_raw_rows"] = len(phy_rows)
    audit["phybench_unique_ids"] = len(all_phy_ids)
    audit["phybench_rows_with_nonempty_answer"] = sum(
        bool(str(row.get("answer") or "").strip()) for row in phy_rows
    )
    audit["phybench_verifiable_unique_prompts"] = len(seen_phy_prompts)

    for path, index, row in _source_rows(str(raw_root / "C-Eval/*__dev.jsonl")):
        subject = path.name.split("__", 1)[0]
        if subject not in CEVAL_STEM_SUBJECTS:
            continue
        answer = str(row.get("answer") or "").strip().upper()
        prompt = _ceval_prompt(row)
        if answer not in set("ABCD") or not prompt.strip():
            continue
        pools["C-Eval-dev"].append(
            _candidate("C-Eval-dev", f"ceval-{subject}-{index}", prompt, answer, "choice",
                       path, index, "dev", subject)
        )

    gsm_path = raw_root / "GSM8K/main__train.jsonl"
    gsm_rows = read_jsonl(gsm_path)
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, row in enumerate(gsm_rows):
        explanation = str(row.get("answer") or "")
        if "####" not in explanation:
            continue
        complexity = explanation.count("<<") * 100 + min(len(explanation), 1000)
        ranked.append((complexity, index, row))
    ranked.sort(reverse=True, key=lambda value: (value[0], -value[1]))
    hard_half = ranked[: max(1, len(ranked) // 2)]
    for complexity, index, row in hard_half:
        explanation = str(row["answer"])
        answer = explanation.rsplit("####", 1)[1].strip()
        pools["GSM8K"].append(
            _candidate("GSM8K", f"gsm8k-train-{index}", str(row.get("question") or ""),
                       answer, "numeric", gsm_path, index, f"complexity-{complexity}", "arithmetic")
        )
    audit["gsm8k_raw_rows"] = len(gsm_rows)
    audit["gsm8k_hard_proxy_pool"] = len(hard_half)
    return pools, audit


def screen_and_select(
    pools: dict[str, list[dict[str, Any]]],
    heldout: set[str],
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    seen: set[str] = set()
    for dataset in ("MATH", "PHYBench", "C-Eval-dev", "GSM8K"):
        eligible: list[dict[str, Any]] = []
        reasons: Counter[str] = Counter()
        for row in pools[dataset]:
            fingerprint = row["prompt_fingerprint"]
            if not fingerprint:
                reasons["empty_prompt"] += 1
            elif fingerprint in heldout:
                reasons["heldout_fingerprint_overlap"] += 1
            elif fingerprint in seen:
                reasons["candidate_duplicate"] += 1
            else:
                seen.add(fingerprint)
                eligible.append(row)
        quota = DEFAULT_QUOTAS[dataset]
        if dataset == "MATH":
            level5 = [row for row in eligible if row["difficulty"] == "Level 5"]
            level4 = [row for row in eligible if row["difficulty"] == "Level 4"]
            level5.sort(key=lambda row: _stable_key(seed + ":math-l5", row["task_id"]))
            level4.sort(key=lambda row: _stable_key(seed + ":math-l4", row["task_id"]))
            chosen = level5[:32] + level4[:16]
        else:
            eligible.sort(key=lambda row: _stable_key(seed + ":" + dataset, row["task_id"]))
            chosen = eligible[:quota]
        if len(chosen) != quota:
            raise ValueError(f"{dataset}: only {len(chosen)} eligible rows for quota {quota}")
        selected.extend(chosen)
        report[dataset] = {
            "raw_candidates": len(pools[dataset]),
            "eligible_unique_leakage_free": len(eligible),
            "excluded": dict(reasons),
            "selected": len(chosen),
        }

    # A deterministic interleave prevents long single-dataset runs if the data
    # loader is configured without shuffling.
    buckets = {name: [row for row in selected if row["dataset"] == name] for name in DEFAULT_QUOTAS}
    ordered: list[dict[str, Any]] = []
    pattern = ["MATH", "PHYBench", "MATH", "C-Eval-dev", "MATH", "GSM8K"]
    while any(buckets.values()):
        progressed = False
        for name in pattern:
            if buckets[name]:
                ordered.append(buckets[name].pop(0))
                progressed = True
        if not progressed:
            break
    if len(ordered) != 80 or len({row["prompt_fingerprint"] for row in ordered}) != 80:
        raise AssertionError("curriculum must contain exactly 80 unique prompts")
    return ordered, report


def training_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    ground_truth = {
        "dataset": row["dataset"],
        "task_id": row["task_id"],
        "answer_type": row["answer_type"],
        "answer": row["answer"],
    }
    record = {
        "data_source": "step120_opensource_recovery_v1",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPTS[row["dataset"]]},
            {"role": "user", "content": row["prompt"]},
        ],
        "ability": row["dataset"],
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": index,
            "split": "train",
            "dataset": row["dataset"],
            "task_id": row["task_id"],
            "difficulty": row["difficulty"],
            "subject": row["subject"],
            "source_path": row["source_path"],
            "source_row": row["source_row"],
            "prompt_fingerprint": row["prompt_fingerprint"],
            "heldout_fingerprint_overlap": False,
        },
    }
    gold_response = rf"\boxed{{{row['answer']}}}"
    result = compute_score(record["data_source"], gold_response, ground_truth, record["extra_info"])
    if result["score"] != 1.0:
        raise ValueError(f"gold answer failed self-verification for {row['task_id']}")
    return record


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> None:
    heldout, heldout_counts = heldout_fingerprints(args.raw_root)
    pools, audit = collect_candidates(args.raw_root)
    selected, selection_report = screen_and_select(pools, heldout, args.seed)
    records = [training_record(row, index) for index, row in enumerate(selected)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "opensource_step120_train.jsonl"
    manifest_path = args.output_dir / "opensource_step120_manifest.jsonl"
    write_jsonl(jsonl_path, records)
    write_jsonl(manifest_path, selected)
    report = {
        "contract": "step120-opensource-recovery-v1",
        "seed": args.seed,
        "groups_per_step": 4,
        "new_policy_steps": 20,
        "selected_total": len(records),
        "selected_by_dataset": dict(Counter(row["dataset"] for row in selected)),
        "selected_math_levels": dict(Counter(row["difficulty"] for row in selected if row["dataset"] == "MATH")),
        "heldout_policy": (
            "canonical punctuation-insensitive prompt fingerprints; "
            "evaluation-only datasets are never sampled"
        ),
        "heldout_unique_fingerprints": heldout_counts,
        "selection": selection_report,
        "source_audit": audit,
        "train_jsonl_sha256": sha256_file(jsonl_path),
        "reward": "strict binary final-answer correctness; no wrong-answer format or process reward",
    }
    (args.output_dir / "opensource_step120_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def convert(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    if len(rows) != 80:
        raise ValueError(f"expected 80 training rows, got {len(rows)}")
    from datasets import Dataset

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(args.output))
    reloaded = Dataset.from_parquet(str(args.output))
    if len(reloaded) != 80:
        raise ValueError(f"Parquet round trip produced {len(reloaded)} rows")
    print(json.dumps({"rows": len(reloaded), "output": str(args.output), "sha256": sha256_file(args.output)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--raw-root", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--seed", default=DEFAULT_SEED)
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--input", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build(args)
    else:
        convert(args)


if __name__ == "__main__":
    main()
