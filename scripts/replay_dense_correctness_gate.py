#!/usr/bin/env python3
"""Replay continuous final-answer correctness over saved GRPO trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq

from llin_verl.pi_reward import dense_final_answer_correctness, extract_final_assistant_answer


_USER_TURN_RE = re.compile(r"(?:^|\n)user\n(.*?)(?=\nassistant\n)", re.DOTALL)


def prompt_text_from_input(text: str) -> str:
    turns = _USER_TURN_RE.findall(text or "")
    if not turns:
        raise ValueError("rollout input has no user turn")
    return turns[-1].strip()


def load_truth(dataset: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in pq.read_table(dataset).to_pylist():
        prompt = row["prompt"][-1]["content"].strip()
        if prompt in mapping:
            raise ValueError(f"duplicate dataset prompt: {prompt[:80]}")
        mapping[prompt] = row["reward_model"]["ground_truth"]
    return mapping


def read_rows(directory: Path, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl"), key=lambda p: int(p.stem)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_phase"] = phase
                row["_file_step"] = int(path.stem)
                rows.append(row)
    return rows


def replay(rows: list[dict[str, Any]], truth_by_prompt: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    unmapped: list[str] = []
    for row in rows:
        prompt = prompt_text_from_input(str(row.get("input") or ""))
        truth = truth_by_prompt.get(prompt)
        if truth is None:
            unmapped.append(prompt)
            continue
        expected = json.loads(truth["expected_value_json"])
        answer = extract_final_assistant_answer(str(row.get("output") or ""))
        row["_dense"] = dense_final_answer_correctness(
            answer,
            str(truth["answer_type"]),
            expected,
            float(truth["abs_tol"]),
            float(truth["rel_tol"]),
        )
        digest = hashlib.sha256(str(row["input"]).encode("utf-8")).hexdigest()
        groups[(row["_phase"], row["_file_step"], digest)].append(row)

    valid_groups = [group for group in groups.values() if len(group) == 4]
    boss_mixed = [g for g in valid_groups if len({float(r["boss_answer_correct"]) for r in g}) > 1]
    strict_mixed = [g for g in valid_groups if len({float(r["final_answer_correct"]) for r in g}) > 1]
    all_binary_wrong = [g for g in valid_groups if not any(float(r["boss_answer_correct"]) for r in g)]

    def spread(group: list[dict[str, Any]]) -> float:
        values = [float(row["_dense"]) for row in group]
        return max(values) - min(values)

    def ranks_correctly(group: list[dict[str, Any]], field: str) -> bool:
        right = [float(r["_dense"]) for r in group if float(r[field]) > 0]
        wrong = [float(r["_dense"]) for r in group if float(r[field]) == 0]
        return bool(right and wrong and mean(right) > mean(wrong))

    dense_values = [float(r["_dense"]) for group in valid_groups for r in group]
    no_answer = [r for group in valid_groups for r in group if not float(r.get("has_final_answer") or 0)]
    meaningful = [g for g in valid_groups if spread(g) >= 0.05]
    wrong_meaningful = [g for g in all_binary_wrong if spread(g) >= 0.05]
    boss_rank_count = sum(ranks_correctly(g, "boss_answer_correct") for g in boss_mixed)
    boss_rank_rate = boss_rank_count / len(boss_mixed) if boss_mixed else 0.0
    strict_rank_count = sum(ranks_correctly(g, "final_answer_correct") for g in strict_mixed)
    strict_rank_rate = strict_rank_count / len(strict_mixed) if strict_mixed else 0.0
    gate = {
        "mapping_complete": not unmapped,
        "valid_group_rate_at_least_99pct": len(valid_groups) >= 0.99 * len(groups),
        "meaningful_dense_group_rate_at_least_50pct": len(meaningful) >= 0.50 * len(valid_groups),
        "all_wrong_group_dense_rate_at_least_40pct": len(wrong_meaningful) >= 0.40 * len(all_binary_wrong),
        "boss_mixed_rank_rate_at_least_90pct": boss_rank_rate >= 0.90,
        "strict_mixed_rank_rate_at_least_95pct": strict_rank_rate >= 0.95,
        "no_final_answer_always_zero": all(float(r["_dense"]) == 0 for r in no_answer),
    }
    return {
        "row_count": len(rows),
        "mapped_row_count": sum(len(g) for g in groups.values()),
        "unmapped_row_count": len(unmapped),
        "group_count": len(groups),
        "valid_group_count": len(valid_groups),
        "invalid_group_count": len(groups) - len(valid_groups),
        "boss_mixed_group_count": len(boss_mixed),
        "strict_mixed_group_count": len(strict_mixed),
        "all_binary_wrong_group_count": len(all_binary_wrong),
        "meaningful_dense_group_count": len(meaningful),
        "meaningful_dense_group_rate": len(meaningful) / len(valid_groups),
        "all_wrong_meaningful_dense_group_count": len(wrong_meaningful),
        "all_wrong_meaningful_dense_group_rate": len(wrong_meaningful) / len(all_binary_wrong),
        "boss_mixed_rank_correct_count": boss_rank_count,
        "boss_mixed_rank_correct_rate": boss_rank_rate,
        "strict_mixed_rank_correct_count": strict_rank_count,
        "strict_mixed_rank_correct_rate": strict_rank_rate,
        "dense_score_mean": mean(dense_values),
        "dense_score_min": min(dense_values),
        "dense_score_max": max(dense_values),
        "no_final_answer_count": len(no_answer),
        "gate_checks": gate,
        "gate_passed": all(gate.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rollout-dir", action="append", type=Path, required=True)
    parser.add_argument("--phase", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.rollout_dir) != len(args.phase):
        parser.error("--rollout-dir and --phase counts must match")
    rows = []
    for directory, phase in zip(args.rollout_dir, args.phase, strict=True):
        rows.extend(read_rows(directory, phase))
    result = replay(rows, load_truth(args.dataset))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["gate_passed"] else 3)


if __name__ == "__main__":
    main()
