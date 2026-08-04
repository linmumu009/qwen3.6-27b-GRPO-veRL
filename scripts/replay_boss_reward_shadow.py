#!/usr/bin/env python3
"""Replay boss KB/DWH trajectories through the candidate shadow reward."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from llin_verl.boss_reward_shadow import (
    boss_task_to_ground_truth,
    compute_shadow_score,
    final_answer_from_openai,
    openai_messages_to_pi_events,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError(f"{label} row is missing task_id")
        if task_id in output:
            raise ValueError(f"duplicate {label} task_id: {task_id}")
        output[task_id] = row
    return output


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = Counter(row["task_family"] for row in results)
    boss_verdicts = Counter(
        (row["task_family"], row.get("boss_verdict", "missing")) for row in results
    )
    dwh = [row for row in results if row["task_family"] == "dwh"]
    kb = [row for row in results if row["task_family"] == "kb"]
    eligible_dwh = [row for row in dwh if row.get("online_eligible")]
    unanswerable_kb = [row for row in kb if not row.get("answerable")]
    score_distribution = {
        family: {
            str(score): count
            for score, count in sorted(
                Counter(row["score"] for row in results if row["task_family"] == family).items()
            )
        }
        for family in ("dwh", "kb")
    }
    return {
        "contract": "boss-kb-dwh-shadow-v1",
        "deployment_status": "shadow_only",
        "rows": len(results),
        "by_family": dict(by_family),
        "online_eligible": sum(bool(row.get("online_eligible")) for row in results),
        "strict_correct": sum(bool(row.get("acc")) for row in results),
        "requires_semantic_judge": sum(
            bool(row.get("requires_semantic_judge")) for row in results
        ),
        "gold_sql_verified": sum(bool(row.get("gold_sql_verified")) for row in results),
        "kb_source_documents_ok": sum(
            row["task_family"] == "kb" and bool(row.get("source_documents_ok"))
            for row in results
        ),
        "score_distribution": score_distribution,
        "dwh": {
            "rows": len(dwh),
            "online_eligible": len(eligible_dwh),
            "strict_correct": sum(bool(row.get("acc")) for row in eligible_dwh),
            "final_answer_correct": sum(
                bool(row.get("final_answer_correct")) for row in eligible_dwh
            ),
            "sql_evidence_correct": sum(
                bool(row.get("sql_evidence_correct")) for row in eligible_dwh
            ),
            "safe": sum(bool(row.get("safe")) for row in eligible_dwh),
            "boss_correct": sum(row.get("boss_verdict") == "correct" for row in dwh),
            "boss_correct_and_candidate_strict": sum(
                row.get("boss_verdict") == "correct" and bool(row.get("acc"))
                for row in eligible_dwh
            ),
            "candidate_strict_but_boss_not_correct": sum(
                row.get("boss_verdict") != "correct" and bool(row.get("acc"))
                for row in eligible_dwh
            ),
        },
        "kb": {
            "rows": len(kb),
            "answerable": sum(bool(row.get("answerable")) for row in kb),
            "unanswerable": len(unanswerable_kb),
            "source_documents_ok": sum(bool(row.get("source_documents_ok")) for row in kb),
            "gold_numbers_ok": sum(bool(row.get("gold_numbers_ok")) for row in kb),
            "gold_anchors_ok": sum(bool(row.get("gold_anchors_ok")) for row in kb),
            "abstention_detected": sum(bool(row.get("abstention_detected")) for row in kb),
            "unanswerable_boss_correct": sum(
                row.get("boss_verdict") == "correct" for row in unanswerable_kb
            ),
            "unanswerable_boss_correct_without_abstention": sum(
                row.get("boss_verdict") == "correct"
                and not bool(row.get("abstention_detected"))
                for row in unanswerable_kb
            ),
        },
        "boss_verdicts": {
            f"{family}:{verdict}": count
            for (family, verdict), count in sorted(boss_verdicts.items())
        },
        "invariants": {
            "task_id_join_only": True,
            "answer_text_does_not_count_as_document_access": True,
            "kb_never_online_eligible_without_semantic_judge": True,
            "unsafe_or_invalid_protocol_is_zero": True,
        },
    }


def replay(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = index_unique(read_jsonl(args.manifest), "manifest")
    trajectories = index_unique(read_jsonl(args.trajectories), "trajectory")
    verdicts = (
        index_unique(read_jsonl(args.boss_verdicts), "boss verdict")
        if args.boss_verdicts
        else {}
    )
    if args.sandbox_root:
        os.environ["PI_AGENT_SANDBOX_LOWER"] = str(args.sandbox_root)

    results: list[dict[str, Any]] = []
    for task_id, trajectory in trajectories.items():
        task = tasks.get(task_id)
        if task is None:
            raise ValueError(f"trajectory task_id is absent from manifest: {task_id}")
        family = str(task.get("type") or "").casefold()
        if family not in {"dwh", "kb"}:
            continue
        messages = trajectory.get("messages") or []
        if not isinstance(messages, list):
            raise ValueError(f"trajectory {task_id} has invalid messages")
        truth = boss_task_to_ground_truth(task)
        result = compute_shadow_score(
            "boss_shadow_replay",
            final_answer_from_openai(messages),
            truth,
            {"pi_tool_events": openai_messages_to_pi_events(messages)},
        )
        old = verdicts.get(task_id) or {}
        results.append(
            {
                **result,
                "boss_verdict": old.get("verdict"),
                "boss_verdict_fine": (old.get("evidence") or {}).get("verdict_fine"),
            }
        )
    return results, summarize(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--sandbox-root", type=Path)
    parser.add_argument("--boss-verdicts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, summary = replay(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in results),
            encoding="utf-8",
        )
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
