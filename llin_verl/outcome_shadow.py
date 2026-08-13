"""Final-outcome-only shadow scoring for PI trajectories.

This module deliberately ignores SQL, tool evidence, process quality, and the
online reward.  It is a diagnostic scorer, not an online reward switch.
"""

from __future__ import annotations

import json
from typing import Any

from llin_verl.pi_reward import (
    dense_final_answer_correctness,
    extract_final_assistant_answer,
    final_answer_correct,
)


def expected_value_from_ground_truth(ground_truth: dict[str, Any]) -> Any:
    """Read the hidden expected value without consulting verifier SQL."""
    if "expected_value_json" in ground_truth:
        value = ground_truth["expected_value_json"]
        return json.loads(value) if isinstance(value, str) else value
    if "expected_value" in ground_truth:
        return ground_truth["expected_value"]
    raise ValueError("ground truth has no expected final value")


def score_final_outcome(solution: str, ground_truth: dict[str, Any]) -> dict[str, Any]:
    """Return binary and dense final-answer-only diagnostics.

    The binary score is the proposed pure-outcome shadow signal.  Dense credit
    is emitted only for diagnosis and is never substituted for the binary
    result in group routing.
    """
    answer_type = str(ground_truth.get("answer_type") or "")
    if answer_type not in {"numeric", "table"}:
        raise ValueError(f"unsupported answer_type: {answer_type!r}")
    expected = expected_value_from_ground_truth(ground_truth)
    abs_tol = float(ground_truth.get("abs_tol", 1e-3))
    rel_tol = float(ground_truth.get("rel_tol", 1e-5))
    final_answer = extract_final_assistant_answer(solution)
    correct = final_answer_correct(final_answer, answer_type, expected, abs_tol, rel_tol)
    dense = dense_final_answer_correctness(
        final_answer,
        answer_type,
        expected,
        abs_tol,
        rel_tol,
    )
    return {
        "outcome_only_score": float(correct),
        "final_answer_correct": float(correct),
        "dense_final_answer_correctness": float(dense),
        "has_final_answer": float(bool(final_answer)),
        "answer_type": answer_type,
    }

