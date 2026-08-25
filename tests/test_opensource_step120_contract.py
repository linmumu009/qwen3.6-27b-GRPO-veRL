from __future__ import annotations

from pathlib import Path

from llin_verl.opensource_reward import compute_score, extract_explicit_answer, math_equal
from scripts.prepare_opensource_step120_data import DEFAULT_QUOTAS, screen_and_select


ROOT = Path(__file__).resolve().parents[1]


def _score(solution: str, answer: str, answer_type: str = "math") -> dict[str, float]:
    return compute_score(
        "step120_opensource_recovery_v1",
        solution,
        {"dataset": "test", "task_id": "t", "answer": answer, "answer_type": answer_type},
        {},
    )


def test_reward_requires_correct_final_answer_and_never_rewards_wrong_format() -> None:
    correct = _score(r"Reasoning. \boxed{\frac{1}{2}}", "0.5")
    formatted_wrong = _score(r"Reasoning. \boxed{0.6}", "0.5")
    unboxed_equation = _score(r"Therefore the result equals 0.5.", "0.5")

    assert correct["score"] == 1.0
    assert formatted_wrong["score"] == 0.0
    assert unboxed_equation["score"] == 0.0
    assert formatted_wrong["explicit_final"] == 1.0


def test_reward_handles_nested_box_choice_and_gsm_numeric() -> None:
    answer, present = extract_explicit_answer(r"Hence \boxed{\frac{3}{\sqrt{5}}}.")
    assert present is True
    assert answer == r"\frac{3}{\sqrt{5}}"
    assert _score(r"The answer is \boxed{C}.", "C", "choice")["score"] == 1.0
    assert _score(r"We obtain \boxed{1,234}.", "1234", "numeric")["score"] == 1.0


def test_math_equality_has_exact_and_numeric_safe_paths() -> None:
    assert math_equal(r"\frac{3}{4}", "0.75")
    assert math_equal(r"\frac 35", "3/5")
    assert math_equal(r"4\sqrt{15}-14", r"4\sqrt{15}-14")
    assert math_equal(r"\sqrt{12}", r"2\sqrt{3}")
    assert math_equal(r"4\sqrt{15}-14", r"-14+4\sqrt{15}")
    assert math_equal(r"F_{\max}=5m\omega^2R", r"5R m\omega^2")
    assert not math_equal("0.7501", "0.75")


def _candidate(dataset: str, index: int, difficulty: str = "") -> dict[str, object]:
    return {
        "dataset": dataset,
        "task_id": f"{dataset}-{index}",
        "prompt": f"prompt {dataset} {index}",
        "answer": str(index),
        "answer_type": "math",
        "source_path": "source.jsonl",
        "source_row": index,
        "difficulty": difficulty,
        "subject": "test",
        "prompt_fingerprint": f"fp-{dataset}-{index}",
    }


def test_curriculum_selection_is_exactly_80_and_matches_frozen_mixture() -> None:
    pools = {
        "MATH": (
            [_candidate("MATH", i, "Level 5") for i in range(50)]
            + [_candidate("MATH", 100 + i, "Level 4") for i in range(30)]
        ),
        "PHYBench": [_candidate("PHYBench", i) for i in range(30)],
        "C-Eval-dev": [_candidate("C-Eval-dev", i) for i in range(20)],
        "GSM8K": [_candidate("GSM8K", i) for i in range(20)],
    }
    heldout = {"fp-PHYBench-0", "fp-GSM8K-0"}
    selected, report = screen_and_select(pools, heldout, "test-seed")

    counts = {name: sum(row["dataset"] == name for row in selected) for name in DEFAULT_QUOTAS}
    assert len(selected) == 80
    assert counts == DEFAULT_QUOTAS
    assert sum(row["dataset"] == "MATH" and row["difficulty"] == "Level 5" for row in selected) == 32
    assert sum(row["dataset"] == "MATH" and row["difficulty"] == "Level 4" for row in selected) == 16
    assert not ({row["prompt_fingerprint"] for row in selected} & heldout)
    assert report["PHYBench"]["excluded"]["heldout_fingerprint_overlap"] == 1


def test_launcher_reuses_step100_contract_and_names_step120_opensource() -> None:
    runner = (ROOT / "scripts/run_opensource_step100_to_step120.sh").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/launch_opensource_step100_to_step120.sh").read_text(encoding="utf-8")
    waiter = (ROOT / "scripts/wait_and_launch_opensource_step120_host.sh").read_text(encoding="utf-8")

    assert "llin-step120-opensource-" in runner
    assert "FINAL_POLICY_STEP=120" in runner
    assert "TOTAL_ROLLOUT_GROUPS=480" in runner
    assert "GROUPS_PER_STEP=4" in runner
    assert "LEARNING_RATE=\"${LEARNING_RATE:-1e-7}\"" in runner
    assert "global_step_100" in runner
    assert "resume-views/llin-step100-opensource/global_step_100" in runner
    assert "actor_rollout_ref.rollout.multi_turn.enable=False" in runner
    assert "opensource_reward.py" in runner
    assert "OPTIMIZER_CPU_OFFLOAD=false" in runner
    assert "ENGINE_OPTIMIZER_OFFLOAD=false" in runner
    assert "save_contents=[model,optimizer,extra]" in runner
    assert "expected_global_step_120" in launcher
    assert "REQUIRED_IDLE_CHECKS=\"${REQUIRED_IDLE_CHECKS:-3}\"" in waiter
    assert "prepare_pi_step100_resume_view.sh' trainer" in waiter
    assert "prepare_pi_step100_resume_view.sh' rollout" in waiter
    assert '-e "PYTHONPATH=${CONTAINER_PROJECT_ROOT}"' in waiter
    assert '"scripts/prepare_opensource_step120_data.py" convert' in waiter
    assert "AMO-Bench" not in runner
    assert "Omni-MATH" not in runner
    assert "ATLAS" not in runner
