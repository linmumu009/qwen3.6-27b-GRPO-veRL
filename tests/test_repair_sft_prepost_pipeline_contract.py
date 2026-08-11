from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_prepost_pipeline_runs_post_replay_and_boss_original_judge():
    script = (ROOT / "scripts" / "run_repair_sft_prepost_pipeline_host.sh").read_text(encoding="utf-8")

    assert "wait_step120_baseline" in script
    assert "launch_post_sft_replay" in script
    assert "reward_judge.py" in script
    assert "compare_boss_exact_evaluations.py" in script
    assert "--left-label step120" in script
    assert "--right-label post_sft" in script
    assert '"minimum_exact_successes": 14' in script
    assert '"heldout_claim_allowed": False' in script


def test_prepost_pipeline_does_not_stop_ray_or_claim_heldout_accuracy():
    script = (ROOT / "scripts" / "run_repair_sft_prepost_pipeline_host.sh").read_text(encoding="utf-8")

    assert "ray stop" not in script
    assert "evaluation_split=train236_same_task_not_heldout" not in script
    assert "heldout_claim_allowed" in script
