from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_chosen_only_first_action_forward_is_calibration_only_and_no_optimizer() -> None:
    script = (
        ROOT / "scripts" / "run_chosen_only_first_action_teacher_forced.sh"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT / "scripts" / "run_teacher_forced_component_diagnostic.py"
    ).read_text(encoding="utf-8")

    assert "chosen_only_schema_action_calibration16.parquet" in script
    assert "task_count=16" in script
    assert "forward_only=true" in script
    assert "optimizer_initialized=false" in script
    assert "checkpoint_saved=false" in script
    assert "training_allowed=false" in script
    assert "promotion_allowed=false" in script
    assert "engine.forward_only=true" in script
    assert "'checkpoint.load_contents=[]'" in script
    assert "'checkpoint.save_contents=[]'" in script
    assert "training_client.infer_batch" in runner
    assert "training_client.train_batch" not in runner


def test_final_answer_component_is_optional_for_first_action_diagnostic() -> None:
    runner = (
        ROOT / "scripts" / "run_teacher_forced_component_diagnostic.py"
    ).read_text(encoding="utf-8")

    required_block = runner.split("COMPONENT_MASKS =", 1)[1].split(
        "OPTIONAL_COMPONENT_MASKS", 1
    )[0]
    optional_block = runner.split("OPTIONAL_COMPONENT_MASKS =", 1)[1].split(
        "def component_sft_loss", 1
    )[0]
    assert '"sql_shell": "sql_shell_mask"' in required_block
    assert "final_answer" not in required_block
    assert '"final_answer": "final_answer_mask"' in optional_block
