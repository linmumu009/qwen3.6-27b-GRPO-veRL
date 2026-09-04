import json
from pathlib import Path

from scripts.summarize_logistics_cpt_exposure_curve import summarize_curve


ROOT = Path(__file__).resolve().parents[1]


def _checkpoint(run_dir: Path, step: int) -> None:
    checkpoint = run_dir / "checkpoints" / f"global_step_{step}"
    model = checkpoint / "model" / "dist_ckpt"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"metadata")
    (model / "__0_0.distcp").write_bytes(b"weights")
    (checkpoint / "ckpt_contents.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "save_contents": ["model", "extra"],
                "contents": {"model": {"format": "megatron_dist_checkpoint", "path": "model/dist_ckpt"}},
            }
        ),
        encoding="utf-8",
    )


def test_exposure_curve_contract_is_continuous_and_saves_only_2x_and_4x():
    script = (ROOT / "scripts" / "run_logistics_cpt_book_exposure_curve.sh").read_text(encoding="utf-8")

    assert "purpose=single_book_cpt_exposure_curve_2x_4x" in script
    assert "checkpoint_initialization=model_only_dist_ckpt" in script
    assert "optimizer_state=fresh_continuous_through_step_116" in script
    assert "optimizer_reset_at_step_58=false" in script
    assert "evaluation_checkpoints=58,116" in script
    assert "trainer.total_epochs=${TOTAL_EXPOSURES}" in script
    assert "trainer.total_training_steps=${TOTAL_STEPS}" in script
    assert "trainer.save_freq=${CHECKPOINT_STEPS}" in script
    assert "trainer.max_ckpt_to_keep=2" in script
    assert "'checkpoint.save_contents=[model,extra]'" in script
    assert "promotion_allowed=false" in script


def test_summarize_exposure_curve_validates_each_epoch_and_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "curve"
    stdout = run_dir / "torchrun_logs" / "run" / "attempt_0" / "0" / "stdout.log"
    stdout.parent.mkdir(parents=True)
    rows = []
    for step in range(1, 5):
        rows.append(
            f"step:{step} - perf/max_memory_allocated_gb:38 - perf/max_memory_reserved_gb:39 "
            f"- perf/cpu_memory_used_gb:88 - train/loss:{3 - step / 10} - train/grad_norm:{step} "
            f"- train/lr:{step}e-8 - train/global_tokens:3 - train/total_tokens(B):{step * 3e-9}"
        )
    stdout.write_text("\n".join(rows) + "\n", encoding="utf-8")
    stdout.with_name("stderr.log").write_text(
        "safe_import failed with: Traceback (most recent call last):\n"
        "AttributeError: optional compatibility symbol is absent\n",
        encoding="utf-8",
    )
    _checkpoint(run_dir, 2)
    _checkpoint(run_dir, 4)

    result = summarize_curve(
        run_dir,
        steps_per_exposure=2,
        total_exposures=2,
        sequence_tokens_per_exposure=6,
        checkpoint_exposures=(1, 2),
    )

    assert result["total_steps"] == 4
    assert result["total_sequence_tokens_with_eos"] == 12
    assert [row["global_step"] for row in result["checkpoints"]] == [2, 4]
    assert [row["sequence_tokens_with_eos"] for row in result["exposures"]] == [6, 6]
