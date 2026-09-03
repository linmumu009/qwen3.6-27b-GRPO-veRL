import json
from pathlib import Path

from scripts.summarize_logistics_cpt_run import summarize


def test_summarize_complete_direct_sft_run(tmp_path: Path):
    run_dir = tmp_path / "pilot"
    stdout = run_dir / "torchrun_logs" / "run" / "attempt_0" / "8" / "stdout.log"
    stderr = stdout.with_name("stderr.log")
    stdout.parent.mkdir(parents=True)
    stdout.write_text(
        "step:1 - perf/max_memory_allocated_gb:38.0 - perf/max_memory_reserved_gb:39.0 "
        "- perf/cpu_memory_used_gb:88.0 - train/loss:2.5 - train/grad_norm:4.0 "
        "- train/lr:5e-7 - train/mfu:0.1 - train/global_tokens:6 "
        "- train/total_tokens(B):6e-9\n",
        encoding="utf-8",
    )
    stderr.write_text("", encoding="utf-8")

    checkpoint = run_dir / "checkpoints" / "global_step_1"
    model = checkpoint / "model" / "dist_ckpt"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"metadata")
    (model / "__0_0.distcp").write_bytes(b"weights")
    (checkpoint / "ckpt_contents.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "save_contents": ["model", "extra"],
                "contents": {
                    "model": {
                        "format": "megatron_dist_checkpoint",
                        "path": "model/dist_ckpt",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = summarize(run_dir, expected_steps=1, expected_tokens=6)

    assert result["status"] == "complete"
    assert result["loss"]["mean"] == 2.5
    assert result["checkpoint"]["layout"] == "direct_sft"
    assert result["checkpoint"]["optimizer_declared"] is False
    assert result["source_content_included"] is False
