from pathlib import Path

from scripts.patch_verl_val_only_force_dist_sync import patch


ROOT = Path(__file__).resolve().parents[1]


def test_force_dist_sync_patch_is_opt_in_and_idempotent(tmp_path: Path):
    target = tmp_path / "trainer.py"
    target.write_text(
        'from __future__ import annotations\n\n'
        '        val_only_base_model = (\n'
        '            self.config.trainer.get("val_only", False)\n'
        '            and self.config.trainer.get("resume_mode", "disable") == "disable"\n'
        '        )\n',
        encoding="utf-8",
    )

    assert patch(target) == "patched"
    text = target.read_text(encoding="utf-8")
    assert 'os.environ.get("LLIN_VAL_ONLY_FORCE_DIST_SYNC") != "1"' in text
    assert patch(target) == "already-patched"


def test_replay_uses_boss_tools_greedy_n1_and_same_16_train_tasks():
    script = (ROOT / "scripts" / "run_repair_sft_replay.sh").read_text(encoding="utf-8")

    assert "repair_sft_replay.parquet" in script
    assert "evaluation_rows=16" in script
    assert "evaluation_split=train236_same_task_not_heldout" in script
    assert "sampling=greedy_n1" in script
    assert "system_tools=boss_exact" in script
    assert "MAX_ASSISTANT_TURNS=26" in script
    assert "MAX_USER_TURNS=25" in script
    assert "LLIN_VAL_ONLY_FORCE_DIST_SYNC=1" in script
    assert "actor_rollout_ref.actor.megatron.dist_checkpointing_path" in script
