from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import prepare_qwen38_approved43_outcome_training as prep
from scripts.patch_verl_fastest_k_oversampling import patch_agent_loop
from scripts.patch_verl_hard_gate_resampling import patch as patch_hard_gate
from scripts.attest_verified_process_structural_audit import attest


ROOT = Path(__file__).resolve().parents[1]


def test_launcher_freezes_qwen38_reward_kl_staleness_and_final_only_save() -> None:
    text = (ROOT / "scripts" / "run_pi_qwen38_approved43_4x_outcome_gated_v5.sh").read_text(encoding="utf-8")

    for required in (
        "MODEL_PATH:-/models/Qwen3.8-27B",
        "TOTAL_ROLLOUT_GROUPS=172",
        "RESPONSES_PER_GROUP=8",
        "OVERSAMPLE_CANDIDATES=16",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "algorithm.use_kl_in_reward=False",
        "STALENESS_THRESHOLD=0",
        "actor_rollout_ref.actor.optim.lr=5e-8",
        "compute_score_correctness_gated_process_v5",
        "trainer.save_freq=\"${TOTAL_NOMINAL_STEPS}\"",
        "trainer.max_actor_ckpt_to_keep=1",
        "checkpoint.save_contents=[model,extra]",
    ):
        assert required in text
    assert "Qwen3.6" not in text
    assert "Step120" not in text
    assert "compute_score_strict_correctness_v3" not in text


def test_prepare_schedule_hash_binds_evidence_and_repeats_exact_members(tmp_path: Path, monkeypatch) -> None:
    approved_rows = []
    manifest_rows = []
    tasks = []
    for index in range(43):
        instruction = f"{index:064x}"
        approved_rows.append(
            {
                "extra_info": {
                    "instruction_sha256": instruction,
                    "global_index": index,
                    "training_allowed": False,
                },
                "reward_model": {"ground_truth": {"environment_id": f"sft/v{index}", "verification_sql": "SELECT 1"}},
            }
        )
        manifest_rows.append({"instruction_sha256": instruction, "source_task_index": index})
        tasks.append(
            {
                "evidence_plan": {"task_type": "aggregation"},
                "expected_tables": ["fact"],
                "verification_criteria": {"must_use_fields": ["value"]},
            }
        )
    approved = tmp_path / "approved.parquet"
    manifest = tmp_path / "manifest.jsonl"
    tasks_path = tmp_path / "tasks.jsonl"
    pq.write_table(pa.Table.from_pylist(approved_rows), approved)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8")
    tasks_path.write_text("".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8")
    monkeypatch.setattr(prep, "PARQUET_SHA256", prep.file_sha256(approved))
    monkeypatch.setattr(prep, "MANIFEST_SHA256", prep.file_sha256(manifest))
    output = tmp_path / "schedule.parquet"
    summary = prep.prepare(approved, manifest, tasks_path, output, tmp_path / "safe.json")

    rows = pq.read_table(output).to_pylist()
    assert len(rows) == 172
    assert summary["unique_evidence_binding_hashes"] == 43
    assert {row["extra_info"]["exposure_index"] for row in rows} == {0, 1, 2, 3}
    assert all(row["extra_info"]["approved43_authorization"] for row in rows)


def test_hard_gate_patch_selects_only_eligible_and_fills_fail_closed_placeholders(tmp_path: Path) -> None:
    source = ROOT / "reference" / "verl" / "verl" / "experimental" / "agent_loop" / "agent_loop.py"
    target = tmp_path / "agent_loop.py"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    patch_agent_loop(target)
    assert patch_hard_gate(target) == "patched"
    text = target.read_text(encoding="utf-8")
    assert "LLIN_HARD_GATE_RESAMPLE_QUORUM" in text
    assert 'reward_info.get("online_eligible", 0)' in text
    assert "hard_gate_cap_exhausted" in text


def test_structural_audit_does_not_claim_human_precision(tmp_path: Path) -> None:
    packet = tmp_path / "packet.jsonl"
    row = {
        "process_verified": 1,
        "successful_sql_count": 1,
        "answer_bearing_sql_count": 1,
        "last_answer_bearing_consistent": 1,
        "numeric_final_parse_ambiguous": 0,
        "audit_checklist": {"verified_process_requires_answer_bearing_successful_sql": True},
    }
    packet.write_text("".join(json.dumps(row) + "\n" for _ in range(20)), encoding="utf-8")
    result = attest(packet, tmp_path / "safe.json")
    assert result["status"] == "pass"
    assert result["structural_precision_proxy"] == 1.0
    assert result["human_precision_established"] is False
    assert result["process_bonus_promotion_allowed"] is False
