from __future__ import annotations

import ast
import asyncio
from concurrent.futures import Future
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from scripts import prepare_qwen38_approved43_outcome_training as prep
from scripts.patch_verl_fastest_k_oversampling import patch_agent_loop
from scripts.patch_verl_hard_gate_resampling import (
    normalize_non_tensor_chunks,
    patch as patch_hard_gate,
)
from scripts.patch_verl_grpo_strict_variance_gate import (
    patch_fully_async_trainer,
    patch_trainer,
)
from scripts import prepare_qwen38_tiered_canary_data as tiered_canary
from scripts import prepare_qwen38_tiered_canary_sealed8 as tiered_sealed
from scripts.attest_verified_process_structural_audit import attest
from llin_verl.grpo_group_gate import (
    apply_strict_correctness_group_gate,
    strict_correctness_group_stats,
)


ROOT = Path(__file__).resolve().parents[1]


def _verl_source(*relative: str) -> Path:
    candidates = (
        ROOT / "reference" / "verl" / "verl" / Path(*relative),
        Path("/verl/verl") / Path(*relative),
    )
    return next(path for path in candidates if path.is_file())


def _patched_method(source: str, name: str):
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    method.decorator_list = []
    method.returns = None
    for argument in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs):
        argument.annotation = None
    namespace = {
        "asyncio": asyncio,
        "datetime": datetime,
        "marked_timer": _marked_timer,
        "reduce_metrics": lambda values: {
            key: float(item[0] if isinstance(item, list) else item)
            for key, item in values.items()
        },
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "<patched-method>", "exec"), namespace)
    return namespace[name]


@contextmanager
def _marked_timer(*_args, **_kwargs):
    yield


class _RemoteResult:
    def __init__(self, value):
        self._future = Future()
        self._future.set_result(value)

    def future(self):
        return self._future


class _RemoteMethod:
    def __init__(self, function):
        self._function = function

    def remote(self):
        return _RemoteResult(self._function())


class _WindowedRollouter:
    def __init__(self):
        self.produced_groups = 2
        self.allowance = 0
        self.reset_calls = 0
        self.reset_staleness = _RemoteMethod(self._reset_staleness)

    def _reset_staleness(self):
        self.reset_calls += 1
        self.allowance = 2
        return {"fully_async/rollouter/step_generated_samples": 2}

    def produce_group(self) -> int | None:
        if self.allowance <= 0:
            return None
        self.allowance -= 1
        self.produced_groups += 1
        return self.produced_groups


def test_launcher_freezes_qwen38_tristate_reward_kl_staleness_and_final_only_save() -> None:
    text = (ROOT / "scripts" / "run_pi_qwen38_approved43_4x_grounded_tristate_v6.sh").read_text(encoding="utf-8")

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
        "compute_score_grounded_tristate_v6",
        "reward=train_mask*success",
        "unknown_behavior=mask_and_resample",
        "guess_correct_reward=0",
        "HUMAN_344_CALIBRATION_APPROVED",
        "VERIFIER_CASEPACK_APPROVED",
        "trainer.save_freq=\"${TOTAL_NOMINAL_STEPS}\"",
        "trainer.max_actor_ckpt_to_keep=1",
        "checkpoint.save_contents=[model,extra]",
    ):
        assert required in text
    assert "Qwen3.6" not in text
    assert "Step120" not in text
    assert "compute_score_strict_correctness_v3" not in text
    assert "PI_PROCESS_BONUS_ALPHA" not in text

    retired = (ROOT / "scripts" / "run_pi_qwen38_approved43_4x_outcome_gated_v5.sh").read_text(encoding="utf-8")
    assert "superseded: v5 H*C can reward an ungrounded guessed answer" in retired
    assert retired.index("exit 3") < retired.index("PROJECT_ROOT=")


def test_tiered_canary_launcher_freezes_actual_update_contract() -> None:
    text = (ROOT / "scripts" / "run_pi_qwen38_approved43_tiered_canary_v1.sh").read_text(encoding="utf-8")

    for required in (
        "MODEL_PATH:-/models/Qwen3.8-27B",
        "TARGET_ACTUAL_OPTIMIZER_STEPS=5",
        "MAX_NOMINAL_GROUPS=20",
        "GROUPS_PER_STEP=2",
        "RESPONSES_PER_GROUP=8",
        "OVERSAMPLE_CANDIDATES=16",
        "compute_score_tiered_query_cost_v1",
        "algorithm.use_kl_in_reward=False",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "+actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True",
        "OPTIMIZER_CPU_OFFLOAD=true ENGINE_OPTIMIZER_OFFLOAD=false",
        "STALENESS_THRESHOLD=0",
        "trainer.val_before_train=true",
        "trainer.test_freq=\"${TARGET_ACTUAL_OPTIMIZER_STEPS}\"",
        "actor_rollout_ref.rollout.val_kwargs.n=4",
        "prepare_qwen38_tiered_canary_sealed8.py",
        'export PYTHONPATH="${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"',
        "trainer.max_actor_ckpt_to_keep=1",
        "full_training_allowed=false_pending_main_thread_review",
        "e2c3b44e4e198e94fcd74903983fc8997f8e504a21575e397f9d59db1cc2fc8f",
        "LC_ALL=C sha256sum \"${model_shards[@]}\" | sha256sum",
        "LC_ALL=C_sha256sum_absolute_sorted_glob_lines_then_sha256",
    ):
        assert required in text
    assert "Qwen3.6/Step120" not in text
    assert "compute_score_grounded_tristate_v6" not in text

    host_text = (ROOT / "scripts" / "launch_qwen38_tiered_canary5_host.sh").read_text(encoding="utf-8")
    for required in (
        "staging_rollout_data",
        "canary20.sensitive.parquet",
        "sealed8.sensitive.parquet",
        "train_sha_local",
        "train_sha_remote",
        "sealed_sha_local",
        "sealed_sha_remote",
        "cross_host_identical",
        "test_runtime_gate_consumes_success_without_legacy_acc",
        "staging_bound_pi_sandbox",
        "stage_bound_pi_sandbox.py",
        "bound_pi_sandbox_cross_host.safe.json",
        "PI_AGENT_SANDBOX_LOWER",
        "PI_AGENT_TOKENIZER_PATH",
    ):
        assert required in host_text
    assert host_text.index("staging_rollout_data") < host_text.index("starting_isolated_ray")
    assert host_text.index("staging_bound_pi_sandbox") < host_text.index("starting_isolated_ray")


def test_runtime_gate_consumes_success_without_legacy_acc() -> None:
    import torch

    class TensorDictLike:
        """Exercise key membership without relying on mapping iteration."""

        def __init__(self, values):
            self.values = values

        def __contains__(self, key):
            return key in self.values

        def __iter__(self):
            # TensorDict iteration is not a plain iterable of hashable keys.
            return iter([self.values])

        def __getitem__(self, key):
            return self.values[key]

        def __setitem__(self, key, value):
            self.values[key] = value

    batch = SimpleNamespace(
        non_tensor_batch={
            "uid": np.asarray(["mixed"] * 8, dtype=object),
            "success": np.asarray([0, 1] * 4),
            "train_mask": np.ones(8, dtype=np.int64),
        },
        batch=TensorDictLike({
            "advantages": torch.ones(8, 2),
            "returns": torch.ones(8, 2),
            "response_mask": torch.ones(8, 2),
        }),
        meta_info={},
    )

    gated, metrics = apply_strict_correctness_group_gate(batch)

    assert gated.meta_info["strict_group_should_update_actor"] is True
    assert metrics["grpo/strict_mixed_groups"] == 1.0
    assert torch.count_nonzero(gated.batch["response_mask"]) == 16


def test_tiered_formal_launcher_freezes_full_contract() -> None:
    text = (ROOT / "scripts" / "run_pi_qwen38_approved43_tiered_formal_v1.sh").read_text(encoding="utf-8")

    for required in (
        "MODEL_PATH:-/models/Qwen3.8-27B",
        "TOTAL_NOMINAL_GROUPS=172",
        "TOTAL_NOMINAL_BATCHES=86",
        "GROUPS_PER_STEP=2",
        "RESPONSES_PER_GROUP=8",
        "OVERSAMPLE_CANDIDATES=16",
        "compute_score_tiered_query_cost_v1",
        "algorithm.use_kl_in_reward=False",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "+actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True",
        "STALENESS_THRESHOLD=0",
        "trainer.val_before_train=true",
        "prepare_qwen38_approved43_outcome_training.py",
        "prepare_qwen38_tiered_canary_sealed8.py",
        "unset LLIN_CANARY_TARGET_OPTIMIZER_STEPS",
        "checkpoint.save_contents=[model,extra]",
        "formal_training_allowed=true",
        "d86b53d906806b150d43a508dce9b0dd6d05105c07e03961e8e7bf9439ccd944",
        "1426bc09a3dbaf4709fd89227790603afb7a2bf11beeba80946057d490e0f424",
        "e2c3b44e4e198e94fcd74903983fc8997f8e504a21575e397f9d59db1cc2fc8f",
    ):
        assert required in text
    assert "TARGET_ACTUAL_OPTIMIZER_STEPS" not in text
    assert "compute_score_grounded_tristate_v6" not in text

    host_text = (ROOT / "scripts" / "launch_qwen38_tiered_formal_host.sh").read_text(encoding="utf-8")
    for required in (
        "staging_rollout_data",
        "approved43x4.sensitive.parquet",
        "sealed8.sensitive.parquet",
        "train_sha_local",
        "train_sha_remote",
        "sealed_sha_local",
        "sealed_sha_remote",
        "cross_host_identical",
    ):
        assert required in host_text
    assert host_text.index("staging_rollout_data") < host_text.index("starting_isolated_ray")


def test_prepare_tiered_sealed8_is_disjoint_and_balanced(tmp_path: Path, monkeypatch) -> None:
    approved_rows = []
    raw_rows = []
    for index in range(100):
        instruction = f"{index:064x}"
        answer_type = "numeric" if index % 2 == 0 else "table"
        row = {
            "extra_info": {"instruction_sha256": instruction, "training_allowed": False},
            "reward_model": {"ground_truth": {"answer_type": answer_type}},
        }
        raw_rows.append(row)
        if index < 43:
            approved_rows.append(row)
    approved = tmp_path / "approved.parquet"
    raw = tmp_path / "raw.parquet"
    output = tmp_path / "sealed.parquet"
    summary_path = tmp_path / "safe.json"
    pq.write_table(pa.Table.from_pylist(approved_rows), approved)
    pq.write_table(pa.Table.from_pylist(raw_rows), raw)
    monkeypatch.setattr(tiered_sealed, "APPROVED43_SHA256", tiered_sealed.file_sha256(approved))
    monkeypatch.setattr(tiered_sealed, "RAW100_SHA256", tiered_sealed.file_sha256(raw))

    summary = tiered_sealed.prepare(approved, raw, output, summary_path)
    rows = pq.read_table(output).to_pylist()
    identity = lambda row: str((row.get("extra_info") or {}).get("instruction_sha256") or "")
    approved_ids = {identity(row) for row in approved_rows}
    sealed_ids = {identity(row) for row in rows}
    assert summary["answer_type_counts"] == {"numeric": 4, "table": 4}
    assert len(rows) == len(sealed_ids) == 8
    assert approved_ids.isdisjoint(sealed_ids)
    assert all(row["extra_info"]["training_allowed"] is False for row in rows)
    assert all(row["extra_info"]["sealed_evaluation_only"] is True for row in rows)


def test_actual_optimizer_patch_covers_parent_and_fully_async_versioning(tmp_path: Path) -> None:
    parent = tmp_path / "ray_trainer.py"
    async_trainer = tmp_path / "fully_async_trainer.py"
    parent.write_text(
        _verl_source("experimental", "separation", "ray_trainer.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    async_trainer.write_text(
        _verl_source("experimental", "fully_async_policy", "fully_async_trainer.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert patch_trainer(parent) in {"patched", "already-patched"}
    assert patch_fully_async_trainer(async_trainer) in {
        "patched",
        "upgraded-v5",
        "already-patched",
    }
    parent_text = parent.read_text(encoding="utf-8")
    async_text = async_trainer.read_text(encoding="utf-8")
    assert 'getattr(self, "current_param_version", self.global_steps - 1)' in parent_text
    assert 'strict_expected_group_size' in parent_text
    assert 'self.strict_optimizer_steps += 1' in parent_text
    assert 'actor/update_skipped_no_strict_mixed' in async_text
    assert 'training/weight_sync_skipped_no_optimizer_step' in async_text
    assert 'LLIN_SKIP_ROLLOUT_WINDOW_RESET_V5' in async_text
    assert 'training/rollout_window_reset_without_optimizer_step' in async_text
    assert 'return await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())' in async_text
    assert 'LLIN_CANARY_TARGET_OPTIMIZER_STEPS' in async_text
    assert patch_trainer(parent) == "already-patched"
    assert patch_fully_async_trainer(async_trainer) == "already-patched"


@pytest.mark.parametrize("skip_kind", ["uniform", "unknown"])
def test_skipped_batch_keeps_optimizer_policy_and_adam_but_reopens_rollout_window(
    tmp_path: Path,
    skip_kind: str,
) -> None:
    parent = tmp_path / "ray_trainer.py"
    async_trainer = tmp_path / "fully_async_trainer.py"
    parent.write_text(
        _verl_source("experimental", "separation", "ray_trainer.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    async_trainer.write_text(
        _verl_source("experimental", "fully_async_policy", "fully_async_trainer.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    patch_trainer(parent)
    patch_fully_async_trainer(async_trainer)

    train_mask = [1] * 8 if skip_kind == "uniform" else [1] * 7 + [0]
    gated, metrics = apply_strict_correctness_group_gate(
        SimpleNamespace(
            non_tensor_batch={
                "uid": [skip_kind] * 8,
                "success": [0] * 8,
                "train_mask": train_mask,
            },
            batch={
                "advantages": torch.ones(8, 2),
                "returns": torch.ones(8, 2),
                "response_mask": torch.ones(8, 2),
            },
            meta_info={"strict_expected_group_size": 8},
        )
    )
    assert gated.meta_info["strict_group_should_update_actor"] is False
    assert torch.count_nonzero(gated.batch["advantages"]) == 0
    if skip_kind == "uniform":
        assert metrics["grpo/skipped_uniform_groups"] == 1.0
    else:
        assert metrics["grpo/skipped_hard_gate_groups"] == 1.0

    parent_source = parent.read_text(encoding="utf-8")
    async_source = async_trainer.read_text(encoding="utf-8")
    update_actor = _patched_method(parent_source, "_fit_update_actor")
    update_local_step = _patched_method(async_source, "_fit_update_local_step")
    update_weights = _patched_method(async_source, "_fit_update_weights")
    rollouter = _WindowedRollouter()
    trainer = SimpleNamespace(
        metrics={},
        timing_raw={},
        strict_optimizer_steps=0,
        current_param_version=0,
        local_trigger_step=1,
        trigger_parameter_sync_step=1,
        global_steps=1,
        config=SimpleNamespace(trainer=SimpleNamespace(critic_warmup=0)),
        rollouter=rollouter,
        actor_parameter_state="actor-base-hash",
        adam_state="adam-empty-hash",
        optimizer_calls=0,
    )

    def forbidden_update(_batch):
        trainer.optimizer_calls += 1
        trainer.actor_parameter_state = "changed"
        trainer.adam_state = "changed"
        raise AssertionError("skipped group must not call actor optimizer")

    trainer._update_actor = forbidden_update
    before = (
        trainer.actor_parameter_state,
        trainer.adam_state,
        trainer.strict_optimizer_steps,
        trainer.current_param_version,
    )
    update_actor(trainer, gated)
    update_local_step(trainer)
    assert rollouter.produce_group() is None
    reset_metrics = asyncio.run(update_weights(trainer))
    after = (
        trainer.actor_parameter_state,
        trainer.adam_state,
        trainer.strict_optimizer_steps,
        trainer.current_param_version,
    )

    assert after == before
    assert trainer.optimizer_calls == 0
    assert trainer.metrics["training/policy_version_advanced"] == 0.0
    assert trainer.metrics["training/weight_sync_skipped_no_optimizer_step"] == 1.0
    assert trainer.metrics["training/rollout_window_reset_without_optimizer_step"] == 1.0
    assert reset_metrics["fully_async/rollouter/step_generated_samples"] == 2
    assert rollouter.reset_calls == 1
    assert rollouter.produce_group() == 3


def test_mixed_group_updates_and_advances_policy_while_stale_group_fails_closed(tmp_path: Path) -> None:
    parent = tmp_path / "ray_trainer.py"
    async_trainer = tmp_path / "fully_async_trainer.py"
    parent.write_text(
        _verl_source("experimental", "separation", "ray_trainer.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    async_trainer.write_text(
        _verl_source("experimental", "fully_async_policy", "fully_async_trainer.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    patch_trainer(parent)
    patch_fully_async_trainer(async_trainer)
    update_actor = _patched_method(parent.read_text(encoding="utf-8"), "_fit_update_actor")
    update_local_step = _patched_method(async_trainer.read_text(encoding="utf-8"), "_fit_update_local_step")

    mixed_batch = SimpleNamespace(meta_info={"strict_group_should_update_actor": True})
    trainer = SimpleNamespace(
        metrics={},
        timing_raw={},
        strict_optimizer_steps=0,
        current_param_version=0,
        local_trigger_step=1,
        trigger_parameter_sync_step=1,
        global_steps=1,
        config=SimpleNamespace(trainer=SimpleNamespace(critic_warmup=0)),
        actor_parameter_state="actor-base-hash",
        adam_state="adam-empty-hash",
        optimizer_calls=0,
    )

    def real_update(_batch):
        trainer.optimizer_calls += 1
        trainer.actor_parameter_state = "actor-updated-hash"
        trainer.adam_state = "adam-updated-hash"
        return SimpleNamespace(meta_info={"metrics": {"actor/loss": [0.25]}})

    trainer._update_actor = real_update
    update_actor(trainer, mixed_batch)
    update_local_step(trainer)
    assert trainer.optimizer_calls == 1
    assert trainer.strict_optimizer_steps == 1
    assert trainer.actor_parameter_state == "actor-updated-hash"
    assert trainer.adam_state == "adam-updated-hash"
    assert trainer.current_param_version == 1
    assert trainer.metrics["training/policy_version_advanced"] == 1.0

    stale_mask, stale_metrics = strict_correctness_group_stats(
        ["stale"] * 8,
        [0, 1] * 4,
        eligibility=[1] * 8,
        policy_versions=[0] * 8,
        expected_policy_version=1,
        expected_group_size=8,
    )
    assert stale_mask == [False] * 8
    assert stale_metrics["grpo/skipped_stale_policy_groups"] == 1.0


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


def test_prepare_tiered_canary_alternates_ten_numeric_ten_table(tmp_path: Path, monkeypatch) -> None:
    approved_rows = []
    manifest_rows = []
    tasks = []
    for index in range(43):
        instruction = f"{index:064x}"
        answer_type = "numeric" if index < 21 else "table"
        approved_rows.append(
            {
                "extra_info": {
                    "instruction_sha256": instruction,
                    "global_index": index,
                    "training_allowed": False,
                },
                "reward_model": {
                    "ground_truth": {
                        "environment_id": f"sft/v{index}",
                        "answer_type": answer_type,
                        "verification_sql": "SELECT value FROM fact",
                    }
                },
            }
        )
        manifest_rows.append({"instruction_sha256": instruction})
        tasks.append(
            {
                "evidence_plan": {"aggregation": "sum"},
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

    output = tmp_path / "canary.parquet"
    summary = tiered_canary.build(
        approved,
        manifest,
        tasks_path,
        output,
        tmp_path / "canary.safe.json",
    )
    rows = pq.read_table(output).to_pylist()
    kinds = [row["reward_model"]["ground_truth"]["answer_type"] for row in rows]
    assert len(rows) == 20
    assert kinds == ["numeric", "table"] * 10
    assert summary["nominal_groups"] == 20
    assert summary["target_actual_optimizer_steps"] == 5
    assert all(row["extra_info"]["canary_training_authorized"] for row in rows)
    assert all(row["extra_info"]["pi_reward_database_root"] == "/pi_sandbox" for row in rows)


def test_tristate_patch_selects_pass_fail_and_resamples_unknown(tmp_path: Path) -> None:
    source = ROOT / "reference" / "verl" / "verl" / "experimental" / "agent_loop" / "agent_loop.py"
    target = tmp_path / "agent_loop.py"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    patch_agent_loop(target)
    assert patch_hard_gate(target) == "patched"
    text = target.read_text(encoding="utf-8")
    assert "LLIN_TRISTATE_UNKNOWN_RESAMPLE_V2" in text
    assert "LLIN_NON_TENSOR_CONCAT_SCHEMA_V1" in text
    assert "all_non_tensor_keys" in text
    assert "missing[:] = None" in text
    assert 'reward_info.get("train_mask", 0)' in text
    assert "tristate_cap_exhausted" in text
    assert patch_hard_gate(target) == "already-patched"


def test_pi_agent_loop_keeps_optional_workspace_evidence_schema_stable() -> None:
    text = (ROOT / "llin_verl" / "pi_agent_loop.py").read_text(encoding="utf-8")

    for required_default in (
        '"pi_workspace_request_id": ""',
        '"pi_environment_id": environment_id',
        '"pi_trajectory_request_id": request_id',
        '"pi_trajectory_environment_id": environment_id',
        '"pi_tool_call_count": 0',
        '"pi_tool_success_count": 0',
        '"pi_workspace_elapsed_seconds": 0.0',
        '"pi_workspace_released": False',
    ):
        assert required_default in text


def test_mixed_tool_no_tool_timeout_chunk_union_preserves_64_identities_and_order() -> None:
    class Chunk(SimpleNamespace):
        def __len__(self) -> int:
            return len(self.non_tensor_batch["identity"])

    identities = [f"task-{index // 8}:sample-{index % 8}" for index in range(64)]
    chunks = [
        Chunk(
            non_tensor_batch={
                "identity": np.array(identities[:21], dtype=object),
                "pi_tool_events": np.array([[{"ok": True}]] * 21, dtype=object),
                "pi_workspace_request_id": np.array([f"req-{i}" for i in range(21)], dtype=object),
            }
        ),
        Chunk(
            non_tensor_batch={
                "identity": np.array(identities[21:43], dtype=object),
                "trajectory_timeout": np.array([False] * 22, dtype=object),
            }
        ),
        Chunk(
            non_tensor_batch={
                "identity": np.array(identities[43:], dtype=object),
                "trajectory_timeout": np.array([True] * 21, dtype=object),
            }
        ),
    ]

    normalized = normalize_non_tensor_chunks(chunks)

    assert normalized is chunks
    assert [value for chunk in chunks for value in chunk.non_tensor_batch["identity"]] == identities
    assert len(set(identities)) == 64
    assert all(set(chunk.non_tensor_batch) == set(chunks[0].non_tensor_batch) for chunk in chunks)
    assert all(value is None for value in chunks[1].non_tensor_batch["pi_tool_events"])
    assert all(value is None for value in chunks[2].non_tensor_batch["pi_workspace_request_id"])


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
