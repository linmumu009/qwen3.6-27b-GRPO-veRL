from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_qwen38_adaptive_dwh_three_wave_queue import (
    launch_wave,
    model_identity,
    validate_topology,
)


ROOT = Path(__file__).resolve().parents[1]


def topology(**changes) -> Namespace:
    values = {
        "reasoning_effort": "medium",
        "tensor_parallel_size": 4,
        "data_parallel_size": 4,
        "rollout_npus": 16,
        "task_batch_size": 32,
        "max_num_seqs": 16,
        "rolling_window_trajectories": 80,
        "rolling_window_max_multiplier": 1.25,
    }
    values.update(changes)
    return Namespace(**values)


def test_tp4_dp4_and_tp4_dp3_topologies_are_admitted() -> None:
    full = validate_topology(topology())
    partial = validate_topology(
        topology(
            data_parallel_size=3,
            rollout_npus=12,
            task_batch_size=24,
            rolling_window_trajectories=60,
        )
    )
    assert full["physical_sequence_capacity"] == 64
    assert full["logical_window_trajectories"] == 80
    assert partial["physical_sequence_capacity"] == 48
    assert partial["logical_window_trajectories"] == 60


@pytest.mark.parametrize(
    "changes",
    [
        {"reasoning_effort": "low"},
        {"rollout_npus": 12},
        {"rolling_window_trajectories": 64},
        {"rolling_window_trajectories": 81},
    ],
)
def test_qwen38_topology_contract_fails_closed(changes: dict) -> None:
    with pytest.raises(ValueError):
        validate_topology(topology(**changes))


def test_model_gate_accepts_native_step0_and_verified_step70_export(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    identity = model_identity(model, 0)
    assert identity["kind"] == "native_hf_checkpoint"
    (model / "llin_export_manifest.json").write_text(
        '{"actor_checkpoint":"/runs/formal/global_step_70/actor",'
        '"verification":{"valid":true}}',
        encoding="utf-8",
    )
    trained = model_identity(model, 70)
    assert trained["kind"] == "llin_megatron_to_hf_export"
    assert trained["policy_step"] == 70
    with pytest.raises(ValueError, match="policy step mismatch"):
        model_identity(model, 69)


def test_qwen38_wave_launcher_passes_medium_and_configurable_topology(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def fake_run(command, *, env, check):
        captured.update({"command": command, "env": env, "check": check})

    args = Namespace(
        project_root=tmp_path,
        model=tmp_path / "model",
        model_label="qwen38-27b-grpo-step70",
        policy_step=70,
        reasoning_effort="medium",
        task_batch_size=32,
        max_num_seqs=16,
        tensor_parallel_size=4,
        data_parallel_size=4,
        rollout_npus=16,
        rolling_window_trajectories=80,
        rolling_window_max_multiplier=1.25,
        ray_address="ray",
        rollout_resource="qwen38_rollout",
        monitor_first_card=2,
        monitor_num_cards=8,
    )
    monkeypatch.setattr(
        "scripts.run_qwen38_adaptive_dwh_three_wave_queue.subprocess.run", fake_run
    )
    launch_wave(args, tmp_path / "wave.parquet", tmp_path / "run", 250)
    environment = captured["env"]
    assert environment["MODEL_LABEL"] == "qwen38-27b-grpo-step70"
    assert environment["POLICY_STEP"] == "70"
    assert environment["REASONING_EFFORT"] == "medium"
    assert environment["TENSOR_PARALLEL_SIZE"] == "4"
    assert environment["DATA_PARALLEL_SIZE"] == "4"
    assert environment["ROLLOUT_NPUS"] == "16"
    assert environment["MAX_NUM_SEQS"] == "16"
    assert environment["ROLLING_WINDOW_TRAJECTORIES"] == "80"
    assert environment["MONITOR_FIRST_CARD"] == "2"
    assert captured["check"] is True


def test_standalone_runtime_wires_reasoning_and_topology_to_verl() -> None:
    runner = (ROOT / "scripts" / "run_runtime_parity_verl_standalone.py").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts" / "launch_multisandbox_dwh_standalone.sh").read_text(
        encoding="utf-8"
    )
    for fragment in (
        "data.apply_chat_template_kwargs.reasoning_effort={args.reasoning_effort}",
        "tensor_model_parallel_size={args.tensor_parallel_size}",
        "data_parallel_size={args.data_parallel_size}",
        "args.tensor_parallel_size * args.data_parallel_size != args.rollout_npus",
        'parser.add_argument("--reasoning-effort"',
        'parser.add_argument("--tensor-parallel-size"',
        'parser.add_argument("--data-parallel-size"',
        'parser.add_argument("--rollout-npus"',
    ):
        assert fragment in runner
    for argument in (
        "--reasoning-effort",
        "--tensor-parallel-size",
        "--data-parallel-size",
        "--rollout-npus",
    ):
        assert argument in launcher
