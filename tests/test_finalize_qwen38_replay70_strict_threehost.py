from argparse import Namespace
from pathlib import Path

from scripts.finalize_qwen38_replay70_strict_threehost import replay_command


def test_replay_command_sets_container_pythonpath() -> None:
    args = Namespace(
        container_project="/workspace/llin-verl-grpo",
        eval_name="replay70-test",
    )

    command = replay_command(args, "m05")

    assert command[:3] == [
        "env",
        "PYTHONPATH=/workspace/llin-verl-grpo",
        "python3",
    ]
    assert Path(command[3]).name == "replay_strict_table_reward_gate.py"
