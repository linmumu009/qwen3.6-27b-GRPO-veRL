import json
from pathlib import Path

from scripts.analyze_accuracy_gate import rollout_signal


def _write_group(path: Path, correctness: list[int]) -> None:
    rows = [
        {
            "input": f"prompt-{path.stem}",
            "final_answer_correct": value,
        }
        for value in correctness
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_rollout_signal_filters_to_requested_policy_window(tmp_path: Path):
    _write_group(tmp_path / "122.jsonl", [0] * 8)
    _write_group(tmp_path / "126.jsonl", [0, 1] * 4)
    _write_group(tmp_path / "127.jsonl", [0, 1] * 4)

    signal = rollout_signal(
        tmp_path,
        expected_group_size=8,
        min_rollout_step=122,
        max_rollout_step=126,
    )

    assert signal["group_count"] == 2
    assert signal["valid_group_count"] == 2
    assert signal["mixed_correct_group_count"] == 1
