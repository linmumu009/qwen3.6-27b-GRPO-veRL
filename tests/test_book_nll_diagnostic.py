import json
from pathlib import Path

from scripts.prepare_book_nll_cases import build_windows
from scripts.run_vllm_prompt_nll import chosen_logprob
from scripts.run_vllm_book_continuation import normalize_template_token_ids


class _Logprob:
    def __init__(self, value: float) -> None:
        self.logprob = value


def test_build_windows_are_fixed_nonoverlapping_and_reproducible() -> None:
    records = [list(range(32)), list(range(100, 132))]
    first = build_windows(records, window_tokens=8, case_count=6, seed=7)
    second = build_windows(records, window_tokens=8, case_count=6, seed=7)
    assert first == second
    assert len({row["case_id"] for row in first}) == 6
    assert all(len(row["token_ids"]) == 8 for row in first)
    windows = [set(row["token_ids"]) for row in first]
    assert all(not (left & right) for i, left in enumerate(windows) for right in windows[i + 1 :])


def test_chosen_logprob_supports_vllm_objects_and_json_keys() -> None:
    assert chosen_logprob({3: _Logprob(-0.25)}, 3) == -0.25
    assert chosen_logprob({"3": {"logprob": -0.5}}, 3) == -0.5


def test_normalize_template_token_ids_accepts_batch_encoding_shape() -> None:
    assert normalize_template_token_ids({"input_ids": [[1, 2, 3]]}) == [1, 2, 3]
