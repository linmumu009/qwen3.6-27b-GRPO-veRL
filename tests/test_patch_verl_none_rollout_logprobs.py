from pathlib import Path

from scripts.patch_verl_none_rollout_logprobs import (
    DETACH_MARKER,
    MARKER,
    V2_AGENT_BLOCK,
    normalize_param_versions,
    patch,
    patch_detach_utils,
)


def test_missing_rollout_logprobs_are_zero_loss_masked(tmp_path: Path) -> None:
    path = tmp_path / "agent_loop.py"
    path.write_text(
        """        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
""",
        encoding="utf-8",
    )
    assert patch(path) == "patched"
    result = path.read_text(encoding="utf-8")
    assert MARKER in result
    assert "torch.zeros_like(input.response_mask)" in result
    assert "torch.zeros_like(input.response_ids, dtype=torch.float32)" in result
    assert 'input.extra_fields.get("min_global_steps") is None' in result
    assert 'input.extra_fields.get("max_global_steps") is None' in result
    assert "if inputs:" in result
    assert patch(path) == "already-patched"


def test_v2_agent_patch_upgrades_to_all_rollout_metadata(tmp_path: Path) -> None:
    path = tmp_path / "agent_loop.py"
    path.write_text(V2_AGENT_BLOCK, encoding="utf-8")
    assert patch(path) == "upgraded"
    result = path.read_text(encoding="utf-8")
    assert MARKER in result
    assert "missing_rollout_metadata" in result
    assert "missing_rollout_logprobs" not in result


def test_missing_param_versions_are_pairwise_normalized_without_partial_span() -> None:
    starts, ends = normalize_param_versions([3, None, 4, None], [5, None, None, 6])
    assert starts == [3, 6, 4, 6]
    assert ends == [5, 6, 4, 6]
    assert [abs(end - start) for start, end in zip(starts, ends, strict=True)] == [2, 0, 0, 0]


def test_all_missing_param_versions_fall_back_to_zero() -> None:
    assert normalize_param_versions([None, None], [None, None]) == ([0, 0], [0, 0])


def test_detach_utils_normalizes_none_versions_before_statistics(tmp_path: Path) -> None:
    path = tmp_path / "detach_utils.py"
    path.write_text(
        '''    param_version_start = final_batch.non_tensor_batch["min_global_steps"]
    param_version_end = final_batch.non_tensor_batch["max_global_steps"]
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
''',
        encoding="utf-8",
    )
    assert patch_detach_utils(path) == "patched"
    result = path.read_text(encoding="utf-8")
    assert DETACH_MARKER in result
    assert "fallback_param_version = max(known_param_versions, default=0)" in result
    assert 'final_batch.non_tensor_batch["max_global_steps"] = param_version_end' in result
    assert patch_detach_utils(path) == "already-patched"
