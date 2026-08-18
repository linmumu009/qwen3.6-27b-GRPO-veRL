from pathlib import Path

from scripts.patch_verl_none_rollout_logprobs import MARKER, patch


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
    assert "if inputs:" in result
    assert patch(path) == "already-patched"
