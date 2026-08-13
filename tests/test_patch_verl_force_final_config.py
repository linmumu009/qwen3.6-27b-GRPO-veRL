from scripts.patch_verl_force_final_config import patch


def test_patch_adds_optional_fields_idempotently(tmp_path):
    target = tmp_path / "rollout.py"
    target.write_text(
        "from typing import Optional\n"
        "class MultiTurnConfig:\n"
        "    num_repeat_rollouts: Optional[int] = None\n",
        encoding="utf-8",
    )
    assert patch(target) == "patched"
    assert patch(target) == "already-patched"
    text = target.read_text(encoding="utf-8")
    assert "force_final_after_assistant_turns: int = 0" in text
    assert "force_final_reserve_response_tokens: int = 0" in text
    assert "force_final_max_response_tokens: int = 0" in text
    assert "force_final_max_retries: int = 0" in text
    assert "agent_timeout_seconds: float = 0.0" in text
    compile(text, str(target), "exec")


def test_patch_upgrades_existing_two_field_installation(tmp_path):
    target = tmp_path / "rollout.py"
    target.write_text(
        "class MultiTurnConfig:\n"
        "    force_final_after_assistant_turns: int = 0\n"
        "    force_final_reserve_response_tokens: int = 0\n",
        encoding="utf-8",
    )
    assert patch(target) == "patched"
    assert patch(target) == "already-patched"
    text = target.read_text(encoding="utf-8")
    assert "force_final_max_response_tokens: int = 0" in text
    assert "force_final_max_retries: int = 0" in text
    assert "agent_timeout_seconds: float = 0.0" in text
