import pytest

from llin_verl.pi_workspace_identity import workspace_binding_state


def test_timeout_snapshot_is_valid_after_workspace_was_already_released():
    fields = {
        "pi_workspace_request_id": "request-timeout",
        "pi_environment_id": "environment-1",
        "pi_workspace_released": True,
        "trajectory_timeout": True,
        "pi_tool_events": [{"ok": True}],
    }

    assert workspace_binding_state(
        fields,
        request_id="request-timeout",
        environment_id="environment-1",
    ) == "released"


def test_live_workspace_requires_exact_request_and_environment_identity():
    fields = {
        "pi_workspace_request_id": "request-live",
        "pi_environment_id": "environment-1",
        "pi_workspace_released": False,
    }

    assert workspace_binding_state(
        fields,
        request_id="request-live",
        environment_id="environment-1",
    ) == "live"
    with pytest.raises(RuntimeError, match="request identity changed"):
        workspace_binding_state(
            fields,
            request_id="request-other",
            environment_id="environment-1",
        )
    with pytest.raises(RuntimeError, match="environment identity changed"):
        workspace_binding_state(
            fields,
            request_id="request-live",
            environment_id="environment-2",
        )


def test_no_tool_trajectory_has_no_workspace_binding():
    assert workspace_binding_state(
        {
            "pi_workspace_request_id": "",
            "pi_environment_id": "environment-1",
            "pi_workspace_released": False,
        },
        request_id="request-no-tool",
        environment_id="environment-1",
    ) == "none"
