"""Custom veRL agent loop that owns the full PI workspace lifecycle."""

from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

from llin_verl.pi_workspace_tools import WORKSPACES


class PiAgentLoop(ToolAgentLoop):
    """ToolAgentLoop plus trajectory-level PI evidence capture and cleanup."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        output = await super().run(sampling_params, **kwargs)
        request_id = output.extra_fields.get("pi_workspace_request_id")
        if request_id:
            output.extra_fields.update(WORKSPACES.snapshot(str(request_id)))
            await WORKSPACES.release(str(request_id))
            output.extra_fields["pi_workspace_released"] = True
        return output

