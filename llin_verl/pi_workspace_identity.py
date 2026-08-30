"""Pure request/environment identity checks for PI workspace lifecycles."""

from __future__ import annotations

from typing import Any, Mapping


def workspace_binding_state(
    fields: Mapping[str, Any],
    *,
    request_id: str,
    environment_id: str,
) -> str:
    """Validate persisted workspace identity and classify its lifecycle state.

    A timed-out trajectory snapshots and releases its workspace while building
    the shape-preserving UNKNOWN output.  Its persisted snapshot remains valid
    evidence, but must not be looked up in the live registry a second time.
    """

    workspace_request_id = str(fields.get("pi_workspace_request_id") or "")
    if not workspace_request_id:
        return "none"
    if workspace_request_id != request_id:
        raise RuntimeError("workspace request identity changed before reward")

    workspace_environment_id = str(fields.get("pi_environment_id") or "")
    if workspace_environment_id != environment_id:
        raise RuntimeError("workspace environment identity changed before reward")
    if bool(fields.get("pi_workspace_released")):
        return "released"
    return "live"
