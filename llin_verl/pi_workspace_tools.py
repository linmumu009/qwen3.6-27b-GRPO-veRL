"""Full PI workspace tools for veRL multi-turn trajectories.

The four tools share one copied workspace for the lifetime of an agent request.
The copy is removed by :class:`llin_verl.pi_agent_loop.PiAgentLoop` after tool
evidence has been attached to the rollout output.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from llin_verl.pi_tool_contract import (
    ENVIRONMENT_PATTERN,
    command_is_safe,
    extract_table_names,
    route_sqlite_cli,
)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cap_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n[truncated {len(value) - limit} characters]"


def truncate_tool_text(
    value: str,
    max_bytes: int,
    max_lines: int,
    *,
    keep_tail: bool,
) -> tuple[str, bool]:
    """Apply the original PI 2000-line/50KB text-output contract."""
    lines = value.splitlines(keepends=True)
    line_truncated = len(lines) > max_lines
    if line_truncated:
        lines = lines[-max_lines:] if keep_tail else lines[:max_lines]
    selected = "".join(lines)
    raw = selected.encode("utf-8")
    byte_truncated = len(raw) > max_bytes
    if byte_truncated:
        raw = raw[-max_bytes:] if keep_tail else raw[:max_bytes]
        selected = raw.decode("utf-8", errors="ignore")
    return selected, line_truncated or byte_truncated


def resolve_workspace_path(workspace: Path, raw_value: Any) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("path must be a non-empty string")
    raw = raw_value.strip()
    if raw == "/workspace":
        candidate = workspace
    elif raw.startswith("/workspace/"):
        candidate = workspace / raw[len("/workspace/") :]
    else:
        candidate = workspace / raw
    candidate = candidate.resolve()
    candidate.relative_to(workspace.resolve())
    return candidate


@dataclass
class WorkspaceState:
    request_id: str
    environment_id: str
    path: Path
    created_at: float = field(default_factory=time.monotonic)
    events: list[dict[str, Any]] = field(default_factory=list)


class WorkspaceRegistry:
    """Per-process registry shared by the four PI tool instances."""

    def __init__(self) -> None:
        self._states: dict[str, WorkspaceState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, request_id: str) -> asyncio.Lock:
        return self._locks.setdefault(request_id, asyncio.Lock())

    async def ensure(
        self,
        request_id: str,
        environment_id: str,
        lower_root: Path,
        run_root: Path,
        run_tag: str,
    ) -> WorkspaceState:
        if not request_id:
            raise ValueError("agent request is missing request_id")
        if not ENVIRONMENT_PATTERN.fullmatch(environment_id):
            raise ValueError(f"invalid PI environment_id: {environment_id!r}")
        async with self._lock(request_id):
            existing = self._states.get(request_id)
            if existing is not None:
                if existing.environment_id != environment_id:
                    raise ValueError("one trajectory cannot switch PI environments")
                return existing

            root = lower_root.resolve(strict=True)
            source = (root / environment_id).resolve(strict=True)
            source.relative_to(root)
            if not source.is_dir():
                raise FileNotFoundError(f"PI environment does not exist: {environment_id}")

            parent = (run_root / run_tag).resolve()
            parent.mkdir(parents=True, exist_ok=True)
            try:
                parent.chmod(0o711)
            except OSError:
                pass
            destination = (parent / request_id).resolve()
            destination.relative_to(parent)
            if destination.exists():
                raise FileExistsError(f"PI rollout workspace already exists: {destination}")
            await asyncio.to_thread(
                shutil.copytree,
                source,
                destination,
                copy_function=shutil.copy2,
                symlinks=True,
            )
            state = WorkspaceState(request_id, environment_id, destination)
            self._states[request_id] = state
            return state

    def record(self, request_id: str, event: dict[str, Any]) -> None:
        self._states[request_id].events.append(event)

    def snapshot(self, request_id: str) -> dict[str, Any]:
        state = self._states.get(request_id)
        if state is None:
            return {}
        events = list(state.events)
        return {
            "pi_workspace_request_id": request_id,
            "pi_environment_id": state.environment_id,
            "pi_tool_events": events,
            "pi_tool_event_contract": "runtime-captured-structured-tool-events-v2",
            "pi_tool_call_count": len(events),
            "pi_tool_success_count": sum(bool(event.get("ok")) for event in events),
            "pi_workspace_elapsed_seconds": round(time.monotonic() - state.created_at, 6),
        }

    async def release(self, request_id: str) -> None:
        state = self._states.pop(request_id, None)
        self._locks.pop(request_id, None)
        if state is not None and state.path.exists():
            await asyncio.to_thread(shutil.rmtree, state.path)


WORKSPACES = WorkspaceRegistry()


class PiWorkspaceTool(BaseTool):
    """One operation from the production PI ``bash/read/write/edit`` toolset."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.operation = str(config.get("operation") or self.name)
        if self.operation not in {"bash", "read", "write", "edit"}:
            raise ValueError(f"unsupported PI operation: {self.operation}")
        self.lower_root = Path(
            os.environ.get("PI_AGENT_SANDBOX_LOWER")
            or config.get("sandbox_root", "/pi_sandbox")
        )
        self.run_root = Path(config.get("run_root", "/workspace/grpo_run/pi_workspaces"))
        self.run_tag = str(config.get("run_tag") or os.environ.get("PI_AGENT_RUN_TAG", "unscoped"))
        self.max_tool_timeout = int(config.get("max_tool_timeout", 60))
        self.max_tool_output = int(config.get("max_tool_output", 50 * 1024))
        self.max_tool_lines = int(config.get("max_tool_lines", 2000))

    @staticmethod
    def _create_kwargs(agent_data: Any, name: str) -> dict[str, Any]:
        tool_kwargs = (getattr(agent_data, "tools_kwargs", {}) or {}).get(name, {})
        value = tool_kwargs.get("create_kwargs", {})
        return value if isinstance(value, dict) else {}

    async def _state(self, agent_data: Any) -> WorkspaceState:
        kwargs = self._create_kwargs(agent_data, self.name)
        environment_id = str(kwargs.get("environment_id") or "")
        state = await WORKSPACES.ensure(
            str(agent_data.request_id),
            environment_id,
            self.lower_root,
            self.run_root,
            self.run_tag,
        )
        agent_data.extra_fields["pi_workspace_request_id"] = state.request_id
        agent_data.extra_fields["pi_environment_id"] = state.environment_id
        return state

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        del instance_id
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            raise ValueError("PI workspace tools require AgentData")
        state = await self._state(agent_data)
        started = time.monotonic()
        try:
            response, ok = await asyncio.to_thread(self._execute_sync, state.path, parameters)
        except Exception as exc:
            response = f"{type(exc).__name__}: {exc}"
            ok = False
        elapsed = time.monotonic() - started
        command = str(parameters.get("command") or "") if self.operation == "bash" else ""
        event = {
            "name": self.operation,
            "arguments": parameters,
            "ok": ok,
            "elapsed_seconds": round(elapsed, 6),
            "tables": extract_table_names(command),
            "response_preview": response[:4000],
            "observed_tool_response": True,
            "call_parse_valid": True,
            "assistant_turn_index": int(getattr(agent_data, "assistant_turns", 0) or 0),
        }
        WORKSPACES.record(state.request_id, event)
        agent_data.extra_fields["pi_tool_events"] = list(WORKSPACES._states[state.request_id].events)
        metrics = {"pi_tool_ok": float(ok), "pi_tool_elapsed_seconds": elapsed}
        return ToolResponse(text=response), 0.0, metrics

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        # veRL releases a BaseTool after every call. The shared workspace is
        # intentionally released once by PiAgentLoop after the whole trajectory.
        del instance_id, kwargs

    def _execute_sync(self, workspace: Path, parameters: dict[str, Any]) -> tuple[str, bool]:
        if self.operation == "bash":
            return self._bash(workspace, parameters)
        if self.operation == "read":
            return self._read(workspace, parameters), True
        if self.operation == "write":
            return self._write(workspace, parameters), True
        return self._edit(workspace, parameters), True

    def _bash(self, workspace: Path, parameters: dict[str, Any]) -> tuple[str, bool]:
        command = parameters.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("bash.command must be a non-empty string")
        if len(command) > 100_000:
            raise ValueError("bash.command is too long")
        if not command_is_safe(command):
            raise PermissionError("network, host escape, and destructive commands are disabled")
        timeout = min(max(int(parameters.get("timeout", self.max_tool_timeout)), 1), self.max_tool_timeout)
        mapped = route_sqlite_cli(command).replace("/workspace", str(workspace))
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "HOME": str(workspace),
            "PWD": str(workspace),
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "socks5://127.0.0.1:9",
            "NO_PROXY": "",
        }
        try:
            process = subprocess.run(
                ["bash", "-lc", mapped],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or "") + (exc.stderr or "")
            output = f"{partial}\nCommand timed out after {timeout} seconds.".strip()
            return self._format_bash_output(workspace, output), False
        output = process.stdout
        if process.stderr:
            output += ("\n" if output else "") + process.stderr
        if not output.strip():
            output = f"Command completed with exit code {process.returncode}."
        return self._format_bash_output(workspace, output), process.returncode == 0

    def _format_bash_output(self, workspace: Path, output: str) -> str:
        visible, truncated = truncate_tool_text(
            output,
            self.max_tool_output,
            self.max_tool_lines,
            keep_tail=True,
        )
        if not truncated:
            return visible
        output_dir = workspace / ".pi_tool_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"bash-{time.time_ns()}.log"
        output_path.write_text(output, encoding="utf-8", errors="replace")
        relative = output_path.relative_to(workspace).as_posix()
        return f"[output truncated; full output saved to /workspace/{relative}]\n{visible}"

    def _read(self, workspace: Path, parameters: dict[str, Any]) -> str:
        path = resolve_workspace_path(workspace, parameters.get("path"))
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {parameters.get('path')}")
        # Boss PI declares a 1-indexed offset; omitting it starts at line 1.
        offset = max(int(parameters.get("offset", 1)) - 1, 0)
        limit = min(max(int(parameters.get("limit", 2000)), 1), 10_000)
        with path.open(encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        visible, truncated = truncate_tool_text(
            "".join(lines[offset : offset + limit]),
            self.max_tool_output,
            self.max_tool_lines,
            keep_tail=False,
        )
        if truncated:
            next_offset = offset + visible.count("\n") + 1
            visible += f"\n[output truncated; continue with offset={next_offset}]"
        return visible

    def _write(self, workspace: Path, parameters: dict[str, Any]) -> str:
        path = resolve_workspace_path(workspace, parameters.get("path"))
        content = parameters.get("content")
        if not isinstance(content, str):
            raise TypeError("write.content must be a string")
        size = len(content.encode("utf-8"))
        if size > 2_000_000:
            raise ValueError("write.content exceeds 2 MB")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        return f"Wrote {size} bytes to /workspace/{path.relative_to(workspace)}."

    def _edit(self, workspace: Path, parameters: dict[str, Any]) -> str:
        path = resolve_workspace_path(workspace, parameters.get("path"))
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {parameters.get('path')}")
        text = path.read_text(encoding="utf-8", errors="strict")
        edits = parameters.get("edits")
        if edits is None:
            edits = [{"oldText": parameters.get("oldText"), "newText": parameters.get("newText")}]
        if not isinstance(edits, list) or not edits:
            raise ValueError("edit requires a non-empty edits array")
        for edit in edits:
            if not isinstance(edit, dict):
                raise TypeError("each edit must be an object")
            old, new = edit.get("oldText"), edit.get("newText")
            if not isinstance(old, str) or not old:
                raise ValueError("oldText must be a non-empty string")
            if not isinstance(new, str):
                raise TypeError("newText must be a string")
            count = text.count(old)
            if count != 1:
                raise ValueError(f"oldText must match exactly once; matched {count} times")
            text = text.replace(old, new, 1)
        if len(text.encode("utf-8")) > 2_000_000:
            raise ValueError("edited file exceeds 2 MB")
        path.write_text(text, encoding="utf-8", newline="\n")
        return f"Applied {len(edits)} edit(s) to /workspace/{path.relative_to(workspace)}."
