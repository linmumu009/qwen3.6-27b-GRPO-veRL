"""Custom veRL agent loop that owns the full PI workspace lifecycle."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from contextvars import ContextVar, Token
import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop

from llin_verl.force_final_policy import build_force_final_instruction, decide_force_final
from llin_verl.prefix_state_curriculum import (
    validate_runtime_prefix,
    validate_suffix_response_mask,
)
from llin_verl.pi_workspace_tools import WORKSPACES
from llin_verl.trajectory_telemetry import TrajectoryTelemetry


_TRAJECTORY_TELEMETRY: ContextVar[TrajectoryTelemetry | None] = ContextVar(
    "llin_trajectory_telemetry",
    default=None,
)


class PiAgentLoop(ToolAgentLoop):
    """ToolAgentLoop plus trajectory-level PI evidence capture and cleanup."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        multi_turn = self.rollout_config.multi_turn
        self.force_final_after_assistant_turns = int(
            getattr(multi_turn, "force_final_after_assistant_turns", 0) or 0
        )
        self.force_final_reserve_response_tokens = int(
            getattr(multi_turn, "force_final_reserve_response_tokens", 0) or 0
        )
        self.force_final_max_response_tokens = int(
            getattr(multi_turn, "force_final_max_response_tokens", 0) or 0
        )
        self.force_final_max_retries = int(getattr(multi_turn, "force_final_max_retries", 0) or 0)
        self.force_final_instruction = str(getattr(multi_turn, "force_final_instruction", "") or "")
        self.agent_timeout_seconds = float(getattr(multi_turn, "agent_timeout_seconds", 0) or 0)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        request_id = str(kwargs.get("__llin_request_id") or uuid4().hex)
        kwargs["__llin_request_id"] = request_id
        input_extra = kwargs.get("extra_info", {}) or {}
        if not isinstance(input_extra, dict):
            raise TypeError("trajectory extra_info must be a mapping")
        # Prefix-state rows are already adapted to native role/tool messages by
        # the private loader.  Validate that exact prompt and its clean reset
        # contract before a token is generated; quarantine rows fail closed.
        validate_runtime_prefix(input_extra, kwargs.get("raw_prompt", []))
        environment_id = str(input_extra.get("environment_id") or "")
        telemetry = TrajectoryTelemetry.start(input_extra)
        context_token: Token = _TRAJECTORY_TELEMETRY.set(telemetry)
        try:
            return await self._run_with_telemetry(
                sampling_params,
                request_id=request_id,
                environment_id=environment_id,
                telemetry=telemetry,
                **kwargs,
            )
        finally:
            _TRAJECTORY_TELEMETRY.reset(context_token)

    async def _run_with_telemetry(
        self,
        sampling_params: dict[str, Any],
        *,
        request_id: str,
        environment_id: str,
        telemetry: TrajectoryTelemetry,
        **kwargs: Any,
    ) -> AgentLoopOutput:
        """Run one trajectory with task-local telemetry state."""
        if kwargs.get("__llin_request_id") != request_id:
            raise RuntimeError("trajectory request identity changed before execution")
        if self.agent_timeout_seconds > 0:
            task = asyncio.create_task(super().run(sampling_params, **kwargs))
            done, _ = await asyncio.wait({task}, timeout=self.agent_timeout_seconds)
            if task in done:
                output = task.result()
            else:
                output = await self._abort_timed_out_trajectory(task, request_id, kwargs)
        else:
            output = await super().run(sampling_params, **kwargs)
        validate_suffix_response_mask(
            output.prompt_ids,
            output.response_ids,
            output.response_mask,
        )
        curriculum_extra = kwargs.get("extra_info", {}) or {}
        if curriculum_extra.get("prefix_state_id"):
            output.extra_fields.update(
                {
                    "prefix_state_id": str(curriculum_extra["prefix_state_id"]),
                    "prefix_group_base": str(curriculum_extra["prefix_group_base"]),
                    "prefix_prompt_sha256": str(curriculum_extra["prefix_prompt_sha256"]),
                    "prefix_history_gradient_tokens": 0,
                    "prefix_history_process_reward_events": 0,
                    "generated_suffix_response_token_count": len(output.response_ids),
                    "generated_suffix_active_assistant_token_count": sum(
                        int(value) for value in output.response_mask
                    ),
                    "generated_suffix_only_mask_verified": True,
                }
            )
        defaults = {
            # vLLM attaches these version fields to every completed request.  A
            # worker whose whole chunk times out would otherwise omit
            # ``global_steps`` entirely, and the outer DataProto.concat would
            # receive a shorter non-tensor column than the batch.  Keep the key
            # present with an explicit unknown value for timeout placeholders.
            "global_steps": None,
            "min_global_steps": None,
            "max_global_steps": None,
            "force_final_triggered": False,
            "force_final_reason": "",
            "force_final_after_assistant_turns": self.force_final_after_assistant_turns,
            "force_final_remaining_response_tokens": -1,
            "force_final_tool_call_rejected": False,
            "force_final_retry_count": 0,
            "trajectory_timeout": False,
            "trajectory_timeout_seconds": 0.0,
            "trajectory_abort_acknowledged_count": 0,
            "trajectory_abort_physical_request_count": 0,
            "trajectory_abort_error_count": 0,
            "runtime_error": False,
            "pi_tool_events": [],
            "pi_runtime_wrapper_events": [],
            "pi_tool_log_present": False,
            "pi_tool_protocol_complete": False,
            "pi_tool_event_source": "runtime_structured_pi_workspace",
            "pi_tool_event_contract": "runtime-captured-structured-tool-events-v3",
            # Persist the logical trajectory identity even when the model does
            # not call a tool.  Tool-using trajectories must bind their copied
            # workspace and every event to this exact request/environment.
            "request_id": request_id,
            "pi_trajectory_request_id": request_id,
            "pi_trajectory_environment_id": environment_id,
            # Tool-using trajectories receive these fields from the workspace
            # snapshot.  Keep the same non-tensor schema for observed no-tool
            # trajectories and timeout placeholders so AgentLoopManager can
            # concatenate chunks without losing the three-state distinction.
            "pi_workspace_request_id": "",
            "pi_environment_id": environment_id,
            "pi_tool_call_count": 0,
            "pi_tool_success_count": 0,
            "pi_workspace_elapsed_seconds": 0.0,
            "pi_workspace_released": False,
        }
        for key, value in defaults.items():
            output.extra_fields.setdefault(key, value)
        workspace_request_id = output.extra_fields.get("pi_workspace_request_id")
        if workspace_request_id:
            snapshot = WORKSPACES.snapshot(str(workspace_request_id))
            if str(snapshot.get("pi_workspace_request_id") or "") != request_id:
                raise RuntimeError("workspace request identity changed before reward")
            if str(snapshot.get("pi_environment_id") or "") != environment_id:
                raise RuntimeError("workspace environment identity changed before reward")
            output.extra_fields.update(snapshot)
            await WORKSPACES.release(str(workspace_request_id))
            output.extra_fields["pi_workspace_released"] = True
        timed_out = bool(output.extra_fields.get("trajectory_timeout"))
        # A completed no-tool trajectory is an observed model choice, not a
        # missing log.  It must be eligible for an explicit no-tool-guess FAIL.
        output.extra_fields["pi_tool_log_present"] = not timed_out
        output.extra_fields["pi_tool_protocol_complete"] = not timed_out
        if not timed_out:
            telemetry.snapshot(
                response_tokens=len(output.response_ids),
                generated_tokens=sum(int(value) for value in output.response_mask),
                assistant_turns=telemetry.assistant_turns,
                user_turns=telemetry.user_turns,
            )
        output.extra_fields.update(
            telemetry.finish(
                timed_out=timed_out,
                active_generation_tokens=telemetry.timeout_active_generation_tokens,
            )
        )
        return output

    @staticmethod
    def _telemetry() -> TrajectoryTelemetry:
        telemetry = _TRAJECTORY_TELEMETRY.get()
        if telemetry is None:
            raise RuntimeError("trajectory telemetry context is unavailable")
        return telemetry

    def _capture_telemetry(self, agent_data: AgentData) -> None:
        telemetry = self._telemetry()
        telemetry.snapshot(
            response_tokens=len(agent_data.response_mask),
            generated_tokens=sum(int(value) for value in agent_data.response_mask),
            assistant_turns=agent_data.assistant_turns,
            user_turns=agent_data.user_turns,
        )

    async def _abort_timed_out_trajectory(
        self,
        task: asyncio.Task,
        request_id: str,
        kwargs: dict[str, Any],
    ) -> AgentLoopOutput:
        """Physically abort one vLLM request and emit a shape-preserving marker."""
        abort_report: dict[str, Any]
        try:
            abort_report = await self.server_manager.abort_request(
                request_id, reset_prefix_cache=False
            )
        except Exception as exc:  # preserve the batch even if abort telemetry fails
            abort_report = {"error": f"{type(exc).__name__}: {exc}"}
        telemetry = self._telemetry()
        telemetry.timeout_active_generation_tokens = int(
            abort_report.get("partial_response_tokens", 0) or 0
        )
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task

        workspace_snapshot = WORKSPACES.snapshot(request_id)
        await WORKSPACES.release(request_id)
        output = await self._build_timeout_output(kwargs)
        output.extra_fields.update(workspace_snapshot)
        output.extra_fields.update(
            {
                "trajectory_timeout": True,
                "trajectory_timeout_seconds": self.agent_timeout_seconds,
                "trajectory_abort_acknowledged_count": int(
                    abort_report.get("acknowledged_count", 0) or 0
                ),
                "trajectory_abort_physical_request_count": int(
                    abort_report.get("physical_request_count", 0) or 0
                ),
                "trajectory_abort_error_count": int(
                    abort_report.get("error_count", 1 if abort_report.get("error") else 0)
                    or 0
                ),
                "pi_workspace_released": True,
                "pi_tool_log_present": False,
                "pi_tool_protocol_complete": False,
            }
        )
        return output

    async def _build_timeout_output(self, kwargs: dict[str, Any]) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            schemas = [
                self.tools[name].tool_schema.model_dump(exclude_unset=True, exclude_none=True)
                for name in tool_selection
                if name in self.tools
            ]
        else:
            schemas = self.tool_schemas
        if self.enable_continuous_token:
            prompt_ids = await self.ct_build_initial_tokens(messages, tools=schemas)
        else:
            prompt_ids = await self.apply_chat_template(
                messages,
                tools=schemas,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        telemetry = self._telemetry()
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=[],
            response_mask=[],
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            response_logprobs=None,
            num_turns=(
                telemetry.assistant_turns
                + telemetry.user_turns
                + 1
            ),
            metrics={"trajectory_timeout": 1},
            routed_experts=None,
            extra_fields={},
        )

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        tool_calls = min(len(agent_data.tool_calls), self.max_parallel_calls)
        started = time.monotonic()
        try:
            state = await super()._handle_processing_tools_state(agent_data)
        finally:
            self._telemetry().add_tools(
                time.monotonic() - started,
                tool_calls,
            )
            self._capture_telemetry(agent_data)
        if state != AgentState.GENERATING or getattr(agent_data, "_llin_force_final_active", False):
            return state

        decision = decide_force_final(
            assistant_turns=agent_data.assistant_turns,
            response_tokens=len(agent_data.response_mask),
            response_length=self.response_length,
            after_assistant_turns=self.force_final_after_assistant_turns,
            reserve_response_tokens=self.force_final_reserve_response_tokens,
        )
        if not decision.triggered:
            return state

        previous_messages = list(agent_data.messages)
        instruction = build_force_final_instruction(
            decision,
            assistant_turns=agent_data.assistant_turns,
            instruction=self.force_final_instruction,
        )
        agent_data.messages.append({"role": "user", "content": instruction})
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)

        if self.enable_continuous_token:
            merge_result, response_mask, response_logprobs = await self.ct_merge_non_assistant_msg(
                previous_messages,
                agent_data.messages,
                agent_data.prompt_ids,
                agent_data.response_mask,
                agent_data.response_logprobs if agent_data.response_logprobs else None,
                tools=schemas,
            )
            agent_data.prompt_ids = merge_result.token_ids
            agent_data.response_mask = response_mask
            if agent_data.response_logprobs:
                agent_data.response_logprobs = response_logprobs or []
        else:
            response_ids = await self.apply_chat_template(
                [agent_data.messages[-1]],
                remove_system_prompt=True,
            )
            response_ids = self.turn_separator + response_ids
            if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
                return AgentState.TERMINATED
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)

        agent_data.user_turns += 1
        agent_data._llin_force_final_active = True
        agent_data._llin_force_final_retries = 0
        agent_data._llin_force_final_tool_schemas = schemas
        agent_data._active_tools = {}
        agent_data._active_tool_schemas = []
        agent_data.extra_fields.update(
            {
                "force_final_triggered": True,
                "force_final_reason": decision.reason,
                "force_final_after_assistant_turns": self.force_final_after_assistant_turns,
                "force_final_remaining_response_tokens": decision.remaining_response_tokens,
            }
        )
        agent_data.metrics["force_final_triggered"] = 1
        return AgentState.GENERATING

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        if getattr(agent_data, "_llin_force_final_active", False) and self.force_final_max_response_tokens > 0:
            sampling_params = {**sampling_params}
            configured_limit = sampling_params.pop("max_new_tokens", None)
            configured_limit = sampling_params.get("max_tokens", configured_limit)
            final_limit = self.force_final_max_response_tokens
            if configured_limit is not None:
                final_limit = min(final_limit, int(configured_limit))
            sampling_params["max_tokens"] = max(1, final_limit)

        started = time.monotonic()
        try:
            state = await super()._handle_generating_state(
                agent_data,
                sampling_params,
                ignore_termination,
            )
        finally:
            self._telemetry().add_generation(time.monotonic() - started)
            self._capture_telemetry(agent_data)
        if getattr(agent_data, "_llin_force_final_active", False) and state == AgentState.PROCESSING_TOOLS:
            agent_data.extra_fields["force_final_tool_call_rejected"] = True
            agent_data.metrics["force_final_tool_call_rejected"] = 1
            retries = int(getattr(agent_data, "_llin_force_final_retries", 0))
            if self.enable_continuous_token and retries < self.force_final_max_retries:
                previous_messages = list(agent_data.messages)
                for tool_call in agent_data.tool_calls:
                    message = {
                        "role": "tool",
                        "content": "Tool execution is disabled by the finalization policy.",
                    }
                    if tool_call.tool_call_id is not None:
                        message["tool_call_id"] = tool_call.tool_call_id
                    agent_data.messages.append(message)
                agent_data.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your tool call was rejected. Do not call tools again. Return only the concise "
                            "final answer now, using evidence already present in the conversation."
                        ),
                    }
                )
                merge_result, response_mask, response_logprobs = await self.ct_merge_non_assistant_msg(
                    previous_messages,
                    agent_data.messages,
                    agent_data.prompt_ids,
                    agent_data.response_mask,
                    agent_data.response_logprobs if agent_data.response_logprobs else None,
                    tools=getattr(agent_data, "_llin_force_final_tool_schemas", self.tool_schemas),
                )
                if len(response_mask) >= self.response_length:
                    return AgentState.TERMINATED
                agent_data.prompt_ids = merge_result.token_ids
                agent_data.response_mask = response_mask
                if agent_data.response_logprobs:
                    agent_data.response_logprobs = response_logprobs or []
                retries += 1
                agent_data._llin_force_final_retries = retries
                agent_data.extra_fields["force_final_retry_count"] = retries
                agent_data.user_turns += 1
                return AgentState.GENERATING
            return AgentState.TERMINATED
        return state
