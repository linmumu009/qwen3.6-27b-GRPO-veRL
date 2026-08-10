"""Custom veRL agent loop that owns the full PI workspace lifecycle."""

from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop

from llin_verl.force_final_policy import build_force_final_instruction, decide_force_final
from llin_verl.pi_workspace_tools import WORKSPACES


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

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        output = await super().run(sampling_params, **kwargs)
        defaults = {
            "force_final_triggered": False,
            "force_final_reason": "",
            "force_final_after_assistant_turns": self.force_final_after_assistant_turns,
            "force_final_remaining_response_tokens": -1,
            "force_final_tool_call_rejected": False,
            "force_final_retry_count": 0,
        }
        for key, value in defaults.items():
            output.extra_fields.setdefault(key, value)
        request_id = output.extra_fields.get("pi_workspace_request_id")
        if request_id:
            output.extra_fields.update(WORKSPACES.snapshot(str(request_id)))
            await WORKSPACES.release(str(request_id))
            output.extra_fields["pi_workspace_released"] = True
        return output

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
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

        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
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
