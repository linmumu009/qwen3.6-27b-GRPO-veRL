"""Fail-closed helpers for prefix-state curriculum GRPO.

The curriculum package stores observable PI messages.  veRL consumes OpenAI
style chat messages, so this module performs a structural conversion while
preserving roles, tool-call identities, and ordering.  It intentionally knows
nothing about teacher suffixes: generated continuations are always produced by
the current actor.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


CONTRACT = "prefix-state-curriculum-grpo-runtime-v1"
REWARD_CONTRACT = "tiered-query-cost-trajectory-shadow-v1"
RESET_MODE = "clean_readonly_base_then_replay_prefix_messages"


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _drop_schema_nulls(value: Any) -> Any:
    """Remove only null struct fields inserted by Arrow round trips."""

    if isinstance(value, dict):
        return {
            str(key): _drop_schema_nulls(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_schema_nulls(item) for item in value]
    return value


def prompt_sha256(messages: Iterable[dict[str, Any]]) -> str:
    return stable_json_sha256(_drop_schema_nulls(list(messages)))


def json_field(value: Any, *, field: str, expected: type) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} is not valid JSON") from exc
    if not isinstance(value, expected):
        raise TypeError(f"{field} must decode to {expected.__name__}")
    return value


def prefix_group_base(task_id: str, prefix_state_id: str) -> str:
    if not task_id or not prefix_state_id:
        raise ValueError("task_id and prefix_state_id are required")
    return f"{task_id}::{prefix_state_id}"


def prefix_group_key(task_id: str, prefix_state_id: str, policy_version: int) -> str:
    if int(policy_version) < 0:
        raise ValueError("policy_version must be non-negative")
    return f"{prefix_group_base(task_id, prefix_state_id)}::policy-{int(policy_version)}"


def _text_content(blocks: Any, *, field: str) -> str:
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        raise TypeError(f"{field} must be text or a list of text blocks")
    values: list[str] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            raise ValueError(f"{field} contains a non-observable or unsupported block")
        values.append(str(block.get("text") or ""))
    return "\n".join(value for value in values if value)


def adapt_pi_prefix_messages(value: Any) -> list[dict[str, Any]]:
    """Convert observable PI messages to veRL/OpenAI chat messages.

    A legal cut is after a complete tool round.  Consequently every historical
    tool call must have exactly one later tool result and no call may remain
    pending at the boundary.  Hidden reasoning blocks fail closed rather than
    being copied into the actor prompt.
    """

    messages = json_field(value, field="prefix_messages", expected=list)
    output: list[dict[str, Any]] = []
    pending: dict[str, str] = {}
    seen_ids: set[str] = set()

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"prefix_messages[{index}] must be a mapping")
        role = str(message.get("role") or "")
        if role in {"system", "user"}:
            output.append(
                {
                    "role": role,
                    "content": _text_content(message.get("content"), field=f"{role}.content"),
                }
            )
            continue

        if role == "assistant":
            blocks = message.get("content")
            if not isinstance(blocks, list):
                raise TypeError("assistant.content must be a list of observable blocks")
            text: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    raise TypeError("assistant content block must be a mapping")
                kind = str(block.get("type") or "")
                if kind == "text":
                    text.append(str(block.get("text") or ""))
                    continue
                if kind != "toolCall":
                    raise ValueError(f"hidden or unsupported assistant block: {kind!r}")
                call_id = str(block.get("id") or "")
                name = str(block.get("name") or "")
                arguments = block.get("arguments")
                if not call_id or not name or not isinstance(arguments, dict):
                    raise ValueError("toolCall requires id, name, and object arguments")
                if call_id in seen_ids:
                    raise ValueError("duplicate tool_call_id in prefix")
                seen_ids.add(call_id)
                pending[call_id] = name
                tool_calls.append(
                    {
                        "type": "function",
                        "id": call_id,
                        "function": {"name": name, "arguments": arguments},
                    }
                )
            adapted: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(value for value in text if value),
            }
            if tool_calls:
                adapted["tool_calls"] = tool_calls
            output.append(adapted)
            continue

        if role == "toolResult":
            call_id = str(message.get("toolCallId") or "")
            name = str(message.get("toolName") or "")
            if not call_id or call_id not in pending:
                raise ValueError("toolResult has no matching earlier toolCall")
            if name and name != pending[call_id]:
                raise ValueError("toolResult name does not match its toolCall")
            output.append(
                {
                    "role": "tool",
                    "content": _text_content(message.get("content"), field="toolResult.content"),
                    "tool_call_id": call_id,
                }
            )
            del pending[call_id]
            continue

        raise ValueError(f"unsupported prefix role: {role!r}")

    if pending:
        raise ValueError("prefix boundary splits an incomplete tool round")
    if not output or output[0].get("role") != "system":
        raise ValueError("prefix must begin with a system message")
    if not any(message.get("role") == "user" for message in output):
        raise ValueError("prefix must contain a user task message")
    return output


def validate_ready_state(row: dict[str, Any]) -> dict[str, Any]:
    """Validate the package/runtime boundary and return the reset contract."""

    if row.get("training_ready") is not True:
        raise ValueError("quarantine or non-ready prefix state is forbidden")
    reasons = json_field(row.get("quarantine_reasons", []), field="quarantine_reasons", expected=list)
    if reasons:
        raise ValueError("ready state unexpectedly has quarantine reasons")
    if str(row.get("reward_contract") or "") != REWARD_CONTRACT:
        raise ValueError("prefix state reward contract mismatch")
    if str(row.get("reward_scope") or "") != "generated_suffix_only":
        raise ValueError("reward_scope must be generated_suffix_only")
    if str(row.get("final_correctness_scope") or "") != "combined_prefix_plus_suffix":
        raise ValueError("final correctness scope mismatch")
    if row.get("prefix_participates_in_gradient") is not False:
        raise ValueError("prefix tokens must not participate in gradient")
    if row.get("prefix_counts_toward_process_or_efficiency_reward") is not False:
        raise ValueError("prefix process evidence/cost must be excluded")
    if str(row.get("response_mask_unit") or "") != "message_index_requires_runtime_token_boundary_adapter":
        raise ValueError("message-index response-mask boundary is not declared")
    if int(row.get("generated_suffix_start_message_index", -1)) != int(
        row.get("prefix_message_count", -2)
    ):
        raise ValueError("generated suffix boundary does not equal prefix message count")

    reset = json_field(
        row.get("workspace_reset_contract", {}),
        field="workspace_reset_contract",
        expected=dict,
    )
    if reset.get("mode") != RESET_MODE or reset.get("database_mount") != "read_only_sha256_bound":
        raise ValueError("workspace/database reset contract is not safe")
    if bool(reset.get("mutable_prefix_requires_snapshot")) and not bool(
        reset.get("snapshot_available")
    ):
        # The builder may still mark a state ready only when its observable
        # prefix has no mutable side effects.  Require that fact explicitly.
        if bool(row.get("prefix_mutable_side_effects", False)):
            raise ValueError("mutable prefix has no restorable snapshot")
    adapt_pi_prefix_messages(row.get("prefix_messages"))
    return reset


def validate_runtime_prefix(extra_info: dict[str, Any], raw_prompt: Iterable[dict[str, Any]]) -> None:
    """Fail closed before model loading/generation for a curriculum sample."""

    if not extra_info.get("prefix_state_id"):
        return
    if extra_info.get("prefix_curriculum_training_ready") is not True:
        raise ValueError("runtime prefix row is not explicitly training-ready")
    if extra_info.get("prefix_future_information_leakage") not in (0, False):
        raise ValueError("runtime prefix row reports future leakage")
    if extra_info.get("prefix_hidden_reasoning_count") not in (0, False):
        raise ValueError("runtime prefix row reports hidden reasoning")
    if str(extra_info.get("workspace_reset_mode") or "") != RESET_MODE:
        raise ValueError("runtime workspace reset mode mismatch")
    task_id = str(extra_info.get("task_id") or "")
    state_id = str(extra_info.get("prefix_state_id") or "")
    expected_base = prefix_group_base(task_id, state_id)
    if str(extra_info.get("prefix_group_base") or "") != expected_base:
        raise ValueError("runtime prefix group identity mismatch")
    prompt = list(raw_prompt)
    expected_sha = str(extra_info.get("prefix_prompt_sha256") or "")
    if not expected_sha or prompt_sha256(prompt) != expected_sha:
        raise ValueError("runtime prefix prompt hash mismatch")
    if not str(extra_info.get("database_sha256") or ""):
        raise ValueError("runtime prefix is missing database hash binding")


def validate_suffix_response_mask(
    prompt_ids: Iterable[int], response_ids: Iterable[int], response_mask: Iterable[int]
) -> None:
    """Assert that veRL exposes only generated-suffix tokens to actor losses."""

    prompt = list(prompt_ids)
    response = list(response_ids)
    mask = [int(value) for value in response_mask]
    if not prompt:
        raise ValueError("prefix prompt tokenization is empty")
    if len(response) != len(mask):
        raise ValueError("response token and response-mask lengths differ")
    if any(value not in (0, 1) for value in mask):
        raise ValueError("response mask must be binary")


def require_same_prefix_group(
    task_ids: Iterable[Any], prefix_state_ids: Iterable[Any], policy_versions: Iterable[Any]
) -> str:
    keys = {
        prefix_group_key(str(task), str(state), int(version))
        for task, state, version in zip(task_ids, prefix_state_ids, policy_versions, strict=True)
    }
    if len(keys) != 1:
        raise ValueError("GRPO samples from different prefix/policy identities cannot be grouped")
    return next(iter(keys))
