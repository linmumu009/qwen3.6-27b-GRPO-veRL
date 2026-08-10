"""Pure policy helpers for the PI agent force-final sentinel."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_FORCE_FINAL_INSTRUCTION = (
    "FINALIZATION REQUIRED. Do not call any more tools. Use only the evidence already "
    "present in the conversation and return the final answer now. Satisfy the requested "
    "time range, aggregation grain, tables, fields, units, and calculations explicitly. "
    "If the evidence is incomplete, give the best supported result and state the limitation."
)


@dataclass(frozen=True)
class ForceFinalDecision:
    triggered: bool
    reason: str | None
    remaining_response_tokens: int


def decide_force_final(
    *,
    assistant_turns: int,
    response_tokens: int,
    response_length: int,
    after_assistant_turns: int = 0,
    reserve_response_tokens: int = 0,
) -> ForceFinalDecision:
    """Return whether the next assistant turn must be terminal.

    A zero threshold disables that trigger. The helper stays independent of veRL so
    policy behavior can be tested without initializing an accelerator runtime.
    """

    remaining = max(0, int(response_length) - int(response_tokens))
    turn_triggered = after_assistant_turns > 0 and assistant_turns >= after_assistant_turns
    token_triggered = reserve_response_tokens > 0 and remaining <= reserve_response_tokens
    if turn_triggered and token_triggered:
        reason = "turn_and_token_budget"
    elif turn_triggered:
        reason = "assistant_turn_budget"
    elif token_triggered:
        reason = "response_token_budget"
    else:
        reason = None
    return ForceFinalDecision(reason is not None, reason, remaining)


def build_force_final_instruction(
    decision: ForceFinalDecision,
    *,
    assistant_turns: int,
    instruction: str | None = None,
) -> str:
    """Attach compact, auditable trigger context to the intervention prompt."""

    if not decision.triggered:
        raise ValueError("force-final instruction requested for a non-triggered decision")
    instruction = instruction or DEFAULT_FORCE_FINAL_INSTRUCTION
    return (
        f"{instruction}\n\n"
        f"[force-final reason={decision.reason}; assistant_turns={assistant_turns}; "
        f"remaining_response_tokens={decision.remaining_response_tokens}]"
    )
