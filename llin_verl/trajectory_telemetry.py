"""Dependency-free per-trajectory timing state for PI-Agent rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import time


TELEMETRY_CONTRACT = "llin-pi-trajectory-telemetry-v1"
ENQUEUED_EPOCH_NS_KEY = "llin_trajectory_enqueued_epoch_ns"


@dataclass
class TrajectoryTelemetry:
    """Accumulate one trajectory's queue, generation, tool and wall timings."""

    started_monotonic_ns: int
    started_epoch_ns: int
    enqueued_epoch_ns: int | None = None
    generation_seconds: float = 0.0
    tool_seconds: float = 0.0
    generation_calls: int = 0
    tool_calls: int = 0
    response_tokens_observed: int = 0
    generated_tokens_observed: int = 0
    timeout_active_generation_tokens: int = 0
    assistant_turns: int = 0
    user_turns: int = 0

    @classmethod
    def start(
        cls,
        extra_info: dict | None,
        *,
        monotonic_ns: int | None = None,
        epoch_ns: int | None = None,
    ) -> "TrajectoryTelemetry":
        enqueued = None
        if isinstance(extra_info, dict):
            candidate = extra_info.get(ENQUEUED_EPOCH_NS_KEY)
            try:
                enqueued = int(candidate) if candidate is not None else None
            except (TypeError, ValueError):
                enqueued = None
        return cls(
            started_monotonic_ns=time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
            started_epoch_ns=time.time_ns() if epoch_ns is None else epoch_ns,
            enqueued_epoch_ns=enqueued,
        )

    def add_generation(self, elapsed_seconds: float) -> None:
        self.generation_seconds += max(0.0, float(elapsed_seconds))
        self.generation_calls += 1

    def add_tools(self, elapsed_seconds: float, calls: int) -> None:
        self.tool_seconds += max(0.0, float(elapsed_seconds))
        self.tool_calls += max(0, int(calls))

    def snapshot(
        self,
        *,
        response_tokens: int,
        generated_tokens: int,
        assistant_turns: int,
        user_turns: int,
    ) -> None:
        self.response_tokens_observed = max(self.response_tokens_observed, int(response_tokens))
        self.generated_tokens_observed = max(self.generated_tokens_observed, int(generated_tokens))
        self.assistant_turns = max(self.assistant_turns, int(assistant_turns))
        self.user_turns = max(self.user_turns, int(user_turns))

    def finish(
        self,
        *,
        timed_out: bool,
        active_generation_tokens: int = 0,
        monotonic_ns: int | None = None,
    ) -> dict[str, float | int | bool | str]:
        finished_monotonic_ns = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        execution_seconds = max(
            0.0,
            (finished_monotonic_ns - self.started_monotonic_ns) / 1_000_000_000,
        )
        queue_available = (
            self.enqueued_epoch_ns is not None
            and self.enqueued_epoch_ns > 0
            and self.started_epoch_ns >= self.enqueued_epoch_ns
        )
        queue_seconds = (
            (self.started_epoch_ns - self.enqueued_epoch_ns) / 1_000_000_000
            if queue_available
            else -1.0
        )
        total_seconds = execution_seconds + (queue_seconds if queue_available else 0.0)
        overhead_seconds = max(
            0.0,
            execution_seconds - self.generation_seconds - self.tool_seconds,
        )
        active_generation_tokens = max(0, int(active_generation_tokens))
        timeout_response_tokens = (
            self.response_tokens_observed + active_generation_tokens if timed_out else 0
        )
        timeout_generation_tokens = (
            self.generated_tokens_observed + active_generation_tokens if timed_out else 0
        )
        return {
            "trajectory_telemetry_contract": TELEMETRY_CONTRACT,
            "trajectory_queue_wait_available": queue_available,
            "trajectory_queue_wait_seconds": queue_seconds,
            "trajectory_generation_seconds": self.generation_seconds,
            "trajectory_tool_seconds": self.tool_seconds,
            "trajectory_execution_seconds": execution_seconds,
            "trajectory_total_seconds": total_seconds,
            "trajectory_overhead_seconds": overhead_seconds,
            "trajectory_generation_calls": self.generation_calls,
            "trajectory_tool_calls": self.tool_calls,
            "trajectory_assistant_turns": self.assistant_turns,
            "trajectory_user_turns": self.user_turns,
            "trajectory_response_tokens_observed": self.response_tokens_observed,
            "trajectory_generated_tokens_observed": self.generated_tokens_observed,
            "trajectory_timeout_partial_response_tokens": timeout_response_tokens,
            "trajectory_timeout_partial_generation_tokens": timeout_generation_tokens,
        }
