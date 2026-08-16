from llin_verl.trajectory_telemetry import (
    ENQUEUED_EPOCH_NS_KEY,
    TELEMETRY_CONTRACT,
    TrajectoryTelemetry,
)


def test_timeout_telemetry_separates_queue_generation_tools_and_partial_tokens():
    telemetry = TrajectoryTelemetry.start(
        {ENQUEUED_EPOCH_NS_KEY: 1_000_000_000},
        monotonic_ns=10_000_000_000,
        epoch_ns=3_000_000_000,
    )
    telemetry.add_generation(5.0)
    telemetry.add_tools(2.0, calls=3)
    telemetry.snapshot(
        response_tokens=100,
        generated_tokens=70,
        assistant_turns=2,
        user_turns=1,
    )

    result = telemetry.finish(
        timed_out=True,
        active_generation_tokens=30,
        monotonic_ns=20_000_000_000,
    )

    assert result["trajectory_telemetry_contract"] == TELEMETRY_CONTRACT
    assert result["trajectory_queue_wait_available"] is True
    assert result["trajectory_queue_wait_seconds"] == 2.0
    assert result["trajectory_generation_seconds"] == 5.0
    assert result["trajectory_tool_seconds"] == 2.0
    assert result["trajectory_execution_seconds"] == 10.0
    assert result["trajectory_total_seconds"] == 12.0
    assert result["trajectory_overhead_seconds"] == 3.0
    assert result["trajectory_generation_calls"] == 1
    assert result["trajectory_tool_calls"] == 3
    assert result["trajectory_assistant_turns"] == 2
    assert result["trajectory_user_turns"] == 1
    assert result["trajectory_timeout_partial_response_tokens"] == 130
    assert result["trajectory_timeout_partial_generation_tokens"] == 100


def test_normal_telemetry_marks_missing_queue_clock_and_no_timeout_tokens():
    telemetry = TrajectoryTelemetry.start({}, monotonic_ns=1, epoch_ns=1)
    telemetry.snapshot(
        response_tokens=9,
        generated_tokens=7,
        assistant_turns=1,
        user_turns=0,
    )

    result = telemetry.finish(timed_out=False, monotonic_ns=1_000_000_001)

    assert result["trajectory_queue_wait_available"] is False
    assert result["trajectory_queue_wait_seconds"] == -1.0
    assert result["trajectory_execution_seconds"] == 1.0
    assert result["trajectory_total_seconds"] == 1.0
    assert result["trajectory_response_tokens_observed"] == 9
    assert result["trajectory_generated_tokens_observed"] == 7
    assert result["trajectory_timeout_partial_response_tokens"] == 0
    assert result["trajectory_timeout_partial_generation_tokens"] == 0
