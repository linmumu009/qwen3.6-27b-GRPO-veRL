#!/usr/bin/env python3
"""Pure helpers for splitting a repair SFT target into diagnostic token masks."""

from __future__ import annotations

from collections.abc import Sequence
import shlex
from typing import Any


SQLITE_COMMAND_PREFIX = "sqlite3 -json /workspace/logistics.sqlite "


def normalize_assistant_turn_indices(value: Any, turn_count: int) -> list[int]:
    """Normalize an optional parquet-backed list of supervised assistant turns."""

    if turn_count <= 0:
        raise ValueError("assistant turn count must be positive")
    if value is None:
        return list(range(turn_count))
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("supervised_assistant_turn_indices must be a list")
    indices = [int(item) for item in value]
    if not indices:
        raise ValueError("at least one assistant turn must be supervised")
    if indices != sorted(set(indices)):
        raise ValueError("supervised assistant turn indices must be unique and sorted")
    if indices[0] < 0 or indices[-1] >= turn_count:
        raise ValueError("supervised assistant turn index is out of range")
    return indices


def assistant_mask_from_ranges(
    token_count: int,
    turn_ranges: Sequence[tuple[int, int]],
    supervised_turn_indices: Sequence[int],
) -> list[int]:
    """Build a token mask for selected assistant turns only."""

    mask = [0] * token_count
    selected = set(int(index) for index in supervised_turn_indices)
    if not selected:
        raise ValueError("selected assistant turn set is empty")
    if min(selected) < 0 or max(selected) >= len(turn_ranges):
        raise ValueError("selected assistant turn index is out of range")
    for index, (start, end) in enumerate(turn_ranges):
        if not 0 <= start < end <= token_count:
            raise ValueError("invalid assistant turn token range")
        if index in selected:
            mask[start:end] = [1] * (end - start)
    return mask


def build_sql_weighted_loss_mask(
    *,
    tool_structure_mask: Sequence[int],
    sql_shell_mask: Sequence[int],
    final_answer_mask: Sequence[int],
    tool_structure_weight: float,
    sql_payload_weight: float,
    final_answer_weight: float,
) -> list[float]:
    """Compose a positive weighted assistant mask from disjoint components."""

    lengths = {len(tool_structure_mask), len(sql_shell_mask), len(final_answer_mask)}
    if len(lengths) != 1:
        raise ValueError("weighted loss component masks have different lengths")
    weights = (tool_structure_weight, sql_payload_weight, final_answer_weight)
    if any(not 0 < weight <= 32 for weight in weights):
        raise ValueError("component weights must be in (0, 32]")

    output: list[float] = []
    for tool, sql, final in zip(
        tool_structure_mask, sql_shell_mask, final_answer_mask, strict=True
    ):
        if int(bool(tool)) + int(bool(sql)) + int(bool(final)) > 1:
            raise ValueError("weighted loss component masks overlap")
        output.append(
            float(tool) * tool_structure_weight
            + float(sql) * sql_payload_weight
            + float(final) * final_answer_weight
        )
    if not any(output):
        raise ValueError("weighted assistant loss mask is empty")
    return output


def emphasize_critical_sql_token(
    *,
    weighted_loss_mask: Sequence[float],
    sql_shell_mask: Sequence[int],
    critical_sql_token_offset: int,
    critical_weight: float,
) -> tuple[list[float], list[int]]:
    """Override exactly one semantic SQL target token by its in-mask offset."""

    if len(weighted_loss_mask) != len(sql_shell_mask):
        raise ValueError("critical-token loss and SQL masks have different lengths")
    if not 0 < critical_weight <= 64:
        raise ValueError("critical SQL token weight must be in (0, 64]")
    positions = [index for index, value in enumerate(sql_shell_mask) if value]
    offset = int(critical_sql_token_offset)
    if not 0 <= offset < len(positions):
        raise ValueError("critical SQL token offset is out of range")
    critical_position = positions[offset]
    output = [float(value) for value in weighted_loss_mask]
    output[critical_position] = critical_weight
    critical_mask = [0] * len(output)
    critical_mask[critical_position] = 1
    return output, critical_mask


def find_all(sequence: Sequence[int], needle: Sequence[int]) -> list[int]:
    if not needle:
        raise ValueError("token marker must not be empty")
    return [
        index
        for index in range(len(sequence) - len(needle) + 1)
        if list(sequence[index : index + len(needle)]) == list(needle)
    ]


def assistant_turn_ranges(
    input_ids: Sequence[int],
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
    expected_turns: int,
) -> list[tuple[int, int]]:
    """Return half-open assistant-body ranges, including each closing marker."""

    prefix_positions = find_all(input_ids, prefix_ids)
    if len(prefix_positions) != expected_turns:
        raise ValueError(
            f"assistant marker count mismatch: expected {expected_turns}, found {len(prefix_positions)}"
        )

    ranges: list[tuple[int, int]] = []
    cursor = 0
    for prefix_position in prefix_positions:
        body_start = prefix_position + len(prefix_ids)
        if body_start < cursor:
            raise ValueError("assistant turn ranges overlap")
        suffix_candidates = find_all(input_ids[body_start:], suffix_ids)
        if not suffix_candidates:
            raise ValueError("assistant turn has no closing token")
        suffix_position = body_start + suffix_candidates[0]
        turn_end = suffix_position + len(suffix_ids)
        ranges.append((body_start, turn_end))
        cursor = turn_end
    return ranges


def token_mask_for_char_span(
    offsets: Sequence[Sequence[int]], char_start: int, char_end: int
) -> list[int]:
    if not 0 <= char_start < char_end:
        raise ValueError(f"invalid character span: {char_start}:{char_end}")
    mask: list[int] = []
    for offset in offsets:
        token_start, token_end = int(offset[0]), int(offset[1])
        overlaps = token_end > token_start and token_end > char_start and token_start < char_end
        mask.append(int(overlaps))
    if not any(mask):
        raise ValueError("character span did not align to any token")
    return mask


def semantic_sql_shell_char_spans(command: str) -> list[tuple[int, int]]:
    """Map decoded SQL characters to shell-command spans, excluding wrapper quotes."""

    if not command.startswith(SQLITE_COMMAND_PREFIX):
        raise ValueError("repair command does not use the frozen sqlite3 prefix")
    shell_payload = command[len(SQLITE_COMMAND_PREFIX) :]
    parts = shlex.split(command)
    if not parts:
        raise ValueError("repair command cannot be parsed")
    sql = parts[-1]
    if shell_payload != shlex.quote(sql):
        raise ValueError("repair SQL shell payload is not canonical shlex.quote output")

    payload_start = len(SQLITE_COMMAND_PREFIX)
    if shell_payload == sql:
        return [(payload_start, payload_start + len(sql))]
    if not shell_payload.startswith("'") or not shell_payload.endswith("'"):
        raise ValueError("unsafe SQL payload is not enclosed by canonical shell quotes")

    spans: list[tuple[int, int]] = []
    cursor = 1
    for character in sql:
        encoded = "'\"'\"'" if character == "'" else character
        if shell_payload[cursor : cursor + len(encoded)] != encoded:
            raise ValueError("cannot align decoded SQL to shell-quoted payload")
        spans.append(
            (
                payload_start + cursor,
                payload_start + cursor + len(encoded),
            )
        )
        cursor += len(encoded)
    if cursor != len(shell_payload) - 1:
        raise ValueError("shell-quoted SQL alignment did not consume the payload")
    return spans


def token_mask_for_char_spans(
    offsets: Sequence[Sequence[int]], char_spans: Sequence[tuple[int, int]]
) -> list[int]:
    if not char_spans:
        raise ValueError("character span list is empty")
    mask: list[int] = []
    for offset in offsets:
        token_start, token_end = int(offset[0]), int(offset[1])
        overlaps = token_end > token_start and any(
            token_end > span_start and token_start < span_end
            for span_start, span_end in char_spans
        )
        mask.append(int(overlaps))
    if not any(mask):
        raise ValueError("character spans did not align to any token")
    return mask


def build_repair_component_masks(
    *,
    input_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
    rendered_text: str,
    command: str,
    turn_ranges: Sequence[tuple[int, int]],
    tool_turn_index: int = 0,
    final_answer_turn_index: int = 1,
) -> dict[str, list[int]]:
    """Split selected correction and final turns into disjoint component masks."""

    if len(offsets) != len(input_ids):
        raise ValueError("offset and token lengths differ")
    if tool_turn_index == final_answer_turn_index:
        raise ValueError("tool and final assistant turns must differ")
    if not 0 <= tool_turn_index < len(turn_ranges):
        raise ValueError("tool assistant turn index is out of range")
    if not 0 <= final_answer_turn_index < len(turn_ranges):
        raise ValueError("final assistant turn index is out of range")
    if not command.startswith(SQLITE_COMMAND_PREFIX):
        raise ValueError("repair command does not use the frozen sqlite3 prefix")
    if rendered_text.count(command) != 1:
        raise ValueError("repair command must occur exactly once in rendered chat")

    token_count = len(input_ids)
    tool_turn = [0] * token_count
    final_answer = [0] * token_count
    tool_start, tool_end = turn_ranges[tool_turn_index]
    final_start, final_end = turn_ranges[final_answer_turn_index]
    tool_turn[tool_start:tool_end] = [1] * (tool_end - tool_start)
    final_answer[final_start:final_end] = [1] * (final_end - final_start)

    command_start = rendered_text.index(command)
    sql_spans = [
        (command_start + start, command_start + end)
        for start, end in semantic_sql_shell_char_spans(command)
    ]
    sql_shell = token_mask_for_char_spans(offsets, sql_spans)
    if any(sql and not tool for sql, tool in zip(sql_shell, tool_turn, strict=True)):
        raise ValueError("SQL shell payload is not fully contained in the first assistant turn")

    tool_structure = [
        int(tool and not sql) for tool, sql in zip(tool_turn, sql_shell, strict=True)
    ]
    if not any(tool_structure):
        raise ValueError("tool structure mask is empty")
    if any(tool and final for tool, final in zip(tool_turn, final_answer, strict=True)):
        raise ValueError("tool and final assistant masks overlap")

    return {
        "tool_turn_mask": tool_turn,
        "tool_structure_mask": tool_structure,
        "sql_shell_mask": sql_shell,
        "final_answer_mask": final_answer,
    }
