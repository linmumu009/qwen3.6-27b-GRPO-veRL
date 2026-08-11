#!/usr/bin/env python3
"""Pure helpers for splitting a repair SFT target into diagnostic token masks."""

from __future__ import annotations

from collections.abc import Sequence


SQLITE_COMMAND_PREFIX = "sqlite3 -json /workspace/logistics.sqlite "


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


def build_repair_component_masks(
    *,
    input_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
    rendered_text: str,
    command: str,
    turn_ranges: Sequence[tuple[int, int]],
) -> dict[str, list[int]]:
    """Split two assistant turns into structure, shell-SQL and final-answer masks."""

    if len(offsets) != len(input_ids):
        raise ValueError("offset and token lengths differ")
    if len(turn_ranges) != 2:
        raise ValueError(f"repair diagnostic requires exactly two assistant turns, got {len(turn_ranges)}")
    if not command.startswith(SQLITE_COMMAND_PREFIX):
        raise ValueError("repair command does not use the frozen sqlite3 prefix")
    if rendered_text.count(command) != 1:
        raise ValueError("repair command must occur exactly once in rendered chat")

    token_count = len(input_ids)
    tool_turn = [0] * token_count
    final_answer = [0] * token_count
    tool_start, tool_end = turn_ranges[0]
    final_start, final_end = turn_ranges[1]
    tool_turn[tool_start:tool_end] = [1] * (tool_end - tool_start)
    final_answer[final_start:final_end] = [1] * (final_end - final_start)

    command_start = rendered_text.index(command)
    sql_start = command_start + len(SQLITE_COMMAND_PREFIX)
    sql_end = command_start + len(command)
    sql_shell = token_mask_for_char_span(offsets, sql_start, sql_end)
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
