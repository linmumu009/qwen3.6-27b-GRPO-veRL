from __future__ import annotations

import shlex

import pytest

from scripts.teacher_forced_component_masks import (
    SQLITE_COMMAND_PREFIX,
    build_first_action_component_masks,
)


def _inputs(rendered: str) -> tuple[list[int], list[tuple[int, int]]]:
    return list(range(len(rendered))), [
        (index, index + 1) for index in range(len(rendered))
    ]


def test_first_action_masks_partition_one_assistant_tool_turn() -> None:
    sql = "SELECT SUM(amount) FROM metric"
    command = SQLITE_COMMAND_PREFIX + shlex.quote(sql)
    rendered = "reasoning " + command + " closing"
    input_ids, offsets = _inputs(rendered)

    masks = build_first_action_component_masks(
        input_ids=input_ids,
        offsets=offsets,
        rendered_text=rendered,
        command=command,
        turn_ranges=[(0, len(rendered))],
    )

    assert sum(masks["sql_shell_mask"]) == len(sql)
    assert sum(masks["tool_structure_mask"]) > 0
    assert [
        int(structure or sql_token)
        for structure, sql_token in zip(
            masks["tool_structure_mask"], masks["sql_shell_mask"], strict=True
        )
    ] == masks["tool_turn_mask"]
    assert not any(
        structure and sql_token
        for structure, sql_token in zip(
            masks["tool_structure_mask"], masks["sql_shell_mask"], strict=True
        )
    )


def test_first_action_masks_fail_when_sql_is_outside_assistant_range() -> None:
    sql = "SELECT 1"
    command = SQLITE_COMMAND_PREFIX + shlex.quote(sql)
    rendered = "prefix " + command
    input_ids, offsets = _inputs(rendered)

    with pytest.raises(ValueError, match="outside the supervised assistant turn"):
        build_first_action_component_masks(
            input_ids=input_ids,
            offsets=offsets,
            rendered_text=rendered,
            command=command,
            turn_ranges=[(0, len("prefix "))],
        )
