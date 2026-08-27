from __future__ import annotations

import json
from pathlib import Path
import re

from scripts.render_logistics_sandbox_example_animation import WORLD_PHASES, STAGES, main, render_html


def _names(items: object) -> set[str]:
    return {item.name for item in items}  # type: ignore[attr-defined]


def test_replay_models_causal_artifact_handoffs_and_real_repair() -> None:
    assert [stage.key for stage in STAGES] == [
        "intake",
        "prd",
        "factor",
        "taxonomy",
        "schema",
        "data",
        "dwh_tasks",
        "catalog",
        "documents",
        "kb_tasks",
        "hybrid",
        "freeze",
        "runner",
    ]
    assert "input/运营分析.md" in _names(STAGES[0].outputs) & _names(STAGES[1].inputs)
    assert {"prd_运营分析.md", "coverage.json", "filled_template.json"} <= (
        _names(STAGES[1].outputs) & _names(STAGES[2].inputs)
    )
    assert {"entities.json", "states.json", "tools_actions.json"} <= (
        _names(STAGES[2].outputs) & _names(STAGES[3].inputs)
    )
    factor = STAGES[2]
    assert factor.duration == "51:14"
    assert [op.status for op in factor.operations] == [
        "normal",
        "normal",
        "error",
        "repair",
        "success",
    ]
    assert "incident_closed_without_compensation" in factor.operations[3].detail
    assert next(item for item in factor.outputs if item.name == "factor_validation_report.json").produced_by == 4
    assert STAGES[6].handoff_type == "branch-return"
    assert STAGES[9].handoff_type == "merge"
    assert {"tasks.jsonl", "knowledge_tasks.jsonl"} <= _names(STAGES[10].inputs)
    assert STAGES[6].step == "Step 5.1"
    assert STAGES[6].title.startswith("数仓任务：")
    assert STAGES[10].step == "Step 5.3"
    assert STAGES[10].title.startswith("混合任务：")


def test_world_model_phases_explain_semantics_not_just_files() -> None:
    assert [phase.key for phase in WORLD_PHASES] == [stage.key for stage in STAGES]
    assert all(len(phase.gains) == len(stage.operations) for phase, stage in zip(WORLD_PHASES, STAGES))
    assert all(len(phase.layers) == len(stage.operations) for phase, stage in zip(WORLD_PHASES, STAGES))
    factor = WORLD_PHASES[2]
    assert factor.formula == "W = (Entities, States, Actions, Invariants)"
    assert "会变化、可行动" in factor.principle
    assert "状态、可执行动作" in factor.capability
    assert "世界里实际发生了什么" in WORLD_PHASES[6].capability
    assert "现实中的物流运行是否符合世界的规则" in WORLD_PHASES[10].capability


def test_replay_visually_builds_a_world_and_has_a_terminal_end_state() -> None:
    html = render_html()

    assert html.startswith("<!doctype html>")
    assert "从业务意图到可执行物流世界" in html
    assert "W = (Entities, States, Actions, Rules, Observations)" in html
    assert 'class="world-canvas"' in html
    assert 'data-layer="entities"' in html
    assert 'data-layer="states"' in html
    assert 'data-layer="actions"' in html
    assert 'data-layer="taxonomy"' in html
    assert 'data-layer="data"' in html
    assert 'data-layer="rules"' in html
    assert 'data-layer="hybrid"' in html
    assert "STEP 5.1<br><strong>数仓任务" in html
    assert "STEP 5.3<br>混合任务" in html
    assert "技术实现（非世界本体）" in html
    assert "animateHandoff" in html
    assert "transfer-orb" in html
    assert "completionBurst" in html
    assert "finishReplay" in html
    assert 'id="endOverlay"' in html
    assert "沙箱生成完成" in html
    assert "state.finished=true" in html
    assert "body.is-finished" in html
    assert "animation-play-state:paused!important" in html
    assert "第一次因子校验" in html
    assert "修复错误引用" in html
    assert "36,101 rows" in html
    assert "65 documents" in html
    assert "500 / 500 passed" in html
    assert 'id="inputZone"' not in html
    assert 'id="outputZone"' not in html
    assert 'class="theatre-grid"' not in html
    assert "pipelineSvg" not in html
    assert 'class="stage-card"' not in html
    assert "mobile-flow" not in html
    assert "render_sandbox_generation_animation" not in html
    assert '<script src="http' not in html
    assert '<link href="http' not in html
    assert "gold_answer" not in html
    assert "SELECT " not in html
    assert "__STAGE_DATA__" not in html
    assert "__WORLD_DATA__" not in html

    match = re.search(
        r'<script id="stageData" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert len(embedded) == 13
    assert all(stage["inputs"] and stage["operations"] and stage["outputs"] for stage in embedded)
    assert all(stage["handoff"] for stage in embedded)

    world_match = re.search(
        r'<script id="worldData" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert world_match is not None
    world_embedded = json.loads(world_match.group(1))
    assert [phase["key"] for phase in world_embedded] == [stage["key"] for stage in embedded]


def test_cli_renders_and_checks_exact_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "logistics-sandbox-generation-theatre.html"
    monkeypatch.setattr("sys.argv", ["render", "--output", str(output)])
    assert main() == 0
    assert output.read_text(encoding="utf-8") == render_html()

    monkeypatch.setattr("sys.argv", ["render", "--output", str(output), "--check"])
    assert main() == 0

    output.write_text("stale", encoding="utf-8")
    assert main() == 1
