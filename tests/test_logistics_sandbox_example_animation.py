from __future__ import annotations

import json
from pathlib import Path
import re

from scripts.render_logistics_sandbox_example_animation import SCENES, WORLD_PHASES, STAGES, main, render_html


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
    assert [scene.key for scene in SCENES] == [
        "brief",
        "mechanics",
        "reality",
        "rules",
        "questions",
        "seal",
    ]
    mapped_keys = [key for scene in SCENES for key in scene.stage_keys]
    assert len(mapped_keys) == len(set(mapped_keys)) == len(STAGES)
    assert set(mapped_keys) == {stage.key for stage in STAGES}


def test_replay_uses_one_waybill_and_one_visible_causal_chain() -> None:
    html = render_html()

    assert html.startswith("<!doctype html>")
    assert "一票运单，如何一步步长成一个物流沙箱？" in html
    assert "const navLabels=['业务描述','世界结构','现实实例','世界规则','观察任务','冻结沙箱']" in html
    assert "W-DEMO-001" in html
    assert "示意运单（非真实数据库行）" in html
    assert "输入：已经有什么" in html
    assert "变化：这一部做什么" in html
    assert "结果：世界因此获得" in html
    assert "查看 13 个工程阶段与落盘凭证（主线的可审计证据）" in html
    assert "Step 5.1 数仓任务" in html
    assert "Step 5.2 知识任务" in html
    assert "Step 5.3 混合任务" in html
    assert "实际发生了什么？" in html
    assert "规则要求什么？" in html
    assert "现实是否符合规则？" in html
    assert "实际时效 T_actual" in html
    assert "SLA 阈值 T_sla" in html
    assert "事实：实际发生了什么" in html
    assert "规则：应该怎样运行" in html
    assert "transitionTo" in html
    assert "function finish()" in html
    assert 'id="end"' in html
    assert "沙箱生成完成" in html
    assert "state.finished=true" in html
    assert "body.finished" in html
    assert "animation-play-state:paused!important" in html
    assert "第一次因子校验" in html
    assert "修复错误引用" in html
    assert "36,101" in html
    assert "65" in html
    assert "500" in html
    assert "评测者保险库" in html
    assert "模型可观察" in html
    assert 'class="world-layout"' not in html
    assert 'class="inherit-panel"' not in html
    assert 'class="gain-list"' not in html
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
    assert "__CHAPTER_DATA__" not in html

    match = re.search(
        r'<script id="stages" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert len(embedded) == 13
    assert all(stage["inputs"] and stage["operations"] and stage["outputs"] for stage in embedded)
    assert all(stage["handoff"] for stage in embedded)

    scene_match = re.search(
        r'<script id="sceneData" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert scene_match is not None
    scene_embedded = json.loads(scene_match.group(1))
    assert len(scene_embedded) == 6
    mapped_keys = [key for scene in scene_embedded for key in scene["stage_keys"]]
    assert len(mapped_keys) == len(set(mapped_keys)) == len(embedded)
    assert set(mapped_keys) == {stage["key"] for stage in embedded}


def test_cli_renders_and_checks_exact_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "logistics-sandbox-generation-theatre.html"
    monkeypatch.setattr("sys.argv", ["render", "--output", str(output)])
    assert main() == 0
    assert output.read_text(encoding="utf-8") == render_html()

    monkeypatch.setattr("sys.argv", ["render", "--output", str(output), "--check"])
    assert main() == 0

    output.write_text("stale", encoding="utf-8")
    assert main() == 1
