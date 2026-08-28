from __future__ import annotations

import json
from pathlib import Path
import re

from scripts.render_logistics_sandbox_example_animation import PROCESS_STEPS, WORLD_PHASES, STAGES, main, render_html


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
    assert [step.key for step in PROCESS_STEPS] == [stage.key for stage in STAGES[:-1]]
    assert [step.phase for step in PROCESS_STEPS] == [
        "understand", "understand", "model", "model", "facts", "facts",
        "facts", "rules", "rules", "rules", "tasks", "seal",
    ]
    assert PROCESS_STEPS[-1].metrics == (57, 76, 80, 64, 36101, 65, 1587, 1520)


def test_replay_uses_one_cumulative_particle_world_to_explain_the_process() -> None:
    html = render_html()

    assert html.startswith("<!doctype html>")
    assert "一份物流 PRD，怎样生长成一个成熟世界？" in html
    assert "同一个世界从一个种子持续分裂" in html
    assert "PRD → 角色与目标" in html
    assert "实体 + 状态 + 动作" in html
    assert "事实分支" in html
    assert "规则分支" in html
    assert "事实 × 规则" in html
    assert "Runner / Raw 隔离" in html
    assert "十二个实际执行阶段" in html
    assert "本阶段动作" in html
    assert "本阶段输入" not in html
    assert "阶段结果" not in html
    assert "世界累计规模" not in html
    assert 'class="metric-grid"' not in html
    assert "查看真实落盘凭证（不参与主叙事）" in html
    assert 'id="worldCanvas"' in html
    assert "粒子世界 · 累积生长" in html
    assert "球体变化：" in html
    assert "共享世界模型" in html
    assert "事实世界" in html
    assert "规则世界" in html
    assert "外部观察任务" in html
    assert "冻结边界" in html
    assert "Step 5.1 数仓任务" in html
    assert "Step 5.2 知识任务" in html
    assert "Step 5.3 混合任务" in html
    assert "57 类实体" in html
    assert "76 个状态" in html
    assert "80 个动作" in html
    assert "15 个变化维度" in html
    assert "115 个采样单元" in html
    assert "27 条组合约束" in html
    assert "64 张表" in html
    assert "36,101 条物流记录" in html
    assert "300 条显式关系桥记录" in html
    assert "65 份制度文档" in html
    assert "1,587 个证据块" in html
    assert "使用 374 对文档引用关系" in html
    assert "1,520" in html
    assert "PRD · 1 个世界种子" in html
    assert "add('seed',1,0,-1" in html
    assert "add('roles',5,1,0" in html
    assert "add('entities',57,2,0" in html
    assert "add('states',76,2,1" in html
    assert "add('actions',80,2,1" in html
    assert "add('taxonomy',115,3,1" in html
    assert "add('tables',64,4,1" in html
    assert "add('facts',36101,5,1" in html
    assert "add('dwh',555,6,3" in html
    assert "add('docanchors',65,7,1" in html
    assert "add('chunks',1587,8,1" in html
    assert "add('kb',465,9,1" in html
    assert "add('hybridA',171,10,0" in html
    assert "add('hybridB',156,10,1" in html
    assert "add('hybridC',173,10,2" in html
    assert "function fib(" in html
    assert "origin(i,count,t)" in html
    assert "birthTimes" in html
    assert "当前可见 ${count.toLocaleString('zh-CN')} 个世界粒子" in html
    assert "事实半球 · 64 表 · 36,101 实例" in html
    assert "规则半球 · 65 文档 · 1,587 证据块" in html
    assert "555 数仓任务 · 观察事实" in html
    assert "465 知识任务 · 观察规则" in html
    assert "500 混合任务 · 比较事实与规则" in html
    assert "壳内 RUNNER · DB + Docs + Schema" in html
    assert "壳外 RAW · 1,520 Tasks + Hidden Gold" in html
    assert "任务始终在世界之外观察它" in html
    assert "绿色冻结壳只封装 Runner 可见世界" in html
    assert "function finish()" in html
    assert 'id="end"' in html
    assert "物流运营沙箱生成完成" in html
    assert "state.finished=true" in html
    assert "body.finished" in html
    assert "animation-play-state:paused!important" in html
    assert "cancelAnimationFrame(worldRaf)" in html
    assert "if(!state.finished&&!reduced)worldRaf=requestAnimationFrame(worldLoop)" in html
    assert "el.play.disabled=state.finished" in html
    assert "el.next.disabled=state.finished" in html
    assert "第一次因子校验" in html
    assert "修复错误引用" in html
    assert "数量来自固定提交 7589170 的 v20 实际运行摘要" in html
    assert "57 是实体类型，36,101 是实例化记录，口径不混用" in html
    assert "W-DEMO-001" not in html
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
    assert "__PROCESS_DATA__" not in html

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

    process_match = re.search(
        r'<script id="processSteps" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert process_match is not None
    process_embedded = json.loads(process_match.group(1))
    assert len(process_embedded) == 12
    assert [step["key"] for step in process_embedded] == [stage["key"] for stage in embedded[:-1]]
    assert process_embedded[-1]["metrics"] == [57, 76, 80, 64, 36101, 65, 1587, 1520]


def test_cli_renders_and_checks_exact_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "logistics-sandbox-generation-theatre.html"
    monkeypatch.setattr("sys.argv", ["render", "--output", str(output)])
    assert main() == 0
    assert output.read_text(encoding="utf-8") == render_html()

    monkeypatch.setattr("sys.argv", ["render", "--output", str(output), "--check"])
    assert main() == 0

    output.write_text("stale", encoding="utf-8")
    assert main() == 1
