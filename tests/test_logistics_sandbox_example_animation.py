from __future__ import annotations

import json
from pathlib import Path
import re

from scripts.render_logistics_sandbox_example_animation import EXAMPLE_CASES, PROCESS_STEPS, WORLD_PHASES, STAGES, main, render_html


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


def test_replay_explains_each_stage_with_a_finite_real_logistics_example() -> None:
    html = render_html()

    assert html.startswith("<!doctype html>")
    assert "物流运营沙箱生成过程 · 分页讲解" in html
    assert '<main class="deck" id="deck">' in html
    assert "aspect-ratio:4/3" in html
    assert "width:min(100vw,calc(100vh * 4 / 3))" in html
    assert "height:min(100vh,calc(100vw * 3 / 4))" in html
    assert "html,body{width:100%;height:100%;margin:0;overflow:hidden" in html
    assert "grid-template-columns:repeat(12,1fr)" in html
    assert "真实生成步骤" in html
    assert "同样规则批量展开" in html
    assert "下一页为什么能继续" in html
    assert "固定 v20 场景 运营分析-8767b626" in html
    assert "shortNames=['输入','业务理解','世界模型','情境空间','表结构','真实数据','数仓任务','知识目录','制度文档','知识任务','混合任务','冻结交付']" in html
    assert "本阶段输入" not in html
    assert "阶段结果" not in html
    assert "世界累计规模" not in html
    assert '<canvas' not in html
    assert "粒子世界" not in html
    assert "空间球体" not in html
    assert "requestAnimationFrame" not in html
    assert "setInterval" not in html
    assert "Step 5.1 数仓任务" in html
    assert "Step 5.2 知识任务" in html
    assert "Step 5.3 混合任务" in html

    # Every page uses a finite real specimen, with Chinese-first labels.
    assert "业务诉求：分析华东区上周的派送延误原因" in html
    assert "分析上周重庆分拨中心时效下降的原因" in html
    assert "运单（waybill）" in html
    assert "23 个业务属性" in html
    assert "把运单更新为下一状态" in html
    assert "waybill_no、target_status、node_id、operator_id" in html
    assert "waybill_created → waybill_assigned" in html
    assert "incident_closed_without_compensation" in html
    assert "“运输中的运单”成为一种可抽取场景" in html
    assert "已签收的运单，不能再次签收" in html
    assert "sender_name / phone / address → sender_info" in html
    assert "current_status → status" in html
    assert "fact_waybill" in html
    assert "25 个物理字段" in html
    assert "WAY000067" in html
    assert "ROU000651 · 华东干线" in html
    assert "CUS000669 · 广星德" in html
    assert "498.3" in html
    assert "task_000285" in html
    assert "生成冷链运单最新一期的数据概览报告" in html
    assert "product_type（产品类型）" in html
    assert "temp_sensor_id（温度传感器）" in html
    assert "doc_002 ·《各线路时效承诺标准（SLA）》" in html
    assert "达到标准阈值的 80% 即触发预警" in html
    assert "KT-LOG-0080" in html
    assert "HT-0051 · data_then_policy" in html
    assert "本季度华南区运单重货专线产品合规检查" in html
    assert "fact_delivery" in html
    assert "验证结果：979 条派送记录" in html
    assert "task_000257" not in html
    assert "HT-0007" not in html

    # The batch outcome stays attached to the example instead of becoming a
    # decorative increment rail.
    assert "57 类实体" in html
    assert "76 个状态" in html
    assert "80 个动作" in html
    assert "15 个变化维度" in html
    assert "115 个采样单元" in html
    assert "27 条组合约束" in html
    assert "64 张表" in html
    assert "36,101 条实例化记录" in html
    assert "300 条显式关系桥记录" in html
    assert "65 份制度文档" in html
    assert "1,587 个证据块" in html
    assert "374 条引用关系" in html
    assert "1,520" in html
    assert "事实→规则 171 个" in html
    assert "规则→事实 156 个" in html
    assert "合规判断 173 个" in html
    assert "DB + Docs + Schema" in html
    assert "Tasks + Hidden Gold" in html
    assert "function finish()" in html
    assert 'id="complete"' in html
    assert "沙箱生成完成" in html
    assert "state.finished=true" in html
    assert "state.playing=false" in html
    assert "el.play.disabled=state.finished" in html
    assert "el.next.disabled=state.finished||state.transitioning" in html
    assert "if(!state.playing||state.finished||state.transitioning)return" in html
    assert "只有点击“重新观看”才会重新播放讲解" in html
    assert "固定提交：758917009d0e…" in html
    assert "W-DEMO-001" not in html
    assert "本阶段新增" not in html
    assert "数字增量" not in html
    assert "render_sandbox_generation_animation" not in html
    assert '<script src="http' not in html
    assert '<link href="http' not in html
    assert "gold_answer" not in html
    assert "SELECT " not in html
    assert "__STAGE_DATA__" not in html
    assert "__PROCESS_DATA__" not in html
    assert "__EXAMPLE_DATA__" not in html

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

    example_match = re.search(
        r'<script id="exampleCases" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert example_match is not None
    examples = json.loads(example_match.group(1))
    assert len(EXAMPLE_CASES) == len(examples) == 12
    assert [case["key"] for case in examples] == [step["key"] for step in process_embedded]
    assert all(3 <= len(case["nodes"]) <= 4 for case in examples)
    assert all(case["scale"] and case["why"] and case["evidence"] for case in examples)


def test_cli_renders_and_checks_exact_output(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "logistics-sandbox-generation-theatre.html"
    monkeypatch.setattr("sys.argv", ["render", "--output", str(output)])
    assert main() == 0
    assert output.read_text(encoding="utf-8") == render_html()

    monkeypatch.setattr("sys.argv", ["render", "--output", str(output), "--check"])
    assert main() == 0

    output.write_text("stale", encoding="utf-8")
    assert main() == 1
