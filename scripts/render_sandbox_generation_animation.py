#!/usr/bin/env python3
"""Render the source-backed sandbox generation animation.

The page is self-contained so it can be reviewed from Git or opened directly
without a web service.  Its stage order and artifact labels are derived from
the runtime snapshot audited on machine 5 on 2026-08-27.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "sandbox_generation_animation_20260827.html"
REMOTE_SNAPSHOT = (
    "/data3/llin/qwen3.6-27b-verl-grpo/source_snapshots/"
    "rjx_sandbox_pipeline_20260814/generator_source"
)


@dataclass(frozen=True)
class Stage:
    key: str
    step: str
    title: str
    subtitle: str
    detail: str
    artifacts: tuple[str, ...]
    path_hint: str
    x: int
    y: int
    boundary: str = "generation"


STAGES = (
    Stage(
        key="brief",
        step="输入",
        title="业务场景说明",
        subtitle="一个文档 = 一个场景",
        detail="扫描受支持的业务说明文件，计算内容指纹，并为每个文档创建独立 scene_id。",
        artifacts=("scene.md", "document_registry.json"),
        path_hint="runtime/document_registry.json",
        x=24,
        y=78,
    ),
    Stage(
        key="scene",
        step="隔离",
        title="场景工作区",
        subtitle="状态、事件、产物彼此隔离",
        detail="复制输入，建立 manifest、运行配置和可恢复状态；所有阶段只写当前场景目录。",
        artifacts=("manifest.json", "runtime_config.json", "state.json", "runtime.events.jsonl"),
        path_hint="scenes/<scene_id>/runtime/",
        x=250,
        y=78,
    ),
    Stage(
        key="prd",
        step="Step 0",
        title="结构化 PRD",
        subtitle="把粗需求变成业务规格",
        detail="从业务说明提炼范围、参与角色、约束和覆盖目标，并在进入下一阶段前校验 PRD 完整性。",
        artifacts=("prd_<scene_id>.md", "coverage.json", "filled_template.json", "validation.json"),
        path_hint="step0_prd/",
        x=476,
        y=78,
    ),
    Stage(
        key="factor",
        step="Step 1",
        title="因子解耦",
        subtitle="实体 · 状态 · 动作",
        detail="把业务规格拆成可组合的实体、状态机与工具动作，并拦截截断或规模异常的 JSON 产物。",
        artifacts=("entities.json", "states.json", "tools_actions.json"),
        path_hint="step1_factor/",
        x=702,
        y=78,
    ),
    Stage(
        key="taxonomy",
        step="Step 2",
        title="任务分类体系",
        subtitle="采样单元与覆盖约束",
        detail="把因子组织为可采样的层次结构，定义组合约束和覆盖计划，驱动后续环境与任务生成。",
        artifacts=(
            "factor_taxonomies.json",
            "sampling_units.json",
            "combination_constraints.json",
            "coverage_plan.json",
        ),
        path_hint="step2_taxonomy/",
        x=928,
        y=78,
    ),
    Stage(
        key="schema",
        step="Step 3.1",
        title="DWH Schema",
        subtitle="表结构与枚举字典",
        detail="根据业务实体和覆盖计划生成事实表、维表、字段类型、枚举字典与数据库级校验规则。",
        artifacts=("schema_overview.json", "tables/*.json", "enum_dictionary.json", "validation_rules.json"),
        path_hint="step3.1_schema/",
        x=928,
        y=354,
    ),
    Stage(
        key="data",
        step="Step 4.1",
        title="合成业务数据",
        subtitle="规则对齐后写入 SQLite",
        detail="按 schema 和业务规则生成表数据、对齐外键与时间链，校验后同时落盘 JSONL 和 SQLite。",
        artifacts=("jsonl/*.jsonl", "database/*.sqlite", "reports/generation_summary.json"),
        path_hint="step4.1_data/",
        x=702,
        y=354,
    ),
    Stage(
        key="tasks",
        step="Step 5.1",
        title="评测任务",
        subtitle="题面 · hidden gold · verifier",
        detail="基于分类体系和真实 SQLite 结果生成任务；任务文件属于评测资产，不直接暴露给模型工作区。",
        artifacts=("tasks/tasks.jsonl", "runtime_configs/task_gen_config.json"),
        path_hint="step5.1_tasks/",
        x=476,
        y=354,
    ),
    Stage(
        key="freeze",
        step="冻结",
        title="校验和与登记",
        subtitle="完整环境才可冻结",
        detail="Step 0–5 全部完成后，为 DB、Docs、Tasks 计算 SHA256，生成不可变 sandbox_id 并写入 registry。",
        artifacts=("db_sha256", "docs_sha256", "tasks_sha256", "sandbox_registry.jsonl"),
        path_hint="trajectory_store/sandbox_registry.jsonl",
        x=250,
        y=354,
    ),
    Stage(
        key="rollout",
        step="边界外",
        title="Rollout Engine",
        subtitle="独立消费冻结沙箱",
        detail="轨迹采集不再属于环境生成器。独立引擎只消费已登记资产，并在运行前复核校验和与可见文件边界。",
        artifacts=("logistics.sqlite", "schema_dictionary.md", "documents/", "tasks hidden"),
        path_hint="独立系统：不计入 Step 0–5",
        x=24,
        y=354,
        boundary="external",
    ),
)


CONNECTIONS = tuple((STAGES[index].key, STAGES[index + 1].key) for index in range(len(STAGES) - 1))


SOURCE_NOTES = (
    {
        "label": "阶段顺序与扁平目录映射",
        "path": f"{REMOTE_SNAPSHOT}/autonomous_pipeline_runtime/core/constants.py",
    },
    {
        "label": "场景生命周期、状态事件与完成后冻结",
        "path": f"{REMOTE_SNAPSHOT}/autonomous_pipeline_runtime/execution/manager.py",
    },
    {
        "label": "各阶段关键产物门禁",
        "path": f"{REMOTE_SNAPSHOT}/autonomous_pipeline_runtime/execution/stage_registry.py",
    },
    {
        "label": "DB / Docs / Tasks 校验和与 registry",
        "path": f"{REMOTE_SNAPSHOT}/autonomous_pipeline_runtime/execution/sandbox_freezer.py",
    },
    {
        "label": "生成链路与历史语义错位审计",
        "path": "docs/boss_sandbox_generation_root_cause_audit_20260814.md",
    },
)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>沙箱生成过程</title>
  <style>
    :root {
      --page: #07111f;
      --page-2: #0b1728;
      --surface: rgba(13, 29, 49, .78);
      --surface-strong: #10233a;
      --surface-soft: rgba(22, 43, 68, .68);
      --text: #eef6ff;
      --muted: #9fb3c8;
      --line: #2b4662;
      --line-dim: #20374f;
      --cyan: #59d8ff;
      --blue: #6e92ff;
      --mint: #54e2b7;
      --amber: #ffca6a;
      --pink: #ff8db4;
      --danger: #ff7777;
      --shadow: 0 24px 70px rgba(0, 0, 0, .32);
    }

    * { box-sizing: border-box; }

    html { min-width: 320px; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 14% 8%, rgba(89, 216, 255, .13), transparent 32rem),
        radial-gradient(circle at 88% 0%, rgba(110, 146, 255, .14), transparent 34rem),
        linear-gradient(160deg, var(--page), var(--page-2));
      font-family: Inter, "SF Pro Display", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
      -webkit-font-smoothing: antialiased;
    }

    button, select, input { font: inherit; }

    button, select { color: inherit; }

    .page {
      width: min(1480px, calc(100% - 36px));
      margin: 0 auto;
      padding: 34px 0 44px;
    }

    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 28px;
      align-items: end;
      margin-bottom: 22px;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 9px;
      margin-bottom: 11px;
      color: var(--cyan);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .16em;
      text-transform: uppercase;
    }

    .eyebrow::before {
      width: 22px;
      height: 2px;
      content: "";
      background: currentColor;
      box-shadow: 9px 0 18px currentColor;
    }

    h1 {
      margin: 0;
      font-size: clamp(30px, 4.2vw, 58px);
      font-weight: 740;
      letter-spacing: -.045em;
      line-height: 1.02;
    }

    .subtitle {
      max-width: 760px;
      margin: 13px 0 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.72;
    }

    .scope {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 250px;
      padding: 13px 16px;
      border: 1px solid rgba(84, 226, 183, .28);
      border-radius: 14px;
      background: rgba(14, 38, 48, .62);
      color: #c7f7e8;
      box-shadow: inset 0 1px rgba(255, 255, 255, .04);
      font-size: 13px;
      line-height: 1.45;
    }

    .scope-dot {
      width: 10px;
      height: 10px;
      flex: 0 0 auto;
      border-radius: 999px;
      background: var(--mint);
      box-shadow: 0 0 18px rgba(84, 226, 183, .9);
    }

    .shell {
      overflow: hidden;
      border: 1px solid rgba(126, 174, 214, .18);
      border-radius: 24px;
      background: linear-gradient(180deg, rgba(16, 35, 58, .82), rgba(8, 21, 37, .88));
      box-shadow: var(--shadow), inset 0 1px rgba(255, 255, 255, .045);
      backdrop-filter: blur(18px);
    }

    .controls {
      display: grid;
      grid-template-columns: auto minmax(180px, 1fr) auto;
      gap: 18px;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid rgba(126, 174, 214, .14);
      background: rgba(7, 20, 35, .5);
    }

    .transport { display: flex; gap: 9px; }

    .control-button,
    .speed-select {
      min-height: 40px;
      border: 1px solid rgba(126, 174, 214, .23);
      border-radius: 10px;
      background: rgba(24, 48, 75, .72);
      transition: border-color .18s ease, background .18s ease, transform .18s ease;
    }

    .control-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 0 14px;
      cursor: pointer;
    }

    .control-button:hover {
      border-color: rgba(89, 216, 255, .65);
      background: rgba(31, 63, 96, .88);
      transform: translateY(-1px);
    }

    .control-button.primary {
      min-width: 102px;
      border-color: rgba(89, 216, 255, .45);
      background: linear-gradient(135deg, rgba(55, 142, 187, .9), rgba(67, 89, 179, .9));
      font-weight: 700;
    }

    .button-icon { width: 16px; text-align: center; }

    .progress-wrap {
      display: grid;
      grid-template-columns: auto minmax(110px, 1fr) auto;
      gap: 12px;
      align-items: center;
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--cyan);
      cursor: pointer;
    }

    .speed-select { padding: 0 10px; cursor: pointer; }

    .visual-region {
      position: relative;
      min-height: 610px;
      padding: 16px 18px 0;
      background-image:
        linear-gradient(rgba(120, 169, 207, .045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(120, 169, 207, .045) 1px, transparent 1px);
      background-size: 26px 26px;
    }

    .flow-label {
      position: absolute;
      z-index: 2;
      top: 28px;
      left: 34px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .12em;
      text-transform: uppercase;
    }

    .flow-label::before {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--cyan);
      content: "";
      box-shadow: 0 0 14px rgba(89, 216, 255, .75);
    }

    #pipelineSvg {
      display: block;
      width: 100%;
      height: auto;
      min-height: 570px;
      overflow: visible;
    }

    .lane-title {
      fill: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .08em;
    }

    .lane-line {
      stroke: var(--line-dim);
      stroke-width: 1;
      stroke-dasharray: 4 8;
    }

    .flow-path {
      fill: none;
      stroke: var(--line-dim);
      stroke-width: 2;
      stroke-linecap: round;
      transition: stroke .35s ease, opacity .35s ease;
    }

    .flow-path.done { stroke: rgba(84, 226, 183, .56); }
    .flow-path.active { stroke: var(--cyan); filter: drop-shadow(0 0 7px rgba(89, 216, 255, .7)); }
    .flow-path.external-link { stroke-dasharray: 6 7; }

    .flow-particle {
      fill: var(--cyan);
      opacity: 0;
      filter: drop-shadow(0 0 8px rgba(89, 216, 255, .95));
    }

    .flow-particle.active { opacity: 1; }

    .stage-card {
      cursor: pointer;
      outline: none;
    }

    .stage-card .node-bg {
      fill: rgba(13, 31, 51, .95);
      stroke: var(--line);
      stroke-width: 1.5;
      transition: fill .28s ease, stroke .28s ease, filter .28s ease;
    }

    .stage-card:hover .node-bg,
    .stage-card:focus-visible .node-bg { stroke: rgba(89, 216, 255, .75); }

    .stage-card.done .node-bg {
      fill: rgba(22, 57, 59, .92);
      stroke: rgba(84, 226, 183, .48);
    }

    .stage-card.active .node-bg {
      fill: rgba(19, 54, 82, .98);
      stroke: var(--cyan);
      stroke-width: 2;
      filter: drop-shadow(0 0 13px rgba(89, 216, 255, .28));
    }

    .stage-card.external .node-bg {
      fill: rgba(31, 29, 52, .92);
      stroke: rgba(255, 141, 180, .48);
      stroke-dasharray: 5 5;
    }

    .stage-card.external.active .node-bg { stroke: var(--pink); }

    .node-step {
      fill: var(--cyan);
      font-size: 11px;
      font-weight: 750;
      letter-spacing: .08em;
    }

    .external .node-step { fill: var(--pink); }

    .node-title {
      fill: var(--text);
      font-size: 16px;
      font-weight: 750;
    }

    .node-subtitle {
      fill: var(--muted);
      font-size: 11px;
    }

    .status-ring {
      fill: rgba(7, 17, 31, .9);
      stroke: var(--line);
      stroke-width: 1.5;
      transition: stroke .28s ease, fill .28s ease;
    }

    .done .status-ring { fill: var(--mint); stroke: var(--mint); }
    .active .status-ring { fill: var(--cyan); stroke: var(--cyan); }
    .external.active .status-ring { fill: var(--pink); stroke: var(--pink); }

    .status-mark {
      fill: none;
      stroke: #061321;
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
      opacity: 0;
      transition: opacity .2s ease;
    }

    .done .status-mark,
    .active .status-mark { opacity: 1; }

    .active-halo {
      fill: none;
      stroke: var(--cyan);
      stroke-width: 1.5;
      opacity: 0;
      transform-box: fill-box;
      transform-origin: center;
    }

    .active .active-halo {
      opacity: .35;
      animation: halo 1.6s ease-out infinite;
    }

    .external.active .active-halo { stroke: var(--pink); }

    @keyframes halo {
      0% { transform: scale(1); opacity: .46; }
      100% { transform: scale(1.08, 1.14); opacity: 0; }
    }

    .boundary-line {
      stroke: rgba(255, 141, 180, .42);
      stroke-width: 1;
      stroke-dasharray: 4 7;
    }

    .boundary-label {
      fill: var(--pink);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .08em;
    }

    .mobile-flow { display: none; }

    .mobile-stage {
      position: relative;
      display: grid;
      grid-template-columns: 42px minmax(0, 1fr);
      gap: 12px;
      padding: 0 0 16px;
      border: 0;
      background: transparent;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }

    .mobile-stage:not(:last-child)::after {
      position: absolute;
      z-index: 0;
      top: 32px;
      bottom: 0;
      left: 20px;
      width: 2px;
      background: var(--line-dim);
      content: "";
    }

    .mobile-stage.done:not(:last-child)::after { background: rgba(84, 226, 183, .56); }

    .mobile-index {
      position: relative;
      z-index: 1;
      display: grid;
      width: 42px;
      height: 42px;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 50%;
      background: var(--surface-strong);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .mobile-stage.done .mobile-index { border-color: var(--mint); background: var(--mint); color: #061321; }
    .mobile-stage.active .mobile-index { border-color: var(--cyan); background: var(--cyan); color: #061321; box-shadow: 0 0 20px rgba(89, 216, 255, .35); }
    .mobile-stage.external .mobile-index { border-style: dashed; border-color: var(--pink); }
    .mobile-stage.external.active .mobile-index { background: var(--pink); }

    .mobile-copy { padding-top: 1px; }
    .mobile-step,
    .mobile-title,
    .mobile-subtitle { display: block; }
    .mobile-step { color: var(--cyan); font-size: 10px; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }
    .mobile-stage.external .mobile-step { color: var(--pink); }
    .mobile-title { margin-top: 3px; font-size: 15px; font-weight: 720; }
    .mobile-subtitle { margin-top: 3px; color: var(--muted); font-size: 12px; }

    .detail-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(300px, .7fr);
      gap: 1px;
      border-top: 1px solid rgba(126, 174, 214, .15);
      background: rgba(126, 174, 214, .15);
    }

    .detail-panel,
    .artifact-panel {
      min-width: 0;
      padding: 22px 24px 24px;
      background: rgba(7, 20, 35, .92);
    }

    .detail-kicker {
      color: var(--cyan);
      font-size: 11px;
      font-weight: 750;
      letter-spacing: .11em;
      text-transform: uppercase;
    }

    .external-copy .detail-kicker { color: var(--pink); }

    .detail-title {
      margin: 7px 0 6px;
      font-size: clamp(20px, 2vw, 28px);
      font-weight: 740;
      letter-spacing: -.025em;
    }

    .detail-copy {
      max-width: 780px;
      min-height: 48px;
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.7;
    }

    .path-hint {
      display: inline-flex;
      max-width: 100%;
      margin-top: 13px;
      padding: 7px 9px;
      overflow-wrap: anywhere;
      border: 1px solid rgba(126, 174, 214, .16);
      border-radius: 7px;
      background: rgba(21, 43, 68, .72);
      color: #c7d9ea;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 11px;
    }

    .artifact-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 13px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 750;
      letter-spacing: .1em;
      text-transform: uppercase;
    }

    .artifact-count {
      color: var(--mint);
      font-variant-numeric: tabular-nums;
    }

    .artifacts {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      min-height: 70px;
      align-content: flex-start;
    }

    .artifact-chip {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 5px 9px;
      border: 1px solid rgba(84, 226, 183, .22);
      border-radius: 7px;
      background: rgba(17, 52, 55, .58);
      color: #c6f6e7;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 11px;
      animation: chip-in .35s both;
    }

    .artifact-chip.hidden {
      border-color: rgba(255, 141, 180, .3);
      background: rgba(67, 32, 52, .52);
      color: #ffd1e1;
    }

    @keyframes chip-in {
      from { opacity: 0; transform: translateY(5px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .timeline {
      display: grid;
      grid-template-columns: repeat(10, minmax(72px, 1fr));
      gap: 8px;
      padding: 15px 18px 18px;
      background: rgba(7, 20, 35, .74);
    }

    .timeline-button {
      min-width: 0;
      padding: 9px 7px;
      overflow: hidden;
      border: 1px solid transparent;
      border-radius: 8px;
      background: transparent;
      color: var(--muted);
      text-overflow: ellipsis;
      white-space: nowrap;
      cursor: pointer;
      font-size: 10px;
      transition: color .2s ease, background .2s ease, border-color .2s ease;
    }

    .timeline-button:hover { color: var(--text); background: rgba(31, 57, 85, .55); }
    .timeline-button.done { color: #bfeedd; }
    .timeline-button.active { border-color: rgba(89, 216, 255, .45); background: rgba(32, 78, 111, .55); color: var(--text); }
    .timeline-button.external.active { border-color: rgba(255, 141, 180, .5); background: rgba(72, 37, 58, .55); }

    .notes {
      display: grid;
      grid-template-columns: 210px minmax(0, 1fr);
      gap: 24px;
      margin-top: 20px;
      padding: 0 4px;
    }

    .notes h2 {
      margin: 0;
      font-size: 14px;
      font-weight: 720;
    }

    .notes p { margin: 6px 0 0; color: var(--muted); font-size: 12px; line-height: 1.6; }

    .source-list {
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .source-list li {
      display: grid;
      grid-template-columns: minmax(160px, .45fr) minmax(0, 1fr);
      gap: 12px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.5;
    }

    .source-list code {
      overflow-wrap: anywhere;
      color: #c6d7e7;
      font-family: "SFMono-Regular", Consolas, monospace;
    }

    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }

    @media (max-width: 900px) {
      .hero { grid-template-columns: 1fr; align-items: start; }
      .scope { width: fit-content; }
      .controls { grid-template-columns: 1fr; }
      .progress-wrap { order: 3; }
      .detail-grid { grid-template-columns: 1fr; }
      .timeline { grid-template-columns: repeat(5, minmax(74px, 1fr)); }
      .notes { grid-template-columns: 1fr; }
    }

    @media (max-width: 700px) {
      .page { width: min(100% - 22px, 1480px); padding-top: 22px; }
      .shell { border-radius: 18px; }
      .desktop-flow { display: none; }
      .mobile-flow { display: block; padding: 22px 8px 4px; }
      .visual-region { min-height: 0; padding: 4px 16px 0; }
      .flow-label { position: static; margin: 18px 0 0 8px; }
      .controls { padding: 14px; gap: 13px; }
      .transport { display: grid; grid-template-columns: 1fr auto; }
      .control-button.primary { min-width: 0; }
      .progress-wrap { grid-template-columns: auto minmax(90px, 1fr) auto; }
      .detail-panel, .artifact-panel { padding: 19px 18px 21px; }
      .timeline { display: none; }
      .source-list li { grid-template-columns: 1fr; gap: 2px; }
    }

    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        scroll-behavior: auto !important;
        animation-duration: .001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .001ms !important;
      }
    }
  </style>
</head>
<body>
  <main class="page" id="sandboxAnimation">
    <header class="hero">
      <div>
        <div class="eyebrow">Sandbox Factory · Source-backed</div>
        <h1>沙箱是怎样生成的？</h1>
        <p class="subtitle">从一份业务场景说明开始，逐阶段构建可验证的数据库环境与评测任务；只有完整产物通过门禁后，才会冻结为可供后续轨迹采集使用的沙箱。</p>
      </div>
      <div class="scope"><span class="scope-dot" aria-hidden="true"></span><span><strong>生成边界：Step 0–5</strong><br>Rollout Engine 已独立解耦</span></div>
    </header>

    <section class="shell" aria-labelledby="animationTitle">
      <h2 class="sr-only" id="animationTitle">沙箱生成阶段动画</h2>
      <div class="controls">
        <div class="transport">
          <button class="control-button primary" id="playButton" type="button" aria-label="播放沙箱生成动画">
            <span class="button-icon" id="playIcon" aria-hidden="true">▶</span>
            <span id="playLabel">播放</span>
          </button>
          <button class="control-button" id="restartButton" type="button" aria-label="从头播放">
            <span aria-hidden="true">↺</span><span>重置</span>
          </button>
        </div>
        <label class="progress-wrap" for="progressRange">
          <span id="progressStep">01</span>
          <input id="progressRange" type="range" min="0" max="9" value="0" step="1" aria-label="沙箱生成阶段">
          <span id="progressTotal">/ 10</span>
        </label>
        <label>
          <span class="sr-only">播放速度</span>
          <select class="speed-select" id="speedSelect" aria-label="播放速度">
            <option value="0.6">0.6×</option>
            <option value="1" selected>1×</option>
            <option value="1.5">1.5×</option>
            <option value="2">2×</option>
          </select>
        </label>
      </div>

      <div class="visual-region">
        <div class="flow-label">environment generation</div>
        <div class="desktop-flow">
          <svg id="pipelineSvg" viewBox="0 0 1140 560" role="img" aria-labelledby="svgTitle svgDesc">
            <title id="svgTitle">沙箱生成流水线</title>
            <desc id="svgDesc">业务场景依次经过隔离、PRD、因子、分类体系、DWH Schema、合成数据、评测任务和冻结登记，之后由独立 Rollout Engine 消费。</desc>
            <g id="laneLayer"></g>
            <g id="connectorLayer"></g>
            <g id="nodeLayer"></g>
          </svg>
        </div>
        <div class="mobile-flow" id="mobileFlow" aria-label="沙箱生成垂直流程"></div>
      </div>

      <div class="detail-grid" id="detailGrid">
        <div class="detail-panel" id="detailPanel">
          <div class="detail-kicker" id="detailStep">输入</div>
          <div class="detail-title" id="detailTitle">业务场景说明</div>
          <p class="detail-copy" id="detailCopy">扫描受支持的业务说明文件，计算内容指纹，并为每个文档创建独立 scene_id。</p>
          <div class="path-hint" id="pathHint">runtime/document_registry.json</div>
        </div>
        <div class="artifact-panel">
          <div class="artifact-title"><span>本阶段落盘 / 可见资产</span><span class="artifact-count" id="artifactCount">2 files</span></div>
          <div class="artifacts" id="artifactList"></div>
        </div>
      </div>

      <nav class="timeline" id="timeline" aria-label="跳转到生成阶段"></nav>
      <div class="sr-only" id="liveStatus" aria-live="polite"></div>
    </section>

    <footer class="notes">
      <div>
        <h2>图示依据</h2>
        <p>2026-08-27 只读核对 5 号机快照；图中不展示任务内容、数据库行、轨迹或凭据。</p>
      </div>
      <ul class="source-list" id="sourceList"></ul>
    </footer>
  </main>

  <script id="stageData" type="application/json">__STAGE_DATA__</script>
  <script id="connectionData" type="application/json">__CONNECTION_DATA__</script>
  <script id="sourceData" type="application/json">__SOURCE_DATA__</script>
  <script>
    (() => {
      "use strict";

      const root = document.getElementById("sandboxAnimation");
      const stages = JSON.parse(document.getElementById("stageData").textContent);
      const connections = JSON.parse(document.getElementById("connectionData").textContent);
      const sources = JSON.parse(document.getElementById("sourceData").textContent);
      const svgNS = "http://www.w3.org/2000/svg";
      const nodeWidth = 188;
      const nodeHeight = 108;
      const stageIndex = new Map(stages.map((stage, index) => [stage.key, index]));
      const nodes = new Map();
      const paths = new Map();
      const particles = new Map();
      const mobileNodes = new Map();
      const timelineNodes = new Map();
      const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const state = { index: 0, playing: false, speed: 1, lastFrame: 0, elapsed: 0 };
      const stepDuration = 1850;

      const playButton = document.getElementById("playButton");
      const playIcon = document.getElementById("playIcon");
      const playLabel = document.getElementById("playLabel");
      const restartButton = document.getElementById("restartButton");
      const progressRange = document.getElementById("progressRange");
      const progressStep = document.getElementById("progressStep");
      const speedSelect = document.getElementById("speedSelect");
      const detailPanel = document.getElementById("detailPanel");
      const detailStep = document.getElementById("detailStep");
      const detailTitle = document.getElementById("detailTitle");
      const detailCopy = document.getElementById("detailCopy");
      const pathHint = document.getElementById("pathHint");
      const artifactList = document.getElementById("artifactList");
      const artifactCount = document.getElementById("artifactCount");
      const liveStatus = document.getElementById("liveStatus");
      const nodeLayer = document.getElementById("nodeLayer");
      const connectorLayer = document.getElementById("connectorLayer");
      const laneLayer = document.getElementById("laneLayer");
      const mobileFlow = document.getElementById("mobileFlow");
      const timeline = document.getElementById("timeline");
      const sourceList = document.getElementById("sourceList");

      function svgElement(name, attributes = {}) {
        const element = document.createElementNS(svgNS, name);
        Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
        return element;
      }

      function addText(parent, text, className, x, y) {
        const element = svgElement("text", { x, y, class: className });
        element.textContent = text;
        parent.appendChild(element);
        return element;
      }

      function connectorPath(from, to) {
        const fromCenter = { x: from.x + nodeWidth / 2, y: from.y + nodeHeight / 2 };
        const toCenter = { x: to.x + nodeWidth / 2, y: to.y + nodeHeight / 2 };
        if (Math.abs(from.y - to.y) < 10) {
          const forward = to.x > from.x;
          const startX = forward ? from.x + nodeWidth : from.x;
          const endX = forward ? to.x : to.x + nodeWidth;
          return `M ${startX} ${fromCenter.y} L ${endX} ${toCenter.y}`;
        }
        const startY = from.y + nodeHeight;
        const endY = to.y;
        return `M ${fromCenter.x} ${startY} C ${fromCenter.x} ${startY + 64}, ${toCenter.x} ${endY - 64}, ${toCenter.x} ${endY}`;
      }

      function buildLanes() {
        const topTitle = addText(laneLayer, "阶段产物逐级落盘", "lane-title", 24, 48);
        topTitle.setAttribute("aria-hidden", "true");
        laneLayer.appendChild(svgElement("line", { x1: 24, y1: 255, x2: 1116, y2: 255, class: "lane-line" }));
        const bottomTitle = addText(laneLayer, "环境资产校验与冻结", "lane-title", 24, 328);
        bottomTitle.setAttribute("aria-hidden", "true");
        laneLayer.appendChild(svgElement("line", { x1: 223, y1: 326, x2: 223, y2: 520, class: "boundary-line" }));
        const boundary = addText(laneLayer, "生成边界", "boundary-label", 213, 540);
        boundary.setAttribute("text-anchor", "end");
      }

      function buildConnectors() {
        connections.forEach(([fromKey, toKey]) => {
          const from = stages[stageIndex.get(fromKey)];
          const to = stages[stageIndex.get(toKey)];
          const d = connectorPath(from, to);
          const path = svgElement("path", {
            d,
            class: `flow-path${to.boundary === "external" ? " external-link" : ""}`,
            "data-from": fromKey,
            "data-to": toKey,
          });
          connectorLayer.appendChild(path);
          paths.set(toKey, path);
          const particle = svgElement("circle", { r: 4.5, class: "flow-particle" });
          connectorLayer.appendChild(particle);
          particles.set(toKey, particle);
        });
      }

      function buildNodes() {
        stages.forEach((stage, index) => {
          const group = svgElement("g", {
            class: `stage-card${stage.boundary === "external" ? " external" : ""}`,
            transform: `translate(${stage.x} ${stage.y})`,
            role: "button",
            tabindex: "0",
            "aria-label": `${stage.step}：${stage.title}。${stage.subtitle}`,
            "data-key": stage.key,
          });
          group.appendChild(svgElement("rect", { x: -5, y: -5, width: nodeWidth + 10, height: nodeHeight + 10, rx: 18, class: "active-halo" }));
          group.appendChild(svgElement("rect", { width: nodeWidth, height: nodeHeight, rx: 14, class: "node-bg" }));
          addText(group, stage.step, "node-step", 15, 23);
          addText(group, stage.title, "node-title", 15, 54);
          addText(group, stage.subtitle, "node-subtitle", 15, 77);
          const fileCount = addText(group, `${stage.artifacts.length} 项资产`, "node-subtitle", 15, 96);
          fileCount.setAttribute("opacity", ".68");
          group.appendChild(svgElement("circle", { cx: 165, cy: 23, r: 9, class: "status-ring" }));
          group.appendChild(svgElement("path", { d: "M160.5 23.2 l3 3.1 l5.4 -6.1", class: "status-mark" }));
          group.addEventListener("click", () => selectStage(index, false));
          group.addEventListener("keydown", event => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              selectStage(index, false);
            }
          });
          nodeLayer.appendChild(group);
          nodes.set(stage.key, group);
        });
      }

      function buildMobileFlow() {
        stages.forEach((stage, index) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = `mobile-stage${stage.boundary === "external" ? " external" : ""}`;
          button.setAttribute("aria-label", `${stage.step}：${stage.title}`);
          button.innerHTML = `<span class="mobile-index">${String(index + 1).padStart(2, "0")}</span><span class="mobile-copy"><span class="mobile-step">${stage.step}</span><span class="mobile-title">${stage.title}</span><span class="mobile-subtitle">${stage.subtitle}</span></span>`;
          button.addEventListener("click", () => selectStage(index, false));
          mobileFlow.appendChild(button);
          mobileNodes.set(stage.key, button);
        });
      }

      function buildTimeline() {
        stages.forEach((stage, index) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = `timeline-button${stage.boundary === "external" ? " external" : ""}`;
          button.textContent = stage.step;
          button.title = stage.title;
          button.setAttribute("aria-label", `跳转到 ${stage.step} ${stage.title}`);
          button.addEventListener("click", () => selectStage(index, false));
          timeline.appendChild(button);
          timelineNodes.set(stage.key, button);
        });
      }

      function buildSources() {
        sources.forEach(source => {
          const item = document.createElement("li");
          const label = document.createElement("span");
          const path = document.createElement("code");
          label.textContent = source.label;
          path.textContent = source.path;
          item.append(label, path);
          sourceList.appendChild(item);
        });
      }

      function updateParticle(path, particle, progress) {
        if (!path || !particle) return;
        const length = path.getTotalLength();
        const point = path.getPointAtLength(length * progress);
        particle.setAttribute("cx", point.x);
        particle.setAttribute("cy", point.y);
      }

      function setPlayState(playing) {
        state.playing = playing;
        playIcon.textContent = playing ? "Ⅱ" : state.index === stages.length - 1 ? "↺" : "▶";
        playLabel.textContent = playing ? "暂停" : state.index === stages.length - 1 ? "重播" : "播放";
        playButton.setAttribute("aria-label", playing ? "暂停沙箱生成动画" : state.index === stages.length - 1 ? "重新播放沙箱生成动画" : "播放沙箱生成动画");
        if (!playing) state.elapsed = 0;
      }

      function selectStage(index, keepPlaying) {
        state.index = Math.max(0, Math.min(stages.length - 1, Number(index)));
        state.elapsed = 0;
        if (!keepPlaying) setPlayState(false);
        render();
      }

      function render() {
        const current = stages[state.index];
        stages.forEach((stage, index) => {
          const status = index < state.index ? "done" : index === state.index ? "active" : "pending";
          [nodes.get(stage.key), mobileNodes.get(stage.key), timelineNodes.get(stage.key)].forEach(element => {
            if (!element) return;
            element.classList.remove("done", "active", "pending");
            element.classList.add(status);
            if (index === state.index) element.setAttribute("aria-current", "step");
            else element.removeAttribute("aria-current");
          });
        });

        connections.forEach(([, toKey]) => {
          const index = stageIndex.get(toKey);
          const path = paths.get(toKey);
          const particle = particles.get(toKey);
          path.classList.remove("done", "active");
          particle.classList.remove("active");
          if (index < state.index) path.classList.add("done");
          if (index === state.index) {
            path.classList.add("active");
            particle.classList.add("active");
            updateParticle(path, particle, .72);
          }
        });

        detailPanel.classList.toggle("external-copy", current.boundary === "external");
        detailStep.textContent = current.step;
        detailTitle.textContent = current.title;
        detailCopy.textContent = current.detail;
        pathHint.textContent = current.path_hint;
        artifactList.replaceChildren();
        current.artifacts.forEach((artifact, index) => {
          const chip = document.createElement("span");
          chip.className = `artifact-chip${artifact.includes("hidden") ? " hidden" : ""}`;
          chip.textContent = artifact;
          chip.style.animationDelay = `${index * 70}ms`;
          artifactList.appendChild(chip);
        });
        artifactCount.textContent = `${current.artifacts.length} items`;
        progressRange.value = String(state.index);
        progressStep.textContent = String(state.index + 1).padStart(2, "0");
        liveStatus.textContent = `第 ${state.index + 1} 阶段，共 ${stages.length} 阶段：${current.step}，${current.title}。${current.detail}`;
      }

      function animate(timestamp) {
        if (!state.lastFrame) state.lastFrame = timestamp;
        const delta = Math.min(80, timestamp - state.lastFrame);
        state.lastFrame = timestamp;
        if (state.playing) {
          state.elapsed += delta * state.speed;
          const currentStage = stages[state.index];
          const path = paths.get(currentStage.key);
          const particle = particles.get(currentStage.key);
          if (path && particle) updateParticle(path, particle, Math.min(1, state.elapsed / stepDuration));
          if (state.elapsed >= stepDuration) {
            state.elapsed = 0;
            if (state.index < stages.length - 1) {
              state.index += 1;
              render();
            } else {
              setPlayState(false);
              render();
            }
          }
        }
        window.requestAnimationFrame(animate);
      }

      playButton.addEventListener("click", () => {
        if (!state.playing && state.index === stages.length - 1) state.index = 0;
        setPlayState(!state.playing);
        render();
      });
      restartButton.addEventListener("click", () => selectStage(0, false));
      progressRange.addEventListener("input", event => selectStage(event.target.value, false));
      speedSelect.addEventListener("change", event => { state.speed = Number(event.target.value); });
      root.addEventListener("keydown", event => {
        if (event.target.matches("input, select, button")) return;
        if (event.key === "ArrowRight") selectStage(state.index + 1, false);
        if (event.key === "ArrowLeft") selectStage(state.index - 1, false);
      });

      buildLanes();
      buildConnectors();
      buildNodes();
      buildMobileFlow();
      buildTimeline();
      buildSources();
      render();
      if (!reducedMotion) setPlayState(true);
      window.requestAnimationFrame(animate);
    })();
  </script>
</body>
</html>
'''


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_html() -> str:
    html = HTML_TEMPLATE
    html = html.replace("__STAGE_DATA__", _safe_json([asdict(stage) for stage in STAGES]))
    html = html.replace("__CONNECTION_DATA__", _safe_json(CONNECTIONS))
    html = html.replace("__SOURCE_DATA__", _safe_json(SOURCE_NOTES))
    return html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成沙箱生成过程自包含动态图")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="只检查现有文件是否与生成器一致")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    expected = render_html()
    output = args.output.resolve()
    if args.check:
        if not output.is_file():
            print(f"missing animation: {output}", file=sys.stderr)
            return 1
        if output.read_text(encoding="utf-8") != expected:
            print(f"animation is stale: {output}", file=sys.stderr)
            return 1
        print(f"animation is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
