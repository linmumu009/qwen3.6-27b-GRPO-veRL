#!/usr/bin/env python3
"""Render a source-backed replay of one real logistics sandbox build.

The replay combines the committed v20 runtime state and completion summaries
from sf_my_sandbox with the archived logistics assets audited on machine 5.
No task prompt, hidden gold, SQL, database row, or credential is embedded.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

if __package__:
    from .render_sandbox_generation_animation import HTML_TEMPLATE, _safe_json
else:
    from render_sandbox_generation_animation import HTML_TEMPLATE, _safe_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "logistics_sandbox_generation_example_animation_20260827.html"
GITHUB_REVISION = "758917009d0ebb0fb36561197171f6abdd279d96"
GITHUB_BASE = f"https://github.com/renjunxiang/sf_my_sandbox/blob/{GITHUB_REVISION}"
SCENE_PATH = ".runtime_state/v20/scenes/运营分析-8767b626"
MACHINE5_ASSET_ROOT = (
    "/data3/llin/qwen3.6-27b-verl-grpo/sandboxes/source/"
    "20260628_v15_boss"
)


@dataclass(frozen=True)
class ExampleStage:
    key: str
    step: str
    title: str
    subtitle: str
    detail: str
    artifacts: tuple[str, ...]
    path_hint: str
    x: int
    y: int
    phase: str
    duration_seconds: int
    replay_ms: int
    boundary: str = "generation"


@dataclass(frozen=True)
class ExampleConnection:
    from_key: str
    to_key: str
    route: str = "auto"


STAGES = (
    ExampleStage(
        key="input",
        step="13:15:20",
        title="物流运营说明入场",
        subtitle="7,105 B · 指纹隔离",
        detail="真实输入《运营分析.md》被复制到独立场景 运营分析-8767b626。场景描述德运物流的零担快运网络、五条产品线与分析角色，源文件先计算 SHA256，后续阶段只写本场景目录。",
        artifacts=("运营分析.md · 7,105 B", "scene_id=运营分析-8767b626", "sha256=f96fc90c…ca4db", "status=running"),
        path_hint=f"{SCENE_PATH}/input/运营分析.md",
        x=24,
        y=78,
        phase="common",
        duration_seconds=0,
        replay_ms=900,
    ),
    ExampleStage(
        key="prd",
        step="Step 0 · 05:15",
        title="物流业务 PRD",
        subtitle="5 个文件 · 14/14 通过",
        detail="把原始说明展开为全国性零担快运业务规格：仓库经理、网点主管、区域运营负责人、运营分析师和总部管理者；覆盖查询、对比、趋势、归因和探索诊断。",
        artifacts=("prd_运营分析.md", "coverage.json", "filled_template.json", "validation.json", "5 个角色"),
        path_hint="artifacts/pipeline/output/运营分析-8767b626/prd/",
        x=250,
        y=78,
        phase="common",
        duration_seconds=315,
        replay_ms=1300,
    ),
    ExampleStage(
        key="factor",
        step="Step 1 · 51:14",
        title="业务因子解耦",
        subtitle="本次运行的耗时瓶颈",
        detail="从物流 PRD 中抽出运单、仓库、线路、承运商、温度记录等业务对象，建立运单 20 态等状态机并补齐可调用动作；首次验证发现 1 个状态引用错误，修正后 9/9 通过。",
        artifacts=("57 个实体", "76 个状态", "80 个工具动作", "1 个 hard error 已修复", "17 个 soft warnings"),
        path_hint="artifacts/pipeline/output/运营分析-8767b626/factor/",
        x=476,
        y=78,
        phase="common",
        duration_seconds=3074,
        replay_ms=3600,
    ),
    ExampleStage(
        key="taxonomy",
        step="Step 2 · 09:04",
        title="任务语义空间",
        subtitle="15 维 · 115 采样单元",
        detail="把实体、状态和动作组织成可采样语义空间，加入组合约束与四类场景覆盖。真实产物包含 56 个 medium 与 59 个 high 复杂度采样单元，覆盖比例归一化为 1.0000。",
        artifacts=("15 个变化维度", "115 个采样单元", "9 条非法约束", "6 条必需组合", "7/7 校验通过"),
        path_hint="artifacts/pipeline/output/运营分析-8767b626/taxonomy/",
        x=702,
        y=78,
        phase="common",
        duration_seconds=544,
        replay_ms=1800,
    ),
    ExampleStage(
        key="schema",
        step="Step 3.1 · 01:27",
        title="物流数仓 Schema",
        subtitle="64 张表 · 验证通过",
        detail="生成物流数据库结构：6 张平台支撑表、3 张平台事实表、54 张业务事实表和 1 张关系表。核心对象包括运单、路线、承运商、仓库、温控记录、货损与质量事件。",
        artifacts=("64 tables/*.json", "54 张业务事实表", "schema_overview.json · 23 KB", "validation_rules.json · 14 KB", "10/10 校验通过"),
        path_hint="artifacts/pipeline/output/运营分析-8767b626/schema/",
        x=928,
        y=294,
        phase="dwh",
        duration_seconds=87,
        replay_ms=1000,
    ),
    ExampleStage(
        key="data",
        step="Step 4.1 · 02:31",
        title="合成物流数据",
        subtitle="36,101 行写入 SQLite",
        detail="按 Schema 为运单、成本、仓储、配送、温控等表生成 JSONL，再构建 logistics.sqlite。真实 v20 运行写入 36,101 条记录、65 个 JSONL 文件，五步数据校验全部通过。",
        artifacts=("36,101 条记录", "65 个 JSONL", "logistics.sqlite", "5/5 数据门禁", "0 failures"),
        path_hint="artifacts/pipeline/output/运营分析-8767b626/generated_data/",
        x=702,
        y=294,
        phase="dwh",
        duration_seconds=151,
        replay_ms=1300,
    ),
    ExampleStage(
        key="dwh_tasks",
        step="Step 5.1 · 01:29",
        title="数仓评测任务",
        subtitle="6,000 → 1,000 → 555",
        detail="先生成 6,000 个候选任务，再由 TaskSelector 选出 1,000 个，加入 44 个链式任务并做语义去重；最终 tasks.jsonl 保留 555 条，验证报告为 0 error、0 warning。",
        artifacts=("6,000 候选", "1,000 个筛选入围", "+44 链式任务", "555 条最终任务", "hidden gold 不展示"),
        path_hint="artifacts/pipeline/output/运营分析-8767b626/tasks/",
        x=476,
        y=294,
        phase="dwh",
        duration_seconds=89,
        replay_ms=1100,
    ),
    ExampleStage(
        key="catalog",
        step="Step 3.2 · 02:09",
        title="物流知识目录",
        subtitle="65 份文档定义",
        detail="从因子与 taxonomy 派生政策、手册、FAQ、培训等知识目录，并关联实际数仓表。该次运行生成 65 份目录定义；374 个双向引用等 soft warning 被保留审计，但不阻塞下游。",
        artifacts=("document_catalog.json · 157 KB", "65 份文档定义", "4 种文档类型", "374 soft warnings", "overall_passed=true"),
        path_hint="artifacts/pipeline/output/运营分析-8767b626/knowledge_catalog/",
        x=928,
        y=466,
        phase="knowledge",
        duration_seconds=129,
        replay_ms=1100,
    ),
    ExampleStage(
        key="documents",
        step="Step 4.2 · 03:12",
        title="合成物流知识库",
        subtitle="65 文档 · 1,587 chunks",
        detail="按照 catalog 依次生成价格政策、线路 SLA、理赔规范、退货流程、异常件手册等 Markdown 文档，再构建索引。实际得到 176,139 字、1,587 个检索块，缺失文档和额外文档均为 0。",
        artifacts=("65 个 Markdown", "1,587 chunks", "176,139 字", "document_index.json", "0 missing / 0 extra"),
        path_hint="artifacts/pipeline/output/运营分析-8767b626/documents/",
        x=702,
        y=466,
        phase="knowledge",
        duration_seconds=192,
        replay_ms=1400,
    ),
    ExampleStage(
        key="kb_tasks",
        step="Step 5.2 · 01:43",
        title="知识库评测任务",
        subtitle="500 目标 → 465 唯一任务",
        detail="从文档事实生成单文档、跨文档、时效与多跳检索任务。模糊去重后因独立原材料不足保留 465 条，而不是强行补足 500；8 项验证全部通过，无失败和警告。",
        artifacts=("465 条任务", "365 answerable", "100 unanswerable", "8/8 校验通过", "不伪造重复任务"),
        path_hint="artifacts/pipeline/output/运营分析-8767b626/knowledge_tasks/",
        x=476,
        y=466,
        phase="knowledge",
        duration_seconds=103,
        replay_ms=1200,
    ),
    ExampleStage(
        key="hybrid",
        step="Step 5.3 · 02:26",
        title="数据 × 政策混合任务",
        subtitle="500/500 双源覆盖",
        detail="把数据库事实和物流政策文档合在同一任务中：171 条先查数据再对照政策、156 条先查政策再验证数据、173 条执行合规检查。每条都同时绑定 expected_tables 与 source_documents。",
        artifacts=("500 条唯一任务", "data→policy 171", "policy→data 156", "compliance 173", "双源覆盖 100%"),
        path_hint="artifacts/pipeline/output/运营分析-8767b626/hybrid_tasks/",
        x=250,
        y=380,
        phase="hybrid",
        duration_seconds=146,
        replay_ms=1500,
    ),
    ExampleStage(
        key="freeze",
        step="14:35:51 · 总计 80:31",
        title="场景完成并登记",
        subtitle="sandbox_id 已生成",
        detail="十个生成阶段全部完成后，状态文件写入 sandbox_id 57a3cf55…6c6c。新版 freezer 会把任务与 hidden gold 留在 raw 层，只向 runner 暴露数据库、文档、schema dictionary 与 manifest。",
        artifacts=("scene_status=completed", "sandbox_id=57a3cf55…6c6c", "raw: tasks + gold", "runner: DB + Docs", "registry: checksums"),
        path_hint=f"{SCENE_PATH}/runtime/state.json",
        x=24,
        y=380,
        phase="freeze",
        duration_seconds=0,
        replay_ms=1200,
    ),
    ExampleStage(
        key="rollout",
        step="运行边界外",
        title="模型进入物流沙箱",
        subtitle="只能看到 runner 资产",
        detail="Rollout Engine 校验登记信息后挂载 logistics.sqlite 与文档库。模型可以查询运单和政策，但看不到 tasks.jsonl、gold answer 或验证 SQL；本动画也只展示汇总数字。",
        artifacts=("logistics.sqlite", "documents/", "schema_dictionary.md", "sandbox_manifest.json", "tasks + gold hidden"),
        path_hint="rollout_engine/sandbox_registry.py",
        x=24,
        y=590,
        phase="external",
        duration_seconds=0,
        replay_ms=1000,
        boundary="external",
    ),
)


CONNECTIONS = (
    ExampleConnection("input", "prd"),
    ExampleConnection("prd", "factor"),
    ExampleConnection("factor", "taxonomy"),
    ExampleConnection("taxonomy", "schema"),
    ExampleConnection("schema", "data"),
    ExampleConnection("data", "dwh_tasks"),
    ExampleConnection("taxonomy", "catalog", "right-rail"),
    ExampleConnection("catalog", "documents"),
    ExampleConnection("documents", "kb_tasks"),
    ExampleConnection("dwh_tasks", "hybrid", "merge-left"),
    ExampleConnection("kb_tasks", "hybrid", "merge-left"),
    ExampleConnection("hybrid", "freeze"),
    ExampleConnection("freeze", "rollout"),
)


SOURCE_NOTES = (
    {
        "label": "v20 真实阶段状态、开始/完成时间与 sandbox_id",
        "path": f"{GITHUB_BASE}/{SCENE_PATH}/runtime/state.json",
    },
    {
        "label": "v20 各阶段真实执行摘要与验证统计",
        "path": f"{GITHUB_BASE}/{SCENE_PATH}/runtime/raw_responses/",
    },
    {
        "label": "本仓库留存的物流 Schema、数据与任务质量摘要",
        "path": "sandbox/v15_qwen38_27b_100_20260821/",
    },
    {
        "label": "5 号机物流沙箱实物归档（DB / Docs / 三类任务）",
        "path": MACHINE5_ASSET_ROOT,
    },
)


def _replace_required(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"upstream animation template changed; missing: {old[:80]!r}")
    return text.replace(old, new, 1)


def render_html() -> str:
    html = HTML_TEMPLATE
    replacements = (
        ("<title>沙箱生成过程</title>", "<title>物流运营沙箱 · 真实生成重放</title>"),
        (
            '<div class="eyebrow">sf_my_sandbox · GitHub main 7589170</div>',
            '<div class="eyebrow">真实运行重放 · v20 · 2026-06-23</div>',
        ),
        ("<h1>沙箱是怎样生成的？</h1>", "<h1>一个物流沙箱怎样真实长出来？</h1>"),
        (
            '<p class="subtitle">公共主干把场景规格化，再分流生成数据仓库与知识库，两路证据汇入 Hybrid 任务；最后通过 raw、runner、registry 三层冻结，把评测秘密隔离在模型工作区之外。</p>',
            '<p class="subtitle">重放真实场景 <strong>运营分析-8767b626</strong>：从 7,105 字节的物流说明开始，经历 80 分 31 秒生成实体、状态、64 张表、36,101 行数据、65 份文档和三类评测任务，最后登记为可运行沙箱。</p>',
        ),
        (
            '<div class="scope"><span class="scope-dot" aria-hidden="true"></span><span><strong>权威源：GitHub main</strong><br>5 号机 20260814 快照作为历史对照</span></div>',
            '<div class="scope"><span class="scope-dot" aria-hidden="true"></span><span><strong>真实结果：completed</strong><br>墙钟 13:15:20 → 14:35:51</span></div>',
        ),
        ('<div class="flow-label">environment generation</div>', '<div class="flow-label">logistics sandbox · compressed replay</div>'),
        ("<title id=\"svgTitle\">沙箱生成流水线</title>", "<title id=\"svgTitle\">物流运营沙箱真实生成流水线</title>"),
        (
            '<desc id="svgDesc">业务场景经过公共规格化主干，再分成数据仓库和知识库两路生成任务，汇入 Hybrid，经过三层冻结后由独立 Rollout Engine 消费 runner 层。</desc>',
            '<desc id="svgDesc">真实物流运营场景从输入说明开始，生成 PRD、因子与分类体系，再分成数仓和知识库两路，汇合为混合任务并登记为沙箱。</desc>',
        ),
        (
            '<p>2026-08-27 对照 GitHub main@7589170 与 5 号机 20260814 快照后采用新版；图中不展示任务正文、hidden gold、数据库行、轨迹或凭据。</p>',
            '<p>按 v20 状态文件、阶段完成摘要和 5 号机归档实物重放；时间和数量均取实际记录。为防泄露，不展示任务正文、hidden gold、SQL、数据库行或凭据。</p>',
        ),
        ("const stepDuration = 1850;", "const defaultStepDuration = 1850;"),
        ("state.elapsed / stepDuration", "state.elapsed / (currentStage.replay_ms || defaultStepDuration)"),
        ("state.elapsed >= stepDuration", "state.elapsed >= (currentStage.replay_ms || defaultStepDuration)"),
    )
    for old, new in replacements:
        html = _replace_required(html, old, new)
    html = html.replace("__STAGE_DATA__", _safe_json([asdict(stage) for stage in STAGES]))
    html = html.replace("__CONNECTION_DATA__", _safe_json([asdict(edge) for edge in CONNECTIONS]))
    html = html.replace("__SOURCE_DATA__", _safe_json(SOURCE_NOTES))
    return html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成物流运营沙箱真实生成重放动画")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="检查 HTML 是否与生成器一致")
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
