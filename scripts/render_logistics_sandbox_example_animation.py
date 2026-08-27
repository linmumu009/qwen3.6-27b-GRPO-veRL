#!/usr/bin/env python3
"""Render a causal replay of one real logistics sandbox build.

The page deliberately shows transformations, not a pipeline overview: every
stage reads concrete upstream artifacts, performs ordered work, materialises
new artifacts, and hands those artifacts to the next stage.  Only aggregate
evidence is embedded; prompts, hidden gold, SQL and database rows are omitted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "logistics_sandbox_generation_example_animation_20260827.html"
GITHUB_REVISION = "758917009d0ebb0fb36561197171f6abdd279d96"
SCENE = "运营分析-8767b626"


@dataclass(frozen=True)
class Artifact:
    name: str
    kind: str
    meta: str
    preview: tuple[str, ...]
    produced_by: int = -1
    tone: str = "blue"


@dataclass(frozen=True)
class Operation:
    label: str
    detail: str
    outcome: str
    status: str = "normal"


@dataclass(frozen=True)
class Stage:
    key: str
    step: str
    title: str
    clock: str
    duration: str
    branch: str
    inputs: tuple[Artifact, ...]
    operations: tuple[Operation, ...]
    outputs: tuple[Artifact, ...]
    handoff: str
    handoff_type: str = "sequential"
    replay_ms: int = 900


@dataclass(frozen=True)
class WorldPhase:
    key: str
    principle: str
    formula: str
    inherited: tuple[str, ...]
    gains: tuple[str, ...]
    capability: str
    layers: tuple[str, ...]


@dataclass(frozen=True)
class Chapter:
    key: str
    number: str
    title: str
    promise: str
    start_stage: int
    end_stage: int


def artifact(
    name: str,
    kind: str,
    meta: str,
    *preview: str,
    produced_by: int = -1,
    tone: str = "blue",
) -> Artifact:
    return Artifact(name, kind, meta, tuple(preview), produced_by, tone)


def operation(
    label: str,
    detail: str,
    outcome: str,
    status: str = "normal",
) -> Operation:
    return Operation(label, detail, outcome, status)


STAGES = (
    Stage(
        key="intake",
        step="准备",
        title="把物流说明隔离成一个场景",
        clock="13:15:20",
        duration="00:00",
        branch="公共主干",
        inputs=(
            artifact(
                "运营分析.md",
                "业务说明",
                "7,105 B · 原始输入",
                "德运物流全国零担快运网络",
                "五条产品线与多级运营角色",
                "查询 / 对比 / 趋势 / 归因 / 诊断",
            ),
        ),
        operations=(
            operation("计算输入指纹", "对原文件计算 SHA256，后续每一步都能追溯同一输入。", "f96fc90c…ca4db"),
            operation("创建独立场景", "建立只属于本次生成的目录，避免和其他行业场景混写。", SCENE),
            operation("复制输入并写状态", "原始说明进入 scene/input，状态切换为 running。", "2 个场景文件落盘"),
        ),
        outputs=(
            artifact(f"scenes/{SCENE}/", "场景目录", "隔离空间", "input/", "runtime/", "artifacts/", produced_by=1, tone="purple"),
            artifact("input/运营分析.md", "受控输入", "SHA 已登记", "source_size = 7,105 B", "sha256 = f96fc90c…ca4db", produced_by=2),
            artifact("runtime/state.json", "运行状态", "status = running", "created_at = 13:15:20", f"scene_id = {SCENE}", produced_by=2, tone="green"),
        ),
        handoff="受控输入 input/运营分析.md 交给 Step 0；原文件不再被后续阶段直接改写。",
        replay_ms=750,
    ),
    Stage(
        key="prd",
        step="Step 0",
        title="把业务说明展开成可生成的 PRD",
        clock="13:15:20 → 13:20:35",
        duration="05:15",
        branch="公共主干",
        inputs=(
            artifact("input/运营分析.md", "上一步交付", "7,105 B", "物流网络与产品线", "运营角色与分析诉求"),
        ),
        operations=(
            operation("识别业务边界", "提取角色、对象、目标与分析任务，不补写输入中不存在的业务。", "5 类运营角色"),
            operation("填充 PRD 模板", "把散文说明变成可由后续程序稳定读取的结构化业务规格。", "prd_运营分析.md"),
            operation("生成覆盖账本", "逐项记录原始要求如何落到 PRD，防止静默遗漏。", "coverage.json"),
            operation("执行 PRD 门禁", "检查必填段、角色、任务类型与覆盖映射。", "14 / 14 通过"),
        ),
        outputs=(
            artifact("prd_运营分析.md", "业务规格", "结构化 PRD", "仓库经理 / 网点主管", "区域运营 / 分析师 / 总部", produced_by=1, tone="purple"),
            artifact("filled_template.json", "结构数据", "模板字段已填充", "actors[]", "business_goals[]", "analysis_tasks[]", produced_by=1),
            artifact("coverage.json", "覆盖账本", "原始要求 → PRD", "query · compare · trend", "attribution · diagnosis", produced_by=2, tone="amber"),
            artifact("validation.json", "质量门禁", "14 / 14 passed", "missing = 0", "blocking_error = 0", produced_by=3, tone="green"),
        ),
        handoff="PRD、模板字段和覆盖账本一起交给 Step 1；Factor 不再猜测原始散文的结构。",
        replay_ms=780,
    ),
    Stage(
        key="factor",
        step="Step 1",
        title="把 PRD 拆成实体、状态与可执行动作",
        clock="13:20:35 → 14:11:49",
        duration="51:14",
        branch="公共主干 · 本次瓶颈",
        inputs=(
            artifact("prd_运营分析.md", "上一步交付", "业务规格", "物流对象、角色、目标"),
            artifact("coverage.json", "上一步交付", "覆盖约束", "原始需求不得丢失"),
            artifact("filled_template.json", "上一步交付", "结构字段", "后续脚本可直接解析"),
        ),
        operations=(
            operation("抽取业务对象", "识别运单、仓库、线路、承运商、温度记录、质量事件等对象。", "57 个实体"),
            operation("建立状态机与动作", "为实体生成状态变化，并补齐工具动作的参数和返回结构。", "76 个状态 · 80 个动作"),
            operation("第一次因子校验", "检查动作引用的状态是否真实存在。", "发现 1 个 hard error", "error"),
            operation("修复错误引用", "quality_incident.triggers 引用了不存在的 incident_closed，改为实际状态 incident_closed_without_compensation。", "1 处引用已修复", "repair"),
            operation("重新执行校验", "保留 17 个不阻塞的 soft warning，硬错误清零。", "9 / 9 通过", "success"),
        ),
        outputs=(
            artifact("entities.json", "实体定义", "57 entities", "waybill · warehouse · route", "carrier · temperature_record", produced_by=0, tone="purple"),
            artifact("states.json", "状态机", "76 states", "运单 20 态", "异常 / 理赔 / 完结状态", produced_by=1),
            artifact("tools_actions.json", "工具动作", "80 actions", "typed arguments", "typed returns", produced_by=1, tone="amber"),
            artifact("factor_validation_report.json", "修复证据", "9 / 9 passed", "hard errors: 1 → 0", "soft warnings: 17", produced_by=4, tone="green"),
        ),
        handoff="57 个实体、76 个状态和 80 个动作成为 Step 2 的输入；修复前版本不会向下游流动。",
        replay_ms=1050,
    ),
    Stage(
        key="taxonomy",
        step="Step 2",
        title="把业务因子组织成可采样的任务空间",
        clock="14:11:49 → 14:20:53",
        duration="09:04",
        branch="公共主干",
        inputs=(
            artifact("entities.json", "上一步交付", "57 个实体", "任务可以围绕哪些对象变化"),
            artifact("states.json", "上一步交付", "76 个状态", "任务可以要求哪些状态判断"),
            artifact("tools_actions.json", "上一步交付", "80 个动作", "模型在环境里能做什么"),
        ),
        operations=(
            operation("定义变化维度", "将对象、时间、区域、状态、产品、指标等组织成正交变化轴。", "15 个维度"),
            operation("展开采样单元", "组合维度并标记任务难度，为任务生成建立抽样底座。", "56 medium + 59 high"),
            operation("写入组合约束", "显式禁止不合法组合，并要求关键因子共同出现。", "9 illegal · 6 required · 8 mutex"),
            operation("归一化覆盖计划", "覆盖比例合计必须等于 1，所有引用必须能回到 Factor。", "7 / 7 通过 · ratio 1.0000"),
        ),
        outputs=(
            artifact("factor_taxonomies.json", "变化维度", "15 dimensions", "object · region · time", "state · product · metric", produced_by=0, tone="purple"),
            artifact("sampling_units.json", "采样单元", "115 units", "medium 56", "high 59", produced_by=1),
            artifact("combination_constraints.json", "组合约束", "27 条核心约束", "illegal 9 · required 6", "mutex 8 · scene 4", produced_by=2, tone="amber"),
            artifact("coverage_plan.json", "覆盖计划", "7 / 7 passed", "coverage ratio = 1.0000", produced_by=3, tone="green"),
        ),
        handoff="Taxonomy 在这里分叉：同一批因子一份进入数仓支路，一份保存在上下文总线，稍后进入知识支路。",
        handoff_type="fork",
        replay_ms=820,
    ),
    Stage(
        key="schema",
        step="Step 3.1",
        title="把任务空间变成物流数仓结构",
        clock="14:20:53 → 14:22:20",
        duration="01:27",
        branch="数仓支路",
        inputs=(
            artifact("Factor bundle", "上下文总线", "57 entities · 80 actions", "业务对象决定表的主题"),
            artifact("Taxonomy bundle", "上一步交付", "15 dimensions · 115 units", "采样空间决定字段覆盖"),
        ),
        operations=(
            operation("规划数据分层", "先放平台支撑表与平台事实表，再映射业务实体。", "6 support + 3 platform fact"),
            operation("展开业务表", "逐实体生成事实表和关系表，并声明主外键。", "54 business fact + 1 relation"),
            operation("生成枚举与校验规则", "把 Factor 状态写入枚举字典，补齐字段级检查。", "enum_dictionary + validation_rules"),
            operation("执行 Schema 门禁", "检查表引用、字段类型、主外键与业务覆盖。", "10 / 10 通过 · 0 errors"),
        ),
        outputs=(
            artifact("schema_overview.json", "结构总览", "64 tables", "6 support · 3 platform", "54 business · 1 relation", produced_by=0, tone="purple"),
            artifact("tables/", "表定义目录", "64 个 JSON", "fact_waybill · fact_route", "fact_temperature_record · … +61", produced_by=1),
            artifact("enum_dictionary.json", "枚举字典", "状态映射已固化", "waybill status", "incident status", produced_by=2, tone="amber"),
            artifact("schema_validation_report.json", "质量门禁", "10 / 10 passed", "errors 0 · warnings 16", produced_by=3, tone="green"),
        ),
        handoff="64 张表定义和枚举字典交给 Step 4.1；Data 只能按这些结构写行。",
        replay_ms=760,
    ),
    Stage(
        key="data",
        step="Step 4.1",
        title="按 Schema 逐表合成物流数据",
        clock="14:22:20 → 14:24:51",
        duration="02:31",
        branch="数仓支路",
        inputs=(
            artifact("tables/", "上一步交付", "64 个表定义", "字段、类型、主外键"),
            artifact("enum_dictionary.json", "上一步交付", "合法枚举", "生成值不能越界"),
            artifact("validation_rules.json", "上一步交付", "字段门禁", "每批写入后立即检查"),
        ),
        operations=(
            operation("逐表生成 JSONL", "按依赖顺序先维表、再事实表，主外键只引用已存在记录。", "65 个 JSONL"),
            operation("构建 SQLite", "将通过行级检查的数据装载为一个可查询物流数据库。", "36,101 rows"),
            operation("执行数据门禁", "检查行数、枚举、非空、引用完整性和数据库可读性。", "5 / 5 通过 · 0 failures"),
        ),
        outputs=(
            artifact("generated_data/*.jsonl", "逐表数据", "65 files", "运单 · 仓储 · 配送", "温控 · 成本 · 质量", produced_by=0),
            artifact("database/logistics.sqlite", "物流数据库", "36,101 rows", "64-table schema", "queryable = true", produced_by=1, tone="purple"),
            artifact("data_validation_report.json", "质量门禁", "5 / 5 passed", "failures 0 · warnings 0", produced_by=2, tone="green"),
        ),
        handoff="数据库、Schema 和 Taxonomy 一起进入 Step 5.1，任务必须能由刚生成的数据回答或明确判定不可答。",
        replay_ms=820,
    ),
    Stage(
        key="dwh_tasks",
        step="Step 5.1",
        title="数仓任务：用物流数据库生成评测任务",
        clock="14:24:52 → 14:26:21",
        duration="01:29",
        branch="数仓支路",
        inputs=(
            artifact("database/logistics.sqlite", "上一步交付", "36,101 rows", "任务证据来自真实已写入数据"),
            artifact("schema_overview.json", "结构字典", "64 tables", "任务引用必须落到实际表"),
            artifact("sampling_units.json", "覆盖计划", "115 units", "控制主题和难度分布"),
        ),
        operations=(
            operation("批量生成候选", "同时构造可回答、不可回答和超范围三类候选。", "6,000 candidates"),
            operation("TaskSelector 筛选", "按覆盖、难度与重复度选出候选子集。", "1,000 selected"),
            operation("补充链式任务", "加入需要多步关联的物流分析任务。", "+44 chain tasks"),
            operation("语义去重并验证", "去除同义任务，检查每条任务的证据绑定；正文与 gold 不进入本动画。", "555 final · 0 errors"),
        ),
        outputs=(
            artifact("candidates.jsonl", "候选池", "6,000 tasks", "answerable 3,600", "unanswerable 1,500 · OOS 900", produced_by=0),
            artifact("selected_tasks.jsonl", "筛选结果", "1,000 tasks", "coverage balanced", "difficulty balanced", produced_by=1),
            artifact("tasks.jsonl", "数仓任务包", "555 unique", "task content sealed", "hidden gold sealed", produced_by=3, tone="purple"),
            artifact("task_validation_report.json", "质量门禁", "passed", "errors 0 · warnings 0", produced_by=3, tone="green"),
        ),
        handoff="数仓任务包先进入汇合缓冲区；生成器回到 Step 2 保存的 Factor + Taxonomy，开始知识支路。",
        handoff_type="branch-return",
        replay_ms=880,
    ),
    Stage(
        key="catalog",
        step="Step 3.2",
        title="从同一批因子生成物流知识目录",
        clock="14:26:21 → 14:28:30",
        duration="02:09",
        branch="知识支路",
        inputs=(
            artifact("Factor bundle", "上下文总线回放", "entities · states · actions", "与数仓支路同源"),
            artifact("Taxonomy bundle", "上下文总线回放", "dimensions · sampling units", "与数仓支路同源"),
            artifact("schema_overview.json", "数仓结构引用", "64 tables", "知识条目可绑定实际表"),
        ),
        operations=(
            operation("规划文档类型", "把业务因子映射为政策、手册、FAQ、培训资料。", "4 种文档类型"),
            operation("展开文档定义", "逐主题声明标题、章节、事实来源和关联表。", "65 document definitions"),
            operation("检查双向引用", "保留可审计的软警告，但阻断缺源或非法表引用。", "passed · 374 soft warnings"),
        ),
        outputs=(
            artifact("document_catalog.json", "知识目录", "65 definitions · 157 KB", "policy · manual", "FAQ · training", produced_by=1, tone="purple"),
            artifact("catalog_validation_report.json", "质量门禁", "overall_passed = true", "soft warnings 374", "blocking errors 0", produced_by=2, tone="green"),
        ),
        handoff="65 份文档定义逐项交给 Step 4.2；下一步不是复制目录，而是按每个定义真正写出正文和索引。",
        replay_ms=780,
    ),
    Stage(
        key="documents",
        step="Step 4.2",
        title="按目录逐份写出物流知识库",
        clock="14:28:30 → 14:31:42",
        duration="03:12",
        branch="知识支路",
        inputs=(
            artifact("document_catalog.json", "上一步交付", "65 definitions", "标题、章节、事实源、关联表"),
        ),
        operations=(
            operation("逐定义生成 Markdown", "生成价格政策、线路 SLA、理赔规范、退货流程、异常件手册等正文。", "65 Markdown"),
            operation("切分并构建索引", "按章节边界切块，保留文档与 chunk 的可追溯关系。", "1,587 chunks · 176,139 words"),
            operation("核对目录完整性", "逐项比对 catalog，缺失和额外文档都必须为 0。", "0 missing · 0 extra"),
        ),
        outputs=(
            artifact("documents/*.md", "物流知识库", "65 documents", "价格政策 · 线路 SLA", "理赔 · 退货 · 异常件", produced_by=0, tone="purple"),
            artifact("document_index.json", "检索索引", "1,587 chunks", "176,139 words", "source trace retained", produced_by=1),
            artifact("document_validation_report.json", "质量门禁", "overall_passed = true", "missing 0 · extra 0", produced_by=2, tone="green"),
        ),
        handoff="文档正文、chunk 索引和 catalog 一起交给 Step 5.2，知识任务必须能回到具体来源文档。",
        replay_ms=820,
    ),
    Stage(
        key="kb_tasks",
        step="Step 5.2",
        title="用文档事实生成知识库评测任务",
        clock="14:31:42 → 14:33:25",
        duration="01:43",
        branch="知识支路",
        inputs=(
            artifact("documents/*.md", "上一步交付", "65 documents", "任务证据来自已生成正文"),
            artifact("document_index.json", "上一步交付", "1,587 chunks", "每条证据可追溯"),
            artifact("document_catalog.json", "目录约束", "65 definitions", "主题与类型覆盖"),
        ),
        operations=(
            operation("生成知识候选", "构造单文档、跨文档、时效与多跳检索问题。", "target 500"),
            operation("事实对齐与去重", "原材料不足时不伪造近重复项，保留真实唯一集合。", "465 unique"),
            operation("执行知识任务门禁", "检查 source_documents、可答性标签与覆盖结构。", "8 / 8 通过"),
        ),
        outputs=(
            artifact("knowledge_tasks.jsonl", "知识任务包", "465 unique", "answerable 365", "unanswerable 100", produced_by=1, tone="purple"),
            artifact("knowledge_task_validation.json", "质量门禁", "8 / 8 passed", "failures 0 · warnings 0", produced_by=2, tone="green"),
        ),
        handoff="知识任务包进入第二个汇合槽；数仓任务包与知识任务包现在同时送往 Step 5.3。",
        handoff_type="merge",
        replay_ms=820,
    ),
    Stage(
        key="hybrid",
        step="Step 5.3",
        title="混合任务：合并数据库证据与政策证据",
        clock="14:33:25 → 14:35:51",
        duration="02:26",
        branch="双支路汇合",
        inputs=(
            artifact("tasks.jsonl", "数仓缓冲区", "555 DWH tasks", "expected_tables 已绑定"),
            artifact("knowledge_tasks.jsonl", "知识缓冲区", "465 KB tasks", "source_documents 已绑定"),
            artifact("logistics.sqlite + documents/", "双源环境", "DB + 65 docs", "每条任务必须同时依赖两侧"),
        ),
        operations=(
            operation("组装 data → policy", "先由数据库得到运营事实，再用政策判断或解释。", "171 tasks"),
            operation("组装 policy → data", "先读政策规则，再到数据库核验实际运营数据。", "156 tasks"),
            operation("组装 compliance check", "把数据库事实直接与政策阈值做合规对照。", "173 tasks"),
            operation("验证双源覆盖", "每条任务都必须同时有 expected_tables 与 source_documents。", "500 / 500 passed"),
        ),
        outputs=(
            artifact("hybrid_tasks.jsonl", "混合任务包", "500 unique", "data→policy 171", "policy→data 156 · compliance 173", produced_by=2, tone="purple"),
            artifact("hybrid_validation_report.json", "双源门禁", "500 / 500", "expected_tables nonempty", "source_documents nonempty", produced_by=3, tone="green"),
        ),
        handoff="DB、Docs 与三类任务已经齐备，最后进入冻结器：可运行资产和评测秘密将在这里分层。",
        replay_ms=850,
    ),
    Stage(
        key="freeze",
        step="完成",
        title="冻结资产、隔离秘密并登记沙箱",
        clock="14:35:51",
        duration="总计 80:31",
        branch="交付边界",
        inputs=(
            artifact("logistics.sqlite", "环境资产", "36,101 rows", "模型运行时需要"),
            artifact("documents/ + index", "环境资产", "65 docs · 1,587 chunks", "模型运行时需要"),
            artifact("DWH + KB + Hybrid tasks", "评测资产", "555 + 465 + 500", "模型运行时不得看见"),
        ),
        operations=(
            operation("计算资产校验和", "为数据库、文档、Schema 和任务包建立不可混淆的版本指纹。", "checksums ready"),
            operation("分离 raw 与 runner", "raw 保存任务与 hidden gold；runner 只保留模型可见的 DB、Docs、Schema 与 manifest。", "secret boundary enforced"),
            operation("写入 registry", "登记 sandbox_id、路径、校验和与完成状态。", "57a3cf55…6c6c"),
        ),
        outputs=(
            artifact("raw/", "评测者专用", "tasks + hidden gold", "not mounted to model", "audit only", produced_by=1, tone="amber"),
            artifact("runner/", "模型可见", "DB + Docs + Schema", "tasks absent", "gold absent", produced_by=1, tone="purple"),
            artifact("sandbox_registry.jsonl", "沙箱登记", "status = completed", "sandbox_id = 57a3cf55…6c6c", "wall clock = 80:31", produced_by=2, tone="green"),
        ),
        handoff="生成结束：Rollout Engine 只挂载 runner/ 并核验 registry；raw/ 永远留在模型工作区之外。",
        handoff_type="release",
        replay_ms=850,
    ),
    Stage(
        key="runner",
        step="运行边界外",
        title="模型进入已经生成好的物流沙箱",
        clock="生成后",
        duration="不计入 80:31",
        branch="Rollout Engine",
        inputs=(
            artifact("runner/", "冻结器交付", "DB + Docs + Schema", "只有模型可见资产"),
            artifact("sandbox_registry.jsonl", "登记信息", "sandbox_id + checksums", "挂载前先验证"),
        ),
        operations=(
            operation("校验登记与指纹", "拒绝未登记或校验和不一致的环境资产。", "registry verified"),
            operation("只读挂载 runner", "模型可以查询物流数据库和文档库。", "sandbox mounted"),
            operation("确认秘密不可见", "tasks、hidden gold、验证 SQL 和原始评测记录不在工作区。", "leakage surface closed"),
        ),
        outputs=(
            artifact("live logistics sandbox", "可运行环境", "ready", "logistics.sqlite mounted", "65 documents searchable", produced_by=1, tone="purple"),
            artifact("visibility check", "隔离证明", "passed", "tasks hidden", "gold hidden", produced_by=2, tone="green"),
        ),
        handoff="完整重放结束。点击“重新播放”可再次观察每个产物怎样成为下一步的输入。",
        handoff_type="end",
        replay_ms=850,
    ),
)


WORLD_PHASES = (
    WorldPhase(
        "intake",
        "先定义世界边界，再谈世界内容",
        "W₀ = Boundary(scene_id, source_hash)",
        ("一段物流业务叙述", "尚未区分角色、对象与规则"),
        ("唯一输入指纹", "独立世界边界", "可追溯的初始状态"),
        "从这一刻起，所有生成物属于同一个物流世界，不能与其他场景串线。",
        ("boundary", "", ""),
    ),
    WorldPhase(
        "prd",
        "把人的意图变成世界必须支持的观察",
        "Intent → Actors + Goals + Observations",
        ("已经隔离的物流世界", "业务叙述中的角色与目标"),
        ("5 类世界观察者", "角色所关心的业务目标", "可观察问题与覆盖账本", "14/14 需求闭环"),
        "世界知道谁在观察它、为什么观察，以及哪些现实问题必须能够回答。",
        ("roles", "", "", ""),
    ),
    WorldPhase(
        "factor",
        "世界不是名词表，而是会变化、可行动的系统",
        "W = (Entities, States, Actions, Invariants)",
        ("角色、目标与观察需求", "尚未形式化的物流业务概念"),
        ("57 类实体进入世界", "76 个状态 + 80 个动作", "发现状态引用断裂", "修复世界中的非法跃迁", "9/9 一致性通过"),
        "运单、仓库、线路和承运商不再只是文字：它们拥有状态、可执行动作与合法变化边界。",
        ("entities", "states", "repair", "actions", ""),
    ),
    WorldPhase(
        "taxonomy",
        "在同一个世界模型上系统地产生不同情境",
        "Situations = Sample(W, Dimensions, Constraints)",
        ("实体、状态、动作与不变量", "一个已经可执行但尚未展开的世界"),
        ("15 条变化轴", "115 个情境采样单元", "27 条组合约束", "覆盖分布归一化"),
        "世界获得情境生成器：可以改变区域、时间、产品和状态，但不会组合出业务上不存在的世界。",
        ("taxonomy", "", "constraints", ""),
    ),
    WorldPhase(
        "schema",
        "Schema 是世界状态的投影，不是世界本身",
        "Persist: World State → 64 Tables",
        ("业务对象、关系与状态机", "受约束的情境空间"),
        ("规划世界状态分层", "把实体关系映射到 64 张表", "把状态边界固化为枚举", "存储投影一致性通过"),
        "世界中每一种状态和关系都有稳定的存储地址，之后产生的事实都能回指语义对象。",
        ("schema", "", "", ""),
    ),
    WorldPhase(
        "data",
        "让世界从类型系统变成有当前状态与历史的实例",
        "Instantiate(W) → 36,101 Facts",
        ("64 张表承载的世界结构", "状态枚举与引用约束"),
        ("实例化仓、线、网点与运单", "36,101 条事实写入世界", "验证状态与关系闭合"),
        "物流世界开始运转：运单沿线路移动，仓库和承运商拥有真实可查询的当前状态。",
        ("data", "", ""),
    ),
    WorldPhase(
        "dwh_tasks",
        "数仓任务是对世界状态的受控观察",
        "DWH Task = Observe(State(W), EvidencePlan)",
        ("已经实例化的物流世界", "可查询的状态、关系与历史"),
        ("从世界状态生成 6,000 个观察", "按覆盖选择 1,000 个", "加入多跳关系观察", "收敛为 555 个数仓任务"),
        "Step 5.1 数仓任务询问“世界里实际发生了什么”，每个问题都必须落到真实世界状态。",
        ("dwh", "", "", ""),
    ),
    WorldPhase(
        "catalog",
        "现实状态之外，世界还需要规范它的规则层",
        "Rules(W) = Policies + Manuals + Procedures",
        ("同源的实体、动作与情境", "数仓世界中的真实对象"),
        ("规划政策、手册、FAQ 与培训", "65 条知识定义绑定世界对象", "检查规则与对象双向引用"),
        "世界获得规范层：不仅知道发生了什么，也开始知道价格、SLA、理赔和异常处理应该怎样。",
        ("rules", "", ""),
    ),
    WorldPhase(
        "documents",
        "把抽象规则变成世界中可检索、可引用的制度记忆",
        "Memory(W) = Express(Rules(W))",
        ("65 条规则与知识定义", "规则关联的实体、动作与数据表"),
        ("写出 65 份制度记忆", "切分为 1,587 个可检索证据块", "确认规则记忆无缺失"),
        "政策不再是标签，而是能够被世界中的角色读取、引用和用于判断的制度记忆。",
        ("docs", "", ""),
    ),
    WorldPhase(
        "kb_tasks",
        "知识任务是对世界规则与制度记忆的观察",
        "KB Task = Observe(Rules(W), SourceDocs)",
        ("可检索的政策与流程记忆", "每个规则的来源关系"),
        ("从规则层生成知识观察", "去重后保留 465 个唯一任务", "8/8 来源与可答性通过"),
        "Step 5.2 知识任务询问“这个世界应该怎样运行”，每个回答都必须回到具体制度来源。",
        ("kb", "", ""),
    ),
    WorldPhase(
        "hybrid",
        "混合任务把事实世界与规则世界放在同一个判断里",
        "Hybrid = Compare(State(W), Rules(W))",
        ("555 个世界状态观察", "465 个规则观察", "同源 DB + Docs"),
        ("现实 → 规则：171 个", "规则 → 现实：156 个", "合规判断：173 个", "500/500 双源闭环"),
        "Step 5.3 混合任务不再只是查询：它判断现实中的物流运行是否符合世界的规则。",
        ("hybrid", "", "", ""),
    ),
    WorldPhase(
        "freeze",
        "冻结的是完整世界和观察边界，而不是一个文件目录",
        "Sandbox = Seal(W, Observations, Visibility)",
        ("实体 + 状态 + 动作 + 约束", "事实状态 + 制度规则 + 三类观察"),
        ("固定世界版本与校验和", "隔离可见世界和评测秘密", "登记唯一 sandbox_id"),
        "世界模型被封装成可重复进入的环境；模型只能看到世界，不能看到评测者预先设计的观察答案。",
        ("freeze", "", ""),
    ),
    WorldPhase(
        "runner",
        "智能体进入世界，以有限观察和合法动作与世界交互",
        "Agent ↔ Observe(W) / Act(W)",
        ("已经冻结的可执行物流世界", "明确的可见性与动作边界"),
        ("验证世界身份", "挂载可观察状态与制度记忆", "确认秘密处于世界之外"),
        "世界建模完成：智能体面对的是会变化、有规则、可行动、可验证的物流世界，而不是一堆文件。",
        ("runner", "", ""),
    ),
)


CHAPTERS = (
    Chapter("define", "01", "定义世界", "先说明这个世界服务谁、边界在哪里", 0, 1),
    Chapter("mechanics", "02", "建立运行机制", "对象怎样变化，角色能够采取什么动作", 2, 3),
    Chapter("facts", "03", "实例化事实世界", "把语义世界变成会流动、可查询的现实状态", 4, 6),
    Chapter("rules", "04", "建立规则世界", "给现实加入政策、SLA 与制度记忆", 7, 9),
    Chapter("reason", "05", "生成混合判断", "同时观察现实状态与制度规则", 10, 10),
    Chapter("seal", "06", "冻结并交付", "封装世界、隔离答案，让智能体安全进入", 11, 12),
)


SOURCE_NOTES = (
    {
        "label": "v20 场景状态与阶段完成摘要",
        "url": f"https://github.com/renjunxiang/sf_my_sandbox/tree/{GITHUB_REVISION}/.runtime_state/v20/scenes/{SCENE}",
    },
    {
        "label": "5 号机归档结构核对",
        "url": "/data3/llin/qwen3.6-27b-verl-grpo/sandboxes/source/20260628_v15_boss",
    },
)


def _safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>物流沙箱生成实录 · 产物如何逐步长出来</title>
<style>
:root {
  color-scheme: dark;
  --bg: #07101f;
  --panel: #0d192b;
  --panel-2: #111f34;
  --line: #263a56;
  --text: #eef5ff;
  --muted: #9eb0c8;
  --blue: #60a5fa;
  --purple: #b18cff;
  --green: #51d69a;
  --amber: #f4bf5f;
  --red: #ff7185;
  --cyan: #54d4e8;
}
* { box-sizing: border-box; }
html { background: var(--bg); }
body {
  margin: 0;
  min-width: 1180px;
  color: var(--text);
  background:
    radial-gradient(circle at 15% 0%, rgba(96,165,250,.12), transparent 32%),
    radial-gradient(circle at 85% 8%, rgba(177,140,255,.10), transparent 30%),
    var(--bg);
  font-family: Inter, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
}
button, select { font: inherit; }
button:focus-visible, select:focus-visible { outline: 3px solid rgba(96,165,250,.55); outline-offset: 2px; }
.app { width: 100%; min-height: 100vh; padding: 24px 30px 22px; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
.topbar { display: flex; align-items: flex-start; justify-content: space-between; gap: 30px; }
.eyebrow { color: var(--cyan); font-size: 12px; letter-spacing: .16em; text-transform: uppercase; margin-bottom: 7px; }
h1 { margin: 0; font-size: 30px; font-weight: 600; letter-spacing: -.02em; }
.lede { margin: 8px 0 0; color: var(--muted); max-width: 820px; font-size: 14px; line-height: 1.65; }
.run-fact { min-width: 280px; padding: 12px 15px; background: rgba(17,31,52,.8); border: 1px solid var(--line); border-radius: 12px; }
.run-fact strong { display: block; font-size: 14px; margin-bottom: 4px; }
.run-fact span { color: var(--muted); font-size: 12px; }
.controls { margin-top: 18px; display: flex; align-items: center; gap: 8px; }
.control-btn, .speed-select {
  border: 1px solid var(--line); color: var(--text); background: var(--panel-2);
  border-radius: 8px; min-height: 36px; padding: 7px 12px; cursor: pointer;
}
.control-btn.primary { background: #eaf3ff; color: #091321; border-color: #eaf3ff; min-width: 92px; }
.control-btn:hover, .speed-select:hover { border-color: #4d6b90; }
.control-btn:disabled { opacity: .38; cursor: default; }
.progress-wrap { flex: 1; margin-left: 10px; }
.progress-meta { display: flex; justify-content: space-between; color: var(--muted); font-size: 11px; margin-bottom: 6px; }
.progress-track { height: 4px; background: #1c2b42; border-radius: 99px; overflow: hidden; }
.progress-fill { height: 100%; width: 0; background: linear-gradient(90deg, var(--blue), var(--purple)); transition: width .35s ease; }
.stage-rail { margin-top: 16px; display: grid; grid-template-columns: repeat(13, 1fr); gap: 5px; }
.stage-tab { border: 0; background: transparent; color: #70839d; padding: 8px 3px 6px; cursor: pointer; border-top: 2px solid #24364e; font-size: 10px; white-space: nowrap; }
.stage-tab.done { color: #a9bad0; border-color: #446488; }
.stage-tab.current { color: var(--text); border-color: var(--blue); background: linear-gradient(180deg, rgba(96,165,250,.10), transparent); }
.stage-tab small { display: block; color: inherit; opacity: .8; margin-top: 3px; font-size: 9px; }
.buffer-bar { margin: 10px 0; min-height: 44px; display: flex; align-items: center; gap: 9px; color: var(--muted); font-size: 11px; }
.buffer-label { letter-spacing: .08em; text-transform: uppercase; margin-right: 3px; }
.memory-chip, .buffer-slot { padding: 7px 10px; border-radius: 8px; border: 1px dashed #35506f; background: rgba(11,24,41,.75); transition: .3s ease; }
.memory-chip.ready { color: #c8d9ec; border-style: solid; border-color: #456b91; }
.buffer-slot.ready { color: var(--text); border-style: solid; border-color: #6176a4; background: rgba(177,140,255,.09); }
.buffer-slot.empty::after { content: "等待产物"; color: #5f728c; margin-left: 6px; }
.theatre { position: relative; }
.stage-banner { display: flex; align-items: center; justify-content: space-between; gap: 20px; padding: 13px 16px; background: rgba(13,25,43,.76); border: 1px solid var(--line); border-bottom: 0; border-radius: 14px 14px 0 0; }
.stage-identity { display: flex; align-items: baseline; gap: 13px; }
.step-badge { color: var(--cyan); font-size: 12px; font-weight: 700; letter-spacing: .08em; }
.stage-title { font-size: 19px; font-weight: 600; }
.stage-time { color: var(--muted); font-size: 12px; text-align: right; }
.stage-time strong { color: var(--text); font-weight: 600; margin-left: 8px; }
.theatre-grid {
  position: relative; display: grid; grid-template-columns: minmax(285px, .9fr) 58px minmax(390px, 1.2fr) 58px minmax(300px, .95fr);
  min-height: 480px; background: rgba(7,16,31,.55); border: 1px solid var(--line); border-radius: 0 0 14px 14px; overflow: hidden;
}
.zone { padding: 17px 15px 16px; min-width: 0; }
.zone-head { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(38,58,86,.75); padding-bottom: 10px; margin-bottom: 12px; }
.zone-label { font-size: 11px; letter-spacing: .14em; color: var(--muted); }
.zone-count { font-size: 11px; color: #6f829b; }
.arrow-lane { position: relative; display: flex; align-items: center; justify-content: center; }
.arrow-lane::before { content: ""; width: 38px; height: 2px; background: #31506f; }
.arrow-lane::after { content: ""; border-left: 8px solid #49759d; border-top: 5px solid transparent; border-bottom: 5px solid transparent; }
.arrow-pulse { position: absolute; width: 8px; height: 8px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 13px var(--cyan); animation: pulseAcross 1.45s linear infinite; }
@keyframes pulseAcross { from { transform: translateX(-20px); opacity: 0; } 20%,80% { opacity: 1; } to { transform: translateX(20px); opacity: 0; } }
.artifact-list { display: grid; gap: 9px; align-content: start; }
.artifact-card { position: relative; background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--blue); border-radius: 10px; padding: 10px 11px; overflow: hidden; }
.artifact-card.tone-purple { border-left-color: var(--purple); }
.artifact-card.tone-green { border-left-color: var(--green); }
.artifact-card.tone-amber { border-left-color: var(--amber); }
.artifact-card.entering { animation: artifactBirth .42s cubic-bezier(.2,.8,.2,1) both; }
@keyframes artifactBirth { from { opacity: 0; transform: translateY(14px) scale(.97); } to { opacity: 1; transform: translateY(0) scale(1); } }
.artifact-top { display: flex; justify-content: space-between; align-items: start; gap: 9px; }
.artifact-name { font-size: 12px; font-weight: 650; overflow-wrap: anywhere; }
.artifact-kind { color: var(--muted); font-size: 9px; border: 1px solid #2a405d; border-radius: 99px; padding: 2px 6px; white-space: nowrap; }
.artifact-meta { color: #bed0e6; font-size: 11px; margin-top: 4px; }
.artifact-preview { margin-top: 8px; padding: 7px 8px; background: #091525; border-radius: 6px; color: #849bb7; font: 10px/1.55 ui-monospace, SFMono-Regular, Consolas, monospace; }
.artifact-preview div::before { content: "› "; color: #4c6c91; }
.empty-output { display: grid; place-items: center; min-height: 245px; color: #5e718a; text-align: center; font-size: 12px; border: 1px dashed #263b57; border-radius: 10px; }
.empty-output strong { display: block; color: #8195ad; font-size: 13px; margin-bottom: 5px; }
.operation-list { display: grid; gap: 7px; }
.operation { display: grid; grid-template-columns: 30px 1fr auto; gap: 9px; align-items: start; padding: 10px; border-radius: 9px; border: 1px solid transparent; color: #667a94; transition: .32s ease; }
.operation.active { color: var(--text); background: rgba(96,165,250,.08); border-color: rgba(96,165,250,.36); transform: translateX(3px); }
.operation.done { color: #a7b9cf; }
.operation.error.active, .operation.error.done { background: rgba(255,113,133,.08); border-color: rgba(255,113,133,.32); }
.operation.repair.active, .operation.repair.done { background: rgba(244,191,95,.08); border-color: rgba(244,191,95,.32); }
.operation.success.active, .operation.success.done { background: rgba(81,214,154,.08); border-color: rgba(81,214,154,.30); }
.op-marker { width: 26px; height: 26px; border: 1px solid #30455f; border-radius: 50%; display: grid; place-items: center; font-size: 10px; color: #6f8299; }
.operation.active .op-marker { border-color: var(--blue); color: var(--blue); box-shadow: 0 0 0 4px rgba(96,165,250,.09); }
.operation.done .op-marker { border-color: var(--green); color: var(--green); }
.operation.error .op-marker { border-color: var(--red); color: var(--red); }
.operation.repair .op-marker { border-color: var(--amber); color: var(--amber); }
.op-label { font-size: 12px; font-weight: 650; margin-top: 1px; }
.op-detail { font-size: 10px; line-height: 1.5; margin-top: 3px; color: #8295ad; }
.op-outcome { align-self: center; min-width: 94px; text-align: right; color: #8095ad; font-size: 10px; }
.operation.active .op-outcome { color: #d8e7f8; }
.operation.error .op-outcome { color: var(--red); }
.operation.repair .op-outcome { color: var(--amber); }
.operation.success .op-outcome, .operation.done.success .op-outcome { color: var(--green); }
.handoff-strip { margin-top: 10px; min-height: 52px; display: flex; align-items: center; gap: 11px; padding: 10px 14px; border: 1px solid var(--line); border-radius: 10px; background: rgba(13,25,43,.8); }
.handoff-icon { width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; background: rgba(84,212,232,.1); color: var(--cyan); flex: 0 0 auto; }
.handoff-text { font-size: 12px; color: #bed0e4; line-height: 1.45; }
.handoff-text strong { color: var(--text); }
.transition-curtain { position: absolute; inset: 0; z-index: 8; display: grid; place-items: center; pointer-events: none; background: rgba(7,16,31,.60); opacity: 0; transition: opacity .2s; }
.transition-curtain.show { opacity: 1; }
.transition-message { max-width: 680px; padding: 15px 22px; border-radius: 12px; border: 1px solid #3a5877; background: #0d1b2e; box-shadow: 0 20px 60px rgba(0,0,0,.45); color: #dceafb; font-size: 13px; text-align: center; }
.cargo-clone { position: fixed; z-index: 99; margin: 0; pointer-events: none; box-shadow: 0 15px 45px rgba(0,0,0,.48); }
.footer { margin-top: 12px; display: flex; justify-content: space-between; align-items: center; gap: 20px; color: #71849d; font-size: 10px; }
.footer a { color: #8fa8c4; text-decoration: none; }
.footer a:hover { text-decoration: underline; }
.live-status { color: var(--cyan); }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
}
</style>
</head>
<body>
<main class="app">
  <header class="topbar">
    <div>
      <div class="eyebrow">Real build replay · v20 scene</div>
      <h1>物流沙箱生成实录</h1>
      <p class="lede">不是流程图：画面按真实顺序演示“上一步交付什么 → 本步实际做什么 → 新产物怎样逐个落盘 → 这些产物怎样进入下一步”。所有数量与修复事件来自真实场景 <strong>运营分析-8767b626</strong>。</p>
    </div>
    <div class="run-fact"><strong>13:15:20 → 14:35:51 · 80 分 31 秒</strong><span>v20 completed · sandbox_id 57a3cf55…6c6c</span></div>
  </header>

  <section class="controls" aria-label="播放控制">
    <button id="prevBtn" class="control-btn" type="button">← 上一步</button>
    <button id="playBtn" class="control-btn primary" type="button">暂停</button>
    <button id="nextBtn" class="control-btn" type="button">下一步 →</button>
    <button id="restartBtn" class="control-btn" type="button">重新播放</button>
    <label class="sr-only" for="speedSelect">播放速度</label>
    <select id="speedSelect" class="speed-select" aria-label="播放速度">
      <option value="0.75">0.75×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option>
    </select>
    <div class="progress-wrap">
      <div class="progress-meta"><span id="progressLabel">准备</span><span id="liveStatus" class="live-status" aria-live="polite">正在读取输入</span></div>
      <div class="progress-track"><div id="progressFill" class="progress-fill"></div></div>
    </div>
  </section>

  <nav id="stageRail" class="stage-rail" aria-label="生成阶段"></nav>

  <section class="buffer-bar" aria-label="跨支路上下文与汇合缓冲区">
    <span class="buffer-label">持久交付</span>
    <span id="factorMemory" class="memory-chip">Factor bundle</span>
    <span id="taxonomyMemory" class="memory-chip">Taxonomy bundle</span>
    <span class="buffer-label" style="margin-left:12px">汇合缓冲区</span>
    <span id="dwhBuffer" class="buffer-slot empty">DWH</span>
    <span id="kbBuffer" class="buffer-slot empty">Knowledge</span>
  </section>

  <section class="theatre" aria-label="当前生成阶段">
    <div class="stage-banner">
      <div class="stage-identity"><span id="stepBadge" class="step-badge"></span><span id="stageTitle" class="stage-title"></span></div>
      <div id="stageTime" class="stage-time"></div>
    </div>
    <div class="theatre-grid">
      <section id="inputZone" class="zone input-zone">
        <div class="zone-head"><span class="zone-label">INPUT · 上一步交付</span><span id="inputCount" class="zone-count"></span></div>
        <div id="inputList" class="artifact-list"></div>
      </section>
      <div class="arrow-lane" aria-hidden="true"><span class="arrow-pulse"></span></div>
      <section id="operationZone" class="zone operation-zone">
        <div class="zone-head"><span class="zone-label">WORK · 本步实际生成动作</span><span id="opCount" class="zone-count"></span></div>
        <div id="operationList" class="operation-list"></div>
      </section>
      <div class="arrow-lane" aria-hidden="true"><span class="arrow-pulse"></span></div>
      <section id="outputZone" class="zone output-zone">
        <div class="zone-head"><span class="zone-label">OUTPUT · 本步新产生</span><span id="outputCount" class="zone-count"></span></div>
        <div id="outputList" class="artifact-list"></div>
      </section>
      <div id="transitionCurtain" class="transition-curtain"><div id="transitionMessage" class="transition-message"></div></div>
    </div>
  </section>

  <section class="handoff-strip" aria-label="本步交付说明">
    <div class="handoff-icon" aria-hidden="true">→</div>
    <div id="handoffText" class="handoff-text"></div>
  </section>

  <footer class="footer">
    <span>安全展示：只含数量、文件名与验证摘要；不含任务正文、hidden gold、SQL、数据库行或凭据。</span>
    <span><a href="https://github.com/renjunxiang/sf_my_sandbox/tree/758917009d0ebb0fb36561197171f6abdd279d96/.runtime_state/v20/scenes/运营分析-8767b626">v20 场景证据</a> · 5 号机归档只用于结构核对</span>
  </footer>
</main>

<script id="stageData" type="application/json">__STAGE_DATA__</script>
<script>
(() => {
  'use strict';
  const stages = JSON.parse(document.getElementById('stageData').textContent);
  const els = {
    rail: document.getElementById('stageRail'), step: document.getElementById('stepBadge'),
    title: document.getElementById('stageTitle'), time: document.getElementById('stageTime'),
    inputs: document.getElementById('inputList'), operations: document.getElementById('operationList'),
    outputs: document.getElementById('outputList'), inputCount: document.getElementById('inputCount'),
    opCount: document.getElementById('opCount'), outputCount: document.getElementById('outputCount'),
    handoff: document.getElementById('handoffText'), progress: document.getElementById('progressFill'),
    progressLabel: document.getElementById('progressLabel'), live: document.getElementById('liveStatus'),
    play: document.getElementById('playBtn'), prev: document.getElementById('prevBtn'),
    next: document.getElementById('nextBtn'), restart: document.getElementById('restartBtn'),
    speed: document.getElementById('speedSelect'), curtain: document.getElementById('transitionCurtain'),
    transitionMessage: document.getElementById('transitionMessage'), inputZone: document.getElementById('inputZone'),
    factorMemory: document.getElementById('factorMemory'), taxonomyMemory: document.getElementById('taxonomyMemory'),
    dwhBuffer: document.getElementById('dwhBuffer'), kbBuffer: document.getElementById('kbBuffer')
  };
  const state = { stage: 0, op: -1, playing: true, speed: 1, timer: null, transitioning: false };
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));

  function artifactCard(item, entering = false) {
    const preview = item.preview.map(line => `<div>${escapeHtml(line)}</div>`).join('');
    return `<article class="artifact-card tone-${escapeHtml(item.tone)}${entering ? ' entering output-visible' : ''}" data-artifact="${escapeHtml(item.name)}">
      <div class="artifact-top"><span class="artifact-name">${escapeHtml(item.name)}</span><span class="artifact-kind">${escapeHtml(item.kind)}</span></div>
      <div class="artifact-meta">${escapeHtml(item.meta)}</div><div class="artifact-preview">${preview}</div>
    </article>`;
  }

  function renderRail() {
    els.rail.innerHTML = stages.map((stage, index) => {
      const stateClass = index < state.stage ? 'done' : index === state.stage ? 'current' : '';
      return `<button type="button" class="stage-tab ${stateClass}" data-index="${index}" aria-current="${index === state.stage ? 'step' : 'false'}">${escapeHtml(stage.step)}<small>${escapeHtml(stage.duration)}</small></button>`;
    }).join('');
    els.rail.querySelectorAll('button').forEach(button => button.addEventListener('click', () => jumpTo(Number(button.dataset.index))));
  }

  function renderBuffers(stage, complete) {
    const factorReady = state.stage > 2 || (state.stage === 2 && complete);
    const taxonomyReady = state.stage > 3 || (state.stage === 3 && complete);
    const dwhReady = state.stage > 6 || (state.stage === 6 && complete);
    const kbReady = state.stage > 9 || (state.stage === 9 && complete);
    els.factorMemory.classList.toggle('ready', factorReady);
    els.taxonomyMemory.classList.toggle('ready', taxonomyReady);
    els.dwhBuffer.className = `buffer-slot ${dwhReady ? 'ready' : 'empty'}`;
    els.kbBuffer.className = `buffer-slot ${kbReady ? 'ready' : 'empty'}`;
    els.dwhBuffer.textContent = dwhReady ? 'DWH · 555 tasks' : 'DWH';
    els.kbBuffer.textContent = kbReady ? 'Knowledge · 465 tasks' : 'Knowledge';
  }

  function renderStage() {
    const stage = stages[state.stage];
    const complete = state.op >= stage.operations.length - 1;
    els.step.textContent = `${stage.step} · ${stage.branch}`;
    els.title.textContent = stage.title;
    els.time.innerHTML = `${escapeHtml(stage.clock)} <strong>${escapeHtml(stage.duration)}</strong>`;
    els.inputs.innerHTML = stage.inputs.map(item => artifactCard(item)).join('');
    els.inputCount.textContent = `${stage.inputs.length} 项`;
    els.operations.innerHTML = stage.operations.map((op, index) => {
      const phase = index < state.op ? 'done' : index === state.op ? 'active' : '';
      const marker = index < state.op ? '✓' : index === state.op ? (op.status === 'error' ? '!' : op.status === 'repair' ? '↻' : '●') : index + 1;
      return `<div class="operation ${phase} ${escapeHtml(op.status)}">
        <div class="op-marker">${marker}</div><div><div class="op-label">${escapeHtml(op.label)}</div><div class="op-detail">${escapeHtml(op.detail)}</div></div><div class="op-outcome">${escapeHtml(op.outcome)}</div>
      </div>`;
    }).join('');
    els.opCount.textContent = state.op < 0 ? `0 / ${stage.operations.length}` : `${Math.min(state.op + 1, stage.operations.length)} / ${stage.operations.length}`;
    const visibleOutputs = stage.outputs.filter(item => item.produced_by <= state.op);
    els.outputs.innerHTML = visibleOutputs.length
      ? visibleOutputs.map(item => artifactCard(item, true)).join('')
      : '<div class="empty-output"><div><strong>还没有产物</strong>动作完成后，文件会在这里逐个生成</div></div>';
    els.outputCount.textContent = `${visibleOutputs.length} / ${stage.outputs.length}`;
    els.handoff.innerHTML = complete ? `<strong>本步完成并交付：</strong>${escapeHtml(stage.handoff)}` : '<strong>生成中：</strong>右侧只显示已经真正产生的文件，未完成产物不会提前出现。';
    const stageFraction = (state.op + 1) / Math.max(stage.operations.length, 1);
    const totalProgress = ((state.stage + Math.max(0, stageFraction)) / stages.length) * 100;
    els.progress.style.width = `${Math.min(100, totalProgress)}%`;
    els.progressLabel.textContent = `${stage.step} · ${state.stage + 1}/${stages.length}`;
    els.live.textContent = complete ? '本步产物已齐备，准备交付' : state.op < 0 ? '正在读取上一步产物' : `正在执行：${stage.operations[state.op].label}`;
    els.prev.disabled = state.stage === 0 && state.op <= -1;
    els.next.disabled = state.transitioning;
    els.play.textContent = state.playing ? '暂停' : '继续播放';
    renderRail();
    renderBuffers(stage, complete);
  }

  function clearTimer() {
    if (state.timer !== null) { window.clearTimeout(state.timer); state.timer = null; }
  }

  function schedule() {
    clearTimer();
    if (!state.playing || state.transitioning) return;
    const stage = stages[state.stage];
    const delay = (state.op < 0 ? 520 : stage.replay_ms) / state.speed;
    state.timer = window.setTimeout(tick, delay);
  }

  function tick() {
    const stage = stages[state.stage];
    if (state.op < stage.operations.length - 1) {
      state.op += 1;
      renderStage();
      schedule();
      return;
    }
    if (state.stage === stages.length - 1) {
      state.playing = false;
      els.live.textContent = '完整生成重放结束';
      renderStage();
      return;
    }
    animateHandoff(state.stage + 1);
  }

  function transferSources(type) {
    if (type === 'merge') return [els.dwhBuffer, els.kbBuffer];
    if (type === 'branch-return') return [els.factorMemory, els.taxonomyMemory];
    return Array.from(els.outputs.querySelectorAll('.output-visible')).slice(-3);
  }

  function animateHandoff(nextIndex) {
    if (state.transitioning) return;
    state.transitioning = true;
    clearTimer();
    const current = stages[state.stage];
    els.transitionMessage.textContent = current.handoff;
    els.curtain.classList.add('show');
    const sources = transferSources(current.handoff_type);
    const target = els.inputZone.getBoundingClientRect();
    const animations = [];
    if (!reducedMotion && sources.length) {
      sources.forEach((source, index) => {
        const rect = source.getBoundingClientRect();
        const clone = source.cloneNode(true);
        clone.classList.add('cargo-clone');
        clone.style.left = `${rect.left}px`; clone.style.top = `${rect.top}px`;
        clone.style.width = `${rect.width}px`; clone.style.height = `${rect.height}px`;
        document.body.appendChild(clone);
        const dx = target.left + 12 - rect.left;
        const dy = target.top + 76 + index * 48 - rect.top;
        const animation = clone.animate([
          { transform: 'translate(0,0) scale(1)', opacity: 1 },
          { transform: `translate(${dx * .48}px, ${dy * .28 - 22}px) scale(.88)`, opacity: .96, offset: .45 },
          { transform: `translate(${dx}px, ${dy}px) scale(.68)`, opacity: .12 }
        ], { duration: 900 / state.speed, easing: 'cubic-bezier(.35,.05,.18,1)', fill: 'forwards' });
        animations.push(animation.finished.catch(() => undefined).finally(() => clone.remove()));
      });
    }
    Promise.all(animations).then(() => window.setTimeout(() => {
      state.stage = nextIndex; state.op = -1; state.transitioning = false;
      els.curtain.classList.remove('show'); renderStage(); schedule();
    }, reducedMotion ? 80 : 220 / state.speed));
  }

  function jumpTo(index) {
    clearTimer(); state.stage = Math.max(0, Math.min(stages.length - 1, index));
    state.op = -1; state.transitioning = false; els.curtain.classList.remove('show');
    renderStage(); schedule();
  }

  els.play.addEventListener('click', () => { state.playing = !state.playing; renderStage(); schedule(); });
  els.prev.addEventListener('click', () => jumpTo(state.stage - 1));
  els.next.addEventListener('click', () => {
    if (state.op < stages[state.stage].operations.length - 1) { state.op += 1; renderStage(); schedule(); }
    else if (state.stage < stages.length - 1) animateHandoff(state.stage + 1);
  });
  els.restart.addEventListener('click', () => { state.playing = true; jumpTo(0); });
  els.speed.addEventListener('change', () => { state.speed = Number(els.speed.value); schedule(); });
  document.addEventListener('keydown', event => {
    if (event.key === 'ArrowRight') els.next.click();
    if (event.key === 'ArrowLeft') els.prev.click();
    if (event.key === ' ') { event.preventDefault(); els.play.click(); }
  });

  renderStage();
  requestAnimationFrame(() => schedule());
})();
</script>
</body>
</html>
'''


WORLD_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>物流世界建模实录 · 从业务意图到可执行沙箱</title>
<style>
:root {
  color-scheme: dark;
  --void:#030812; --deep:#07111f; --panel:#0b1728; --panel2:#101f34;
  --line:#263c59; --text:#edf6ff; --muted:#91a7c1; --cyan:#4de1ff;
  --blue:#5b9dff; --violet:#ae7cff; --green:#4fe0a0; --amber:#ffc562; --red:#ff657d;
}
* { box-sizing:border-box; }
html { background:var(--void); }
body { margin:0; min-width:1180px; color:var(--text); background:var(--void); font-family:Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif; }
button,select { font:inherit; }
button:focus-visible,select:focus-visible { outline:3px solid rgba(77,225,255,.55); outline-offset:2px; }
.sr-only { position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0; }
.app { position:relative; min-height:100vh; padding:22px 28px 18px; overflow:hidden; isolation:isolate; background:
  radial-gradient(circle at 13% 6%,rgba(55,128,255,.18),transparent 28%),
  radial-gradient(circle at 86% 8%,rgba(174,124,255,.15),transparent 28%),
  linear-gradient(180deg,#06101e 0%,#030812 100%); }
.app::before { content:"";position:fixed;inset:0;z-index:-2;pointer-events:none;opacity:.16;background-image:
  linear-gradient(rgba(77,225,255,.14) 1px,transparent 1px),linear-gradient(90deg,rgba(77,225,255,.14) 1px,transparent 1px);
  background-size:54px 54px;transform:perspective(600px) rotateX(58deg) scale(1.6) translateY(14%);transform-origin:center bottom;animation:gridDrift 14s linear infinite; }
.app::after { content:"";position:fixed;inset:-35%;z-index:-1;pointer-events:none;background:conic-gradient(from 0deg,transparent,rgba(77,225,255,.04),transparent,rgba(174,124,255,.05),transparent);animation:auroraSpin 24s linear infinite; }
@keyframes gridDrift { to { background-position:0 54px,54px 0; } }
@keyframes auroraSpin { to { transform:rotate(360deg); } }
.topbar { display:flex;justify-content:space-between;align-items:flex-start;gap:28px; }
.eyebrow { color:var(--cyan);font-size:11px;letter-spacing:.2em;text-transform:uppercase;margin-bottom:7px;text-shadow:0 0 16px rgba(77,225,255,.55); }
h1 { margin:0;font-size:29px;letter-spacing:-.025em;font-weight:650; }
.lede { margin:7px 0 0;max-width:860px;color:var(--muted);font-size:13px;line-height:1.65; }
.world-equation { min-width:310px;padding:12px 15px;border:1px solid rgba(77,225,255,.28);border-radius:12px;background:linear-gradient(135deg,rgba(77,225,255,.08),rgba(174,124,255,.07));box-shadow:inset 0 0 24px rgba(77,225,255,.035),0 10px 38px rgba(0,0,0,.2); }
.world-equation strong { display:block;color:#dffbff;font:13px ui-monospace,SFMono-Regular,Consolas,monospace;margin-bottom:4px; }
.world-equation span { color:var(--muted);font-size:11px; }
.controls { margin-top:15px;display:flex;align-items:center;gap:7px; }
.control-btn,.speed-select { min-height:34px;padding:6px 11px;border:1px solid var(--line);border-radius:8px;background:rgba(13,27,46,.88);color:var(--text);cursor:pointer; }
.control-btn.primary { min-width:90px;color:#05101b;background:linear-gradient(135deg,#bdf6ff,#d8caff);border-color:transparent;box-shadow:0 0 24px rgba(77,225,255,.16); }
.control-btn:disabled { opacity:.42;cursor:default;box-shadow:none; }
.progress-wrap { flex:1;margin-left:9px; }
.progress-meta { display:flex;justify-content:space-between;color:var(--muted);font-size:10px;margin-bottom:5px; }
.live-status { color:var(--cyan);text-shadow:0 0 12px rgba(77,225,255,.35); }
.progress-track { height:4px;background:#15263c;border-radius:10px;overflow:hidden; }
.progress-fill { height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--blue),var(--violet));box-shadow:0 0 14px var(--cyan);transition:width .35s ease; }
.stage-rail { margin-top:13px;display:grid;grid-template-columns:repeat(13,1fr);gap:4px; }
.stage-tab { border:0;border-top:2px solid #1f334d;background:transparent;color:#667d98;padding:7px 2px 6px;cursor:pointer;font-size:9px;white-space:nowrap; }
.stage-tab small { display:block;margin-top:3px;font-size:8px;opacity:.72; }
.stage-tab.done { color:#9bb1c9;border-color:#42688c; }
.stage-tab.current { color:var(--text);border-color:var(--cyan);background:linear-gradient(180deg,rgba(77,225,255,.12),transparent);text-shadow:0 0 10px rgba(77,225,255,.3); }
.theatre { position:relative;margin-top:11px;border:1px solid rgba(60,91,128,.75);border-radius:15px;overflow:hidden;background:rgba(4,11,21,.68);box-shadow:0 24px 80px rgba(0,0,0,.34),inset 0 1px rgba(255,255,255,.025); }
.stage-banner { position:relative;overflow:hidden;display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:20px;min-height:58px;padding:11px 16px;border-bottom:1px solid var(--line);background:linear-gradient(90deg,rgba(13,28,48,.94),rgba(8,18,32,.88)); }
.stage-banner::after { content:"";position:absolute;inset:0;width:32%;background:linear-gradient(90deg,transparent,rgba(77,225,255,.10),transparent);transform:skewX(-20deg);animation:bannerScan 3.8s ease-in-out infinite; }
@keyframes bannerScan { 0%,20% { transform:translateX(-160%) skewX(-20deg); } 75%,100% { transform:translateX(430%) skewX(-20deg); } }
.stage-id { position:relative;z-index:1; }
.step-badge { color:var(--cyan);font-size:11px;letter-spacing:.09em;font-weight:700; }
.stage-title { display:block;margin-top:3px;font-size:18px;font-weight:650; }
.principle { position:relative;z-index:1;text-align:center;color:#bed2e8;font-size:12px; }
.principle::before { content:"世界建模原则";display:block;color:#64809f;font-size:8px;letter-spacing:.18em;margin-bottom:3px; }
.stage-time { position:relative;z-index:1;text-align:right;color:var(--muted);font-size:10px; }
.stage-time strong { display:block;color:var(--text);font-size:12px;margin-top:4px; }
.world-layout { position:relative;display:grid;grid-template-columns:225px minmax(600px,1fr) 310px;min-height:590px; }
.inherit-panel,.build-panel { padding:15px 14px;background:rgba(8,18,32,.64); }
.inherit-panel { border-right:1px solid rgba(38,60,89,.7); }
.build-panel { border-left:1px solid rgba(38,60,89,.7); }
.panel-label { color:#7087a3;font-size:9px;letter-spacing:.16em;text-transform:uppercase;margin-bottom:10px; }
.inherit-list { display:grid;gap:8px; }
.concept-chip { position:relative;padding:10px 10px 10px 28px;border:1px solid #263f5f;border-radius:9px;background:linear-gradient(135deg,rgba(22,43,69,.72),rgba(9,22,39,.72));color:#c3d5e8;font-size:11px;line-height:1.4;animation:conceptIn .45s ease both; }
.concept-chip::before { content:"";position:absolute;left:11px;top:15px;width:7px;height:7px;border-radius:50%;background:var(--cyan);box-shadow:0 0 12px var(--cyan); }
@keyframes conceptIn { from { opacity:0;transform:translateX(-12px); } to { opacity:1;transform:none; } }
.formula-box { margin-top:13px;padding:11px;border-left:2px solid var(--violet);background:rgba(174,124,255,.06);color:#d6c9f7;font:10px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere; }
.inherit-note { margin-top:12px;color:#6f86a1;font-size:10px;line-height:1.55; }
.world-main { position:relative;padding:14px 15px 15px;min-width:0; }
.world-head { display:flex;align-items:center;justify-content:space-between;margin-bottom:9px; }
.world-head span:first-child { color:var(--cyan);font-size:10px;letter-spacing:.18em;text-shadow:0 0 12px rgba(77,225,255,.4); }
.world-head span:last-child { color:#647d99;font-size:9px; }
.world-canvas { position:relative;height:500px;overflow:hidden;border:1px solid #294363;border-radius:14px;background:
  radial-gradient(circle at 50% 48%,rgba(40,114,182,.18),transparent 29%),
  linear-gradient(rgba(54,100,145,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(54,100,145,.08) 1px,transparent 1px),
  linear-gradient(180deg,#071525,#040c17);background-size:auto,35px 35px,35px 35px,auto;box-shadow:inset 0 0 80px rgba(0,0,0,.48),0 14px 44px rgba(0,0,0,.25); }
.world-canvas::before { content:"";position:absolute;inset:-45%;background:conic-gradient(from 0deg,transparent,rgba(77,225,255,.055),transparent,rgba(174,124,255,.06),transparent);animation:worldAura 13s linear infinite; }
@keyframes worldAura { to { transform:rotate(360deg); } }
.world-caption { position:absolute;left:14px;bottom:12px;right:14px;z-index:30;display:flex;align-items:center;gap:10px;padding:8px 10px;border:1px solid rgba(77,225,255,.2);border-radius:9px;background:rgba(3,10,19,.84);backdrop-filter:blur(8px); }
.change-pulse { width:8px;height:8px;border-radius:50%;background:var(--cyan);box-shadow:0 0 14px var(--cyan);animation:changePulse 1.1s ease-in-out infinite; }
@keyframes changePulse { 50% { transform:scale(1.7);opacity:.45; } }
.world-caption strong { color:#e6faff;font-size:11px; }
.world-caption span { color:#91a9c4;font-size:10px; }
.world-layer { position:absolute;inset:0;z-index:3;opacity:0;transform:scale(.96);transition:opacity .55s ease,transform .55s cubic-bezier(.2,.8,.2,1);pointer-events:none; }
.world-layer.visible { opacity:1;transform:scale(1); }
.world-layer.current { filter:drop-shadow(0 0 12px rgba(77,225,255,.28)); }
.world-boundary-ring { position:absolute;left:8%;top:8%;width:84%;height:78%;border:1px solid rgba(77,225,255,.48);border-radius:47% 42% 45% 43%;box-shadow:inset 0 0 35px rgba(77,225,255,.05),0 0 32px rgba(77,225,255,.08); }
.world-boundary-ring::before { content:"WORLD BOUNDARY · 运营分析-8767b626";position:absolute;left:9%;top:-8px;padding:2px 7px;background:#071525;color:var(--cyan);font-size:8px;letter-spacing:.12em; }
.role { position:absolute;z-index:8;width:78px;text-align:center;color:#b9cce0;font-size:9px; }
.role::before { content:"";display:block;width:25px;height:25px;margin:0 auto 5px;border:1px solid var(--blue);border-radius:50% 50% 42% 42%;background:radial-gradient(circle at 50% 36%,var(--blue) 0 4px,transparent 5px),linear-gradient(135deg,rgba(91,157,255,.3),transparent);box-shadow:0 0 18px rgba(91,157,255,.25); }
.role.r1 { left:4%;top:35%; }.role.r2 { right:3%;top:37%; }.role.r3 { left:44%;top:3%; }
.network-svg { position:absolute;inset:0;width:100%;height:100%;overflow:visible; }
.route-path { fill:none;stroke:rgba(91,157,255,.48);stroke-width:2;stroke-dasharray:7 8;animation:routeFlow 2.6s linear infinite; }
.route-path.knowledge { stroke:rgba(174,124,255,.42); }
@keyframes routeFlow { to { stroke-dashoffset:-30; } }
.entity-node { position:absolute;z-index:12;transform:translate(-50%,-50%);min-width:78px;padding:8px 9px;border:1px solid #3b6086;border-radius:10px;background:rgba(9,25,43,.94);text-align:center;color:#dcecff;font-size:10px;box-shadow:0 8px 24px rgba(0,0,0,.35),inset 0 0 18px rgba(91,157,255,.06); }
.entity-node b { display:block;color:var(--cyan);font-size:8px;letter-spacing:.08em;margin-bottom:3px; }
.entity-node.hub { left:50%;top:48%; }.entity-node.warehouse { left:22%;top:30%; }.entity-node.outlet { left:78%;top:29%; }.entity-node.carrier { left:22%;top:70%; }.entity-node.customer { left:78%;top:70%; }
.state-ring { position:absolute;left:50%;top:48%;width:178px;height:178px;transform:translate(-50%,-50%);border:1px dashed rgba(79,224,160,.7);border-radius:50%;animation:stateSpin 12s linear infinite; }
.state-ring::before,.state-ring::after { content:"运输中";position:absolute;padding:3px 6px;border-radius:8px;background:#0c2b27;color:var(--green);font-size:8px; }
.state-ring::before { left:3px;top:22px; }.state-ring::after { content:"已签收";right:-4px;bottom:27px; }
.state-label { position:absolute;left:50%;top:calc(48% + 103px);transform:translateX(-50%);color:var(--green);font-size:8px;letter-spacing:.12em; }
@keyframes stateSpin { to { transform:translate(-50%,-50%) rotate(360deg); } }
.action-beam { position:absolute;height:1px;background:linear-gradient(90deg,transparent,var(--amber),transparent);transform-origin:left center;box-shadow:0 0 8px var(--amber);animation:beamPulse 1.7s ease-in-out infinite; }
.action-beam.a1 { left:27%;top:34%;width:175px;transform:rotate(16deg); }.action-beam.a2 { left:51%;top:49%;width:170px;transform:rotate(-27deg); }.action-beam.a3 { left:25%;top:69%;width:190px;transform:rotate(-17deg); }
.action-name { position:absolute;color:var(--amber);font:8px ui-monospace,monospace; }.action-name.n1 { left:35%;top:31%; }.action-name.n2 { left:64%;top:43%; }.action-name.n3 { left:34%;top:67%; }
@keyframes beamPulse { 50% { opacity:.28;filter:blur(1px); } }
.taxonomy-orbit { position:absolute;left:50%;top:48%;width:480px;height:365px;transform:translate(-50%,-50%);border:1px dashed rgba(174,124,255,.55);border-radius:50%;animation:orbitBreath 3.2s ease-in-out infinite; }
.axis { position:absolute;padding:3px 6px;border:1px solid rgba(174,124,255,.38);border-radius:8px;background:rgba(32,19,57,.88);color:#cfb7ff;font-size:8px; }.axis.x1{left:12%;top:14%}.axis.x2{right:9%;top:15%}.axis.x3{left:5%;bottom:15%}.axis.x4{right:7%;bottom:13%}.axis.x5{left:45%;top:5%}
@keyframes orbitBreath { 50% { box-shadow:0 0 38px rgba(174,124,255,.14);transform:translate(-50%,-50%) scale(1.025); } }
.constraint-stack { position:absolute;right:9%;top:51%;display:grid;gap:5px; }.constraint { padding:4px 7px;border-left:2px solid var(--red);background:rgba(70,16,30,.72);color:#ff9aaa;font-size:8px; }.constraint.ok { border-color:var(--green);background:rgba(15,61,48,.72);color:#85efbd; }
.schema-deck { position:absolute;left:9%;bottom:12%;width:210px;display:grid;grid-template-columns:repeat(4,1fr);gap:4px;transform:perspective(300px) rotateX(50deg);transform-origin:center bottom; }
.table-tile { height:29px;border:1px solid rgba(91,157,255,.5);background:linear-gradient(180deg,rgba(31,72,111,.9),rgba(10,28,48,.9));box-shadow:0 5px 0 rgba(13,36,61,.9);color:#9cc8ff;font-size:6px;display:grid;place-items:center;text-align:center; }
.schema-count { position:absolute;left:12%;bottom:5%;color:var(--blue);font-size:9px;letter-spacing:.12em; }
.shipment { position:absolute;z-index:18;width:8px;height:8px;border-radius:50%;background:var(--cyan);box-shadow:0 0 15px var(--cyan),0 0 3px #fff;animation:shipmentMove 3.2s ease-in-out infinite; }.shipment.s2{animation-delay:-1.1s}.shipment.s3{animation-delay:-2.2s}
@keyframes shipmentMove { 0%{left:22%;top:30%;opacity:0}12%{opacity:1}50%{left:50%;top:48%}88%{opacity:1}100%{left:78%;top:70%;opacity:0} }
.fact-counter { position:absolute;left:43%;bottom:8%;padding:6px 9px;border:1px solid rgba(77,225,255,.35);background:rgba(4,19,31,.86);color:var(--cyan);font:9px ui-monospace,monospace;box-shadow:0 0 18px rgba(77,225,255,.1); }
.probe { position:absolute;z-index:20;width:95px;height:95px;border:1px solid;border-radius:50%;display:grid;place-items:center;text-align:center;font-size:8px;line-height:1.4;animation:probeScan 2.4s ease-in-out infinite; }.probe.dwh{left:7%;top:43%;border-color:var(--blue);color:#9cc8ff;background:radial-gradient(circle,rgba(91,157,255,.14),transparent 66%)}.probe.kb{right:6%;top:43%;border-color:var(--violet);color:#d1baff;background:radial-gradient(circle,rgba(174,124,255,.14),transparent 66%)}
@keyframes probeScan { 50%{box-shadow:0 0 0 16px transparent,0 0 28px currentColor;transform:scale(1.06)} }
.policy-cloud { position:absolute;right:7%;top:9%;width:190px;display:grid;gap:5px; }.policy { padding:6px 8px;border:1px solid rgba(174,124,255,.38);border-radius:7px;background:rgba(29,18,50,.88);color:#d4c1f5;font-size:8px;box-shadow:5px 5px 0 rgba(25,15,43,.55); }.policy b{color:var(--violet);margin-right:5px}
.doc-stack { position:absolute;right:12%;top:21%;width:128px;height:100px; }.doc-page { position:absolute;inset:0;border:1px solid #765cb0;border-radius:6px;background:linear-gradient(135deg,#251a3d,#0e1728);box-shadow:0 8px 22px rgba(0,0,0,.3); }.doc-page:nth-child(1){transform:translate(-12px,12px) rotate(-5deg)}.doc-page:nth-child(2){transform:translate(-6px,6px) rotate(-2deg)}.doc-page:nth-child(3){padding:13px 10px;color:#cfbaff;font-size:8px}.doc-page i{display:block;height:2px;margin:6px 0;background:#5b477f}
.hybrid-core { position:absolute;left:50%;top:48%;z-index:25;width:130px;height:130px;transform:translate(-50%,-50%);border-radius:50%;display:grid;place-items:center;text-align:center;color:#f0eaff;font-size:9px;background:radial-gradient(circle,rgba(174,124,255,.46),rgba(77,225,255,.12) 38%,transparent 68%);border:1px solid rgba(211,191,255,.7);box-shadow:0 0 38px rgba(174,124,255,.33),inset 0 0 25px rgba(77,225,255,.2);animation:hybridPulse 1.8s ease-in-out infinite; }
.hybrid-core strong { display:block;font-size:17px;margin-bottom:2px; }.hybrid-core::before,.hybrid-core::after { content:"";position:absolute;inset:-24px;border:1px solid rgba(77,225,255,.22);border-radius:50%;animation:hybridOrbit 4s linear infinite; }.hybrid-core::after{inset:-42px;border-color:rgba(174,124,255,.18);animation-direction:reverse;animation-duration:6s}
@keyframes hybridPulse{50%{transform:translate(-50%,-50%) scale(1.06)}}@keyframes hybridOrbit{to{transform:rotate(360deg)}}
.repair-alert { position:absolute;left:50%;top:11%;z-index:26;transform:translateX(-50%);padding:7px 11px;border:1px solid var(--red);border-radius:8px;background:rgba(62,13,26,.94);color:#ff9aac;font-size:8px;box-shadow:0 0 22px rgba(255,101,125,.24);animation:alertShake .35s ease 2; }.repair-alert.resolved { border-color:var(--green);background:rgba(11,55,42,.94);color:#83efbb; }
@keyframes alertShake{25%{transform:translateX(calc(-50% - 4px))}75%{transform:translateX(calc(-50% + 4px))}}
.freeze-shell { position:absolute;left:50%;top:48%;z-index:28;width:570px;height:410px;transform:translate(-50%,-50%);border:2px solid rgba(79,224,160,.72);border-radius:48% 43% 46% 42%;box-shadow:inset 0 0 58px rgba(79,224,160,.08),0 0 50px rgba(79,224,160,.14); }.freeze-shell::before { content:"SEALED WORLD · CHECKSUM VERIFIED";position:absolute;left:50%;top:-10px;transform:translateX(-50%);padding:3px 10px;background:#061421;color:var(--green);font-size:8px;letter-spacing:.14em;white-space:nowrap; }
.agent-node { position:absolute;z-index:29;right:1%;top:42%;width:67px;height:67px;border:1px solid var(--amber);border-radius:16px;display:grid;place-items:center;text-align:center;color:var(--amber);font-size:8px;background:rgba(47,33,10,.9);box-shadow:0 0 24px rgba(255,197,98,.18); }.agent-beam { position:absolute;z-index:27;right:8%;top:49%;width:165px;height:2px;background:linear-gradient(90deg,var(--amber),transparent);box-shadow:0 0 10px var(--amber); }
.gain-list { display:grid;gap:6px; }
.gain-step { display:grid;grid-template-columns:25px 1fr;gap:8px;padding:8px;border:1px solid transparent;border-radius:8px;color:#617995;transition:.3s ease; }
.gain-step.active { color:var(--text);border-color:rgba(77,225,255,.35);background:rgba(77,225,255,.065);transform:translateX(-3px); }
.gain-step.done { color:#a8bdd3; }
.gain-marker { width:23px;height:23px;border:1px solid #314b68;border-radius:50%;display:grid;place-items:center;font-size:8px; }
.gain-step.active .gain-marker { color:var(--cyan);border-color:var(--cyan);box-shadow:0 0 14px rgba(77,225,255,.3); }.gain-step.done .gain-marker{color:var(--green);border-color:var(--green)}
.gain-title { font-size:10px;font-weight:650; }.gain-op { color:#7087a1;font-size:8px;margin-top:3px;line-height:1.4; }.gain-result { color:#94abc3;font-size:8px;margin-top:3px; }
.capability { margin-top:12px;padding:11px;border:1px solid #29425f;border-radius:10px;background:linear-gradient(135deg,rgba(20,40,65,.75),rgba(9,22,38,.75)); }
.capability::before { content:"WORLD CAPABILITY";display:block;color:#617b99;font-size:8px;letter-spacing:.15em;margin-bottom:6px; }
.capability p { margin:0;color:#8ca3bd;font-size:10px;line-height:1.55; }
.capability.ready { border-color:rgba(79,224,160,.4);box-shadow:inset 0 0 25px rgba(79,224,160,.045); }.capability.ready p{color:#c3e8d7}
.evidence-label { margin-top:12px;color:#607995;font-size:8px;letter-spacing:.14em; }
.evidence-list { margin-top:7px;display:flex;flex-wrap:wrap;gap:5px; }
.evidence-chip { max-width:100%;padding:4px 6px;border:1px solid #253c58;border-radius:6px;background:rgba(7,17,30,.74);color:#7991ab;font:7px ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere;animation:evidenceIn .35s ease both; }
@keyframes evidenceIn{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
.handoff { min-height:44px;display:flex;align-items:center;gap:10px;padding:8px 14px;border-top:1px solid var(--line);background:rgba(9,20,34,.88);color:#a8bfd7;font-size:10px; }
.handoff-symbol { color:var(--cyan);font-size:18px;text-shadow:0 0 14px var(--cyan); }.handoff strong{color:var(--text)}
.transition-curtain { position:absolute;inset:58px 0 44px;z-index:60;display:grid;place-items:center;pointer-events:none;background:rgba(2,8,15,.7);backdrop-filter:blur(4px);opacity:0;transition:opacity .2s; }.transition-curtain.show{opacity:1}.transition-message{max-width:680px;padding:13px 20px;border:1px solid rgba(77,225,255,.42);border-radius:12px;background:#081829;color:#d8f4ff;font-size:11px;text-align:center;box-shadow:0 0 45px rgba(77,225,255,.12)}
.transfer-orb,.transfer-spark { position:fixed;z-index:99;pointer-events:none;border-radius:50%;background:var(--cyan);box-shadow:0 0 18px var(--cyan),0 0 42px rgba(77,225,255,.4); }.transfer-orb{width:18px;height:18px}.transfer-spark{width:4px;height:4px}
.end-overlay { position:absolute;inset:0;z-index:80;display:grid;place-items:center;text-align:center;background:radial-gradient(circle at 50% 45%,rgba(30,99,92,.32),rgba(3,9,17,.94) 55%);opacity:0;visibility:hidden;transition:opacity .6s ease; }.end-overlay.show{opacity:1;visibility:visible}
.end-seal { position:relative;width:430px;padding:28px 28px 24px;border:1px solid rgba(79,224,160,.52);border-radius:18px;background:linear-gradient(145deg,rgba(10,36,38,.94),rgba(8,18,31,.96));box-shadow:0 0 80px rgba(79,224,160,.17),inset 0 0 42px rgba(79,224,160,.04); }
.end-check { width:76px;height:76px;margin:0 auto 16px;border:2px solid var(--green);border-radius:50%;display:grid;place-items:center;color:var(--green);font-size:37px;box-shadow:0 0 0 12px rgba(79,224,160,.05),0 0 34px rgba(79,224,160,.3);animation:sealIn .75s cubic-bezier(.2,.85,.2,1) both; }
@keyframes sealIn{from{transform:scale(.2) rotate(-90deg);opacity:0}to{transform:none;opacity:1}}
.end-kicker { color:var(--green);font-size:9px;letter-spacing:.2em; }.end-title { margin:7px 0 5px;font-size:24px;font-weight:650; }.end-id { color:#92b5ad;font:9px ui-monospace,monospace; }.end-metrics { display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:18px 0; }.end-metric { padding:9px 5px;border-top:1px solid rgba(79,224,160,.25);color:#8eaaa7;font-size:8px; }.end-metric strong{display:block;color:#ddfff4;font-size:14px;margin-bottom:3px}.end-restart{padding:7px 14px;border:1px solid rgba(79,224,160,.5);border-radius:8px;background:rgba(79,224,160,.09);color:#cffff0;cursor:pointer}
.completion-particle{position:absolute;left:50%;top:45%;width:5px;height:5px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green);pointer-events:none}
.footer { margin-top:10px;display:flex;justify-content:space-between;color:#5f7690;font-size:9px; }.footer a{color:#7f9ab6;text-decoration:none}
body.is-finished .app::before,body.is-finished .app::after,body.is-finished .stage-banner::after,body.is-finished .world-canvas::before,body.is-finished .route-path,body.is-finished .state-ring,body.is-finished .taxonomy-orbit,body.is-finished .shipment,body.is-finished .probe,body.is-finished .hybrid-core,body.is-finished .hybrid-core::before,body.is-finished .hybrid-core::after,body.is-finished .change-pulse { animation-play-state:paused!important; }
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
</style>
</head>
<body>
<main class="app">
  <header class="topbar">
    <div><div class="eyebrow">World modeling replay · v20 logistics scene</div><h1>从业务意图到可执行物流世界</h1><p class="lede">主角不是文件，而是一个逐步获得边界、角色、实体、状态、动作、约束、事实与规则的物流世界。文件只在右侧作为世界模型已经落盘的技术凭证。</p></div>
    <div class="world-equation"><strong>W = (Entities, States, Actions, Rules, Observations)</strong><span>运营分析-8767b626 · 80 分 31 秒后封装为可运行沙箱</span></div>
  </header>
  <section class="controls" aria-label="播放控制">
    <button id="prevBtn" class="control-btn" type="button">← 上一阶段</button><button id="playBtn" class="control-btn primary" type="button">暂停</button><button id="nextBtn" class="control-btn" type="button">下一动作 →</button><button id="restartBtn" class="control-btn" type="button">重新建模</button>
    <label class="sr-only" for="speedSelect">播放速度</label><select id="speedSelect" class="speed-select" aria-label="播放速度"><option value=".75">0.75×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select>
    <div class="progress-wrap"><div class="progress-meta"><span id="progressLabel">准备</span><span id="liveStatus" class="live-status" aria-live="polite">正在建立世界边界</span></div><div class="progress-track"><div id="progressFill" class="progress-fill"></div></div></div>
  </section>
  <nav id="stageRail" class="stage-rail" aria-label="世界建模阶段"></nav>
  <section class="theatre" aria-label="物流世界建模过程">
    <div class="stage-banner"><div class="stage-id"><span id="stepBadge" class="step-badge"></span><span id="stageTitle" class="stage-title"></span></div><div id="principle" class="principle"></div><div id="stageTime" class="stage-time"></div></div>
    <div class="world-layout">
      <aside class="inherit-panel"><div class="panel-label">Inherited world · 继承的世界</div><div id="inheritList" class="inherit-list"></div><div id="formulaBox" class="formula-box"></div><div class="inherit-note">这些是上一阶段已经赋予世界的语义能力，不是待加工文件列表。</div></aside>
      <section class="world-main"><div class="world-head"><span>LIVE WORLD MODEL</span><span>scene · 运营分析-8767b626</span></div>
        <div id="worldCanvas" class="world-canvas" role="img" aria-label="持续生长的物流世界模型">
          <div class="world-layer" data-layer="boundary"><div class="world-boundary-ring"></div></div>
          <div class="world-layer" data-layer="roles"><div class="role r1">仓库经理</div><div class="role r2">区域运营</div><div class="role r3">运营分析师</div></div>
          <div class="world-layer" data-layer="entities"><svg class="network-svg" viewBox="0 0 700 500" preserveAspectRatio="none"><path class="route-path" d="M155 150 C260 130 300 210 350 240"/><path class="route-path" d="M350 240 C445 180 505 150 545 145"/><path class="route-path" d="M155 350 C250 330 300 270 350 240"/><path class="route-path" d="M350 240 C440 270 485 335 545 350"/></svg><div class="entity-node warehouse"><b>ENTITY</b>仓库</div><div class="entity-node outlet"><b>ENTITY</b>网点</div><div class="entity-node carrier"><b>ENTITY</b>承运商</div><div class="entity-node customer"><b>ENTITY</b>客户</div><div class="entity-node hub"><b>ENTITY</b>运单 / 线路</div></div>
          <div class="world-layer" data-layer="states"><div class="state-ring"></div><div class="state-label">76 STATES · 运单 20 态</div></div>
          <div class="world-layer" data-layer="actions"><div class="action-beam a1"></div><div class="action-beam a2"></div><div class="action-beam a3"></div><span class="action-name n1">dispatch()</span><span class="action-name n2">trace()</span><span class="action-name n3">claim()</span></div>
          <div class="world-layer" data-layer="repair"><div id="repairAlert" class="repair-alert">⚠ 非法状态跃迁：incident_closed 不存在</div></div>
          <div class="world-layer" data-layer="taxonomy"><div class="taxonomy-orbit"></div><span class="axis x1">区域</span><span class="axis x2">时间</span><span class="axis x3">产品</span><span class="axis x4">状态</span><span class="axis x5">指标 · …15维</span></div>
          <div class="world-layer" data-layer="constraints"><div class="constraint-stack"><span class="constraint">× 9 illegal</span><span class="constraint ok">✓ 6 required</span><span class="constraint">↔ 8 mutex</span></div></div>
          <div class="world-layer" data-layer="schema"><div class="schema-deck">''' + ''.join(f'<span class="table-tile">{name}</span>' for name in ("waybill","route","warehouse","carrier","delivery","temperature","quality","cost","SLA","claim","event","… +53")) + r'''</div><div class="schema-count">WORLD STATE → 64 TABLES</div></div>
          <div class="world-layer" data-layer="data"><i class="shipment s1"></i><i class="shipment s2"></i><i class="shipment s3"></i><div class="fact-counter">36,101 instantiated facts</div></div>
          <div class="world-layer" data-layer="dwh"><div class="probe dwh">STEP 5.1<br><strong>数仓任务</strong><br>观察世界状态<br>555</div></div>
          <div class="world-layer" data-layer="rules"><div class="policy-cloud"><div class="policy"><b>RULE</b>线路 SLA</div><div class="policy"><b>RULE</b>价格政策</div><div class="policy"><b>RULE</b>理赔规范</div></div></div>
          <div class="world-layer" data-layer="docs"><div class="doc-stack"><div class="doc-page"></div><div class="doc-page"></div><div class="doc-page">制度记忆<i></i><i></i><i></i>65 docs</div></div></div>
          <div class="world-layer" data-layer="kb"><div class="probe kb">STEP 5.2<br><strong>知识任务</strong><br>观察世界规则<br>465</div></div>
          <div class="world-layer" data-layer="hybrid"><div class="hybrid-core"><div><strong>500</strong>STEP 5.3<br>混合任务<br>STATE × RULES</div></div></div>
          <div class="world-layer" data-layer="freeze"><div class="freeze-shell"></div></div>
          <div class="world-layer" data-layer="runner"><div class="agent-beam"></div><div class="agent-node">AGENT<br>Observe / Act</div></div>
          <div class="world-caption"><i class="change-pulse"></i><strong id="worldChangeTitle">世界尚未变化</strong><span id="worldChangeDetail">等待第一项建模动作</span></div>
          <div id="endOverlay" class="end-overlay" role="status" aria-live="polite"><div id="completionParticles"></div><div class="end-seal"><div class="end-check">✓</div><div class="end-kicker">WORLD MODEL SEALED</div><div class="end-title">沙箱生成完成</div><div class="end-id">sandbox_id · 57a3cf55-3bb7-4241-b314-8d6ace6d6c6c</div><div class="end-metrics"><div class="end-metric"><strong>36,101</strong>世界事实</div><div class="end-metric"><strong>65</strong>制度文档</div><div class="end-metric"><strong>3 类</strong>观察任务</div></div><button id="endRestartBtn" class="end-restart" type="button">重新观察建模过程</button></div></div>
        </div>
      </section>
      <aside class="build-panel"><div class="panel-label">World gains · 本步让世界获得</div><div id="gainList" class="gain-list"></div><div id="capability" class="capability"><p></p></div><div class="evidence-label">落盘凭证 · 技术实现（非世界本体）</div><div id="evidenceList" class="evidence-list"></div></aside>
      <div id="transitionCurtain" class="transition-curtain"><div id="transitionMessage" class="transition-message"></div></div>
    </div>
    <div class="handoff"><span class="handoff-symbol">⟶</span><span id="handoffText"></span></div>
  </section>
  <footer class="footer"><span>世界建模视角：任务是对世界状态或规则的观察；文件只是模型持久化的载体。</span><span><a href="https://github.com/renjunxiang/sf_my_sandbox/tree/758917009d0ebb0fb36561197171f6abdd279d96/.runtime_state/v20/scenes/运营分析-8767b626">v20 真实场景证据</a> · 不展示任务正文、hidden gold、SQL 或数据库行</span></footer>
</main>
<script id="stageData" type="application/json">__STAGE_DATA__</script><script id="worldData" type="application/json">__WORLD_DATA__</script>
<script>
(()=>{'use strict';
const stages=JSON.parse(document.getElementById('stageData').textContent),phases=JSON.parse(document.getElementById('worldData').textContent);
const navLabels={intake:'准备',prd:'0 · PRD',factor:'1 · Factor',taxonomy:'2 · 情境',schema:'3.1 · Schema',data:'4.1 · 数据',dwh_tasks:'5.1 · 数仓任务',catalog:'3.2 · 规则',documents:'4.2 · 文档',kb_tasks:'5.2 · 知识任务',hybrid:'5.3 · 混合任务',freeze:'冻结',runner:'交付'};
const els={rail:document.getElementById('stageRail'),step:document.getElementById('stepBadge'),title:document.getElementById('stageTitle'),principle:document.getElementById('principle'),time:document.getElementById('stageTime'),inherit:document.getElementById('inheritList'),formula:document.getElementById('formulaBox'),gains:document.getElementById('gainList'),capability:document.getElementById('capability'),evidence:document.getElementById('evidenceList'),handoff:document.getElementById('handoffText'),progress:document.getElementById('progressFill'),progressLabel:document.getElementById('progressLabel'),live:document.getElementById('liveStatus'),play:document.getElementById('playBtn'),prev:document.getElementById('prevBtn'),next:document.getElementById('nextBtn'),restart:document.getElementById('restartBtn'),endRestart:document.getElementById('endRestartBtn'),speed:document.getElementById('speedSelect'),curtain:document.getElementById('transitionCurtain'),transitionMessage:document.getElementById('transitionMessage'),world:document.getElementById('worldCanvas'),changeTitle:document.getElementById('worldChangeTitle'),changeDetail:document.getElementById('worldChangeDetail'),end:document.getElementById('endOverlay'),particles:document.getElementById('completionParticles')};
const state={stage:0,op:-1,playing:true,speed:1,timer:null,transitioning:false,finished:false};
const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
function renderRail(){els.rail.innerHTML=stages.map((s,i)=>`<button type="button" class="stage-tab ${i<state.stage?'done':i===state.stage?'current':''}" data-index="${i}" aria-current="${i===state.stage?'step':'false'}">${esc(navLabels[s.key]||s.step)}<small>${esc(s.duration)}</small></button>`).join('');els.rail.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>jumpTo(Number(b.dataset.index))));}
function visibleLayers(){const set=new Set();for(let i=0;i<state.stage;i++)phases[i].layers.forEach(x=>{if(x&&x!=='repair')set.add(x)});for(let i=0;i<=state.op;i++){const x=phases[state.stage].layers[i];if(x)set.add(x)}return set;}
function renderWorld(){const active=phases[state.stage].layers[state.op]||'';const visible=visibleLayers();els.world.querySelectorAll('.world-layer').forEach(layer=>{const key=layer.dataset.layer;layer.classList.toggle('visible',visible.has(key));layer.classList.toggle('current',key===active)});const repair=els.world.querySelector('[data-layer="repair"]');if(state.stage===2&&state.op>=2&&state.op<=3){repair.classList.add('visible');document.getElementById('repairAlert').classList.toggle('resolved',state.op===3);document.getElementById('repairAlert').textContent=state.op===3?'✓ 非法跃迁已修复，世界重新闭合':'⚠ 非法状态跃迁：incident_closed 不存在'}else repair.classList.remove('visible');}
function renderStage(){const s=stages[state.stage],p=phases[state.stage],complete=state.op>=s.operations.length-1;els.step.textContent=s.key==='dwh_tasks'?'Step 5.1 · 数仓任务':s.key==='hybrid'?'Step 5.3 · 混合任务':`${s.step} · ${s.branch}`;els.title.textContent=s.title;els.principle.textContent=p.principle;els.time.innerHTML=`${esc(s.clock)}<strong>${esc(s.duration)}</strong>`;els.inherit.innerHTML=p.inherited.map((x,i)=>`<div class="concept-chip" style="animation-delay:${i*.08}s">${esc(x)}</div>`).join('');els.formula.textContent=p.formula;els.gains.innerHTML=p.gains.map((gain,i)=>{const phase=i<state.op?'done':i===state.op?'active':'';const marker=i<state.op?'✓':i===state.op?'●':i+1;const op=s.operations[i];return `<div class="gain-step ${phase}"><span class="gain-marker">${marker}</span><div><div class="gain-title">${esc(gain)}</div><div class="gain-op">${esc(op.label)}</div><div class="gain-result">${esc(op.outcome)}</div></div></div>`}).join('');els.capability.classList.toggle('ready',complete);els.capability.querySelector('p').textContent=complete?p.capability:'世界能力正在形成；右侧动作完成后才会成为下一阶段可继承的能力。';const outputs=s.outputs.filter(x=>x.produced_by<=state.op);els.evidence.innerHTML=outputs.map(x=>`<span class="evidence-chip">${esc(x.name)}</span>`).join('')||'<span class="evidence-chip">尚未落盘</span>';els.handoff.innerHTML=complete?`<strong>世界能力交付：</strong>${esc(s.handoff)}`:'<strong>建模中：</strong>主画布正在改变世界的语义结构；文件只记录变化结果。';if(state.op<0){els.changeTitle.textContent='继承上一阶段的世界';els.changeDetail.textContent='等待本阶段第一项建模动作'}else{els.changeTitle.textContent=p.gains[state.op];els.changeDetail.textContent=s.operations[state.op].detail}const frac=(state.op+1)/Math.max(s.operations.length,1),total=((state.stage+Math.max(0,frac))/stages.length)*100;els.progress.style.width=`${state.finished?100:Math.min(100,total)}%`;els.progressLabel.textContent=state.finished?'完成 · 13/13':`${navLabels[s.key]||s.step} · ${state.stage+1}/${stages.length}`;els.live.textContent=state.finished?'✓ 沙箱生成完成':complete?'本阶段世界能力已形成，准备交付':state.op<0?'正在继承世界状态':`正在建模：${p.gains[state.op]}`;els.prev.disabled=state.finished||(state.stage===0&&state.op<0);els.next.disabled=state.finished||state.transitioning;els.play.disabled=state.finished;els.play.textContent=state.finished?'已结束':state.playing?'暂停':'继续播放';renderWorld();renderRail();}
function clearTimer(){if(state.timer!==null){clearTimeout(state.timer);state.timer=null}}
function schedule(){clearTimer();if(!state.playing||state.transitioning||state.finished)return;const s=stages[state.stage],delay=(state.op<0?500:s.replay_ms)/state.speed;state.timer=setTimeout(tick,delay)}
function tick(){const s=stages[state.stage];if(state.op<s.operations.length-1){state.op++;renderStage();schedule();return}if(state.stage===stages.length-1){finishReplay();return}animateHandoff(state.stage+1)}
function animateHandoff(nextIndex){if(state.transitioning||state.finished)return;state.transitioning=true;clearTimer();const current=stages[state.stage];els.transitionMessage.textContent=current.handoff;els.curtain.classList.add('show');const source=els.capability.getBoundingClientRect(),target=els.inherit.getBoundingClientRect(),sx=source.left+source.width/2,sy=source.top+source.height/2,tx=target.left+target.width/2,ty=target.top+60;const animations=[];if(!reduced){const orb=document.createElement('i');orb.className='transfer-orb';orb.style.left=`${sx}px`;orb.style.top=`${sy}px`;document.body.appendChild(orb);const a=orb.animate([{transform:'translate(0,0) scale(.5)',opacity:0},{transform:`translate(${(tx-sx)*.45}px,${(ty-sy)*.15-45}px) scale(1.5)`,opacity:1,offset:.45},{transform:`translate(${tx-sx}px,${ty-sy}px) scale(.25)`,opacity:.1}],{duration:900/state.speed,easing:'cubic-bezier(.28,.1,.2,1)',fill:'forwards'});animations.push(a.finished.catch(()=>{}).finally(()=>orb.remove()));for(let i=0;i<12;i++){const spark=document.createElement('i');spark.className='transfer-spark';spark.style.left=`${sx}px`;spark.style.top=`${sy}px`;document.body.appendChild(spark);const spread=(i-5.5)*5,sa=spark.animate([{transform:'translate(0,0)',opacity:0},{opacity:1,offset:.2},{transform:`translate(${tx-sx+spread}px,${ty-sy+Math.sin(i)*28}px) scale(.2)`,opacity:0}],{duration:(700+i*22)/state.speed,delay:i*24/state.speed,easing:'ease-out',fill:'forwards'});animations.push(sa.finished.catch(()=>{}).finally(()=>spark.remove()))}}Promise.all(animations).then(()=>setTimeout(()=>{state.stage=nextIndex;state.op=-1;state.transitioning=false;els.curtain.classList.remove('show');renderStage();schedule()},reduced?60:170/state.speed))}
function completionBurst(){if(reduced)return;els.particles.innerHTML='';for(let i=0;i<30;i++){const dot=document.createElement('i');dot.className='completion-particle';els.particles.appendChild(dot);const angle=(Math.PI*2*i)/30,distance=110+(i%6)*18;dot.animate([{transform:'translate(-50%,-50%) scale(.2)',opacity:0},{opacity:1,offset:.15},{transform:`translate(calc(-50% + ${Math.cos(angle)*distance}px),calc(-50% + ${Math.sin(angle)*distance}px)) scale(.1)`,opacity:0}],{duration:900+(i%5)*90,delay:(i%6)*28,easing:'cubic-bezier(.2,.7,.2,1)',fill:'forwards'})}}
function finishReplay(){clearTimer();state.finished=true;state.playing=false;state.transitioning=false;document.body.classList.add('is-finished');els.curtain.classList.remove('show');els.end.classList.add('show');completionBurst();renderStage()}
function jumpTo(index){clearTimer();state.stage=Math.max(0,Math.min(stages.length-1,index));state.op=-1;state.transitioning=false;state.finished=false;document.body.classList.remove('is-finished');els.end.classList.remove('show');els.particles.innerHTML='';renderStage();schedule()}
function restart(){state.playing=true;jumpTo(0)}
els.play.addEventListener('click',()=>{if(state.finished)return;state.playing=!state.playing;renderStage();schedule()});els.prev.addEventListener('click',()=>jumpTo(state.stage-1));els.next.addEventListener('click',()=>{const s=stages[state.stage];if(state.op<s.operations.length-1){state.op++;renderStage();schedule()}else if(state.stage<stages.length-1)animateHandoff(state.stage+1);else finishReplay()});els.restart.addEventListener('click',restart);els.endRestart.addEventListener('click',restart);els.speed.addEventListener('change',()=>{state.speed=Number(els.speed.value);schedule()});document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')els.next.click();if(e.key==='ArrowLeft')els.prev.click();if(e.key===' '&&!state.finished){e.preventDefault();els.play.click()}});renderStage();requestAnimationFrame(schedule);
})();
</script></body></html>
'''


CLEAR_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>物流世界是怎样建成的？</title>
<style>
:root{color-scheme:dark;--bg:#020711;--surface:#071423;--surface2:#0b1b2d;--line:#203a57;--text:#f2f8ff;--muted:#91a8c2;--fact:#42d8ff;--rule:#b68cff;--task:#ffc35c;--ok:#4be3a1;--bad:#ff647c}
*{box-sizing:border-box}html{background:var(--bg)}body{margin:0;min-width:1180px;color:var(--text);background:var(--bg);font-family:Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif}button,select,summary{font:inherit}button:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid rgba(66,216,255,.55);outline-offset:2px}.sr-only{position:absolute;width:1px;height:1px;margin:-1px;overflow:hidden;clip:rect(0,0,0,0)}
.app{position:relative;min-height:100vh;padding:22px 28px 18px;overflow:hidden;isolation:isolate;background:radial-gradient(circle at 10% 0,rgba(34,108,208,.18),transparent 28%),radial-gradient(circle at 90% 3%,rgba(182,140,255,.14),transparent 27%),linear-gradient(180deg,#06101d,#020711 75%)}
.app::before{content:"";position:fixed;inset:0;z-index:-2;opacity:.11;pointer-events:none;background-image:linear-gradient(rgba(66,216,255,.16) 1px,transparent 1px),linear-gradient(90deg,rgba(66,216,255,.16) 1px,transparent 1px);background-size:64px 64px;animation:bgdrift 15s linear infinite}@keyframes bgdrift{to{background-position:0 64px,64px 0}}
.top{display:grid;grid-template-columns:1fr auto;align-items:start;gap:24px}.eyebrow{color:var(--fact);font-size:11px;letter-spacing:.2em;text-transform:uppercase;text-shadow:0 0 18px rgba(66,216,255,.45)}h1{margin:6px 0 0;font-size:34px;line-height:1.1;letter-spacing:-.035em}.subtitle{margin:8px 0 0;color:var(--muted);font-size:14px;line-height:1.6;max-width:800px}.legend{display:flex;gap:13px;align-items:center;padding:10px 13px;border:1px solid var(--line);border-radius:10px;background:rgba(7,20,35,.78);font-size:11px;color:var(--muted)}.legend i{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px;box-shadow:0 0 11px currentColor}.fact{color:var(--fact)}.rule{color:var(--rule)}.task{color:var(--task)}.ok{color:var(--ok)}
.controls{margin-top:16px;display:flex;align-items:center;gap:7px}.btn,.speed{min-height:36px;padding:7px 12px;border:1px solid var(--line);border-radius:8px;background:rgba(9,25,43,.9);color:var(--text);cursor:pointer}.btn.primary{min-width:92px;background:linear-gradient(135deg,#c7f7ff,#dccdff);color:#04101b;border:0;box-shadow:0 0 26px rgba(66,216,255,.17)}.btn:disabled{opacity:.38;cursor:default;box-shadow:none}.progress{flex:1;margin-left:10px}.progress-copy{display:flex;justify-content:space-between;margin-bottom:5px;color:var(--muted);font-size:11px}.live{color:var(--fact);text-shadow:0 0 12px rgba(66,216,255,.4)}.track{height:4px;background:#15273d;border-radius:10px;overflow:hidden}.fill{height:100%;width:0;background:linear-gradient(90deg,var(--fact),#668dff,var(--rule));box-shadow:0 0 14px var(--fact);transition:width .35s ease}
.chapters{margin-top:14px;display:grid;grid-template-columns:repeat(6,1fr);gap:7px}.chapter{position:relative;min-height:58px;padding:9px 11px;border:1px solid #1d334c;border-radius:9px;background:rgba(5,15,28,.68);color:#7189a3;text-align:left;cursor:pointer;overflow:hidden}.chapter::after{content:"";position:absolute;left:0;bottom:0;width:0;height:2px;background:var(--fact);transition:width .35s}.chapter.done{color:#a5b9ce;border-color:#365573}.chapter.done::after{width:100%;background:#456a8c}.chapter.current{color:var(--text);border-color:rgba(66,216,255,.48);background:linear-gradient(135deg,rgba(66,216,255,.1),rgba(182,140,255,.055));box-shadow:inset 0 0 28px rgba(66,216,255,.035)}.chapter.current::after{width:100%;box-shadow:0 0 13px var(--fact)}.chapter b{display:block;color:inherit;font-size:12px}.chapter small{display:block;margin-top:5px;font-size:10px;opacity:.8;line-height:1.3}.chapter-no{position:absolute;right:8px;top:7px;font:17px ui-monospace,monospace;opacity:.15}
.story{position:relative;margin-top:10px;border:1px solid #25425f;border-radius:15px;overflow:hidden;background:rgba(3,10,18,.75);box-shadow:0 28px 90px rgba(0,0,0,.38)}.story-head{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:18px;padding:12px 16px;border-bottom:1px solid var(--line);background:linear-gradient(90deg,rgba(10,28,48,.96),rgba(5,15,27,.93))}.stage-token{color:var(--fact);font-size:11px;letter-spacing:.08em}.stage-title{display:block;margin-top:3px;font-size:21px;font-weight:650}.stage-question{text-align:center;font-size:14px;color:#d8e6f4}.stage-question small{display:block;margin-bottom:4px;color:#6f88a3;font-size:9px;letter-spacing:.15em}.stage-meta{text-align:right;color:var(--muted);font-size:10px}.stage-meta strong{display:block;color:var(--text);font-size:13px;margin-top:3px}
.cinema{position:relative;height:540px;overflow:hidden;background:radial-gradient(circle at 50% 47%,rgba(30,103,166,.18),transparent 28%),linear-gradient(rgba(57,105,151,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(57,105,151,.08) 1px,transparent 1px),linear-gradient(180deg,#061422,#020914);background-size:auto,42px 42px,42px 42px,auto;transition:filter .28s ease,transform .28s ease}.cinema.zooming{filter:blur(7px) brightness(.55);transform:scale(.975)}.cinema::before{content:"";position:absolute;inset:-50%;background:conic-gradient(from 0deg,transparent,rgba(66,216,255,.05),transparent,rgba(182,140,255,.045),transparent);animation:aura 16s linear infinite}@keyframes aura{to{transform:rotate(360deg)}}
.hero-change{position:absolute;z-index:50;top:18px;left:50%;transform:translateX(-50%);min-width:420px;padding:10px 18px;border:1px solid rgba(66,216,255,.3);border-radius:12px;background:rgba(3,13,24,.82);backdrop-filter:blur(8px);text-align:center;box-shadow:0 10px 36px rgba(0,0,0,.3),inset 0 0 25px rgba(66,216,255,.04)}.hero-change small{display:block;color:#6d8aa8;font-size:9px;letter-spacing:.16em}.hero-gain{display:block;margin-top:4px;font-size:20px;line-height:1.25}.hero-outcome{display:inline-block;margin-top:5px;color:var(--ok);font-size:11px}.hero-change.pop{animation:heroPop .55s cubic-bezier(.2,.9,.2,1)}@keyframes heroPop{0%{transform:translateX(-50%) scale(.72);opacity:0}65%{transform:translateX(-50%) scale(1.04)}100%{transform:translateX(-50%) scale(1);opacity:1}}
.layer{position:absolute;inset:0;z-index:4;opacity:0;transform:scale(.97);transition:opacity .55s ease,transform .55s cubic-bezier(.2,.85,.2,1),filter .55s}.layer.context{opacity:.13;filter:saturate(.4) brightness(.65);transform:scale(.98)}.layer.focus{opacity:1;transform:scale(1);filter:drop-shadow(0 0 14px rgba(66,216,255,.22))}.layer.born{animation:layerBorn .65s cubic-bezier(.18,.9,.2,1)}@keyframes layerBorn{0%{opacity:0;transform:scale(.72);filter:brightness(2)}70%{opacity:1;transform:scale(1.035)}100%{transform:scale(1)}}
.boundary{position:absolute;left:10%;top:14%;width:80%;height:72%;border:2px solid rgba(66,216,255,.47);border-radius:46% 43% 47% 42%;box-shadow:0 0 45px rgba(66,216,255,.1),inset 0 0 50px rgba(66,216,255,.055)}.boundary::before{content:"这个物流世界的边界";position:absolute;top:-12px;left:50%;transform:translateX(-50%);padding:3px 10px;background:#061422;color:var(--fact);font-size:11px;white-space:nowrap}
.person{position:absolute;width:100px;text-align:center;color:#d4e5f5;font-size:12px}.person::before{content:"";display:block;width:35px;height:35px;margin:0 auto 7px;border:1px solid var(--fact);border-radius:50% 50% 43% 43%;background:radial-gradient(circle at 50% 35%,var(--fact) 0 5px,transparent 6px),linear-gradient(135deg,rgba(66,216,255,.28),transparent);box-shadow:0 0 22px rgba(66,216,255,.22)}.p1{left:8%;top:38%}.p2{right:8%;top:38%}.p3{left:calc(50% - 50px);top:15%}
.map{position:absolute;inset:0;width:100%;height:100%}.route{fill:none;stroke:rgba(66,216,255,.5);stroke-width:2.5;stroke-dasharray:8 8;animation:routeflow 2.6s linear infinite}@keyframes routeflow{to{stroke-dashoffset:-32}}.node{position:absolute;z-index:8;transform:translate(-50%,-50%);width:86px;height:64px;display:grid;place-items:center;border:1px solid #3a739c;border-radius:12px;background:rgba(6,25,43,.95);text-align:center;font-size:12px;box-shadow:0 10px 30px rgba(0,0,0,.34),inset 0 0 20px rgba(66,216,255,.06)}.node b{display:block;color:var(--fact);font-size:9px}.warehouse{left:22%;top:34%}.outlet{left:78%;top:34%}.carrier{left:22%;top:72%}.customer{left:78%;top:72%}.hub{left:50%;top:53%;width:104px;height:76px}
.statewheel{position:absolute;left:50%;top:53%;width:230px;height:230px;transform:translate(-50%,-50%);border:2px dashed var(--ok);border-radius:50%;animation:spin 13s linear infinite;box-shadow:0 0 35px rgba(75,227,161,.08)}.statewheel span{position:absolute;padding:4px 8px;border-radius:10px;background:#093326;color:var(--ok);font-size:10px}.statewheel span:nth-child(1){top:17px;left:4px}.statewheel span:nth-child(2){right:-8px;top:92px}.statewheel span:nth-child(3){left:65px;bottom:-8px}@keyframes spin{to{transform:translate(-50%,-50%) rotate(360deg)}}
.beam{position:absolute;height:2px;background:linear-gradient(90deg,transparent,var(--task),transparent);box-shadow:0 0 12px var(--task);animation:beampulse 1.7s ease-in-out infinite}.b1{left:27%;top:37%;width:210px;transform:rotate(17deg)}.b2{left:52%;top:54%;width:195px;transform:rotate(-30deg)}.b3{left:25%;top:70%;width:230px;transform:rotate(-18deg)}@keyframes beampulse{50%{opacity:.25}}
.repair{position:absolute;left:50%;top:22%;transform:translateX(-50%);padding:9px 13px;border:1px solid var(--bad);border-radius:9px;background:rgba(70,13,28,.93);color:#ff9aac;font-size:12px;box-shadow:0 0 30px rgba(255,100,124,.25)}.repair.fixed{border-color:var(--ok);background:rgba(8,56,41,.94);color:#9bf4c9}
.orbit{position:absolute;left:50%;top:53%;width:610px;height:395px;transform:translate(-50%,-50%);border:2px dashed rgba(182,140,255,.55);border-radius:50%;animation:breath 3s ease-in-out infinite}.axis{position:absolute;padding:5px 9px;border:1px solid rgba(182,140,255,.5);border-radius:10px;background:#211638;color:#d9c7ff;font-size:11px}.x1{left:12%;top:22%}.x2{right:11%;top:22%}.x3{left:9%;bottom:19%}.x4{right:9%;bottom:19%}.x5{left:45%;top:13%}@keyframes breath{50%{box-shadow:0 0 45px rgba(182,140,255,.18);transform:translate(-50%,-50%) scale(1.025)}}
.constraints{position:absolute;right:12%;top:47%;display:grid;gap:7px}.constraint{padding:6px 10px;border-left:3px solid var(--bad);background:rgba(67,15,27,.9);color:#ff9aae;font-size:11px}.constraint.yes{border-color:var(--ok);background:rgba(10,57,43,.9);color:#91f0c1}
.tables{position:absolute;left:50%;top:57%;width:620px;transform:translate(-50%,-50%) perspective(500px) rotateX(53deg);display:grid;grid-template-columns:repeat(8,1fr);gap:6px}.table{height:46px;border:1px solid rgba(66,216,255,.55);background:linear-gradient(#164564,#08223a);box-shadow:0 8px 0 #06182a;display:grid;place-items:center;color:#b7eeff;font-size:10px;text-align:center}.tablecount{position:absolute;left:50%;bottom:17%;transform:translateX(-50%);color:var(--fact);font-size:14px;letter-spacing:.1em}
.shipment{position:absolute;width:10px;height:10px;border-radius:50%;background:#fff;box-shadow:0 0 8px #fff,0 0 25px var(--fact);animation:move 3.1s ease-in-out infinite}.s2{animation-delay:-1s}.s3{animation-delay:-2s}@keyframes move{0%{left:22%;top:34%;opacity:0}12%{opacity:1}50%{left:50%;top:53%}88%{opacity:1}100%{left:78%;top:72%;opacity:0}}.facts{position:absolute;left:50%;bottom:13%;transform:translateX(-50%);padding:8px 13px;border:1px solid var(--fact);border-radius:10px;background:rgba(3,23,37,.9);color:var(--fact);font:14px ui-monospace,monospace;box-shadow:0 0 28px rgba(66,216,255,.15)}
.probe{position:absolute;top:48%;width:126px;height:126px;border:2px solid;border-radius:50%;display:grid;place-items:center;text-align:center;font-size:11px;line-height:1.5;animation:scan 2.3s ease-in-out infinite}.probe strong{font-size:17px}.dwh{left:7%;color:var(--fact);background:radial-gradient(circle,rgba(66,216,255,.17),transparent 68%)}.kb{right:7%;color:var(--rule);background:radial-gradient(circle,rgba(182,140,255,.17),transparent 68%)}@keyframes scan{50%{transform:scale(1.07);box-shadow:0 0 40px currentColor}}
.policies{position:absolute;right:10%;top:27%;display:grid;gap:9px;width:230px}.policy{padding:10px 12px;border:1px solid rgba(182,140,255,.5);border-radius:9px;background:rgba(35,22,57,.94);color:#e0d3ff;font-size:12px;box-shadow:8px 8px 0 rgba(25,14,43,.55)}.policy b{color:var(--rule);margin-right:7px}.docs{position:absolute;right:15%;top:34%;width:170px;height:130px}.page{position:absolute;inset:0;border:1px solid #765bb2;border-radius:8px;background:linear-gradient(135deg,#2a1d44,#0a1728);box-shadow:0 12px 30px rgba(0,0,0,.32)}.page:nth-child(1){transform:translate(-18px,18px) rotate(-5deg)}.page:nth-child(2){transform:translate(-9px,9px) rotate(-2deg)}.page:nth-child(3){padding:22px 17px;color:#ddccff;font-size:13px}.page i{display:block;height:3px;margin:8px 0;background:#5e4983}
.hybrid{position:absolute;left:50%;top:53%;width:170px;height:170px;transform:translate(-50%,-50%);border:2px solid #e8dbff;border-radius:50%;display:grid;place-items:center;text-align:center;background:radial-gradient(circle,rgba(182,140,255,.48),rgba(66,216,255,.13) 44%,transparent 69%);box-shadow:0 0 55px rgba(182,140,255,.38);animation:hybridpulse 1.8s ease-in-out infinite}.hybrid strong{display:block;font-size:28px}.hybrid::before,.hybrid::after{content:"";position:absolute;inset:-28px;border:1px solid rgba(66,216,255,.28);border-radius:50%;animation:spin2 5s linear infinite}.hybrid::after{inset:-52px;border-color:rgba(182,140,255,.22);animation-direction:reverse;animation-duration:8s}@keyframes hybridpulse{50%{transform:translate(-50%,-50%) scale(1.07)}}@keyframes spin2{to{transform:rotate(360deg)}}
.shield{position:absolute;left:50%;top:53%;width:710px;height:430px;transform:translate(-50%,-50%);border:3px solid rgba(75,227,161,.7);border-radius:48% 43% 46% 42%;box-shadow:0 0 65px rgba(75,227,161,.2),inset 0 0 70px rgba(75,227,161,.08)}.shield::before{content:"WORLD SEALED";position:absolute;top:-14px;left:50%;transform:translateX(-50%);padding:4px 13px;background:#061820;color:var(--ok);font-size:12px;letter-spacing:.18em}.agent{position:absolute;right:3%;top:47%;width:92px;height:92px;border:2px solid var(--task);border-radius:20px;display:grid;place-items:center;text-align:center;color:var(--task);background:rgba(52,36,8,.94);box-shadow:0 0 35px rgba(255,195,92,.22)}
.story-strip{display:grid;grid-template-columns:1fr 1.2fr 1fr;border-top:1px solid var(--line);background:rgba(5,15,27,.96)}.story-cell{position:relative;min-height:116px;padding:13px 15px}.story-cell+ .story-cell{border-left:1px solid var(--line)}.story-cell small{display:block;color:#6f88a3;font-size:10px;letter-spacing:.13em;margin-bottom:7px}.story-cell strong{font-size:15px;line-height:1.45}.story-cell p{margin:5px 0 0;color:var(--muted);font-size:12px;line-height:1.5}.story-cell.doing{background:linear-gradient(135deg,rgba(66,216,255,.07),rgba(182,140,255,.045))}.story-cell.doing small{color:var(--fact)}.story-cell.after strong{color:var(--ok)}
.action-bar{padding:10px 13px;border-top:1px solid var(--line);background:#06111f}.action-label{color:#6f88a3;font-size:10px;margin-bottom:7px}.action-steps{display:grid;gap:6px;grid-template-columns:repeat(5,1fr)}.action-step{min-height:48px;padding:7px 8px;border:1px solid #203852;border-radius:8px;background:rgba(8,24,41,.72);color:#66809c;text-align:left;font-size:10px;line-height:1.35;cursor:pointer}.action-step.done{color:#acd0c2;border-color:#315f57}.action-step.current{color:var(--text);border-color:var(--fact);background:rgba(66,216,255,.1);box-shadow:inset 0 0 20px rgba(66,216,255,.04)}.action-step b{display:inline-grid;place-items:center;width:18px;height:18px;margin-right:5px;border:1px solid currentColor;border-radius:50%;font-size:9px}
.evidence{border-top:1px solid var(--line);background:#050e1a}.evidence summary{padding:9px 14px;color:#7890aa;font-size:11px;cursor:pointer}.evidence summary::marker{color:var(--fact)}.evidence-content{padding:0 14px 10px;display:flex;flex-wrap:wrap;gap:5px}.file-chip{padding:4px 7px;border:1px solid #223952;border-radius:6px;color:#7690a9;font:9px ui-monospace,monospace}.transition{position:absolute;inset:0;z-index:90;display:grid;place-items:center;pointer-events:none;background:rgba(1,6,12,.82);backdrop-filter:blur(6px);opacity:0;visibility:hidden;transition:opacity .25s}.transition.show{opacity:1;visibility:visible}.transition-card{text-align:center}.transition-card small{color:var(--fact);font-size:11px;letter-spacing:.18em}.transition-card strong{display:block;margin-top:8px;font-size:28px}.transition-card p{margin:7px 0 0;color:var(--muted);font-size:13px}.transition-ring{width:110px;height:110px;margin:0 auto 16px;border:2px solid var(--fact);border-radius:50%;box-shadow:0 0 0 16px rgba(66,216,255,.04),0 0 48px rgba(66,216,255,.26);animation:ringIn .8s cubic-bezier(.2,.9,.2,1)}@keyframes ringIn{from{transform:scale(.2) rotate(-120deg);opacity:0}to{transform:none;opacity:1}}
.spark{position:absolute;z-index:80;width:5px;height:5px;border-radius:50%;background:var(--fact);box-shadow:0 0 12px var(--fact);pointer-events:none}.end{position:absolute;inset:0;z-index:100;display:grid;place-items:center;background:radial-gradient(circle,rgba(20,89,71,.38),rgba(1,7,13,.97) 58%);opacity:0;visibility:hidden;transition:opacity .6s}.end.show{opacity:1;visibility:visible}.seal{text-align:center;width:500px;padding:30px;border:1px solid rgba(75,227,161,.55);border-radius:20px;background:linear-gradient(145deg,rgba(9,42,36,.96),rgba(6,17,29,.97));box-shadow:0 0 90px rgba(75,227,161,.2)}.check{width:86px;height:86px;margin:0 auto 17px;border:3px solid var(--ok);border-radius:50%;display:grid;place-items:center;color:var(--ok);font-size:42px;box-shadow:0 0 0 14px rgba(75,227,161,.05),0 0 38px rgba(75,227,161,.32)}.seal h2{margin:7px 0;font-size:28px}.seal p{color:#a7c7bd;font-size:13px}.seal-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:19px 0}.seal-metrics div{padding:9px;border-top:1px solid rgba(75,227,161,.28);color:#91aaa2;font-size:10px}.seal-metrics strong{display:block;color:#e3fff5;font-size:17px}.footer{margin-top:9px;display:flex;justify-content:space-between;color:#5c748e;font-size:10px}.footer a{color:#7895b1;text-decoration:none}
body.finished .app::before,body.finished .cinema::before,body.finished .route,body.finished .statewheel,body.finished .beam,body.finished .orbit,body.finished .shipment,body.finished .probe,body.finished .hybrid,body.finished .hybrid::before,body.finished .hybrid::after{animation-play-state:paused!important}@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
</style></head><body><main class="app">
<header class="top"><div><div class="eyebrow">A logistics world being born · real v20 replay</div><h1>一个物流世界，是怎样一步步建成的？</h1><p class="subtitle">每个镜头只看三件事：上一刻的世界有什么、这一刻改变了什么、改变后世界获得什么能力。蓝色是现实状态，紫色是制度规则，金色是从世界产生的任务。</p></div><div class="legend"><span><i class="fact"></i>事实世界</span><span><i class="rule"></i>规则世界</span><span><i class="task"></i>观察任务</span><span><i class="ok"></i>已验证</span></div></header>
<section class="controls" aria-label="播放控制"><button id="prev" class="btn" type="button">← 上一阶段</button><button id="play" class="btn primary" type="button">暂停</button><button id="next" class="btn" type="button">下一动作 →</button><button id="restart" class="btn" type="button">重新播放</button><label class="sr-only" for="speed">速度</label><select id="speed" class="speed" aria-label="播放速度"><option value=".75">0.75×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select><div class="progress"><div class="progress-copy"><span id="progressText"></span><span id="live" class="live" aria-live="polite"></span></div><div class="track"><div id="fill" class="fill"></div></div></div></section>
<nav id="chapters" class="chapters" aria-label="六个建模章节"></nav>
<section class="story"><header class="story-head"><div><span id="stageToken" class="stage-token"></span><strong id="stageTitle" class="stage-title"></strong></div><div id="stageQuestion" class="stage-question"></div><div id="stageMeta" class="stage-meta"></div></header>
<div id="cinema" class="cinema" role="img" aria-label="物流世界模型当前变化"><div id="hero" class="hero-change"><small>这一刻，世界正在获得</small><strong id="heroGain" class="hero-gain"></strong><span id="heroOutcome" class="hero-outcome"></span></div>
<div class="layer" data-layer="boundary"><div class="boundary"></div></div><div class="layer" data-layer="roles"><div class="person p1">仓库经理</div><div class="person p2">区域运营</div><div class="person p3">运营分析师</div></div>
<div class="layer" data-layer="entities"><svg class="map" viewBox="0 0 1000 540" preserveAspectRatio="none"><path class="route" d="M220 184 C350 160 420 245 500 286"/><path class="route" d="M500 286 C640 205 720 175 780 184"/><path class="route" d="M220 389 C350 375 430 320 500 286"/><path class="route" d="M500 286 C625 330 700 385 780 389"/></svg><div class="node warehouse"><b>ENTITY</b>仓库</div><div class="node outlet"><b>ENTITY</b>网点</div><div class="node carrier"><b>ENTITY</b>承运商</div><div class="node customer"><b>ENTITY</b>客户</div><div class="node hub"><b>ENTITY</b>运单 / 线路</div></div>
<div class="layer" data-layer="states"><div class="statewheel"><span>待揽收</span><span>运输中</span><span>已签收</span></div></div><div class="layer" data-layer="actions"><i class="beam b1"></i><i class="beam b2"></i><i class="beam b3"></i></div><div class="layer" data-layer="repair"><div id="repair" class="repair"></div></div>
<div class="layer" data-layer="taxonomy"><div class="orbit"></div><span class="axis x1">区域</span><span class="axis x2">时间</span><span class="axis x3">产品</span><span class="axis x4">状态</span><span class="axis x5">指标 · 15维</span></div><div class="layer" data-layer="constraints"><div class="constraints"><span class="constraint">× 非法组合</span><span class="constraint yes">✓ 必须共同出现</span><span class="constraint">↔ 互斥状态</span></div></div>
<div class="layer" data-layer="schema"><div class="tables">''' + ''.join(f'<span class="table">{x}</span>' for x in ("运单","线路","仓库","承运商","配送","温控","质量","成本","SLA","理赔","事件","…+53","平台","角色","关系","枚举")) + r'''</div><div class="tablecount">世界状态 → 64 张表</div></div><div class="layer" data-layer="data"><i class="shipment s1"></i><i class="shipment s2"></i><i class="shipment s3"></i><div class="facts">36,101 条真实世界状态</div></div>
<div class="layer" data-layer="dwh"><div class="probe dwh"><div><strong>555</strong><br>Step 5.1 数仓任务<br>观察现实状态</div></div></div><div class="layer" data-layer="rules"><div class="policies"><div class="policy"><b>RULE</b>线路 SLA</div><div class="policy"><b>RULE</b>价格政策</div><div class="policy"><b>RULE</b>理赔规范</div></div></div><div class="layer" data-layer="docs"><div class="docs"><div class="page"></div><div class="page"></div><div class="page">制度记忆<i></i><i></i><i></i>65 份文档</div></div></div><div class="layer" data-layer="kb"><div class="probe kb"><div><strong>465</strong><br>Step 5.2 知识任务<br>观察制度规则</div></div></div><div class="layer" data-layer="hybrid"><div class="hybrid"><div><strong>500</strong>Step 5.3 混合任务<br>现实 × 规则</div></div></div><div class="layer" data-layer="freeze"><div class="shield"></div></div><div class="layer" data-layer="runner"><div class="agent">AGENT<br>观察 / 行动</div></div>
<div id="end" class="end"><div id="endParticles"></div><div class="seal"><div class="check">✓</div><span class="ok">WORLD MODEL SEALED</span><h2>物流沙箱生成完成</h2><p>智能体面对的是会变化、有规则、可行动、可验证的物流世界。</p><div class="seal-metrics"><div><strong>36,101</strong>现实状态</div><div><strong>65</strong>制度文档</div><div><strong>3 类</strong>观察任务</div></div><button id="endRestart" class="btn" type="button">重新观看世界诞生</button></div></div></div>
<div class="story-strip"><section class="story-cell"><small>① 上一刻的世界</small><strong id="beforeTitle"></strong><p id="beforeText"></p></section><section class="story-cell doing"><small>② 这一刻发生什么</small><strong id="doingTitle"></strong><p id="doingText"></p></section><section class="story-cell after"><small>③ 世界因此获得</small><strong id="afterTitle"></strong><p id="afterText"></p></section></div>
<div class="action-bar"><div class="action-label">本阶段的建模动作（可点击跳转）</div><div id="actionSteps" class="action-steps"></div></div><details class="evidence"><summary>查看技术落盘凭证（世界模型的载体，不是主角）</summary><div id="files" class="evidence-content"></div></details>
<div id="transition" class="transition"><div class="transition-card"><div class="transition-ring"></div><small id="transitionKicker"></small><strong id="transitionTitle"></strong><p id="transitionText"></p></div></div></section>
<footer class="footer"><span>真实场景：运营分析-8767b626 · 13:15:20—14:35:51 · 80分31秒</span><span><a href="https://github.com/renjunxiang/sf_my_sandbox/tree/758917009d0ebb0fb36561197171f6abdd279d96/.runtime_state/v20/scenes/运营分析-8767b626">v20 执行证据</a> · 不展示任务正文、hidden gold、SQL 或数据库行</span></footer></main>
<script id="stages" type="application/json">__STAGE_DATA__</script><script id="phases" type="application/json">__WORLD_DATA__</script><script id="chapterData" type="application/json">__CHAPTER_DATA__</script>
<script>(()=>{'use strict';const stages=JSON.parse(document.getElementById('stages').textContent),phases=JSON.parse(document.getElementById('phases').textContent),chapterData=JSON.parse(document.getElementById('chapterData').textContent);const labels={intake:'准备 · 世界边界',prd:'Step 0 · PRD',factor:'Step 1 · Factor',taxonomy:'Step 2 · 情境空间',schema:'Step 3.1 · Schema',data:'Step 4.1 · 世界数据',dwh_tasks:'Step 5.1 · 数仓任务',catalog:'Step 3.2 · 规则目录',documents:'Step 4.2 · 制度文档',kb_tasks:'Step 5.2 · 知识任务',hybrid:'Step 5.3 · 混合任务',freeze:'冻结世界',runner:'交付智能体'};const focus={intake:['boundary'],prd:['boundary','roles'],factor:['entities','states','actions','repair'],taxonomy:['entities','taxonomy','constraints'],schema:['entities','schema'],data:['entities','data'],dwh_tasks:['data','dwh'],catalog:['entities','rules'],documents:['rules','docs'],kb_tasks:['docs','kb'],hybrid:['data','dwh','rules','docs','kb','hybrid'],freeze:['entities','data','rules','hybrid','freeze'],runner:['freeze','runner']};const $=id=>document.getElementById(id),el={chapters:$('chapters'),stageToken:$('stageToken'),stageTitle:$('stageTitle'),stageQuestion:$('stageQuestion'),stageMeta:$('stageMeta'),cinema:$('cinema'),hero:$('hero'),heroGain:$('heroGain'),heroOutcome:$('heroOutcome'),beforeTitle:$('beforeTitle'),beforeText:$('beforeText'),doingTitle:$('doingTitle'),doingText:$('doingText'),afterTitle:$('afterTitle'),afterText:$('afterText'),steps:$('actionSteps'),files:$('files'),progressText:$('progressText'),live:$('live'),fill:$('fill'),prev:$('prev'),play:$('play'),next:$('next'),restart:$('restart'),speed:$('speed'),transition:$('transition'),transitionKicker:$('transitionKicker'),transitionTitle:$('transitionTitle'),transitionText:$('transitionText'),end:$('end'),endParticles:$('endParticles'),endRestart:$('endRestart')};const state={stage:0,op:-1,playing:true,speed:1,timer:null,transitioning:false,finished:false};const reduced=matchMedia('(prefers-reduced-motion:reduce)').matches,esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));function chapterIndex(stageIndex=state.stage){return chapterData.findIndex(c=>stageIndex>=c.start_stage&&stageIndex<=c.end_stage)}function accumulated(){const set=new Set;for(let i=0;i<state.stage;i++)phases[i].layers.forEach(x=>{if(x&&x!=='repair')set.add(x)});for(let i=0;i<=state.op;i++){const x=phases[state.stage].layers[i];if(x)set.add(x)}return set}function renderChapters(){const ci=chapterIndex();el.chapters.innerHTML=chapterData.map((c,i)=>`<button class="chapter ${i<ci?'done':i===ci?'current':''}" data-stage="${c.start_stage}" type="button"><span class="chapter-no">${esc(c.number)}</span><b>${esc(c.title)}</b><small>${esc(c.promise)}</small></button>`).join('');el.chapters.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>jump(Number(b.dataset.stage))))}function renderWorld(){const visible=accumulated(),foci=new Set(focus[stages[state.stage].key]||[]),born=phases[state.stage].layers[state.op]||'';el.cinema.querySelectorAll('.layer').forEach(x=>{const key=x.dataset.layer,isVisible=visible.has(key),isFocus=isVisible&&foci.has(key);x.className=`layer ${isVisible?(isFocus?'focus':'context'):''} ${key===born?'born':''}`});const repair=$('repair'),repairLayer=el.cinema.querySelector('[data-layer="repair"]');if(state.stage===2&&state.op>=2&&state.op<=3){repairLayer.className='layer focus born';repair.classList.toggle('fixed',state.op===3);repair.textContent=state.op===3?'✓ 修复非法跃迁，世界重新闭合':'⚠ 状态引用断裂：世界不能继续生成'}else if(repairLayer)repairLayer.className='layer'}function render(){const s=stages[state.stage],p=phases[state.stage],c=chapterData[chapterIndex()],complete=state.op>=s.operations.length-1,current=state.op>=0?state.op:0;el.stageToken.textContent=labels[s.key];el.stageTitle.textContent=s.title;el.stageQuestion.innerHTML=`<small>这一阶段要回答</small>${esc(p.principle)}`;const chapterPos=state.stage-c.start_stage+1,chapterTotal=c.end_stage-c.start_stage+1;el.stageMeta.innerHTML=`本章 ${chapterPos}/${chapterTotal} · 全程 ${state.stage+1}/13<strong>${esc(s.duration)}</strong>`;el.heroGain.textContent=state.op<0?'继承上一阶段的世界':p.gains[state.op];el.heroOutcome.textContent=state.op<0?'准备进入下一项建模动作':s.operations[state.op].outcome;el.hero.classList.remove('pop');void el.hero.offsetWidth;el.hero.classList.add('pop');el.beforeTitle.textContent=p.inherited[0];el.beforeText.textContent=p.inherited.slice(1).join('；');el.doingTitle.textContent=state.op<0?'读取并理解已有世界':s.operations[current].label;el.doingText.textContent=state.op<0?'尚未改变世界，先确认本阶段继承的语义能力。':s.operations[current].detail;el.afterTitle.textContent=complete?'能力已经形成':'完成后将获得';el.afterText.textContent=p.capability;el.steps.style.gridTemplateColumns=`repeat(${s.operations.length},1fr)`;el.steps.innerHTML=p.gains.map((g,i)=>`<button type="button" class="action-step ${i<state.op?'done':i===state.op?'current':''}" data-op="${i}"><b>${i<state.op?'✓':i+1}</b>${esc(g)}</button>`).join('');el.steps.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{state.op=Number(b.dataset.op);render();schedule()}));const outputs=s.outputs.filter(x=>x.produced_by<=state.op);el.files.innerHTML=(outputs.length?outputs:s.outputs).map(x=>`<span class="file-chip">${esc(x.name)}</span>`).join('');const frac=(state.op+1)/Math.max(1,s.operations.length),pct=((state.stage+Math.max(0,frac))/stages.length)*100;el.fill.style.width=`${state.finished?100:Math.min(100,pct)}%`;el.progressText.textContent=state.finished?'完成 · 13/13':`${c.number} ${c.title} · ${labels[s.key]}`;el.live.textContent=state.finished?'✓ 物流世界建模完成':complete?'✓ 本阶段能力已形成':state.op<0?'正在理解继承的世界':`正在发生：${p.gains[state.op]}`;el.prev.disabled=state.finished||(state.stage===0&&state.op<0);el.next.disabled=state.finished||state.transitioning;el.play.disabled=state.finished;el.play.textContent=state.finished?'已结束':state.playing?'暂停':'继续播放';el.next.textContent=state.op<s.operations.length-1?'下一动作 →':state.stage<stages.length-1?'交付到下一阶段 →':'完成建模';renderWorld();renderChapters()}function clearTimer(){if(state.timer!==null){clearTimeout(state.timer);state.timer=null}}function schedule(){clearTimer();if(!state.playing||state.transitioning||state.finished)return;const s=stages[state.stage];state.timer=setTimeout(tick,(state.op<0?600:s.replay_ms)/state.speed)}function tick(){const s=stages[state.stage];if(state.op<s.operations.length-1){state.op++;burst();render();schedule()}else if(state.stage<stages.length-1)transitionTo(state.stage+1);else finish()}function burst(){if(reduced)return;for(let i=0;i<18;i++){const dot=document.createElement('i');dot.className='spark';dot.style.left='50%';dot.style.top='48%';el.cinema.appendChild(dot);const a=Math.PI*2*i/18,d=80+(i%5)*24;dot.animate([{transform:'translate(-50%,-50%) scale(.2)',opacity:0},{opacity:1,offset:.2},{transform:`translate(calc(-50% + ${Math.cos(a)*d}px),calc(-50% + ${Math.sin(a)*d}px)) scale(.1)`,opacity:0}],{duration:650+(i%4)*100,easing:'ease-out'}).finished.finally(()=>dot.remove())}}function transitionTo(nextStage){if(state.transitioning)return;state.transitioning=true;clearTimer();const oldChapter=chapterIndex(),newChapter=chapterIndex(nextStage),next=stages[nextStage];el.transitionKicker.textContent=oldChapter!==newChapter?`CHAPTER ${chapterData[newChapter].number}`:'WORLD CAPABILITY DELIVERED';el.transitionTitle.textContent=oldChapter!==newChapter?chapterData[newChapter].title:labels[next.key];el.transitionText.textContent=oldChapter!==newChapter?chapterData[newChapter].promise:`上一阶段形成的能力，成为“${next.title}”的起点。`;el.transition.classList.add('show');el.cinema.classList.add('zooming');setTimeout(()=>{state.stage=nextStage;state.op=-1;state.transitioning=false;el.transition.classList.remove('show');el.cinema.classList.remove('zooming');render();schedule()},reduced?80:900/state.speed)}function finish(){clearTimer();state.finished=true;state.playing=false;document.body.classList.add('finished');el.end.classList.add('show');burst();render()}function jump(index){clearTimer();state.stage=Math.max(0,Math.min(12,index));state.op=-1;state.finished=false;state.transitioning=false;document.body.classList.remove('finished');el.end.classList.remove('show');el.transition.classList.remove('show');render();schedule()}function restart(){state.playing=true;jump(0)}el.prev.addEventListener('click',()=>jump(state.stage-1));el.play.addEventListener('click',()=>{if(state.finished)return;state.playing=!state.playing;render();schedule()});el.next.addEventListener('click',()=>{const s=stages[state.stage];if(state.op<s.operations.length-1){state.op++;burst();render();schedule()}else if(state.stage<stages.length-1)transitionTo(state.stage+1);else finish()});el.restart.addEventListener('click',restart);el.endRestart.addEventListener('click',restart);el.speed.addEventListener('change',()=>{state.speed=Number(el.speed.value);schedule()});document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')el.next.click();if(e.key==='ArrowLeft')el.prev.click();if(e.key===' '&&!state.finished){e.preventDefault();el.play.click()}});render();requestAnimationFrame(schedule)})();</script></body></html>
'''


def render_html() -> str:
    return (
        CLEAR_HTML_TEMPLATE.replace("__STAGE_DATA__", _safe_json([asdict(stage) for stage in STAGES]))
        .replace("__WORLD_DATA__", _safe_json([asdict(phase) for phase in WORLD_PHASES]))
        .replace("__CHAPTER_DATA__", _safe_json([asdict(chapter) for chapter in CHAPTERS]))
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成物流沙箱逐步生成实录动画")
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
