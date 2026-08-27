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
        title="用物流数据库生成数仓评测任务",
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
        title="把数据库证据与政策证据合成双源任务",
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


def render_html() -> str:
    return HTML_TEMPLATE.replace("__STAGE_DATA__", _safe_json([asdict(stage) for stage in STAGES]))


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
