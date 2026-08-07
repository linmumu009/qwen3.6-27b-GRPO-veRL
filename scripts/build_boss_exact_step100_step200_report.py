#!/usr/bin/env python3
"""Build the canonical report artifact for the Step 100 vs Step 200 boss evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GENERATED_AT = "2026-08-07T15:35:00+08:00"


def _source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    *,
    filters: list[str] | None = None,
    metric_definitions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_json_auto('{path}')",
            "description": description,
            "executed_at": GENERATED_AT,
            "filters": filters or [
                "固定 v15 DWH val20，贪心 n=1",
                "Step 100 与 Step 200 按 task_id 一一配对",
                "直接调用老板原始 reward_judge.py，未启用 LLM judge",
                "原始轨迹、模型和数据库不进入报告载荷",
            ],
            "metric_definitions": metric_definitions or [
                "reward_total = 0.5 × result_score + 0.5 × process_score；has_answer=0 时硬门控为 0",
                "result_score = 0.5 × has_answer + 0.5 × numeric answer correctness",
                "process_score 对 tables、fields、docs、task fit 按 0.3/0.3/0.2/0.2 加权，None 维度跳过并归一",
                "paired win/loss/tie 按同一 task_id 的 Step 200 reward_total 相对 Step 100 判定",
            ],
            "tables_used": [path],
        },
    }


def build_artifact(
    summary: dict[str, Any],
    adapter: dict[str, Any],
    audit: dict[str, Any],
    diagnosis: dict[str, Any],
    first100_training_signal: dict[str, Any],
    training_signal: dict[str, Any],
    failure_review: dict[str, Any],
    runtime_audit: dict[str, Any],
) -> dict[str, Any]:
    step100 = summary["step100"]
    step200 = summary["step200"]
    deltas = summary["numeric_deltas"]
    relative_reward_change = deltas["reward_total_mean"]["delta"] / step100["reward_total_mean"]

    score_comparison = []
    for metric, label in (
        ("reward_total_mean", "老板总奖励"),
        ("result_score_mean", "结果分"),
        ("process_score_mean", "过程分"),
    ):
        for checkpoint, values in (("Step 100", step100), ("Step 200", step200)):
            score_comparison.append(
                {
                    "metric": label,
                    "checkpoint": checkpoint,
                    "value": values[metric],
                    "task_count": 20,
                    "absolute_delta_step200_vs_step100": deltas[metric]["delta"],
                    "same_prompt_count": summary["prompt_identity"]["identical_prompt_count"],
                }
            )

    paired_outcomes = [
        {
            "outcome": label,
            "count": summary["paired_reward"][key],
            "share": summary["paired_reward"][key] / 20,
            "task_count": 20,
        }
        for key, label in (("wins", "Step 200 胜"), ("losses", "Step 200 负"), ("ties", "持平"))
    ]

    driver_contributions = [
        {
            **row,
            "share_percent": 100 * row["share_of_net_decline"],
            "task_count": diagnosis["task_count"],
        }
        for row in diagnosis["driver_rows"]
    ]

    training_signal_comparison = []
    for metric, label in (
        ("score_mean", "训练混合分"),
        ("boss_reward_mean", "训练老板奖励"),
        ("boss_process_score_mean", "训练过程分"),
        ("has_final_answer_mean", "最终回答率"),
        ("boss_answer_correct_mean", "数值正确率"),
    ):
        for window, values in (
            ("前 25 步", training_signal["first_quartile"]),
            ("后 25 步", training_signal["last_quartile"]),
        ):
            training_signal_comparison.append(
                {
                    "metric": label,
                    "window": window,
                    "value": values[metric],
                    "trajectory_count": values["rows"],
                    "step_range": f"{values['step_range'][0]}–{values['step_range'][1]}",
                }
            )

    first_group = first100_training_signal["within_group_signal"]
    second_group = training_signal["within_group_signal"]
    phase_signal_comparison = []
    for key, label, interpretation in (
        ("numeric_correctness_mixed", "数值正确性有对有错", "直接提供二值正确性的组内相对信号"),
        ("numeric_correctness_all_wrong", "四条全错", "二值正确性分量无法区分组内优劣"),
        ("numeric_correctness_all_correct", "四条全对", "二值正确性分量无法区分组内优劣"),
        ("completion_mixed", "最终回答状态有差异", "可提供是否收尾的组内相对信号"),
        ("fields_used_mixed", "字段使用有差异", "可提供字段覆盖的组内相对信号"),
    ):
        count_key = f"{key}_count"
        first_count = first_group[count_key]
        second_count = second_group[count_key]
        denominator = first_group["total_groups"]
        phase_signal_comparison.append(
            {
                "metric": label,
                "first100_count": first_count,
                "first100_rate": first_count / denominator,
                "second100_count": second_count,
                "second100_rate": second_count / second_group["total_groups"],
                "delta_percentage_points": 100
                * (second_count / second_group["total_groups"] - first_count / denominator),
                "interpretation": interpretation,
            }
        )

    failure_rows = [
        {
            "task_id": row["task_id"],
            "reward_delta": row["reward_delta"],
            "category": row["category"],
            "verified_failure": row["verified_failure"],
        }
        for row in failure_review["tasks"]
    ]

    metric_table = [
        {"metric": "老板总奖励均值", "step100": "0.443750", "step200": "0.399685", "delta": "-0.044065", "meaning": "相对下降 9.93%"},
        {"metric": "结果分均值", "step100": "0.400000", "step200": "0.350000", "delta": "-0.050000", "meaning": "结果能力下降"},
        {"metric": "过程分均值", "step100": "0.765000", "step200": "0.723750", "delta": "-0.041250", "meaning": "过程命中也下降"},
        {"metric": "完整收尾", "step100": "13/20", "step200": "13/20", "delta": "0", "meaning": "没有改善"},
        {"metric": "数值正确", "step100": "3/20", "step200": "1/20", "delta": "-2", "meaning": "主要退化来源"},
        {"metric": "必需表命中", "step100": "15/20", "step200": "15/20", "delta": "0", "meaning": "保持不变"},
        {"metric": "必需字段命中均值", "step100": "0.647059", "step200": "0.529412", "delta": "-0.117647", "meaning": "明显下降"},
        {"metric": "平均 SQL 数", "step100": "11.70", "step200": "13.50", "delta": "+1.80", "meaning": "查询更多但未换来更好结果"},
        {"metric": "平均重复命令数", "step100": "21.55", "step200": "20.20", "delta": "-1.35", "meaning": "重复略有改善"},
        {"metric": "平均答案长度", "step100": "4,349", "step200": "5,321", "delta": "+972", "meaning": "答案更长"},
    ]

    comparison_path = "docs/boss_exact_step100_step200_20260807_summary.json"
    adapter_path = "docs/boss_exact_step200_20260807_adapter_summary.json"
    audit_path = "docs/boss_exact_step100_step200_20260807_audit.json"
    diagnosis_path = "docs/boss_exact_step100_step200_20260807_diagnosis.json"
    first100_training_path = "docs/boss_exact_step100_step200_20260807_first100_training_signal.json"
    training_path = "docs/boss_exact_step100_step200_20260807_training_signal.json"
    failure_path = "docs/boss_exact_step100_step200_20260807_failure_review.json"
    runtime_path = "docs/boss_exact_step100_step200_20260807_runtime_audit.json"
    comparison_source = _source(
        "boss_exact_comparison",
        "Step 100 与 Step 200 老板原版评分配对汇总",
        comparison_path,
        "读取两轮 reward_judge 输出并按 task_id 配对，复算总奖励、结果分、过程分、行为指标和胜负平。",
    )
    adapter_source = _source(
        "step200_adapter_audit",
        "Step 200 OpenAI 轨迹转换审计",
        adapter_path,
        "核对评测行数、task_id、工具调用、截断调用、输入与 manifest 哈希。",
    )
    audit_source = _source(
        "boss_exact_audit",
        "老板原版评分器、数据库与输入一致性审计",
        audit_path,
        "记录评分脚本、数据库和两轮 selected manifest 的 SHA256，并核对逐题输入与评分匹配率。",
    )
    diagnosis_source = _source(
        "boss_exact_driver_diagnosis",
        "Step 100→200 老板总分逐题可加总归因",
        diagnosis_path,
        "按老板奖励门控公式把 20 道配对题的总分变化拆为完成状态、数值正确性和过程质量三个互斥组件。",
        filters=[
            "同一固定 val20，按 task_id 一一配对",
            "完成状态发生切换时，将整题变化归入完成状态组件",
            "两轮均有最终回答时，再拆数值正确性与过程质量",
            "老板评分四位小数舍入残差并入过程质量组件",
        ],
        metric_definitions=[
            "三个组件的 reward_sum_delta 精确加总为 -0.8813",
            "share_of_net_decline = 负向组件变化 / 0.8813",
            "reward_mean_delta = reward_sum_delta / 20",
            "评分公式只产生三个互斥驱动，无法在不破坏可加总性的前提下扩成第四类别，因此使用精确三项负向柱图",
        ],
    )
    training_source = _source(
        "training_signal_drift",
        "Step 100→200 的 1,600 条训练 rollout 聚合",
        training_path,
        "按训练 step 汇总老板奖励、过程分、完成、数值正确、字段与组内相对信号；不保存原始轨迹文本。",
        filters=[
            "100 个更新、每步 16 条轨迹、共 400 个完整 GRPO group",
            "前四分位为文件 step 102–126，后四分位为 177–201",
            "组内信号按同 step、同 prompt 的 4 条 response 计算",
        ],
        metric_definitions=[
            "numeric_correctness_mixed 指同 prompt 的 4 条 response 同时包含正确与错误结果",
            "只有组内发生差异的二值正确性，才能直接产生该组件的相对优势信号",
            "last_minus_first 为后 25 步均值减前 25 步均值",
        ],
    )
    phase_comparison_source = {
        "id": "training_phase_signal_comparison",
        "label": "前100步与后100步组内训练信号同口径比较",
        "path": f"{first100_training_path} + {training_path}",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": (
                f"SELECT '前100步' AS phase, * FROM read_json_auto('{first100_training_path}') "
                f"UNION ALL SELECT '后100步' AS phase, * FROM read_json_auto('{training_path}')"
            ),
            "description": "用同一分析器比较两个连续100步阶段各400个完整GRPO group的组内信号分布。",
            "executed_at": GENERATED_AT,
            "filters": [
                "前100步与后100步均为100个更新、每步16条轨迹、每组4条response",
                "mixed correctness 指同一prompt的4条response同时包含正确与错误",
                "两个阶段均要求 invalid_group_size_count=0",
            ],
            "metric_definitions": [
                "mixed correctness rate = 数值正确性有对有错group数 / 400",
                "delta_percentage_points = 后100步比例 - 前100步比例",
            ],
            "tables_used": [first100_training_path, training_path],
        },
    }
    failure_source = _source(
        "failure_trajectory_review",
        "6 道退化题轨迹人工复核",
        failure_path,
        "逐题核对最终答案、统计窗口、期望表、必需字段和是否触发 26 回合边界。",
        filters=["仅包含 Step 200 reward_total 低于 Step 100 的 6 道题"],
        metric_definitions=["reward_delta = Step 200 reward_total - Step 100 reward_total"],
    )
    runtime_source = _source(
        "training_runtime_audit",
        "Step 100→200 续训契约与 fully-async 计数审计",
        runtime_path,
        "记录 optimizer/data cursor 重置、staleness 阈值、处理与丢弃计数，用于界定潜在干扰因素。",
        filters=["最终 global step 200 的 resume_contract 与 driver 指标"],
        metric_definitions=[
            "stale_trajectory_processed 为 bounded fully-async 累计处理的非当前版本轨迹",
            "dropped_stale_samples 为超过阈值后实际丢弃的轨迹数",
        ],
    )

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Step 200 没有超过 Step 100",
            "description": "固定 val20 上使用老板原版评分器的逐题配对复评。",
            "generatedAt": GENERATED_AT,
            "charts": [
                {
                    "id": "score_comparison_chart",
                    "title": "老板原版评分：Step 100 与 Step 200",
                    "subtitle": "同一固定 val20、贪心 n=1；三项分数均为 0–1。",
                    "type": "bar",
                    "dataset": "score_comparison",
                    "sourceId": "boss_exact_comparison",
                    "encodings": {
                        "x": {"field": "metric", "type": "nominal", "label": "评分指标"},
                        "y": {"field": "value", "type": "quantitative", "label": "均值"},
                        "color": {"field": "checkpoint", "type": "nominal", "label": "Checkpoint"},
                    },
                    "valueFormat": "number",
                    "layout": "full",
                    "maxRows": 6,
                },
                {
                    "id": "paired_outcome_chart",
                    "title": "20 道题的逐题配对胜负",
                    "subtitle": "以 Step 200 相对 Step 100 的单题 reward_total 判定。",
                    "type": "bar",
                    "dataset": "paired_outcomes",
                    "sourceId": "boss_exact_comparison",
                    "encodings": {
                        "x": {"field": "outcome", "type": "nominal", "label": "配对结果"},
                        "y": {"field": "count", "type": "quantitative", "label": "题数"},
                    },
                    "valueFormat": "number",
                    "unit": "题",
                    "layout": "full",
                    "maxRows": 3,
                },
                {
                    "id": "driver_contribution_chart",
                    "title": "老板总分下降的可加总归因",
                    "subtitle": "20 道配对题的 reward_total 合计变化；三项精确加总为 -0.8813。",
                    "type": "bar",
                    "dataset": "driver_contributions",
                    "sourceId": "boss_exact_driver_diagnosis",
                    "encodings": {
                        "x": {"field": "driver", "type": "nominal", "label": "评分组件"},
                        "y": {"field": "reward_sum_delta", "type": "quantitative", "label": "总奖励变化"},
                    },
                    "valueFormat": "number",
                    "layout": "full",
                    "maxRows": 3,
                },
                {
                    "id": "training_signal_chart",
                    "title": "训练前后四分位的在线信号",
                    "subtitle": "前 25 步与后 25 步各 400 条轨迹；所有值均为 0–1 均值。",
                    "type": "bar",
                    "dataset": "training_signal_comparison",
                    "sourceId": "training_signal_drift",
                    "encodings": {
                        "x": {"field": "metric", "type": "nominal", "label": "训练指标"},
                        "y": {"field": "value", "type": "quantitative", "label": "均值"},
                        "color": {"field": "window", "type": "nominal", "label": "训练阶段"},
                    },
                    "valueFormat": "number",
                    "layout": "full",
                    "maxRows": 10,
                },
            ],
            "tables": [
                {
                    "id": "metric_table",
                    "title": "关键指标精确对比",
                    "subtitle": "同一 20 道 DWH 任务；变化为 Step 200 减 Step 100。",
                    "dataset": "metric_table",
                    "sourceId": "boss_exact_comparison",
                    "defaultSort": {"field": "metric", "direction": "asc"},
                    "columns": [
                        {"field": "metric", "label": "指标"},
                        {"field": "step100", "label": "Step 100"},
                        {"field": "step200", "label": "Step 200"},
                        {"field": "delta", "label": "变化", "movement": True},
                        {"field": "meaning", "label": "解释"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "failure_task_table",
                    "title": "6 道退化题的直接失败模式",
                    "subtitle": "仅列 Step 200 单题老板总奖励低于 Step 100 的任务。",
                    "dataset": "failure_rows",
                    "sourceId": "failure_trajectory_review",
                    "defaultSort": {"field": "reward_delta", "direction": "asc"},
                    "columns": [
                        {"field": "task_id", "label": "任务"},
                        {"field": "reward_delta", "label": "总奖励变化", "movement": True},
                        {"field": "category", "label": "失败类型"},
                        {"field": "verified_failure", "label": "逐题核验"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "phase_signal_table",
                    "title": "前100步与后100步的组内信号",
                    "subtitle": "两个阶段各400个完整group；变化为后100步减前100步。",
                    "dataset": "phase_signal_comparison",
                    "sourceId": "training_phase_signal_comparison",
                    "defaultSort": {"field": "metric", "direction": "asc"},
                    "columns": [
                        {"field": "metric", "label": "组内状态"},
                        {"field": "first100_count", "label": "前100步组数"},
                        {"field": "first100_rate", "label": "前100步比例", "format": "percent"},
                        {"field": "second100_count", "label": "后100步组数"},
                        {"field": "second100_rate", "label": "后100步比例", "format": "percent"},
                        {"field": "delta_percentage_points", "label": "变化（百分点）", "movement": True},
                        {"field": "interpretation", "label": "训练含义"},
                    ],
                    "layout": "full",
                },
            ],
            "sources": [
                {"id": comparison_source["id"], "label": comparison_source["label"], "path": comparison_source["path"]},
                {"id": adapter_source["id"], "label": adapter_source["label"], "path": adapter_source["path"]},
                {"id": audit_source["id"], "label": audit_source["label"], "path": audit_source["path"]},
                {"id": diagnosis_source["id"], "label": diagnosis_source["label"], "path": diagnosis_source["path"]},
                {"id": training_source["id"], "label": training_source["label"], "path": training_source["path"]},
                {"id": phase_comparison_source["id"], "label": phase_comparison_source["label"], "path": phase_comparison_source["path"]},
                {"id": failure_source["id"], "label": failure_source["label"], "path": failure_source["path"]},
                {"id": runtime_source["id"], "label": runtime_source["label"], "path": runtime_source["path"]},
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Step 200 没有超过 Step 100", "layout": "full"},
                {
                    "id": "executive_summary",
                    "type": "markdown",
                    "body": f"## Executive Summary\n\n- **老板原版总奖励下降。** Step 100 为 `0.443750`，Step 200 为 `0.399685`，绝对下降 `0.044065`，相对下降 `{abs(relative_reward_change):.2%}`。\n- **直接损失主要是数值正确性。** 20 题总奖励合计少 `0.8813`；数值正确性贡献 `-0.5000`（`56.7%`），过程与字段质量贡献 `-0.2876`（`32.6%`），完成状态净变化贡献 `-0.0937`（`10.6%`）。\n- **训练信号在奖励代理上变好，但正确性几乎不动。** 后 25 步相对前 25 步，在线老板奖励提高 `0.0225`、过程分提高 `0.0388`、最终回答率提高 `4.75pp`，但数值正确率只提高 `0.25pp`。\n- **正确性相对信号一直稀疏，并非后100步才突然恶化。** 有对有错group从前100步的 `75/400`（`18.75%`）降至后100步的 `72/400`（`18.00%`），只下降 `0.75pp`；两个阶段都约有82%的group无法从二值正确性上形成直接组内区分。",
                    "layout": "full",
                },
                {
                    "id": "score_finding",
                    "type": "markdown",
                    "body": "## 三项核心评分同时走低\n\n老板总奖励、结果分与过程分全部下降，说明这不是某一个聚合权重造成的表面波动。Step 200 查询更多 SQL、答案更长，但没有把额外探索转化为更准确的数值或更好的字段命中。继续单纯增加训练步数，当前证据下更可能扩大无效探索，而不是自然恢复。",
                    "sourceId": "boss_exact_comparison",
                    "layout": "full",
                },
                {"id": "score_chart_block", "type": "chart", "chartId": "score_comparison_chart", "layout": "full"},
                {
                    "id": "exact_driver_finding",
                    "type": "markdown",
                    "body": "## 57% 的净下降来自数值正确性\n\n把老板评分公式按同一 task_id 拆开后，三个组件可以精确对账：数值正确性 `-0.5000`、过程与字段质量 `-0.2876`、完成状态切换净额 `-0.0937`，合计正好是 `-0.8813`。所以首要问题不是回答有没有写出来，而是最终数值与统计口径不再可靠；过程退化是第二大来源。",
                    "sourceId": "boss_exact_driver_diagnosis",
                    "layout": "full",
                },
                {"id": "driver_contribution_block", "type": "chart", "chartId": "driver_contribution_chart", "layout": "full"},
                {
                    "id": "failure_modes_finding",
                    "type": "markdown",
                    "body": "## 六道退化题集中在三种行为\n\n两道原本答对的题分别漏了整体汇总值、选错了“最新一期”的时间粒度；两道过程题漏掉期望表或必需字段；两道温度题在分组口径上来回探索，达到 26 个 assistant 回合后仍未给最终答案。它们共同指向：模型会查库，但更容易在统计口径、关键输出契约和歧义收敛上失手。",
                    "sourceId": "failure_trajectory_review",
                    "layout": "full",
                },
                {"id": "failure_task_table_block", "type": "table", "tableId": "failure_task_table", "layout": "full"},
                {
                    "id": "training_signal_finding",
                    "type": "markdown",
                    "body": "## 在线奖励上升掩盖了正确性停滞\n\n后 25 步比前 25 步的训练混合分高 `0.0194`、在线老板奖励高 `0.0225`；过程分、最终回答率和字段使用也都提高。数值正确率却只从 `11.00%` 到 `11.25%`。这解释了为什么训练曲线看起来没有坏掉，固定 val20 却下降：优化器主要收到了完成与过程代理的可区分信号，没有收到足够稳定的正确答案相对信号。",
                    "sourceId": "training_signal_drift",
                    "layout": "full",
                },
                {"id": "training_signal_chart_block", "type": "chart", "chartId": "training_signal_chart", "layout": "full"},
                {
                    "id": "group_signal_mechanism",
                    "type": "markdown",
                    "body": "## GRPO 组内正确性信号只有 18%\n\n400 个四响应 group 中，`314` 个四条全错、`14` 个四条全对，只有 `72` 个同时含正确与错误答案。GRPO 的相对优势在同 prompt 的四条 response 内计算，因此前两类 group 的二值正确性分量无法区分哪条更好；相比之下，`189` 个 group 的最终回答状态有差异，`99` 个 group 的字段使用有差异。当前训练天然更容易学到“要收尾、要覆盖过程字段”，而不是“要选对统计窗口并给出正确数值”。",
                    "sourceId": "training_signal_drift",
                    "layout": "full",
                },
                {
                    "id": "phase_signal_stability",
                    "type": "markdown",
                    "body": "## 有对有错比例只小幅下降，低信号是持续性问题\n\n前100步为 `75/400`（`18.75%`），后100步为 `72/400`（`18.00%`），下降 `0.75` 个百分点，只少了3个group。四条全错则从313组变为314组，四条全对从12组变为14组。两个阶段均为400个完整group且没有缺组，因此不能把Step 200退化解释为后100步正确性信号突然塌陷；更准确的结论是两个阶段一直都严重缺少组内正确性对比，后100步仅略微更低。",
                    "sourceId": "training_phase_signal_comparison",
                    "layout": "full",
                },
                {"id": "phase_signal_table_block", "type": "table", "tableId": "phase_signal_table", "layout": "full"},
                {
                    "id": "driver_finding",
                    "type": "markdown",
                    "body": "## 汇总完成率不变，但内部发生了抵消\n\n两轮完整收尾都为 13/20，并不代表完成行为稳定：Step 200 新救回 2 题，同时又让 2 题从有答案变成超时，四道切换题净损失 `0.0937`。必需表命中汇总仍为 15/20，但必需字段命中均值下降 `0.1176`；平均 SQL 增加 1.8 条、答案增加约 972 字符。含义很直接：更多查询和更长答案没有保证模型围绕真正决定答案的字段收敛。",
                    "sourceId": "boss_exact_comparison",
                    "layout": "full",
                },
                {"id": "metric_table_block", "type": "table", "tableId": "metric_table", "layout": "full"},
                {
                    "id": "paired_finding",
                    "type": "markdown",
                    "body": "## 逐题配对也偏向退化\n\n20 道题中只有 3 道提高，6 道下降，其余 11 道持平。胜负比为 1:2；因此均值下降并非由单个极端题独自驱动。样本量仍小，不能据此估计全量任务的泛化差异，但足以否定“Step 200 已经在这套固定评测上优于 Step 100”。",
                    "sourceId": "boss_exact_comparison",
                    "layout": "full",
                },
                {"id": "paired_chart_block", "type": "chart", "chartId": "paired_outcome_chart", "layout": "full"},
                {
                    "id": "comparison_consistency",
                    "type": "markdown",
                    "body": f"## 比较口径经过一致性核对\n\n- 两轮均为固定 v15 DWH val20、贪心 `n=1`，同一 task_id 集合。\n- `{audit['identical_system_user_prompts']}/20` 道题的 system 与 user 输入逐题完全一致；两轮 selected manifest SHA256 同为 `5d7efda6…f87`。\n- 老板三份原始评分脚本 SHA256 与 Step 100 评测时一致，且未启用 LLM judge。\n- `logistics.sqlite` 的最后修改时间早于两轮评测，当前 SHA256 已记录；Step 200 原版评分器 20/20 匹配、无缺题。",
                    "sourceId": "boss_exact_audit",
                    "layout": "full",
                },
                {
                    "id": "adapter_behavior",
                    "type": "markdown",
                    "body": f"## 截断轨迹没有被伪造成答案\n\nStep 200 转换共保留 `{adapter['tool_calls']}` 个工具调用；其中 1 个最终工具调用被 token 边界截断，转换器将其保留为“未响应调用”。这条轨迹仍由老板原版完成门禁判为 incomplete、奖励为 0，没有把调用前的推理文字误算成最终答案。",
                    "sourceId": "step200_adapter_audit",
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## 下一轮先增强正确性信号，再继续训练\n\n1. 暂停从 Step 200 按原奖励与采样方式继续堆步数；保留 checkpoint 作为对照起点。\n2. 让更多 group 在数值正确性上产生组内差异：优先抽取当前正确率处于中间区间的 prompt，或增加候选后按正确性分层组成训练 group；不能只把二值正确奖励权重调大，因为 82% 的 group 该分量仍全部相同。\n3. 把“统计周期/粒度确认、必需字段显式输出、剩余回合强制收尾”做成可验证奖励或轨迹门禁，专门覆盖本次三类失败。\n4. 做固定 400 groups 的 A/B：原策略 vs 新正确性信号策略；比较老板原版 val20、组内 mixed-correct 率和 sealed test20，禁止只看训练 reward。",
                    "layout": "full",
                },
                {
                    "id": "runtime_confounds",
                    "type": "markdown",
                    "body": f"## 两个运行因素会加噪，但不是已证实主因\n\nStep 100 checkpoint 没有 Adam 状态，因此续训时 optimizer 被重置；修正 train236 后数据游标也从头开始。bounded fully-async 还累计处理了 `{runtime_audit['fully_async_final_counters']['stale_trajectory_processed']}` 条非当前版本轨迹，但超过阈值实际丢弃为 `0`。这些因素都可能增加优化路径波动，不过现有证据不能把固定 val20 的下降归因给其中任何一个；最直接、可量化的问题仍是上一节所示的正确性组内信号稀疏。",
                    "sourceId": "training_runtime_audit",
                    "layout": "full",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": "## 仍需回答的问题\n\n- 把 mixed-correct group 比例从 18% 提高后，数值正确率是否真正上升，还是暴露出 gold/题意口径本身的噪声？\n- Step 200 在 sealed test20 上是否也下降；若不一致，val20 的配对差异有多少只是 20 题小样本波动？\n- 保持其他配置不变、仅恢复完整 Adam 状态与同步 rollout，能否复现这次方向；这需要单独 A/B，不能从当前单次续训反推。",
                    "layout": "full",
                },
                {
                    "id": "caveats",
                    "type": "markdown",
                    "body": "## 局限与假设\n\n- 分母只有 20 道题，差异适合做方向性门禁，不足以给出窄置信区间或宣称全量泛化退化。\n- 两轮虽为贪心解码且输入完全相同，但底层工具环境的数据库内容被核验为同一路径而非做逐页快照；本次使用的老板 logistics.sqlite 与评分脚本版本未变化。\n- 本报告比较的是老板原版规则评分，不等价于人工业务专家对答案可用性的最终判断。\n- 原始轨迹、数据库、模型和 checkpoint 仅保留在服务器；仓库只保存聚合、配对结果与 SHA256 审计信息。",
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "score_comparison": score_comparison,
                "paired_outcomes": paired_outcomes,
                "metric_table": metric_table,
                "driver_contributions": driver_contributions,
                "training_signal_comparison": training_signal_comparison,
                "phase_signal_comparison": phase_signal_comparison,
                "failure_rows": failure_rows,
            },
        },
        "sources": [
            comparison_source,
            adapter_source,
            audit_source,
            diagnosis_source,
            training_source,
            phase_comparison_source,
            failure_source,
            runtime_source,
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--first100-training-signal", type=Path, required=True)
    parser.add_argument("--training-signal", type=Path, required=True)
    parser.add_argument("--failure-review", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_artifact(
        json.loads(args.summary.read_text(encoding="utf-8")),
        json.loads(args.adapter.read_text(encoding="utf-8")),
        json.loads(args.audit.read_text(encoding="utf-8")),
        json.loads(args.diagnosis.read_text(encoding="utf-8")),
        json.loads(args.first100_training_signal.read_text(encoding="utf-8")),
        json.loads(args.training_signal.read_text(encoding="utf-8")),
        json.loads(args.failure_review.read_text(encoding="utf-8")),
        json.loads(args.runtime_audit.read_text(encoding="utf-8")),
    )
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
