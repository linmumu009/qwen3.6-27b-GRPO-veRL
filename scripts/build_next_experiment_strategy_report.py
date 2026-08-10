#!/usr/bin/env python3
"""Build a canonical technical report for the post-Step-120 experiment strategy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GENERATED_AT = "2026-08-10T21:00:00+08:00"


def source() -> dict[str, Any]:
    path = "docs/next_experiment_strategy_20260810_summary.json"
    return {
        "id": "next_experiment_summary",
        "label": "Step 120未收尾、显存容量与快速实验聚合",
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_json_auto('{path}')",
            "description": "读取同题val20轨迹边界、老板效率字段、48K实测校准、模型原生位置上限和Step 120耗时。",
            "executed_at": GENERATED_AT,
            "filters": [
                "质量诊断使用Step 120固定val20的20道题",
                "未收尾定义为没有terminal final answer",
                "显存规划保持TP4/PP2/CP2训练、TP8×DP2 rollout和0.80 vLLM预算",
                "GPU耗时估算不含冷启动与模型加载",
            ],
            "metric_definitions": [
                "训练规划峰值 = 既有小上下文实测峰值 + 目标长度直接张量增量 + 10 GiB workspace余量",
                "48K→96K缓存增量按每个跑满活跃序列的full-attention KV与GDN状态差计算",
                "完整训练成本 = Step 120平均step耗时 × 步数 + 最终val20 + checkpoint保存",
            ],
            "tables_used": [path],
        },
    }


def chart(chart_id: str, title: str, subtitle: str, dataset: str, x: str, x_label: str, y: str, y_label: str, max_rows: int) -> dict[str, Any]:
    return {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "type": "bar",
        "dataset": dataset,
        "sourceId": "next_experiment_summary",
        "encodings": {
            "x": {"field": x, "type": "nominal", "label": x_label},
            "y": {"field": y, "type": "quantitative", "label": y_label},
        },
        "valueFormat": "number",
        "layout": "full",
        "maxRows": max_rows,
    }


def build(summary: dict[str, Any]) -> dict[str, Any]:
    audit = summary["completion_audit"]
    capacity = summary["capacity"]
    runtime = summary["runtime_and_experiments"]
    scenario_rows = capacity["scenario_rows"]
    training_rows = []
    seen = set()
    for row in scenario_rows:
        if row["context_k"] in seen:
            continue
        seen.add(row["context_k"])
        training_rows.append(
            {
                "context": f'{row["context_k"]}K',
                "context_tokens": row["context_tokens"],
                "planning_peak_gib": round(row["training_planning_peak_gib"], 2),
                "headroom_gib": round(row["training_headroom_gib"], 2),
                "usable_hbm_gib": capacity["usable_hbm_gib"],
            }
        )
    cache_rows = [
        {
            "active_sequences": int(level),
            "additional_cache_gib": value,
            "current_headroom_low_gib": capacity["current_post_sync_headroom_gib"]["low"],
            "current_headroom_high_gib": capacity["current_post_sync_headroom_gib"]["high"],
        }
        for level, value in capacity["calibrated_increment_by_concurrency_gib"].items()
    ]
    completion_rows = [
        {
            "状态": "未收尾",
            "题数": audit["incomplete"]["tasks"],
            "平均回合": round(audit["incomplete"]["assistant_turns_mean"], 1),
            "平均SQL": round(audit["incomplete"]["sql_calls_mean"], 2),
            "平均重复命令": round(audit["incomplete"]["duplicate_commands_mean"], 2),
            "振荡率": f'{audit["incomplete"]["redundancy_oscillation_rate"]:.0%}',
        },
        {
            "状态": "已收尾",
            "题数": audit["complete"]["tasks"],
            "平均回合": round(audit["complete"]["assistant_turns_mean"], 1),
            "平均SQL": round(audit["complete"]["sql_calls_mean"], 2),
            "平均重复命令": round(audit["complete"]["duplicate_commands_mean"], 2),
            "振荡率": f'{audit["complete"]["redundancy_oscillation_rate"]:.1%}',
        },
    ]
    capacity_table = []
    selected = {(48, 24), (64, 24), (64, 16), (96, 8), (96, 12), (96, 16), (96, 24)}
    for row in scenario_rows:
        key = (row["context_k"], row["max_sequences_per_replica"])
        if key not in selected:
            continue
        capacity_table.append(
            {
                "上下文": f'{row["context_k"]}K',
                "序列/副本": row["max_sequences_per_replica"],
                "训练峰值GiB": f'{row["training_planning_peak_gib"]:.1f}',
                "rollout规划GiB": f'{row["rollout_planning_total_gib"]:.1f}',
                "0.80预算余量GiB": f'{row["rollout_budget_headroom_gib"]:+.1f}',
                "规划门禁": "通过" if row["rollout_planning_fit"] else "不通过",
            }
        )
    experiment_rows = [
        {
            "优先级": row["priority"],
            "实验": row["experiment"],
            "训练步": row["training_steps"],
            "新轨迹": row["new_trajectories"],
            "GPU小时": row["estimated_gpu_hours_excluding_cold_start"],
        }
        for row in runtime["ranked_experiments"]
    ]
    cost_rows = [
        {
            "steps": str(row["steps"]),
            "update_hours": round(row["update_hours"], 2),
            "full_hours": round(row["with_full_val_and_save_hours"], 2),
        }
        for row in runtime["training_cost_projection"]
    ]

    charts = [
        chart("training_capacity", "训练侧上下文容量规划", "micro-batch 1、full recompute；可用HBM为61.27 GiB/卡。", "training_capacity", "context", "上下文", "planning_peak_gib", "规划峰值 GiB/卡", 4),
        chart("step_cost", "不同训练步数的完整墙钟成本", "按Step 120实测均值，包含一次67.6分钟val20和89秒保存，不含冷启动。", "step_cost", "steps", "训练步数", "full_hours", "小时", 4),
    ]
    tables = [
        {"id": "completion_table", "title": "未收尾与已收尾轨迹", "subtitle": "固定Step 120 val20；效率字段由老板原版评分聚合。", "dataset": "completion_table", "sourceId": "next_experiment_summary", "defaultSort": {"field": "状态", "direction": "asc"}, "columns": [{"field": key, "label": key} for key in completion_rows[0]], "layout": "full"},
        {"id": "capacity_table", "title": "上下文与rollout并发门禁", "subtitle": "rollout规划预算为0.80；负余量表示不能让所有序列同时跑满。", "dataset": "capacity_table", "sourceId": "next_experiment_summary", "defaultSort": {"field": "上下文", "direction": "asc"}, "columns": [{"field": key, "label": key} for key in capacity_table[0]], "layout": "full"},
        {"id": "experiment_table", "title": "快速实验优先级", "subtitle": "GPU小时不含冷启动；每项均有独立停止门禁。", "dataset": "experiment_table", "sourceId": "next_experiment_summary", "defaultSort": {"field": "优先级", "direction": "asc"}, "columns": [{"field": key, "label": key} for key in experiment_rows[0]], "layout": "full"},
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# 先修复收尾，再决定是否上96K", "layout": "full"},
        {"id": "technical_summary", "type": "markdown", "body": "## Technical Summary\n\n- **不建议直接启动96K正式训练。** Step 120的4道未收尾题全部在26回合停止并留下1个未返回工具调用；它们首先撞到回合/收尾边界，而不是被证明撞到48K token上限。\n- **96K训练侧大概率能装下，rollout并发不能原样保留。** 模型原生上限262K；训练规划峰值约48.6 GiB/卡。但48K→96K每个跑满序列增加约0.75 GiB缓存，24序列最坏增加18 GiB，超过当前同步后约11 GiB余量。\n- **最省时间的顺序是48K强制收尾→64K+32轮→96K定向诊断。** 96K只对64K仍失败的题运行，并从8序列/副本的容量探针开始。\n- **任何长跑前设置5步可学习性门禁。** 当前100步加一次完整验证约18.1小时；5步加完整val20约2小时。", "sourceId": "next_experiment_summary", "layout": "full"},
        {"id": "completion_finding", "type": "markdown", "body": "## 四道失败题都在重复探索后撞到同一边界\n\n未收尾题平均26回合、24.5条SQL、30.75条重复命令，4/4被判定为冗余振荡；已收尾题平均15回合、8.56条SQL和19.13条重复命令。四道失败题还都以一个没有工具响应的调用结束。这个模式更支持“停止探索并强制合成答案”，而不是无条件给模型更长的跑道。", "sourceId": "next_experiment_summary", "layout": "full"},
        {"id": "completion_table_block", "type": "table", "tableId": "completion_table", "layout": "full"},
        {"id": "training_capacity_finding", "type": "markdown", "body": "## 训练显存不是96K的首要阻塞\n\nQwen3.6-27B配置的原生位置上限为262,144。保持TP4/PP2/CP2、full recompute、micro-batch 1和CPU offload时，48K/64K/80K/96K训练规划峰值约43.9/45.5/47.1/48.6 GiB/卡，均低于61.27 GiB可用HBM。这里仍是规划值；96K必须用1 prompt × 4 responses实跑确认。", "sourceId": "next_experiment_summary", "layout": "full"},
        {"id": "training_capacity_chart", "type": "chart", "chartId": "training_capacity", "layout": "full"},
        {"id": "rollout_capacity_finding", "type": "markdown", "body": "## 96K会用并发换容量\n\n当前48K正式带载约53.8–56.1 GiB/卡，权重同步后约54.4–54.7 GiB，只剩10.8–11.1 GiB。若24序列都跑满，96K相对48K新增约18 GiB缓存；即使规划器通过preemption避免直接OOM，也会把长序列换入换出并显著拉长队列。64K保留24序列只有约3.4 GiB规划余量，也应先以16序列门禁；96K从8序列开始最稳妥。", "sourceId": "next_experiment_summary", "layout": "full"},
        {"id": "capacity_table_block", "type": "table", "tableId": "capacity_table", "layout": "full"},
        {"id": "cost_finding", "type": "markdown", "body": "## 用5步金丝雀替代100步盲跑\n\nStep 120平均每步611.7秒，完整val20另需4054.6秒。按同一效率，5/10/20/100步连同最终验证与保存约2.0/2.85/4.55/18.14小时。5步足以回答奖励和采样是否在选定train prompt上可学习；如果连小范围过拟合都做不到，就没有理由继续扩大步数。", "sourceId": "next_experiment_summary", "layout": "full"},
        {"id": "cost_chart", "type": "chart", "chartId": "step_cost", "layout": "full"},
        {"id": "scope", "type": "markdown", "body": "## 范围、定义与方法\n\n- 停止原因基于Step 120固定val20的20道贪心轨迹，未收尾为没有terminal final answer。\n- 显存模型沿用已实测的Qwen3.6混合全注意力/GDN结构、BF16和当前并行拓扑；rollout使用0.80预算。\n- 训练成本使用Step 120完整driver的平均step、最终验证和保存耗时。\n- 哨兵集建议为4道已知未收尾题加2道已答对/高分题；它只用于快速淘汰，不用于宣称泛化。", "sourceId": "next_experiment_summary", "layout": "full"},
        {"id": "experiment_design", "type": "markdown", "body": "## 分级实验设计\n\n1. 48K保持不变，在第22轮或剩余4K token时禁止新工具调用并要求直接作答。\n2. 只有第一步救回少于2/4失败题，才测64K+32轮；要求额外救回至少1道且单位题耗时增幅不超过60%。\n3. 96K只运行64K仍失败的题；先做8序列/副本容量探针，至少救回剩余失败的一半才进入训练候选。\n4. 零GPU并行做奖励回放：完成硬门控、重复命令/无新证据惩罚、dense权重30/50/70%。\n5. 最终候选跑5步、2 groups/update、40条新轨迹的可学习性金丝雀；训练prompt不改善或哨兵退化即停止。", "layout": "full"},
        {"id": "experiment_table_block", "type": "table", "tableId": "experiment_table", "layout": "full"},
        {"id": "other_options", "type": "markdown", "body": "## 两条更可能节省长期成本的路线\n\n- **难度课程与高信号采样。** 优先选择历史dense方差高、同prompt响应有明显优劣的train任务；不要平均消耗大量全错或全对group。\n- **离线纠错SFT/DPO。** 从train236的未收尾轨迹截取长历史，追加由gold SQL机械核验的正确最终回答，或构造完整/正确响应优于振荡/未收尾响应的偏好对。它不需要在线rollout，可能比继续GRPO探索便宜，但需单独验证不会只记模板。", "layout": "full"},
        {"id": "limitations", "type": "markdown", "body": "## 局限、鲁棒性与失败模式\n\n- 96K rollout尚未实跑；所有并发结论都是48K实测校准后的容量门禁，不是吞吐保证。\n- 更长上下文可能让模型继续无效探索，完成率、数值正确性和单位题耗时必须同时报告。\n- 强制收尾可能过早终止真正需要更多证据的题，因此哨兵集中保留两道既有成功题作为退化护栏。\n- 5步金丝雀允许判断“能否学到”，不能证明密封集泛化；通过后仍需较小的独立评测。", "layout": "full"},
        {"id": "recommendation", "type": "markdown", "body": "## 推荐的下一步\n\n立即实现并运行48K强制收尾sentinel6，同时离线回放完成硬门控与反循环惩罚。若救回≥2/4且成功题不退化，直接进入5步金丝雀，不测96K；只有强制收尾与64K都不足时，才对剩余题做96K定向诊断。", "layout": "full"},
        {"id": "questions", "type": "markdown", "body": "## 仍需回答的问题\n\n- 强制收尾后，新增final answer是否真正提高数值正确性，还是只把incomplete变成incorrect？\n- 64K相对48K的增益来自额外token还是额外回合？需要保持单因素或记录二者共同变化。\n- train236中有多少group适合高信号课程，能否在不看密封集的前提下稳定定义？", "layout": "full"},
    ]
    src = source()
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "先修复收尾，再决定是否上96K",
        "description": "Step 120未收尾原因、64K/80K/96K显存容量与快速实验优先级。",
        "generatedAt": GENERATED_AT,
        "charts": charts,
        "tables": tables,
        "sources": [{key: value for key, value in src.items() if key != "query"}],
        "blocks": blocks,
    }
    snapshot = {
        "version": 1,
        "generatedAt": GENERATED_AT,
        "status": "ready",
        "datasets": {
            "training_capacity": training_rows,
            "cache_increment": cache_rows,
            "step_cost": cost_rows,
            "completion_table": completion_rows,
            "capacity_table": capacity_table,
            "experiment_table": experiment_rows,
        },
    }
    return {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": [src]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(json.loads(args.summary.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
