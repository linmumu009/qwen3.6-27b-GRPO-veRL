#!/usr/bin/env python3
"""Build the canonical report artifact for the Fastest-K efficiency experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GENERATED_AT = "2026-07-31T19:15:00+08:00"


CONFIGURATION_MATRIX = [
    {
        "configuration": "T8D2 · 4→4",
        "rollout_topology": "TP8×DP2",
        "candidates": 4,
        "selected": 4,
        "queue_wait_s": 383.81,
        "step_s": 464.60,
        "step_change_vs_baseline": 0.0,
    },
    {
        "configuration": "T8D2 · 5→4",
        "rollout_topology": "TP8×DP2",
        "candidates": 5,
        "selected": 4,
        "queue_wait_s": 332.16,
        "step_s": 412.34,
        "step_change_vs_baseline": -0.1125,
    },
    {
        "configuration": "T8D2 · 6→4",
        "rollout_topology": "TP8×DP2",
        "candidates": 6,
        "selected": 4,
        "queue_wait_s": 283.85,
        "step_s": 364.69,
        "step_change_vs_baseline": -0.2150,
    },
    {
        "configuration": "T4D4 · 4→4",
        "rollout_topology": "TP4×DP4",
        "candidates": 4,
        "selected": 4,
        "queue_wait_s": 401.67,
        "step_s": 482.61,
        "step_change_vs_baseline": 0.0388,
    },
    {
        "configuration": "T4D4 · 5→4",
        "rollout_topology": "TP4×DP4",
        "candidates": 5,
        "selected": 4,
        "queue_wait_s": 307.72,
        "step_s": 388.97,
        "step_change_vs_baseline": -0.1628,
    },
]


def _source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    sql: str,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": GENERATED_AT,
            "filters": [
                "只使用退出码为 0 的 llin 实验",
                "20-step 稳态定义为 step 2–20；step 1 因 actor 首次编译单独保留",
                "不把模型、原始轨迹、日志或 checkpoint 放入报告载荷",
            ],
            "metric_definitions": [
                "queue_wait_s = trainer 为凑齐 4 个完整 GRPO group 在内存队列上的等待时间",
                "update_actor_s = actor 前反向与优化器更新的实际计算时间",
                "step_s = 从取得训练 batch 到本步训练与权重同步结束的 wall time",
                "strict final correctness = 目标数值只出现在最后一个可见 assistant answer，排除 think、tool call 与 tool result",
            ],
            "tables_used": [path],
        },
    }


def build_artifact(summary: dict[str, Any]) -> dict[str, Any]:
    driver = summary["driver"]
    rollouts = summary["rollouts"]
    stages = driver["steps"]
    steady = driver["steady_state"]
    prewarm = driver["prewarm"]
    total_rows = int(rollouts["rows"])

    step_timing_chart = []
    for row in stages:
        for series, field in (
            ("队列等待", "queue_wait_s"),
            ("Actor 更新", "update_actor_s"),
            ("完整 step", "step_s"),
        ):
            step_timing_chart.append(
                {"step": row["step"], "series": series, "seconds": row[field]}
            )

    step_detail = [
        {
            "step": row["step"],
            "queue_wait_s": row["queue_wait_s"],
            "actor_update_s": row["update_actor_s"],
            "step_s": row["step_s"],
            "wait_share": row["queue_wait_s"] / row["step_s"],
        }
        for row in stages
    ]
    correctness = [
        {
            "metric": "旧轨迹证据口径",
            "correct": int(rollouts["evidence_contains_expected"]),
            "total": total_rows,
            "rate": rollouts["evidence_contains_expected"] / total_rows,
        },
        {
            "metric": "严格最终答案口径",
            "correct": int(rollouts["strict_final_answer_correct"]),
            "total": total_rows,
            "rate": rollouts["strict_final_answer_correct"] / total_rows,
        },
    ]

    current_run_path = "run-summary/efficiency_summary.json"
    matrix_source = _source(
        "configuration_matrix",
        "五组真实单步拓扑与过量采样对照",
        "docs/fastest_k_oversampling_validation_20260731.md",
        "汇总同数据、同训练拓扑下的 TP8×DP2 4→4/5→4/6→4，以及 TP4×DP4 4→4/5→4 单步实测。",
        """SELECT * FROM (VALUES
('T8D2 · 4→4','TP8×DP2',4,4,383.81,464.60,0.0000),
('T8D2 · 5→4','TP8×DP2',5,4,332.16,412.34,-0.1125),
('T8D2 · 6→4','TP8×DP2',6,4,283.85,364.69,-0.2150),
('T4D4 · 4→4','TP4×DP4',4,4,401.67,482.61,0.0388),
('T4D4 · 5→4','TP4×DP4',5,4,307.72,388.97,-0.1628)
) AS t(configuration, rollout_topology, candidates, selected, queue_wait_s, step_s, step_change_vs_baseline)""",
    )
    current_source = _source(
        "current_run_summary",
        "TP8×DP2 6→4 预热 8 组的 20-step 聚合结果",
        current_run_path,
        "解析 LLIN_PREWARM、LLIN_TRAIN_STAGE、LLIN_FASTEST_K 和落盘 rollout 指标。",
        f"SELECT * FROM read_json_auto('{current_run_path}')",
    )

    technical_summary = f"""## 技术摘要

- **单步矩阵中，TP8×DP2 的 `6→最快4` 最快。** 相对同拓扑 `4→4`，trainer 等待从 `383.81s` 降到 `283.85s`，完整 step 从 `464.60s` 降到 `364.69s`（`-21.50%`）。TP4×DP4 的副本数更多，但单 token 推理变慢，未超过 TP8×DP2。
- **20-step 稳定性验证成功，但训练机仍主要在等数据。** 20/20 更新和 320 条训练轨迹全部落盘，退出码为 0；step 2–20 的平均 step 为 `{steady['step_s']['mean']:.2f}s`，其中队列等待 `{steady['queue_wait_s']['mean']:.2f}s`，按累计 wall time 计算的等待占比为 `{steady['trainer_idle_ratio']:.2%}`。8-group 预热只覆盖开头两个 batch，无法弥补长期 rollout 生产率不足。
- **本轮奖励不能当作最终答案正确率，后续代码已修正。** 旧口径在整条轨迹任意位置搜索目标数值，320 条中命中 `{int(rollouts['evidence_contains_expected'])}` 条；排除工具结果、思考和工具调用后，最终可见答案只正确 `{int(rollouts['strict_final_answer_correct'])}` 条（`{rollouts['strict_final_answer_correct'] / total_rows:.2%}`）。按新语义离线 replay 后平均 reward 从 `{rollouts['score']['mean']:.4f}` 降为 `{rollouts['strict_reward_replay']['mean']:.4f}`。
- **检查点完整。** `global_step_20` 保存 model+extra，约 47.57 GiB；索引引用的 13 个 safetensors 分片全部存在。"""

    sources = [matrix_source, current_source]
    manifest_sources = [
        {"id": source["id"], "label": source["label"], "path": source["path"]}
        for source in sources
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Fastest-K 过量采样与双机流水线效率验证",
            "description": "五组单步配置矩阵与 TP8×DP2 6→4、预热 8 组的 20-step fully-async 稳态验证。",
            "generatedAt": GENERATED_AT,
            "charts": [
                {
                    "id": "configuration_step_chart",
                    "title": "五组配置的单步 wall time",
                    "subtitle": "同一 8K 数据与 TP4/PP2/CP2 训练拓扑；单位为秒，跨 rollout 拓扑比较包含各自冷启动噪声。",
                    "type": "bar",
                    "dataset": "configuration_matrix",
                    "sourceId": "configuration_matrix",
                    "encodings": {
                        "x": {"field": "configuration", "type": "nominal", "label": "配置"},
                        "y": {"field": "step_s", "type": "quantitative", "label": "完整 step", "unit": "s"},
                    },
                    "valueFormat": "number",
                    "unit": "s",
                    "layout": "full",
                    "maxRows": 5,
                },
                {
                    "id": "step_timing_chart",
                    "title": "20 个训练步的队列等待、Actor 更新与完整 step",
                    "subtitle": "第 1 步包含 actor 首次编译；step 2–20 用于稳态汇总。",
                    "type": "line",
                    "dataset": "step_timing_chart",
                    "sourceId": "current_run_summary",
                    "encodings": {
                        "x": {"field": "step", "type": "ordinal", "label": "训练步"},
                        "y": {"field": "seconds", "type": "quantitative", "label": "耗时", "unit": "s"},
                        "color": {"field": "series", "type": "nominal", "label": "阶段"},
                    },
                    "valueFormat": "number",
                    "unit": "s",
                    "layout": "full",
                    "maxRows": 60,
                },
                {
                    "id": "correctness_chart",
                    "title": "旧轨迹证据与严格最终答案正确率",
                    "subtitle": "同一 320 条训练轨迹；严格口径排除 think、tool call 与 tool result。",
                    "type": "bar",
                    "dataset": "correctness",
                    "sourceId": "current_run_summary",
                    "encodings": {
                        "x": {"field": "metric", "type": "nominal", "label": "正确性口径"},
                        "y": {"field": "rate", "type": "quantitative", "label": "轨迹占比"},
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                    "maxRows": 2,
                },
            ],
            "tables": [
                {
                    "id": "configuration_table",
                    "title": "单步配置矩阵",
                    "subtitle": "每个配置均完成一次全参数 GRPO 更新。",
                    "dataset": "configuration_matrix",
                    "sourceId": "configuration_matrix",
                    "defaultSort": {"field": "step_s", "direction": "asc"},
                    "columns": [
                        {"field": "configuration", "label": "配置"},
                        {"field": "queue_wait_s", "label": "队列等待(s)", "format": "number"},
                        {"field": "step_s", "label": "完整 step(s)", "format": "number"},
                    ],
                    "layout": "full",
                },
            ],
            "sources": manifest_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Fastest-K 过量采样与双机流水线效率验证", "layout": "full"},
                {"id": "technical_summary", "type": "markdown", "body": technical_summary, "layout": "full"},
                {
                    "id": "matrix_finding",
                    "type": "markdown",
                    "body": "## TP8×DP2 的 6→最快4 是当前单步最优，但不是零成本优化\n\n在 TP8×DP2 内，候选数从 4 增到 5、6 时，完整 step 依次从 464.60 秒降到 412.34、364.69 秒；`6→4` 绕开同一 prompt 的尾部候选最有效。TP4×DP4 增加到 4 个副本后，每个副本 TP 变小导致单 token 推理更慢，`5→4` 仍需 388.97 秒。结论只支持当前模型与 8K 配置下选择 TP8×DP2 6→4，不支持把更高 DP 当作普遍更快。",
                    "sourceId": "configuration_matrix",
                    "layout": "full",
                },
                {"id": "configuration_chart_block", "type": "chart", "chartId": "configuration_step_chart", "layout": "full"},
                {"id": "configuration_table_block", "type": "table", "tableId": "configuration_table", "layout": "full"},
                {
                    "id": "pipeline_finding",
                    "type": "markdown",
                    "body": f"## 预热解决冷启动，却没有解决长期生产率不足\n\n预热 8 个完整 group 用时 `{prewarm['wait_s']:.2f}` 秒、缓存 `{prewarm['queued_tokens']:,}` tokens，使前两步几乎无需等队列；从第 3 步起 backlog 被耗尽。step 2–20 的平均队列等待为 `{steady['queue_wait_s']['mean']:.2f}` 秒，而 actor 更新仅 `{steady['actor_update_s']['mean']:.2f}` 秒，累计等待占完整 step 的 `{steady['trainer_idle_ratio']:.2%}`。因此继续把队列从 8 加到 12 或 16 只能延长启动预热，不能改变长期供需。真正要优化的是 rollout groups/min。",
                    "sourceId": "current_run_summary",
                    "layout": "full",
                },
                {"id": "step_timing_chart_block", "type": "chart", "chartId": "step_timing_chart", "layout": "full"},
                {
                    "id": "reward_finding",
                    "type": "markdown",
                    "body": f"## 吞吐实验同时发现了奖励泄漏\n\n本轮满分判断在整条序列中搜索目标数值，工具返回里出现正确值就可能把轨迹计为正确。320 条训练样本中，旧口径命中 `{int(rollouts['evidence_contains_expected'])}` 条（`{rollouts['evidence_contains_expected'] / total_rows:.2%}`），严格最终答案只命中 `{int(rollouts['strict_final_answer_correct'])}` 条（`{rollouts['strict_final_answer_correct'] / total_rows:.2%}`）。按严格语义离线 replay 后，满分从 83 条降为 `{int(rollouts['strict_full_reward_count'])}` 条，平均 reward 从 `{rollouts['score']['mean']:.4f}` 降为 `{rollouts['strict_reward_replay']['mean']:.4f}`。本次交付已将后续满分条件改为最终可见答案正确；历史运行值不回写。",
                    "sourceId": "current_run_summary",
                    "layout": "full",
                },
                {"id": "correctness_chart_block", "type": "chart", "chartId": "correctness_chart", "layout": "full"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "body": "## 范围、数据与指标定义\n\n- 数据为 4 个真实 prompt 循环采样；每步训练消费 4 个 prompt group，每组保留最快 4 条，共 16 条轨迹。20 步分母为 320 条训练轨迹。\n- 模型为 Qwen3.6-27B；训练侧 TP4×PP2×CP2、16 NPU；rollout 侧 TP8×DP2、16 NPU；上下文 8,192 tokens，assistant/tool-feedback 上限 25/24。\n- `queue_wait_s` 只表示 trainer 等待 4 个完整 group；`update_actor_s` 是真实 actor 计算；`step_s` 还包含权重同步和少量组装开销。\n- 稳态按 step 2–20 定义，因为 step 1 的 actor 更新为 176.23 秒，包含首次图编译；后续中位数约 19.16 秒。\n- Fastest-K 的调度单位仍是完整 n=4 GRPO group，不会把不同 prompt 的候选混合。",
                    "layout": "full",
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": "## 实验设计与验证方法\n\n先分别运行 TP8×DP2 的 4→4、5→4、6→4，以及 TP4×DP4 的 4→4、5→4 单步闭环，固定训练拓扑、数据与训练 group 大小；选出单步 wall time 最低的 TP8×DP2 6→4。随后以 8 个完整 group 预热、staleness=1.0、最多 8 个在途/排队 group 运行 20 步。运行时补丁记录队列等待、反序列化、组装、奖励、advantage、actor 更新和完整 step。结束后只读解析 20 份 rollout JSONL，并用严格最终答案口径复核奖励。最终 checkpoint 通过索引到分片的一致性检查。",
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## 局限、反例与稳健性边界\n\n- 单步矩阵每个格子只有一次运行，可证明工程闭环和方向，不能估计置信区间；跨 TP8×DP2 与 TP4×DP4 的比较还包含各自冷启动噪声。\n- 20-step 只验证了 TP8×DP2 6→4，没有同一时间段、同一运行时计时补丁下的 4→4 长跑对照，因此不能把 20-step 所有变化因果归于 Fastest-K。\n- 日志记录 77 个 Fastest-K quorum，而 trainer 实际消费 80 个 group；quorum 分布以 n=77 为分母，不能把缺失的 3 个 marker 当成成功取消证据。\n- 154 个未选候选中物理 vLLM abort 仍为 0；上层 task 已取消，但本轮没有证明生成中的底层请求被实际中止。\n- 当前训练只有 4 个唯一 prompt，reward 与长度分布不代表 1,500 条任务总体。\n- 严格最终答案正确率极低，意味着本轮只适合评估系统效率，不能用于证明训练质量或收敛。",
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## 奖励已修正；下一步扩 rollout 产能\n\n1. 本次交付已将满分条件改为：成功工具证据 + 必需表 + **最终可见 assistant answer** 含正确数值；tool/thought 中命中只保留为诊断指标。320 条离线 replay 已确认新平均 reward 为 0.19625、满分 3 条。\n2. 用固定 prompt 顺序与 seed 跑 20-step `4→4` / `6→4` 对照，重新建立严格质量基线。\n3. 不继续加深队列。以 TP8×DP2 6→4 为基线，先做每副本 `max_num_seqs 16→24` 的 3-step HBM/吞吐门禁；通过后再跑 20-step。该参数直接提高可同时解码的候选数，更可能提升 groups/min。\n4. 要求下一轮记录物理 abort acknowledgement、zombie request、candidate compute tokens、group production rate 和严格最终正确率；任何吞吐收益都必须在这些 guardrails 下解释。\n5. 若 rollout 长期生产率仍低于 trainer 消费率，单靠缓存、预热或消息队列无法无缝衔接，需要增加 rollout 计算资源或降低每步训练消费速度。",
                    "layout": "full",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": "## 仍需回答的问题\n\n- 物理请求为什么在 154 次候选丢弃中一次都没有命中 abort：候选是否普遍停在工具/回合边界，还是逻辑到物理 request 映射仍有观测盲区？\n- 修正奖励后，Fastest-K 是否仍系统性偏向更短、最终答案更差的轨迹？\n- `max_num_seqs=24` 能否在 60% HBM 预算内提高两个 TP8 副本的有效 decode 并发，还是只增加 KV cache 压力？\n- 4 个 prompt 的长尾结构是否能代表完整任务集；扩大数据后 quorum p95 是否仍约 394 秒？",
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "configuration_matrix": CONFIGURATION_MATRIX,
                "step_timing_chart": step_timing_chart,
                "step_detail": step_detail,
                "correctness": correctness,
            },
        },
        "sources": sources,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    artifact = build_artifact(summary)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
