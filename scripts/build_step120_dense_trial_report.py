#!/usr/bin/env python3
"""Build the canonical portable-report payload for the Step 120 dense trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GENERATED_AT = "2026-08-10T17:10:00+08:00"


def source(source_id: str, label: str, path: str, description: str) -> dict[str, Any]:
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
            "filters": [
                "固定 v15 DWH val20，三版按 task_id 一一配对",
                "老板分使用原版 reward_judge.py，未启用 LLM judge",
                "dense30 按 0.7 × base_score + 0.3 × dense_final_answer_correctness 复算",
                "原始轨迹、模型、数据库和机器绝对路径不进入报告载荷",
            ],
            "metric_definitions": [
                "老板总奖励 = 0.5 × result_score + 0.5 × process_score；has_answer=0 时为 0",
                "dense30 = 0.7 × base_score + 0.3 × dense_final_answer_correctness",
                "置信区间为同题 reward 差值的 20,000 次配对 bootstrap 95% 区间",
                "exact sign p 只使用非平局题，双侧检验",
            ],
            "tables_used": [path],
        },
    }


def chart(chart_id: str, title: str, subtitle: str, chart_type: str, dataset: str,
          x_field: str, x_label: str, y_field: str, y_label: str,
          color_field: str | None = None, max_rows: int = 30) -> dict[str, Any]:
    encodings: dict[str, Any] = {
        "x": {"field": x_field, "type": "nominal" if chart_type == "bar" else "quantitative", "label": x_label},
        "y": {"field": y_field, "type": "quantitative", "label": y_label},
    }
    if color_field:
        encodings["color"] = {"field": color_field, "type": "nominal", "label": "Checkpoint"}
    return {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "type": chart_type,
        "dataset": dataset,
        "sourceId": "step120_dense_summary",
        "encodings": encodings,
        "valueFormat": "number",
        "layout": "full",
        "maxRows": max_rows,
    }


def build(summary: dict[str, Any]) -> dict[str, Any]:
    boss_100_120 = summary["boss_exact"]["step100_vs_step120"]
    boss_120_200 = summary["boss_exact"]["step120_vs_step200"]
    boss_rows = []
    for label, values in (
        ("Step 100", boss_100_120["step100"]),
        ("Step 120", boss_100_120["step120"]),
        ("Step 200", boss_120_200["step200"]),
    ):
        for metric, key in (
            ("老板总奖励", "reward_total_mean"),
            ("结果分", "result_score_mean"),
            ("过程分", "process_score_mean"),
        ):
            boss_rows.append({"metric": metric, "checkpoint": label, "label": f"{label} · {metric}", "value": values[key], "task_count": 20})

    dense_rows = []
    for step in (100, 120, 200):
        means = summary["validation"][f"step{step}"]["means"]
        for metric, key in (
            ("dense30 总分", "dense30_score"),
            ("dense 正确性", "dense_final_answer_correctness"),
            ("base score", "base_score"),
        ):
            dense_rows.append({"metric": metric, "checkpoint": f"Step {step}", "value": means[key], "task_count": 20})

    boss_total_rows = [
        {"checkpoint": row["checkpoint"], "value": row["value"], "task_count": 20}
        for row in boss_rows
        if row["metric"] == "老板总奖励"
    ]

    training_rows = summary["training_run"]["score_metric_rows"]
    p_boss = boss_100_120["paired_reward"]
    p_dense = summary["paired_validation"]["step100_vs_step120"]["dense30_score"]
    p_correct = summary["paired_validation"]["step100_vs_step120"]["dense_final_answer_correctness"]
    b100 = boss_100_120["step100"]
    b120 = boss_100_120["step120"]
    v100 = summary["validation"]["step100"]
    v120 = summary["validation"]["step120"]

    metric_rows = [
        {"metric": "老板总奖励", "step100": f'{b100["reward_total_mean"]:.6f}', "step120": f'{b120["reward_total_mean"]:.6f}', "delta": f'{b120["reward_total_mean"]-b100["reward_total_mean"]:+.6f}', "meaning": "方向性提高；95% CI 跨 0"},
        {"metric": "老板结果分", "step100": f'{b100["result_score_mean"]:.3f}', "step120": f'{b120["result_score_mean"]:.3f}', "delta": f'{b120["result_score_mean"]-b100["result_score_mean"]:+.3f}', "meaning": "小幅提高"},
        {"metric": "老板过程分", "step100": f'{b100["process_score_mean"]:.3f}', "step120": f'{b120["process_score_mean"]:.3f}', "delta": f'{b120["process_score_mean"]-b100["process_score_mean"]:+.3f}', "meaning": "主要提升来源"},
        {"metric": "完整收尾", "step100": f'{b100["complete_count"]}/20', "step120": f'{b120["complete_count"]}/20', "delta": f'{b120["complete_count"]-b100["complete_count"]:+d}', "meaning": "少 3 道超时"},
        {"metric": "数值正确", "step100": f'{b100["correct_numeric_count"]}/20', "step120": f'{b120["correct_numeric_count"]}/20', "delta": f'{b120["correct_numeric_count"]-b100["correct_numeric_count"]:+d}', "meaning": "没有改善，反而少 1 道"},
        {"metric": "必需表命中", "step100": f'{b100["tables_hit_count"]}/20', "step120": f'{b120["tables_hit_count"]}/20', "delta": f'{b120["tables_hit_count"]-b100["tables_hit_count"]:+d}', "meaning": "过程覆盖改善"},
        {"metric": "dense30 总分", "step100": f'{v100["means"]["dense30_score"]:.6f}', "step120": f'{v120["means"]["dense30_score"]:.6f}', "delta": f'{v120["means"]["dense30_score"]-v100["means"]["dense30_score"]:+.6f}', "meaning": "新训练目标几乎不变"},
        {"metric": "dense 正确性", "step100": f'{v100["means"]["dense_final_answer_correctness"]:.6f}', "step120": f'{v120["means"]["dense_final_answer_correctness"]:.6f}', "delta": f'{v120["means"]["dense_final_answer_correctness"]-v100["means"]["dense_final_answer_correctness"]:+.6f}', "meaning": "方向性提高但不确定"},
    ]
    uncertainty_rows = [
        {"metric": "老板总奖励", "mean_delta": f'{boss_100_120["numeric_deltas"]["reward_total_mean"]["delta"]:+.6f}', "ci": f'[{p_boss["mean_delta_bootstrap_95pct_ci"][0]:.6f}, {p_boss["mean_delta_bootstrap_95pct_ci"][1]:.6f}]', "wins_losses_ties": f'{p_boss["wins"]}/{p_boss["losses"]}/{p_boss["ties"]}', "sign_p": f'{p_boss["exact_sign_test_p"]:.5f}', "interpretation": "正向，但样本不足以排除无提升"},
        {"metric": "dense30 总分", "mean_delta": f'{p_dense["mean_delta"]:+.6f}', "ci": f'[{p_dense["paired_bootstrap_95pct_ci"][0]:.6f}, {p_dense["paired_bootstrap_95pct_ci"][1]:.6f}]', "wins_losses_ties": f'{p_dense["wins"]}/{p_dense["losses"]}/{p_dense["ties"]}', "sign_p": f'{p_dense["exact_sign_test_p"]:.5f}', "interpretation": "没有可辨别变化"},
        {"metric": "dense 正确性", "mean_delta": f'{p_correct["mean_delta"]:+.6f}', "ci": f'[{p_correct["paired_bootstrap_95pct_ci"][0]:.6f}, {p_correct["paired_bootstrap_95pct_ci"][1]:.6f}]', "wins_losses_ties": f'{p_correct["wins"]}/{p_correct["losses"]}/{p_correct["ties"]}', "sign_p": f'{p_correct["exact_sign_test_p"]:.5f}', "interpretation": "小幅正向，统计不确定"},
    ]
    run = summary["training_run"]
    runtime_rows = [
        {"phase": "单步平均", "seconds": round(run["mean_step_s"], 1), "minutes": round(run["mean_step_s"] / 60, 1), "interpretation": "其中等待 rollout 平均 428.6 秒"},
        {"phase": "最终验证", "seconds": round(run["validation_time_s"], 1), "minutes": round(run["validation_time_s"] / 60, 1), "interpretation": "约 68 分钟，是结束阶段主要耗时"},
        {"phase": "保存 checkpoint", "seconds": round(run["checkpoint_save_time_s"], 1), "minutes": round(run["checkpoint_save_time_s"] / 60, 1), "interpretation": "约 89 秒；不是此前误判的 69 分钟"},
    ]

    charts = [
        chart("boss_scores", "老板原版总奖励：Step 100 / 120 / 200", "固定同题 val20、贪心 n=1；0–1 均值。", "bar", "boss_total_scores", "checkpoint", "Checkpoint", "value", "均值", None, 3),
    ]
    tables = [
        {"id": "metric_table", "title": "Step 100→120 关键指标", "subtitle": "同一 20 道任务；delta 为 Step 120 减 Step 100。", "dataset": "metric_table", "sourceId": "step120_dense_summary", "columns": [{"field": "metric", "label": "指标"}, {"field": "step100", "label": "Step 100"}, {"field": "step120", "label": "Step 120"}, {"field": "delta", "label": "变化", "movement": True}], "layout": "full"},
        {"id": "uncertainty_table", "title": "同题配对不确定性", "subtitle": "20,000 次配对 bootstrap；胜/负/平以 Step 120 相对 Step 100 判定。", "dataset": "uncertainty_table", "sourceId": "step120_dense_summary", "columns": [{"field": "metric", "label": "指标"}, {"field": "mean_delta", "label": "均值变化", "movement": True}, {"field": "ci", "label": "95% CI"}, {"field": "wins_losses_ties", "label": "胜/负/平"}], "layout": "full"},
        {"id": "runtime_table", "title": "结束阶段耗时复核", "subtitle": "从完整 driver 日志解析；纠正此前把验证耗时误当作保存耗时的判断。", "dataset": "runtime_table", "sourceId": "step120_dense_summary", "columns": [{"field": "phase", "label": "阶段"}, {"field": "seconds", "label": "秒"}, {"field": "minutes", "label": "分钟"}], "layout": "full"},
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Step 120 提高了老板总分，但没有证明正确性提升", "layout": "full"},
        {"id": "summary", "type": "markdown", "body": "## Technical Summary\n\n- **老板原版总奖励从 `0.443750` 升到 `0.563745`，增加 `0.119995`（相对 +27.0%）**，但配对 bootstrap 95% CI 为 `[-0.019375, 0.265610]`，20 题不足以排除无提升。\n- **提升来自完成与过程，不是最终正确性。** 完整收尾 `13→16`、必需表命中 `15→18`、过程分 `0.765→0.840`；数值正确反而 `3→2`。\n- **新 dense30 训练目标几乎没变。** 同题复算 `0.324059→0.324087`，变化只有 `+0.000028`；dense 正确性 `+0.011642`，置信区间仍跨 0。\n- **因此不建议立即继续堆步数。** 先扩充密封评测并提高组内正确性可区分率，再决定是否从 Step 120 续训。", "layout": "full"},
        {"id": "boss_finding", "type": "markdown", "body": "## 老板评分改善是方向性证据，不是定论\n\nStep 120 在 20 道同题上 7 胜、3 负、10 平；均值提升不是单个任务造成，但区间仍跨 0。Step 120 同时明显高于 Step 200，说明它是当前三个 checkpoint 中更值得保留的候选，不过仍应等待更大密封集验证。", "sourceId": "step120_dense_summary", "layout": "full"},
        {"id": "boss_chart", "type": "chart", "chartId": "boss_scores", "layout": "full"},
        {"id": "mechanism", "type": "markdown", "body": "## 提升机制是“更容易收尾并覆盖过程”\n\n老板总分上升与完成题数、必需表命中和字段命中同步；正确数值却少 1 道，说明模型学到的主要是任务执行与输出契约，而不是更可靠的统计口径或最终答案。答案平均长度从 4,349 降到 1,810 字，也表明输出更简洁，但简洁本身不能替代正确性。", "sourceId": "step120_dense_summary", "layout": "full"},
        {"id": "metric_table_block", "type": "table", "tableId": "metric_table", "layout": "full"},
        {"id": "dense_finding", "type": "markdown", "body": "## dense30 试验没有在自身目标上形成清晰增益\n\n把 Step 100、120、200 的同一 val20 全部用相同 dense30 公式离线复算后，Step 100→120 仅增加 `0.000028`。dense 正确性虽增加 `0.011642`，但二值 final-answer correctness 与 strict acc 都下降。当前证据只支持“方向可继续研究”，不支持“奖励改造已经成功”。", "sourceId": "step120_dense_summary", "layout": "full"},
        {"id": "uncertainty", "type": "markdown", "body": "## 小样本结论必须带区间\n\n三个关键指标的配对区间都跨 0。老板总奖励的点估计最大，但仍可能由 20 题波动造成；dense30 的点估计几乎为 0。把这轮结果当成候选筛选门禁是合理的，把它当成泛化提升证明则不合理。", "sourceId": "step120_dense_summary", "layout": "full"},
        {"id": "uncertainty_table_block", "type": "table", "tableId": "uncertainty_table", "layout": "full"},
        {"id": "training_curve_finding", "type": "markdown", "body": "## 在线训练分数高度波动，没有持续上升趋势\n\n日志中可用的 19 个 step 汇总均值为 `0.3010`；前 5 步均值 `0.2848`，后 5 步 `0.2798`。每步 prompt 和 rollout 不同，这条曲线不能直接替代固定集评测，但至少没有显示继续训练会自然带来稳定增长。", "sourceId": "step120_dense_summary", "layout": "full"},
        {"id": "scope", "type": "markdown", "body": "## 范围、数据与方法\n\n- Step 100、120、200 均使用同一固定 v15 DWH val20；task_id、prompt 和 ground truth 20/20 一致，verifier error 为 0。\n- 老板分由当前原版 `reward_judge.py` 重放；Step 200 重放产物与已有结果逐字节一致。\n- dense30、base score 与 dense correctness 从原始 val20 行离线复算，三版统一口径。\n- 对 Step 100→120 做 task-level paired bootstrap 与 exact sign test；没有把 20 题当独立大样本做正态近似。", "sourceId": "step120_dense_summary", "layout": "full"},
        {"id": "runtime", "type": "markdown", "body": "## 约 69 分钟来自最终验证，不是保存模型\n\n完整日志显示最终验证耗时 `4054.6` 秒（67.6 分钟），checkpoint 保存从开始到完成约 `89` 秒。此前把两者之间的墙钟时间都归到保存阶段是错误的；这会直接影响下一轮优化优先级，应优先缩短或拆分最终验证，而不是改 checkpoint 保存格式。", "sourceId": "step120_dense_summary", "layout": "full"},
        {"id": "runtime_table_block", "type": "table", "tableId": "runtime_table", "layout": "full"},
        {"id": "limitations", "type": "markdown", "body": "## 局限与不确定性\n\n- 只有 20 道固定题，置信区间较宽；多次查看同一 val20 也会逐渐形成评测过拟合。\n- 老板评分器是业务代理指标，不等价于专家人工核验；过程分上升可能掩盖数值错误。\n- 在线训练曲线的每一步样本组成不同，不能当成严格的时间序列泛化曲线。\n- Step 100/200 是旧奖励训练，Step 120 是 dense30 续训；跨 checkpoint 比较可评估最终表现，但不能把全部差异因果归于单个奖励项。", "layout": "full"},
        {"id": "next_steps", "type": "markdown", "body": "## 建议的下一步\n\n1. **保留 Step 120，不立刻续训。** 它是当前老板分最好的候选，但正确性证据尚弱。\n2. **先做 80–100 道密封集。** 预先冻结题目、评分器和成功标准；老板总奖励、dense 正确性、二值正确率必须同时报告。\n3. **提升组内正确性可区分率。** 采样或课程设计要让更多 group 同时包含正确和错误响应；仅提高正确性权重无法解决全对/全错 group 无相对信号的问题。\n4. **下一轮做小型 A/B。** 从同一 Step 120 checkpoint 与 optimizer 状态出发，仅改一个因素，先跑 20–30 步；通过密封集门禁后再扩到 100 步。", "layout": "full"},
        {"id": "questions", "type": "markdown", "body": "## 仍需回答的问题\n\n- 扩到 80–100 道密封题后，老板总奖励的 +0.12 是否仍然存在？\n- 哪些 prompt 难度区间最能产生 mixed-correct group，并真正提高 dense 正确性？\n- 最终验证的 67.6 分钟中，轨迹生成、工具执行和评分各占多少，能否并行或分层抽样？", "layout": "full"},
    ]

    compact_source = source("step120_dense_summary", "Step 100/120/200 同题配对与运行耗时汇总", "docs/step120_dense_trial_20260810_summary.json", "读取三版 val20、老板原版重放、配对置信区间和 Step 120 完整 driver 日志的聚合结果。")
    manifest = {"version": 1, "surface": "report", "title": "Step 120 提高了老板总分，但没有证明正确性提升", "description": "固定 val20 上对 Step 100/120/200 的老板原版评分、dense30 复算、配对不确定性和运行耗时诊断。", "generatedAt": GENERATED_AT, "charts": charts, "tables": tables, "sources": [{k: v for k, v in compact_source.items() if k != "query"}], "blocks": blocks}
    snapshot = {"version": 1, "generatedAt": GENERATED_AT, "status": "ready", "datasets": {"boss_scores": boss_rows, "boss_total_scores": boss_total_rows, "dense_scores": dense_rows, "training_curve": training_rows, "metric_table": metric_rows, "uncertainty_table": uncertainty_rows, "runtime_table": runtime_rows}}
    return {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": [compact_source]}


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
