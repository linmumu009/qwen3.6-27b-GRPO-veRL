#!/usr/bin/env python3
"""Create the reproducible Step 120 analysis notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python", "version": "3"}
    nb["cells"] = [
        markdown("""# Step 120 dense30 试验复盘

**TL;DR：** 老板原版评分从 Step 100 的 `0.443750` 升到 Step 120 的 `0.563745`，但提升主要来自完成率和过程覆盖；数值正确由 3/20 降到 2/20。dense30 自身几乎不变，因此应先扩大密封评测、增强组内正确性信号，而不是直接继续堆训练步数。

本 notebook 只读取仓库中的聚合汇总，不包含服务器路径、原始轨迹、模型或数据库。"""),
        code("""from pathlib import Path
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Glyph .* missing from font")
plt.style.use("seaborn-v0_8-whitegrid")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
summary_path = ROOT / "docs" / "step120_dense_trial_20260810_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["analysis"], summary["generated_at"]"""),
        markdown("""## 1. 数据质量与可比性

比较前先验证三版是否真的在同一批题、同一输入、同一 ground truth 上评测，并检查奖励公式复算与 verifier error。"""),
        code("""integrity = summary["validation_integrity"]
quality = pd.DataFrame({
    "检查": ["task 集合一致", "共同任务数", "GTS 逐题一致", "prompt 逐题一致", "verifier error 总数"],
    "结果": [
        integrity["task_sets_identical"],
        integrity["common_task_count"],
        integrity["gts_identical_count"],
        integrity["prompt_identical_count"],
        sum(integrity["verifier_error_counts"].values()),
    ],
})
quality"""),
        markdown("""## 2. 老板原版评分

老板总奖励、结果分和过程分来自原版 `reward_judge.py`。Step 120 的总分点估计提升 0.119995，但 20 道题的配对置信区间仍跨 0。"""),
        code("""c100_120 = summary["boss_exact"]["step100_vs_step120"]
c120_200 = summary["boss_exact"]["step120_vs_step200"]
boss = pd.DataFrame([
    {"checkpoint": label, "老板总奖励": row["reward_total_mean"], "结果分": row["result_score_mean"], "过程分": row["process_score_mean"]}
    for label, row in [
        ("Step 100", c100_120["step100"]),
        ("Step 120", c100_120["step120"]),
        ("Step 200", c120_200["step200"]),
    ]
]).set_index("checkpoint")
boss"""),
        code("""ax = boss.plot(kind="bar", figsize=(9, 4.8), ylim=(0, 1), rot=0, color=["#3B82F6", "#F59E0B", "#10B981"])
ax.set_title("Boss-original score on the same val20")
ax.set_ylabel("Mean score")
ax.set_xlabel("")
plt.tight_layout()
plt.show()"""),
        code("""b100, b120 = c100_120["step100"], c100_120["step120"]
behavior = pd.DataFrame({
    "Step 100": [b100["complete_count"], b100["correct_numeric_count"], b100["tables_hit_count"]],
    "Step 120": [b120["complete_count"], b120["correct_numeric_count"], b120["tables_hit_count"]],
}, index=["完整收尾", "数值正确", "必需表命中"])
paired_boss = c100_120["paired_reward"]
display(behavior)
pd.Series({
    "mean_delta": c100_120["numeric_deltas"]["reward_total_mean"]["delta"],
    "bootstrap_ci_low": paired_boss["mean_delta_bootstrap_95pct_ci"][0],
    "bootstrap_ci_high": paired_boss["mean_delta_bootstrap_95pct_ci"][1],
    "wins": paired_boss["wins"], "losses": paired_boss["losses"], "ties": paired_boss["ties"],
    "exact_sign_p": paired_boss["exact_sign_test_p"],
})"""),
        markdown("""## 3. dense30 同口径复算

三版都用 Step 120 的 dense30 公式重新计算，避免拿旧 reward 和新 reward 直接比较。这里是判断奖励改造是否在自身目标上起效的关键证据。"""),
        code("""dense = pd.DataFrame([
    {
        "checkpoint": f"Step {step}",
        "dense30": summary["validation"][f"step{step}"]["means"]["dense30_score"],
        "dense correctness": summary["validation"][f"step{step}"]["means"]["dense_final_answer_correctness"],
        "base score": summary["validation"][f"step{step}"]["means"]["base_score"],
    }
    for step in (100, 120, 200)
]).set_index("checkpoint")
dense"""),
        code("""ax = dense.plot(kind="bar", figsize=(9, 4.8), ylim=(0, 0.55), rot=0, color=["#8B5CF6", "#EC4899", "#64748B"])
ax.set_title("Dense30 metrics replayed with one formula")
ax.set_ylabel("Mean score")
ax.set_xlabel("")
plt.tight_layout()
plt.show()"""),
        code("""paired = summary["paired_validation"]["step100_vs_step120"]
pd.DataFrame([
    {
        "metric": key,
        "delta": paired[key]["mean_delta"],
        "ci_low": paired[key]["paired_bootstrap_95pct_ci"][0],
        "ci_high": paired[key]["paired_bootstrap_95pct_ci"][1],
        "wins": paired[key]["wins"], "losses": paired[key]["losses"], "ties": paired[key]["ties"],
    }
    for key in ("dense30_score", "dense_final_answer_correctness", "base_score", "boss_reward")
]).set_index("metric")"""),
        markdown("""## 4. 训练曲线与耗时

日志有 19 个在线 score 汇总（Step 102–120），Step 101 汇总行缺失；另有 20 个 stage 耗时记录。在线分数每步样本不同，因此只看波动，不当作固定集泛化曲线。"""),
        code("""run = summary["training_run"]
curve = pd.DataFrame(run["score_metric_rows"])
ax = curve.plot(x="global_step", y="score_mean", marker="o", figsize=(9, 4.2), legend=False, color="#2563EB")
ax.axhline(run["score_mean"], color="#DC2626", linestyle="--", label=f'均值 {run["score_mean"]:.3f}')
ax.set_title("Online training score, Step 102–120")
ax.set_ylabel("score mean")
ax.legend()
plt.tight_layout()
plt.show()"""),
        code("""pd.DataFrame([
    {"阶段": "单步平均", "秒": run["mean_step_s"], "分钟": run["mean_step_s"] / 60},
    {"阶段": "最终验证", "秒": run["validation_time_s"], "分钟": run["validation_time_s"] / 60},
    {"阶段": "保存 checkpoint", "秒": run["checkpoint_save_time_s"], "分钟": run["checkpoint_save_time_s"] / 60},
]).round(2)"""),
        markdown("""## 5. 结论与行动建议

1. Step 120 是当前更值得保留的候选：老板总分和过程完成度优于 Step 100/200。
2. 不能宣称正确性提升：数值正确 3/20→2/20，dense30 几乎不变，关键区间均跨 0。
3. 下一步先做 80–100 道密封集，并同时报告老板总分、dense 正确性和二值正确率。
4. 再训练时先提高 mixed-correct group 比例，并从同一 checkpoint 与 optimizer 状态做 20–30 步单因素 A/B。
5. 运行效率方面，结束阶段瓶颈是 67.6 分钟最终验证；保存 checkpoint 只有约 89 秒。

### 局限

- 固定集只有 20 题，且已多次查看，存在评测过拟合风险。
- 老板评分器是代理指标，不等于业务专家人工正确性。
- 跨 checkpoint 观察不能把所有差异因果归于 dense30 单个奖励改动。"""),
    ]
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Execute all cells before writing")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build()
    if args.execute:
        NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(Path.cwd())}}).execute()
    nbf.write(notebook, args.output)


if __name__ == "__main__":
    main()
