# GRPO 沙箱数据与运行说明

数据与训练方法说明  2026年9月16日

本材料说明 Qwen3.6-27B 的双机沙箱 GRPO 路线。以 v15 DWH 修正版数据和当前 100 步入口说明运行方法；历史 Step100 训练使用 237 条，修正版为 236 条，不混写。

## 一 RL 沙箱数据构成

RL 数据由任务、沙箱环境和评分依据组成。模型在沙箱中实时生成多轮轨迹，同一道任务采样 4 条，再按组内相对奖励更新。历史源轨迹保存在参考文件中，不作为 GRPO 的答案输入。

| 组成 | 内容 |
| --- | --- |
| 任务 | v15 DWH 物流数据查询与分析指令 |
| 环境 | 同源 logistics.sqlite、表结构及工作区 |
| 评分依据 | 参考 SQL、结构化预期结果、答案类型和所需表字段 |
| 在线交互 | bash、read、edit、write；每条轨迹独立工作区 |

| 数据版本 | train | val | test |
| --- | --- | --- | --- |
| 历史 full277 | 237 | 20 | 20 |
| 修正 full276 | 236 | 20 | 20 |

修正版删除了一条相同指令绑定冲突答案的训练任务。276 条参考 SQL 均已通过同源数据库执行及结果匹配，但这不等于全部业务题意已完成人工确认。

## 二 数据形态

下面摘录训练 Parquet 第 70 条记录，任务编号 task_000245。问题要求查询超时订单总数，记录同时保存数值型评分目标和参考 SQL。

```
{
  "data_source":"boss_pi_aligned_v1",
  "agent_name":"pi_agent",
  "prompt":[{"role": "user", "content": "时效分析汇总里，最近这一期超时订单的总数有多少？"}],
  "reward_model":{"style":"rule","ground_truth":{
    "task_id":"task_000245",
    "environment_id":"sft/20260628_v15",
    "answer_type":"numeric",
    "expected_value_json":"481309",
    "verification_sql":"SELECT SUM(exception_order_cnt) FROM fact_dws_timeliness_analysis"
  }},
  "extra_info":{"split":"train"}
}
```

节选省略 system 消息、user 前置沙箱布局，以及工具参数和来源指纹。任务句、SQL、评分值均保持原文；本页展示存储格式，不替代业务题意审核。

prompt 中的任务交给模型；ground_truth 交给评分器，不能拼进模型提示。样本没有预先写好的 assistant 回答，回答和工具轨迹由训练时在线生成。

样本来源：boss_v15_dwh_full276_20260806/dataset/boss_pi_train.parquet，第 70 条。

## 三 脚本运行指令

5 号机负责训练，16 卡 TP4/PP2/CP2；6 号机负责 rollout，16 卡 TP8/DP2。两侧需同步项目、模型和沙箱，使用既定 veRL 环境；在专用空闲环境启动 Ray。

```
# 5 号机训练容器
cd /workspace/llin-verl-grpo
bash scripts/start_ray_m05.sh

# 6 号机 rollout 容器
cd /workspace/llin-verl-grpo
bash scripts/start_ray_m06.sh
```

Ray 两个角色就绪后，在 5 号机训练容器执行修正版数据检查与启动。

```
cd /workspace/llin-verl-grpo
export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PWD/runtime:$PWD:${PYTHONPATH:-}"
export DATA_DIR="$PWD/data/boss_v15_dwh_full276_20260806/dataset"
python3 scripts/check_boss_alignment_contract.py \
  --data-dir "$DATA_DIR"
export RUN_NAME="llin-grpo-full276-$(date +%Y%m%d-%H%M%S)"
bash scripts/launch_pi_formal_100step_12groups.sh
```

该入口从原始 /models/Qwen3.6-27B 训练 100 步。每步更新 4 组任务，每组 4 条轨迹；12 组指可并行在途组数，不是每步更新量。脚本同时检查远端数据并验证最终检查点。

本命令使用修正版 full276，不复刻旧 full277。8 月 25 日的开源续训 Step120 属于另一条 GRPO 路线，不在此启动。

## 依据

本仓库 scripts/prepare_boss_aligned_dataset.py、launch_pi_formal_100step_12groups.sh、run_pi_formal_100step_12groups.sh；docs/training_data_provenance_quality_audit_20260806.artifact.json。
