# 老板 v15 DWH 主奖励 5-step 真实训练复盘

日期：2026-08-04

## 结论

`llin-v15-dwh-bossreward-5step-20260804-03` 已完成 5/5 次全参数 GRPO 更新、最终 val20 贪心评测和 checkpoint 写盘，训练进程退出码为 0。两台本项目 `llin` 容器随后停止，两机 16 张 NPU 均无运行进程，HBM 回到约 2.88–3.13 GiB 的驱动基线。

这次实验同时得到两个不能混淆的结论：

1. **训练与奖励链路跑通，并出现有限正向信号。** 固定 val20 的混合分数从冻结基线 `0.243075` 提升到 `0.391000`，老板奖励从 `0.452250` 提升到 `0.490000`，SQL 证据从 `0/20` 提升到 `1/20`；80 条训练 rollout 的 reward 公式不匹配和 verifier 异常均为 0，19/20 个 GRPO group 有组内奖励方差。
2. **还不能宣称质量收敛，也不能复用本轮权重继续训练。** val20 strict acc 仍为 `0/20`，老板答案正确仍为 `2/20`；PP=2 的在线 HF 导出只写入基础模型 `905/1199` 个 tensor，完整缺失语言模型第 32–63 层等 294 个 tensor。本轮 checkpoint 已标记 `CHECKPOINT_INVALID`，不得加载或部署。

因此，本轮是“真实数据、真实 PI Agent、真实老板奖励和真实参数更新”的工程与短程质量验证，不是可交付模型。下一次训练必须先通过新的 distributed-checkpoint 保存/加载门禁。

## 环境和配置

| 项目 | 配置 |
| --- | --- |
| 训练节点 | 5 号机，16 × Ascend 910 64GB |
| rollout 节点 | 6 号机，16 × Ascend 910 64GB |
| actor 并行 | Megatron TP=4、PP=2、CP=2、DP=1 |
| rollout 并行 | vLLM TP=8、DP=2 |
| 模型 | Qwen3.6-27B，全参数，LoRA 关闭 |
| 参数/优化器 | 参数常驻 NPU；优化器不做阶段间 offload；激活重计算开启 |
| 上下文 | prompt 4,096 + response 45,056，总上限 49,152 tokens |
| Agent | 老板同源 system prompt；`bash/read/write/edit` 四工具；最多 26 次 assistant、25 次工具反馈 |
| 数据 | v15 DWH 审核集：train 237 / val 20 / test 20，本轮实际消费 20 个唯一 train prompt |
| GRPO group | 每 prompt 4 条轨迹，每步消费 4 个完整 group，共 16 条轨迹 |
| 调度 | bounded fully-async；8-group 预热；最多 8 个并发/排队 group；staleness=1.0 |
| 采样 | 精确 `4→4`；没有 Fastest-K 同 prompt 候选过量采样，也没有丢弃候选 |
| 奖励 | 安全/协议/gold 门禁后，`0.7 × boss_reward + 0.3 × evidence_reward` |
| 学习率 | `1e-7` |
| 验证/保存 | 第 5 步在固定 val20 上贪心评测并保存 model+extra |

“8-group 预热”是跨不同 prompt 建立队列 backlog，不等于同一 prompt 的 `6→最快4` Fastest-K。本轮 20 个训练 prompt 各使用一次，20 个 group 全部进入训练，没有按速度永久淘汰任务。

## 运行时间线

- 运行开始：`2026-08-04 08:16:02 UTC`（北京时间 16:16:02）。
- 首次完成 8-group 预热：等待 `1960.216s`（约 32 分 40 秒），队列累计 `846,859 tokens`。
- 完成 5 次参数更新后执行固定 val20；验证耗时 `3313.342s`（约 55 分 13 秒）。
- 运行结束：`2026-08-04 11:13:12 UTC`（北京时间 19:13:12）。
- 总墙钟时间：`2h 57m 10s`。
- 训练退出码：0。

## 每一步训练时间

| Step | 等待 4 个完整 group | Actor 更新 | 整步 | 说明 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.164s | 280.495s | 288.951s | 使用预热队列前 4 个 group |
| 2 | 0.157s | 136.258s | 144.425s | 直接消费预热剩余 4 个 group |
| 3 | 1645.392s | 245.364s | 1898.607s | 队列耗尽，重新受长轨迹支配 |
| 4 | 421.940s | 235.119s | 664.832s | rollout 与训练部分重叠 |
| 5 | 1564.969s | 145.934s | 1718.457s | 数据源接近耗尽，等待最后 4 个 group |

5 步平均队列等待 `726.524s`、平均 actor 更新 `208.634s`、平均整步 `943.054s`；累计队列等待占 5 步总时间 `77.04%`。权重同步约 `7.3–8.1s/次`，不是主要瓶颈。

前两步看似无缝，是因为 32 分 40 秒的预热提前生成了 8 个 group；消费完 backlog 后，第 3、5 步再次等待 26–27 分钟。加深队列只能搬移等待时间，不能改变 rollout 长期生产率低于 trainer 消费率这一事实。

## 80 条训练 rollout

### 完整性与 GRPO 信号

- 5 个 rollout 文件、80/80 条轨迹、20 个完整 group，group size 全部为 4。
- 20 个唯一 prompt，每个只暴露一次；没有 train/val/test 重复或同 prompt 速度筛选。
- reward 公式不匹配：0；verifier 异常：0。
- 19/20 个 group 有组内奖励方差；只有 1 个零方差 group。
- 平均 score `0.352856`，中位数 `0.439500`，范围 `0–0.86`。
- 平均 boss reward `0.413812`；平均 evidence reward `0.215000`。
- strict acc `1/80`；最终答案正确 `10/80`；SQL 证据正确 `5/80`。
- 必需表使用 `73/80`；产生最终回答 `53/80`；bash 成功与协议有效均为 `80/80`。

### 轨迹长度和工具行为

- 平均 response 长度约 `25,939 tokens`（有完整 step 指标的四步）；单步均值范围 `18,188–32,326`。
- 轨迹最大达到 `45,056 tokens`；四个完整指标步的平均 clip ratio 为 `3.125%`。
- 平均消息轮数约 `36.09`；单步均值范围 `28–46`，单条最大 52。
- 每条轨迹工具调用均值 `39.09`，范围 `8–81`；共重放 2,873 条 bash 命令。
- 5/80 条被安全门禁归零：3 条 network、2 条 host-path escape；在线字段与离线安全重放完全一致。新增安全原因指标成功解释了硬归零，不再出现“只知道 safe=0”的盲区。

## 固定 val20：冻结基线与 step 5

| 指标 | 冻结模型 | Step 5 | 变化 |
| --- | ---: | ---: | ---: |
| 混合训练分数均值 | 0.243075 | 0.391000 | +0.147925 |
| 老板奖励均值 | 0.452250 | 0.490000 | +0.037750 |
| 严格证据奖励均值 | 0.132500 | 0.160000 | +0.027500 |
| 老板答案正确 | 2/20 | 2/20 | 不变 |
| 严格最终答案正确 | 1/20 | 1/20 | 不变 |
| SQL 证据正确 | 0/20 | 1/20 | +1 |
| strict acc | 0/20 | 0/20 | 不变 |
| 使用必需表 | 13/20 | 15/20 | +2 |
| 产生最终回答 | 15/20 | 17/20 | +2 |
| safe | 14/20 | 20/20 | +6 |

按答案类型：

| 类型 | 数量 | 冻结 score | Step 5 score | 冻结 boss reward | Step 5 boss reward | Strict correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| numeric | 9 | 0.262222 | 0.532000 | 0.534444 | 0.660000 | 0/9 |
| table | 11 | 0.227409 | 0.275636 | 0.385000 | 0.350909 | 0/11 |

总体分数改善主要来自 numeric 和过程项；table 的 boss reward 反而下降，strict acc 没有改善。5 步样本量太小，不能据此证明模型总体正确率提高，也不能把混合分数增长等同于最终评测增长。

最终 val20 额外核验：20 个唯一 task、无空值、reward 公式不匹配 0、verifier 异常 0；540 条可见 bash 命令没有任何安全命中，在线 `safe=20/20` 与离线重放完全一致。

## 本轮问题、根因与处理

### 1. 冻结基线的 `boss_fields_used=None` 聚合失败

配置：固定 val20、冻结模型、老板主奖励。3 条任务没有 `must_use_fields`，老板 process score 正确地把该项视为“不参与权重”，但旧实现同时向 veRL reducer 暴露 `None`。veRL 对每个 reward 字段求均值时抛出 `TypeError`。

处理：内部公式仍保留 `None` 的省略语义；对外监控字段改为 `1.0`，只修复可聚合协议，不改变奖励。最终 step-5 validation 的 20 条该字段全部是数值。

### 2. 安全硬归零缺少原因

配置：冻结 val20。旧结果有 6 条 `safe=0`，但落盘数据没有 network/destructive/path-escape 等原因，无法判断真实违规还是规则误判。

处理：新增各类纯数值安全诊断。正式训练 5 条安全失败均能被离线命令重放逐项解释；最终 val20 的 540 条 bash 命令与 `safe=20/20` 完全一致。

### 3. `-02` 在 step 0 主动中止

配置与最终 5-step 相同，但尚未应用安全原因观测。发现缺口时 `training/global_step=0`，主动重启本项目两个容器，未产生参数更新或 checkpoint。最终运行使用 `-03`，避免半步状态与正式证据混淆。

### 4. 完整 PI 48K 带来极长预热和队列耗尽

配置：完整 system/四工具、25 次工具反馈、48K、精确 4→4。首次 8-group 预热生成 846,859 tokens，耗时 1960.216s；随后第 3、5 步队列等待分别为 1645.392s 和 1564.969s。

结论：bounded fully-async 的重叠机制生效，但 rollout 供给仍明显慢于训练消费。8-group backlog 只覆盖前两步，不能实现长期无缝。后续应优先优化真实 Agent rollout 产能/终止策略，而不是继续无限加深 staleness 队列。

### 5. `mstx.range_end` 重复报错

配置：Megatron/MindSpeed actor 更新。日志持续打印 `mstx.range_end() missing 1 required positional argument: 'range_id'`。该异常来自性能标记 API 版本不兼容；5 次更新、梯度、权重同步和退出码均正常，因此不是训练失败，但造成大量日志噪声并掩盖真实错误。

处理状态：本轮保留证据，尚未修改上游 profiler。正式长训前应禁用这条不兼容的 range marker 或回移成成对保存 `range_id` 的实现。

### 6. NPU 不支持算子回退 CPU

第 1 步出现 `aten::_jagged_to_padded_dense_forward` 回退 CPU 警告。训练继续成功，但 actor MFU 仍很低，属于后续性能优化项，不能误当作 rollout 等待。

### 7. 最终 HF checkpoint 静默缺失半个 pipeline

配置：Megatron TP4/PP2/CP2，`use_dist_checkpointing=False`，mbridge 在线导出 HF，保存 `model,extra`。训练器返回 0，manifest 也被写入，但日志已报告：`294 tensors from the original checkpoint were not written`。

独立 fail-closed 核验结果：

- 基础模型 index：1,199 tensors。
- 输出 index：905 tensors，13 个被引用且存在的 safetensors 分片。
- 缺失：294 tensors，extra=0。
- 缺失范围：语言模型第 32–63 层、最终 norm 和 MTP，恰好对应 PP=2 的后半 pipeline stage。
- `extra/dist_ckpt` 只保存 RNG state，不含模型权重，无法恢复缺失层。

结论：`global_step_5` 占用约 48 GiB，但不是完整 checkpoint；已在运行目录写入 `checkpoint_integrity.json` 和 `CHECKPOINT_INVALID`。容器停止后完整训练态已释放，本轮权重不能续训或部署。

代码修正：

1. 正式训练改为 `actor_rollout_ref.actor.megatron.use_dist_checkpointing=True`，模型槽保存完整 Megatron 分布式权重；部署前另做独立 HF 转换与验证。
2. 新增 `scripts/verify_checkpoint_integrity.py`。HF checkpoint 必须与基础模型 tensor key 精确一致且所有引用分片非空；Megatron dist checkpoint 必须具备元数据和非空分片。
3. `launch_pi_formal_50step.sh` 在发布成功退出码前自动检查最后一次 checkpoint；失败时写入 `CHECKPOINT_INVALID` 并把作业退出码改为 8。

该修正已通过本地单元测试，但 distributed checkpoint 仍需一次实际“保存 → 新进程加载 → 最小前向”的 1-step 门禁，之后才能开始正式长训。

### 8. 最终 step 训练聚合行缺失

运行有 5 个 `LLIN_TRAIN_STAGE`、5 个 rollout 文件、80 条训练轨迹且参数版本到 5，但标准 `training/global_step` 聚合行只有 4 条；第 5 步紧接最终 validation 与 checkpoint 后退出。结果完整性可由 stage/rollout/param version 交叉证明，但通用分析器只看标准聚合行会误报 4 步。

处理状态：报告使用三类证据交叉计数；后续应让 fully-async trainer 在最终 validation/checkpoint 前先落盘第 5 步聚合指标。

## 下一步门槛

1. 用新的 Megatron distributed checkpoint 配置跑 1-step 保存/加载门禁；检查 metadata、全部分片、新进程恢复和最小前向一致性。
2. 修复或关闭 `mstx.range_end` 日志噪声，并记录 CPU fallback 对 actor 更新时间的影响。
3. checkpoint 门禁通过后，再运行至少 20-step 的 DWH pilot；继续使用固定 val20 与 sealed test20。
4. 扩展训练只看 strict correctness、numeric/table 分项和 reward hacking 审计，不以混合分数单项增长作为通过条件。
5. KB 继续排除，test20 继续封存，不因本轮短程信号提前使用。

## 证据位置

服务器运行目录：`runs/llin-v15-dwh-bossreward-5step-20260804-03/`。其中包含 `driver.log`、5 个 rollout JSONL、最终 validation、聚合摘要、checkpoint 完整性报告和失效标记。原始轨迹、日志、模型和 checkpoint 不进入 Git。
