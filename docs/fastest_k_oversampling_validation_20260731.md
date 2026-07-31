# Fastest-K 过量采样验证报告

日期：2026-07-31

## 1. 目标与结论

本次验证的问题是：保持 GRPO 每个 prompt 使用 4 条轨迹计算组内优势不变，物理上并发生成 6 条候选轨迹，在任意 4 条完成后立即组成训练 group，并取消剩余 2 条，能否减少最慢 rollout 对训练机的阻塞。

结论分为两部分：

1. **吞吐方向成立。** 在模型、数据、并行拓扑和训练 batch 完全相同的单步 A/B 中，`6 取最快 4` 将 trainer 收集首批数据的等待时间从 `383.81s` 降到 `283.85s`，下降 `26.04%`；完整训练步从 `464.60s` 降到 `364.69s`，下降 `21.50%`。
2. **不能直接作为质量无损优化。** 同一轮 A/B 中，平均奖励从 `0.500000` 降到 `0.290625`，完全答对数从 `6/16` 降到 `2/16`，平均输出字符数下降 `12.62%`。这与“最快样本倾向于更短、更容易提前结束”的选择偏差一致。由于每个方案目前只跑了 1 个随机训练步，尚不能把全部质量差异归因于 Fastest-K，但已经足以说明它必须经过多步质量 A/B 后才能作为正式默认值。

## 2. 实现语义

### 2.1 GRPO group 保持完整

训练侧的 `rollout.n` 仍为 4。过量采样只扩大物理生成数量：

```text
一个 prompt
   ├─ candidate 0 ─┐
   ├─ candidate 1 ─┤
   ├─ candidate 2 ─┤── 最先完成的 4 条 → 一个完整 n=4 GRPO group
   ├─ candidate 3 ─┤
   ├─ candidate 4 ─┘
   └─ candidate 5 ── 取消/丢弃
```

这样不会把不同 prompt 的 response 混到同一个 group，也不会用不足 4 条的残缺 group 计算 GRPO advantage。

运行参数为：

```bash
FASTEST_K=4
OVERSAMPLE_CANDIDATES=6
```

将 `OVERSAMPLE_CANDIDATES=4` 即可关闭过量采样，恢复原始 `4→4` 行为。

### 2.2 Quorum 与取消

Agent loop 使用 `asyncio.wait(..., return_when=FIRST_COMPLETED)` 持续收集完成项，达到 4 条 quorum 后：

1. 按真实完成时间选取最先完成的 4 条；
2. 取消仍在运行的 Python task；
3. 通过逻辑 request ID 查找当前物理 vLLM request；
4. 若请求仍在模型生成阶段，调用 server actor 的 `abort_request`；
5. 逐请求取消固定使用 `reset_prefix_cache=False`，避免破坏已经验证约 32% 命中的 prefix cache。

Ray 启动脚本会在 worker 注册前幂等应用补丁，避免 head 和 rollout 节点运行不同版本的 agent loop。

## 3. A/B 实验条件

两组只改变候选数量，其余条件完全一致。

| 项目 | Baseline | Fastest-K |
| --- | --- | --- |
| 物理候选数 | 4 | 6 |
| 选入训练 group | 4 | 最快 4 |
| GRPO `rollout.n` | 4 | 4 |
| prompt groups/step | 4 | 4 |
| 最终训练轨迹数 | 16 | 16 |
| 数据 | `pi_real_env_48k_4prompt.parquet` | 相同 |
| 最大上下文 | 8,192 | 8,192 |
| prompt / response 上限 | 4,096 / 4,096 | 相同 |
| assistant / tool-feedback 上限 | 25 / 24 | 相同 |
| 训练拓扑 | TP4 × PP2 × CP2 | 相同 |
| rollout 拓扑 | TP8 × DP2 | 相同 |
| checkpoint | 关闭 | 关闭 |

本次使用 8K 而不是 48K，是为了先隔离验证 Fastest-K 的调度语义和真实训练闭环；它不改变此前已经完成的 48K 容量结论。

## 4. 实验过程

### 4.1 前置门禁

- 两台机器的四个 veRL runtime 文件均存在 Fastest-K marker。
- 补丁在干净上游副本上首次返回 `patched`，第二次返回 `already-patched`。
- 两侧 Python 编译通过。
- Ray 重启后仍可见 32 张 NPU，训练/rollout 角色分别固定在 5/6 号机。
- 两台机器的输入 Parquet SHA-256 一致：

```text
3f669ef0024b5b0ede9fb4ccdbcaa26d70277e8a0cf5dd5bec6a1803b3d99d38
```

第一次 baseline 启动在生成前失败，原因是新 Parquet 只存在于 5 号机，6 号机的远端 AgentLoop worker 无法打开文件。同步数据并校验哈希后重新运行；该失败没有进入 rollout，也没有产生训练更新。

### 4.2 成功运行

- Baseline：`llin-fastest-k4of4-baseline-8k-1step-20260731-02`
- Fastest-K：`llin-fastest-k6of4-8k-1step-20260731-01`
- 两个运行均完成一个全参数 GRPO 更新并以退出码 `0` 结束。
- 两组最终落盘轨迹均为 16 条。

Fastest-K 四个 prompt group 的 quorum 时间为：

```text
157.998s
212.089s
253.719s
283.583s
```

每个 group 都记录 `candidates=6 selected=4 discarded=2`。

## 5. 结果

### 5.1 吞吐

| 指标 | 4→4 baseline | 6→最快4 | 变化 |
| --- | ---: | ---: | ---: |
| trainer 收集等待 | 383.81s | 283.85s | **-26.04%** |
| 完整训练步 | 464.60s | 364.69s | **-21.50%** |
| rollouter active | 448.28s | 347.43s | -22.50% |
| 初次参数同步 | 13.79s | 13.58s | 基本相同 |
| 最终轨迹数 | 16 | 16 | 相同 |

这证明过量采样可以绕开同一 prompt 内最慢的 1–2 条候选，使完整 group 更早进入 bounded fully-async 队列，从而缩短训练机等数据的时间。

### 5.2 轨迹质量与选择偏差

| 指标 | 4→4 baseline | 6→最快4 |
| --- | ---: | ---: |
| 平均 reward | 0.500000 | 0.290625 |
| reward 范围 | 0.20–1.00 | 0.05–1.00 |
| 完全答对 | 6/16 | 2/16 |
| 使用必需表 | 16/16 | 15/16 |
| 调用工具 | 16/16 | 16/16 |
| 平均 output chars | 9,426.75 | 8,237.44 |

Fastest-K 的输出平均短 `12.62%`。在工具任务中，“更快完成”可能同时意味着：

- 更短的 reasoning；
- 更少的工具探索；
- 更早给出最终答案；
- 较少遇到慢工具调用；
- 较低概率走到需要多轮修正的困难路径。

因此 Fastest-K 改变的不只是系统调度，也会改变进入训练的数据分布。它属于带选择偏差的采样策略，不能仅以 tokens/s 或 trainer 利用率评价。

### 5.3 物理 vLLM abort 的证据边界

本次四个 group 的日志均为：

```text
physical_aborts=0
reset_prefix_cache=False
```

代码中的真取消链路已经建立并通过编译，但本次被丢弃的 task 在 quorum 发生时没有可查询到的活跃物理 vLLM request。最可能的情况是它们当时处于工具执行、回合切换或两次模型请求之间；取消上层 task 已阻止其发起下一轮模型请求，因此无需对 NPU 上的活跃请求执行 abort。

所以本次可以证明：

- Fastest-K quorum、完整 group 和上层取消生效；
- 未选中的候选没有进入训练数据；
- prefix cache 没有被取消逻辑主动重置。

本次**不能**宣称已经观察到一个生成中的物理 vLLM request 被成功 abort。后续长跑应要求 `physical_aborts > 0` 的定向门禁，并同时检查 abort acknowledgement 和 zombie request 计数。

## 6. 判断与下一步

### 6.1 当前判断

Fastest-K 适合作为可配置实验能力，但还不适合作为不加条件的生产默认策略：

- 吞吐收益已被真实全参数训练验证；
- 单步质量指标存在明显下降；
- 物理 abort 路径尚未在运行中被实际命中；
- 单步随机 A/B 无法评估长期 reward、KL 和收敛影响。

### 6.2 建议的正式 A/B

下一轮建议至少运行 20 step/arm，并固定相同的 prompt 调度顺序：

1. `4→4` baseline；
2. `6→最快4`；
3. 如质量下降稳定存在，再测试“6 条中按 deadline 完成、但在已完成候选中做长度/奖励无关的随机 4 条”，区分尾延迟收益和最快选择偏差。

必须同时记录：

- trainer queue wait、step wall time、queue depth；
- candidates/groups per minute、有效训练 tokens/s；
- reward、answer accuracy、KL、clip ratio、grad norm；
- response tokens、工具轮数、工具成功率；
- completed discard、physical abort、abort acknowledgement、zombie request；
- prefix-cache hit rate；
- policy version 与 sample staleness。

若 `6→4` 在多步 A/B 中仍能保持约 20% 的 step 降幅，且 reward/accuracy/KL 与 baseline 无显著恶化，才建议把它设为正式默认值。
