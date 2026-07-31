# 48K 上下文与工具轮次验证报告

> 验证日期：2026-07-31  
> 模型：Qwen3.6-27B，全参数 GRPO，LoRA 关闭  
> 训练：5 号机 16 NPU，Megatron `TP4 × PP2 × CP2`  
> Rollout：6 号机 16 NPU，vLLM `TP8 × DP2`  
> 结论：48K rollout、前向、反向和 CPU Adam 更新均已实跑成功；训练显存足够，但完整 PI 工具环境仍未完全复现

## 1. 验证目标

这次验证回答三个问题：

1. 将最大上下文提高到 `49,152` tokens 后，5 号机的全参数训练显存是否足够。
2. 将多轮上限提高到 25 个 assistant turns、24 个工具反馈批次后，是否仍会被旧的 4/3 轮限制截断。
3. 老板轨迹存在单轮并行工具调用，当前 veRL 配置能否覆盖相应的调用轮数和并行度。

这里的 48K 不是只改 Hydra 参数或做公式估算。最终探针实际完成了：

```text
43,848-token prompt
        ↓
TP8 × DP2 vLLM 多轮工具 rollout
        ↓
4 条 GRPO responses
        ↓
1→16 HCCL 权重同步
        ↓
TP4 × PP2 × CP2 actor 前向、反向与 CPU Adam 更新
        ↓
exit code 0
```

## 2. 环境和最终配置

| 项目 | 配置 |
| --- | --- |
| 训练节点 | 5 号机，16 × 64GB Ascend NPU |
| Rollout 节点 | 6 号机，16 × 64GB Ascend NPU |
| 训练并行 | TP=4、PP=2、CP=2、DP=1 |
| Rollout 并行 | TP=8、DP=2 |
| 参数 | BF16，全参数训练，参数常驻 NPU |
| 优化器 | FP32 Adam，CPU offload |
| 梯度 | CPU offload |
| 激活 | full recompute / gradient checkpointing |
| 最大上下文 | 49,152 tokens |
| 默认 prompt/response 预算 | 4,096 / 45,056 tokens |
| 容量探针 prompt/response 预算 | 45,056 / 4,096 tokens |
| Assistant 回合上限 | 25 |
| 工具反馈批次上限 | 24 |
| 单轮并行工具调用上限 | 4 |
| 单次工具返回上限 | 32,768 字符 |
| vLLM chunked prefill | 8,192 tokens |
| Prefix cache | 开启 |
| Continuous Token | 开启 |

`max_user_turns` 在 veRL ToolAgentLoop 中按“工具反馈批次”计数，不是按同一轮中的每个并行工具调用计数。因此，仅把它提高到 24 仍不足以复现老板轨迹；还必须把默认的 `max_parallel_calls=1` 显式提高到 4。本次已同时修改 One-Step 和 bounded fully-async 启动入口。

## 3. Prompt 和工具轮次对齐

### 3.1 已对齐的部分

- 新 Parquet 保留源轨迹的原始 system message 和 user message，不再替换为项目早期的短 system prompt。
- 抽查首条 prompt：
  - system 文本哈希：`4009345a8c9d`
  - user 文本哈希：`eea2378880c5`
  - 均与 source trajectory 对应消息一致。
- 抽查的 4 条老板源轨迹：

| 任务 | Assistant turns | 工具调用/反馈 | 单轮最大并行调用 |
| --- | ---: | ---: | ---: |
| task_000286 | 8 | 11 | 2 |
| task_000133 | 11 | 20 | 4 |
| task_000214 | 7 | 9 | 4 |
| task_000281 | 6 | 10 | 3 |

当前 `25 / 24 / 4` 的上限覆盖上述源轨迹的实际范围，不再因框架默认值提前终止。

### 3.2 实际生成的工具轮次

| 验证 | 轨迹数 | 实际工具调用范围 | 平均工具调用 | 单轮并行调用峰值 |
| --- | ---: | ---: | ---: | ---: |
| 8K 真实环境 | 16 | 3–15 | 8.125 | 3 |
| 16K 真实环境 | 16 | 4–19 | 11.125 | 3 |
| 32K 容量探针 | 4 | 3–6 | 4.0 | 1 |
| 48K 容量探针 | 4 | 8–12 | 10.75 | 1 |

16K 实验已生成最多 19 次工具调用，接近抽查源轨迹的 20 次；8K 和 16K 中都观测到一轮连续 3 个工具调用，证明不再受旧的单调用限制。模型在本次样本中没有自然产生 4 路并行调用，但配置门禁已允许 4，覆盖老板源数据观测到的最大并行度。

工具调用数与工具反馈数偶尔相差 1，是 response token 上限在最后一个工具调用后截断造成的；这属于 token budget 截断，不是轮次配置回退。

### 3.3 尚未对齐的部分

当前运行时工具仍只有只读 `query_sqlite`。老板 PI runtime 使用的是 `bash/read/edit/write`，两者的：

- 工具 schema；
- 文件系统操作能力；
- 单次工具输出体积；
- 并发执行语义；
- 错误恢复路径

并不相同。因此，本次可以证明“原始 system/user prompt、上下文容量和轮次上限已对齐并可运行”，但不能把它称为“老板完整 PI Agent 环境已经一比一复现”。要完成最后的环境对齐，需要为每条 rollout 建立隔离、可回收的工作区，并实现受限的 `bash/read/edit/write` 工具，而不是直接给模型宿主机写权限。

## 4. 上下文阶梯实测

### 4.1 8K 真实环境

运行：`llin-realenv-8k-25turn-20260731-02`

| 指标 | 结果 |
| --- | ---: |
| 轨迹数 | 16 |
| Response mean / max | 3,358.25 / 4,096 |
| Response clip ratio | 25% |
| 消息轮次 min / mean / max | 8 / 16.375 / 32 |
| Actor allocated / reserved | 26.46 / 28.49 GiB |
| CPU memory | 813.85 GiB |
| 平均奖励 | 0.43125 |
| Grad norm | 1.5173 |
| 单步耗时 | 533.91s |
| 退出码 | 0 |

该实验首先证明 25/24 轮配置实际生效：轨迹总消息轮次已经明显超过旧配置能达到的范围。

### 4.2 16K 真实环境

运行：`llin-realenv-16k-25turn-20260731-01`

| 指标 | 结果 |
| --- | ---: |
| 轨迹数 | 16 |
| Response mean / max | 8,455.81 / 12,288 |
| Response clip ratio | 18.75% |
| 消息轮次 min / mean / max | 6 / 23.625 / 40 |
| 工具调用 min / mean / max | 4 / 11.125 / 19 |
| Actor allocated / reserved | 26.81 / 28.49 GiB |
| CPU memory | 926.53 GiB |
| 平均奖励 | 0.490625 |
| Actor loss | 0.120002 |
| Grad norm | 1.133889 |
| Rollout mean / max | 548.69 / 1,583.18s |
| Actor update | 192.03s |
| 单步耗时 | 1,791.04s |
| 退出码 | 0 |

16K 已覆盖老板抽查轨迹的典型工具调用数量，但长尾非常严重：最慢轨迹约 26.4 分钟，明显超过训练更新时间。

### 4.3 32K 前反向容量探针

运行：`llin-capacity-32k-1prompt-20260731-03`

前两次预检分别暴露并修正：

1. `train_batch_size=1` 时 `ppo_mini_batch_size` 仍为 4；
2. 4 条 GRPO responses 不能平均切给 8 个 AgentLoop workers。

最终使用 `ppo_mini_batch_size=1` 和 `agent.num_workers=4`，完整运行成功。

| 指标 | 结果 |
| --- | ---: |
| 实际 prompt | 27,848 tokens |
| Response mean / max | 2,657.25 / 4,041 |
| 总训练 tokens | 122,021 |
| 消息轮次 min / mean / max | 8 / 10 / 14 |
| Actor allocated / reserved | 27.86 / 31.35 GiB |
| CPU memory | 796.83 GiB |
| Rollout mean / max | 228.16 / 361.89s |
| Actor update | 184.89s |
| 单步耗时 | 561.69s |
| 退出码 | 0 |

这是容量探针，同一 prompt 的 4 条 reward 都是 0.05，因此组内 advantage 和梯度为 0；它证明计算图、显存和 optimizer step 能完成，不用于判断学习质量。

### 4.4 48K 前反向容量探针

运行：`llin-capacity-48k-1prompt-20260731-01`

| 指标 | 结果 |
| --- | ---: |
| 实际 prompt | 43,848 tokens |
| Response mean / max | 3,877 / 4,096 |
| Response clip ratio | 25% |
| 总训练 tokens | 190,900 |
| 消息轮次 min / mean / max | 18 / 22 / 24 |
| 工具调用范围 | 8–12 |
| Actor allocated / reserved | 32.42 / 37.78 GiB |
| CPU memory | 798.93 GiB |
| Rollout mean / max | 216.13 / 229.98s |
| Actor update | 199.00s |
| 权重同步 | 7.51s |
| 单步耗时 | 448.39s |
| 退出码 | 0 |

48K 探针实际走到了 24 个消息轮次，并完成了全参数 actor update。它同时验证了：

- vLLM `TP8 × DP2` 能以 49,152 最大上下文启动并生成；
- Megatron CP2 能处理约 47.7K 的 prompt+response 单序列；
- CPU Adam/梯度 offload 能完成 optimizer step；
- 1→16 权重同步与 48K 配置兼容。

## 5. 显存结论

5 号机单卡实际可用约 `61.27 GiB`。48K 实测峰值为：

```text
allocated = 32.42 GiB
reserved  = 37.78 GiB
reserved headroom = 61.27 - 37.78 = 23.49 GiB
```

因此，当前 `TP4 × PP2 × CP2`、每卡 micro-batch 1、激活重计算、参数常驻 NPU、Adam/梯度 CPU offload 的组合可以运行 48K 全参数训练。

这个结论的适用边界：

- 已实测的是 1 个 prompt × 4 条 GRPO responses。
- 正式配置每步为 4 个 prompt × 4 responses。由于 actor 仍按 micro-batch 1 顺序执行，训练峰值不应按轨迹数线性放大，但 CPU batch、rollout 并发和总步耗时会显著增加。
- 48K 下不应提高训练 micro-batch；也不建议同时关闭 optimizer/gradient offload。
- Rollout 侧已真实成功启动并生成，没有 OOM；但 16 条同时接近 48K 时会受到 KV-cache 调度和 preemption 影响，不能从 4 条探针推导满并发吞吐。

## 6. 生产建议

1. 保留 48K 最大能力，但按实际长度做 token-budget batching，不要求每条都生成到 48K。
2. 保持：
   - `micro_batch_size_per_gpu=1`
   - TP4/PP2/CP2
   - optimizer/gradient CPU offload
   - full recompute
   - vLLM chunked prefill
3. 使用 bounded fully-async，队列单位保持一个完整的 `n=4` GRPO group。
4. 对超长尾设置真实 vLLM abort 的硬超时；不能只取消 Python coroutine。
5. 25/24/4 是容量上限，不是要求每条轨迹必须跑满。应记录实际 assistant turns、工具反馈批次和单轮并行调用数。
6. 在补齐隔离版 PI `bash/read/edit/write` runtime 前，报告和实验名继续标注为 `query_sqlite` 环境，避免把“prompt/轮次对齐”误写成“完整环境对齐”。

## 7. 最终判断

- **训练显存：足够。** 48K 真实前反向 reserved 峰值 37.78 GiB/卡，余量约 23.49 GiB/卡。
- **Rollout 显存：本次 4 条探针足够。** TP8×DP2 在 49,152 上下文下成功生成；满 16 条接近 48K 的吞吐仍需独立压力测试。
- **工具轮次：上限已匹配。** 25 assistant turns、24 工具反馈批次和单轮 4 个并行调用覆盖抽查源轨迹；实际 16K rollout 已达到 19 次工具调用。
- **真实环境：尚未完全一致。** System/user prompt 已按源数据保留，但当前工具实现仍是 `query_sqlite`，不是老板 PI 的 `bash/read/edit/write`。
