# veRL Qwen3.6 Step-120 纠错 SFT 冒烟实测

日期：2026-08-11

## 结论

veRL 自带的 `verl.trainer.sft_trainer` 可以复用当前 Megatron 训练栈，并能从 Step 120 的 distributed model checkpoint 做模型态初始化。在 5 号机 16 张 Ascend NPU 上，TP4/PP2/CP2 已完成一次真实的前向、反向、梯度裁剪和参数更新，进程最终退出码为 `0`。

这不是正式纠错训练，也不允许晋升模型。输入只有一条确定性合成工具调用样本，目的仅是验证工程链路；Step 120 原 checkpoint 未被修改，SFT 优化器和 dataloader 状态均从零建立。

## 已验证链路

- Trainer：veRL 官方 `verl.trainer.sft_trainer`。
- Engine：Megatron，`TP=4, PP=2, CP=2`，16 NPU。
- 初始化：`engine.use_dist_checkpointing=true`，只读取 Step 120 的 `actor/model/dist_ckpt`。
- 恢复语义：`trainer.resume_mode=disable`、`checkpoint.load_contents=[]`，不会加载 GRPO Adam、学习率状态或数据游标。
- 优化器：新建全参 Adam，NPU 参数常驻，optimizer/gradient CPU offload，学习率 `1e-7`。
- 数据：完整 Qwen3.6 system/user/assistant/tool/assistant 对话和工具 schema。
- 损失：只覆盖 assistant 内容、工具调用及其结束 token；system、user 和 tool response 全部遮罩。
- 保存：冒烟测试只保存 `extra` 与 dataloader 状态，不保存模型和优化器。

## Qwen3.6 数据兼容

veRL 默认 `MultiTurnSFTDataset` 会逐条消息调用 chat template。Qwen3.6 的工具模板要求 system、tools 和 user query 作为完整对话共同渲染，因此默认数据集在单独处理 system 消息时会报：

```text
No user query found in messages.
System message must be at the beginning.
```

项目使用 veRL 官方 `data.custom_cls` 扩展点接入 `Qwen36AssistantMaskSFTDataset`：先用模型自带模板一次性渲染完整对话，再按 Qwen assistant 起止控制 token 生成 loss mask。门禁会拒绝 assistant 数量不一致、含歧义控制 token、无 assistant loss token 或超长样本，不以 `ignore_input_ids_mismatch=true` 绕过问题。

合成样本的服务器预检结果：

| 指标 | 数值 |
| --- | ---: |
| 对话总 token | 418 |
| assistant loss token | 65 |
| 被遮罩的 system/user/tool token | 353 |
| 有 assistant loss 的行 | 1/1 |
| 同时遮罩非 assistant 上下文的行 | 1/1 |

## 四次隔离启动

每次启动均使用新目录，失败记录没有覆盖，且都发生在正式模型晋升之前。

| 运行 | 到达阶段 | 结果与修正 |
| --- | --- | --- |
| `-01` | Hydra 配置合成 | 已存在字段误用 `+` 新增语法；改为 `++` 新增或覆盖。 |
| `-02` | Megatron bridge 初始化 | 容器 site-package 中的旧 bridge 不识别 Qwen3.5 架构名；固定为 Step 120 已验证的 `Megatron-Bridge-de93536e`。 |
| `-03` | Step 120 加载、Adam 初始化、`train_batch` | Megatron SFT 明确不支持 `right` padding；恢复官方 `no_padding`。 |
| `-04` | 完整训练与清理 | 成功，退出码 `0`。 |

## 成功运行证据

成功运行：`llin-repair-sft-megatron-smoke-20260811-04`。

| 指标 | 数值 |
| --- | ---: |
| 开始时间（UTC） | `2026-08-11 07:54:38` |
| 结束时间（UTC） | `2026-08-11 07:59:38` |
| 总墙钟 | `5m00s` |
| veRL epoch 进度段 | `3m26s` |
| train loss | `0.9603356` |
| 裁剪前 grad norm | `141.0989` |
| learning rate | `1e-7` |
| global tokens | `418` |
| 单卡峰值 allocated HBM | `26.2696 GiB` |
| 单卡峰值 reserved HBM | `26.6074 GiB` |
| 5 号机进程统计 CPU 内存 | `821.6263 GiB` |
| 整个冒烟运行目录 | `282 MiB` |

梯度范数有限且非零，结合成功完成 `train_batch` 和 checkpoint 清理，可确认这不是只加载模型或只做前向。单条短样本导致 MFU 极低，不应把本次墙钟外推为真实 16–64 条纠错数据的每步吞吐；但峰值显存和 CPU 内存与既有全参 GRPO 的约 `29.63 GiB/卡`、`824.35 GiB` 同量级，说明现有 5 号机容量足够。

## 下一步边界

工程 go/no-go 已通过；下一步可以把 16 条机械验证通过的真实纠错轨迹转换成同一 schema，先做小样本过拟合门禁。正式运行前仍必须逐条验证最终数值、SQL/证据、工具调用去重、完整收尾和 train/dev/test 隔离。本次合成样本及 `-04` 产物均不得用于模型晋升或效果结论。
