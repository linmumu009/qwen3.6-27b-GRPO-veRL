# Semantic-delta pairwise 一步金丝雀

日期：2026-08-12  
结论：一步 reference-free pairwise 更新让全部 16 对的正确候选 margin 朝正确方向移动，但只有 `3/16` 从负值翻为正值，未达到冻结的 `12/16` 概率门槛。按合同停止，不跑短回放、不追加同数据训练步、不晋级该 checkpoint。

## 1. 为什么做这一步

此前同一批首错状态上的两项训练前证据一致指向 plan-to-SQL realization：

- operator oracle 与 full semantic plan 一次生成分别仅恢复 `1/16`、`2/16`；
- Step 120 对 correct-vs-actual-wrong semantic delta 的正确候选偏好为 `0/16`，平均 margin 为 `-1.1877`。

因此本次只回答一个窄问题：一次受控的 pairwise 梯度能否在不引入更早 token 回退的前提下，把正确 SQL edit span 的相对概率推过预设门槛。

## 2. 冻结实验合同

- 起点：Step 120。
- 数据：16 个相邻的 `chosen → rejected` pair，共 32 行；chosen 是机械验证的纠正 SQL，rejected 是同一首错状态下模型实际产生的错误 SQL。
- 目标：只比较 semantic-delta token 的长度归一化 log probability，使用 reference-free logistic ranking loss；该目标不称为 DPO。
- 拓扑：单机 16 卡，TP=4、PP=2、CP=2；global batch 32，microbatch 2，关闭 shuffle，确保每个 microbatch 恰好是一对候选。
- 优化：fresh CPU-offload Adam，学习率 `1e-6`，beta `1.0`，只执行 1 个 optimizer step。
- 保存：只保存 model 与 extra，不保存 optimizer。
- 训练后门槛：正确候选占优至少 `12/16`；逐题 margin 改善至少 `12/16`；更早 non-greedy 回退为 0；冻结 offset 的 target 非法数为 0。

完整流水线在训练前用最终 32 行数据重新生成 Step 120 baseline，在训练后对同一数据与同一 mask 做 forward-only 复测，避免跨数据哈希或旧 token gate 比较。

## 3. 工程故障与修复边界

首次隔离运行在第一个 batch 前退出：固定顺序 sampler 没有提供 veRL trainer 每个 epoch 调用的 `set_epoch()` 接口。该运行没有 optimizer step、没有 checkpoint，保留作失败审计。

修复只增加 epoch-aware 的顺序 sampler：`set_epoch()` 记录 epoch，但迭代顺序始终不变。修复后单元测试、全量测试、容器内语法和固定顺序检查全部通过，并用新的运行编号重新执行完整流水线。

## 4. 实跑结果

### 4.1 唯一训练步

| 指标 | 结果 |
| --- | ---: |
| optimizer steps | 1 |
| train loss | 1.4948 |
| grad norm | 98.01 |
| learning rate | 1e-6 |
| global tokens | 70,019 |
| 单卡峰值已分配显存 | 26.92 GiB |
| 单卡峰值保留显存 | 27.59 GiB |
| CPU 内存 | 786.80 GiB |
| 完整 pipeline 墙钟 | 10 分 02 秒 |
| pipeline exit code | 0 |

最终 checkpoint 约 `51G`，model 与 extra 的分布式 metadata 均存在；审计未发现 optimizer 文件。

### 4.2 冻结概率门禁

| 指标 | Step 120 baseline | Pairwise Step 1 | 门槛 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 正确 semantic delta 占优 | 0/16 | 3/16 | ≥12/16 | 未通过 |
| 平均 margin | -1.1877 | -0.7646 | >0 为正确侧占优 | 改善但仍为负 |
| 中位 margin | -1.1270 | -0.7003 | >0 为正确侧占优 | 改善但仍为负 |
| 逐题 margin 改善 | — | 16/16 | ≥12/16 | 通过 |
| 新增更早 non-greedy 回退 | 0 | 0 | 0 | 通过 |
| 冻结 offset 非法 target | 0 | 0 | 0 | 通过 |

平均 margin 的净变化为 `+0.4232`。按首分叉家族观察：

| 家族 | baseline 平均 margin | Step 1 平均 margin | Step 1 正确侧占优 |
| --- | ---: | ---: | ---: |
| aggregation | -1.1459 | -0.6908 | 2/9 |
| query start | -1.8726 | -1.1011 | 1/3 |
| identifier/literal | -0.9744 | -0.8827 | 0/3 |
| clause | -0.1492 | -0.0642 | 0/1 |

## 5. 解释与停止决策

这一步证明梯度方向是连贯的：`16/16` margin 改善且没有更早分叉回退，不像随机噪声或 mask 符号反向。但它没有证明当前小样本配方足以改变模型决策边界：只有 3 对翻正，整体均值和中位数仍明显为负。

因此 checkpoint 仅保留为诊断资产，`promotion_allowed=false`。严格执行预先承诺的 fail-closed 决策：

- 不运行 semantic-plan 短回放或完整 48K 回放；
- 不在相同 16 对上追加 pairwise step；
- 不把方向性改善表述为准确率提升或泛化改善；
- 不将该 checkpoint 接入后续正式训练。

## 6. 下一步最有价值的动作

下一阶段不应继续压同一 16 对，而应把它们冻结为评价集，并从不重叠任务构造更大的、机械验证的 plan-to-SQL chosen/rejected 训练集。优先覆盖仍未翻正的 aggregation、query-start 与 identifier/literal 家族；在 CPU 门禁中固定来源、首错状态、pair 顺序、semantic mask 和执行等价性后，再设计一个有独立 held-out 证据的短训练矩阵。

在新的不重叠训练数据准备好之前，不再占用 NPU。实验结束后 5 号机 NPU 已确认无运行进程。
