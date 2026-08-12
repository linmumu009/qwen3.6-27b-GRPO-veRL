# Step 120 semantic plan 与 semantic-delta 训练前门禁

日期：2026-08-12  
范围：train236 中同一组 16 条开发题；不作 held-out 或晋升声明。

## 结论

下一训练目标已锁定为 **plan-to-SQL realization and recovery**，并且训练形式应是一次 chosen-vs-rejected pairwise 金丝雀，而不是继续增加 semantic token/span 的正向 SFT 权重。

三臂一次生成门禁中，Control 恢复 `1/16`，operator oracle 仍为 `1/16`，其中 aggregation-critical 恢复 `0/9`（门槛 `4/9`）；full semantic plan 仅恢复 `2/16`（门槛 `8/16`）。因此正确 operator 或完整表/列/join 计划都不足以让 Step 120 可靠地产生下一条正确 SQL，不能把下一训练目标解释为 plan selection 或 schema grounding。

在相同“模型首错 SQL + 实际工具结果”状态上，进一步比较机械验证 correction SQL 与模型实际首错 SQL 的 token edit span：正确候选在 `0/16` 对中获得更高平均 token 概率，平均 margin 为 `-1.1877`，中位数为 `-1.1270`；负值表示模型系统性偏好错误候选。aggregation、query start、identifier/literal 与 clause 四类全部同方向。Step 120 冻结的首个 non-greedy SQL token 在 `16/16` 对中精确重建，排除了数据改写或更早分叉导致的假信号。

## 独立协议问题

48 条三臂生成中，严格“恰好一个 bash”只有 `29/48`；有 `7/48` 行产生并行调用，总计 `55` 次工具调用，包括 `36` 次 bash 和 `19` 次 read。回放循环没有执行这些新调用，因此该指标只描述工具动作纪律，不污染只读 SQL 评分。它说明后续配方还应把 bash-only 作为硬策略约束，但不是当前首要语义目标。

## 冻结的一步停止门槛

- 从 Step 120 初始化，只训练一次 optimizer step；候选为同一 16 对 correct-vs-actual-wrong SQL。
- 训练后正确 semantic delta 至少在 `12/16` 对中占优。
- 至少 `12/16` 对的 margin 相比 Step 120 改善。
- 不允许任何任务出现更早的首个 non-greedy SQL token 回退。
- 概率门禁通过前不跑完整自由回放，不作 checkpoint 晋升。

## 资源与证据边界

两项门禁均未初始化 optimizer、未保存 checkpoint。semantic-plan 生成在工具执行前停止；margin 运行仅做 Step 120 forward-only。完成后两机各 16 张 NPU 均无残留项目进程。

机器路径、原始问题、SQL、工具结果和答案未写入本报告；精确聚合值见同目录安全 JSON 摘要。
