# 老板 v15 DWH 冻结模型基线与 5-step 门禁准备

日期：2026-08-04

## 结论

老板 v15 DWH 的固定 20 条 validation 已完成全部轨迹生成并成功落盘。运行在 veRL 最终汇总阶段因 `boss_fields_used=None` 无法求平均而以退出码 1 结束；故障发生在 rollout 与 reward 计算之后，不影响 20 条结果的完整性，也没有产生任何参数更新。

现有结果可以直接作为冻结模型基线，无需重复消耗约 35 分钟生成同一批贪心轨迹。修复后的奖励输出保证每个监控字段均为数值；没有字段要求的任务将 `boss_fields_used` 记为 `1.0`，但老板 process score 内部仍按原规则省略该字段权重。

## 环境与数据

- 5 号机：16 张 NPU，Megatron `TP4 × PP2 × CP2` 训练角色；本轮为 forward-only 冻结验证，不创建优化器、不更新权重。
- 6 号机：16 张 NPU，vLLM `TP8 × DP2` rollout。
- 数据：`boss_v15_dwh_full277_20260804` 的固定 validation split，共 20 条；11 条 table、9 条 numeric。
- Agent：老板同源 system prompt 与 `bash/read/write/edit` schema，最多 26 个 assistant turns / 25 个工具反馈批次，总上下文 48K。
- 采样：贪心 `n=1`。
- 奖励：`0.7 × boss_reward + 0.3 × strict_evidence_reward`，安全/协议/gold 门禁失败时硬归零。

## 运行记录

- 运行：`llin-v15-dwh-full277-frozen-val20-20260804-01`
- 起止：服务器记录 `07:10 → 07:45`，约 35 分钟。
- 轨迹：`20/20` 唯一 task_id，已写入 `validation/0.jsonl`。
- reward 公式不匹配：0。
- verifier 异常：0。
- 模型更新：0。
- 退出码：1，仅由最终 NumPy 指标聚合 `None` 触发。
- 自动监督器：正确记录 `baseline_failed` 并阻断原计划的 5-step，没有误启动训练。

## 冻结基线指标

| 指标 | 结果 |
| --- | ---: |
| 混合训练分数均值 | 0.243075 |
| 老板奖励均值 | 0.452250 |
| 严格证据奖励均值 | 0.132500 |
| 老板数字答案正确 | 2/20 |
| 严格最终答案正确 | 1/20 |
| SQL 证据正确 | 0/20 |
| strict acc | 0/20 |
| 使用必需表 | 13/20 |
| 产生最终回答 | 15/20 |
| bash 成功 | 20/20 |
| gold SQL 自洽 | 20/20 |

分类型：

| 类型 | 任务数 | 混合分数均值 | 老板奖励均值 | 严格正确 |
| --- | ---: | ---: | ---: | ---: |
| numeric | 9 | 0.262222 | 0.534444 | 0/9 |
| table | 11 | 0.227409 | 0.385000 | 0/11 |

## 本轮发现并修复的问题

### 1. 可选字段指标返回 `None`

老板 process score 在任务没有 `must_use_fields` 时会省略该分量；原实现同时把 `boss_fields_used=None` 暴露给 veRL。veRL 的 validation reducer 会对每个 reward 字段计算均值，因此 3 条无字段要求的任务导致 `TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'`。

修复：内部公式继续使用 `None` 表示“不参与加权”，对外监控字段改为数值 `1.0`。这不会改变任何任务的 reward，只修复指标协议。

### 2. 安全硬归零缺少原因观测

20 条中 6 条记录为 `safe=0`。从最终可见 output 重建的 519 条 bash 命令没有命中当前正式安全规则，但 reward 使用的是工具事件原始参数；旧落盘文件没有保存具体安全原因，无法证明是隐含/截断事件还是规则误杀。

修复：在线 reward 新增纯数值字段：

- `bash_command_count`
- `unsafe_command_count`
- `unsafe_network_count`
- `unsafe_destructive_count`
- `unsafe_host_path_escape_count`
- `unsafe_python_network_count`
- `unsafe_root_scan_count`

后续 rollout 可以直接说明每个硬归零来自哪类安全规则，不记录或暴露完整敏感命令。

### 3. 第一次 5-step 初始化主动终止

`llin-v15-dwh-bossreward-5step-20260804-02` 已通过数据门禁并开始初始化，但在 `global_step=0` 时发现上述安全观测缺口，因此主动重启两个本项目 `llin` 容器；没有参数更新、checkpoint 或半步训练状态。

最终运行使用新编号 `llin-v15-dwh-bossreward-5step-20260804-03`，保留 `-02` 作为可审计的主动中止记录。

## 5-step 判断门槛

第 5 步对同一 20 条 validation 贪心评测。只有满足以下条件才扩展训练：

1. 老板奖励和 strict acc 至少一个改善，另一个没有显著恶化。
2. SQL 证据正确率或最终答案正确率出现可解释提升。
3. reward 公式不匹配和 verifier 异常保持为 0。
4. 安全硬归零原因可解释；如果主要来自规则误杀，先修安全契约，不用错误负反馈继续训练。
5. numeric/table 分项和人工样例没有明显 reward hacking。
