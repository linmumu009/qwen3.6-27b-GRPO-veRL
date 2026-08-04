# 正式 50-step GRPO 质量诊断与整改报告（2026-08-04）

## 结论先行

`llin-pi-formal-grpo-4of4-50step-20260803-03` 在工程上成功：`50/50`、退出码 `0`、800 条训练 rollout 完整、最终 checkpoint 约 48 GiB、两机 NPU 已释放。但它没有证明模型能力提升：五次贪心 validation 的严格准确率、最终答案正确率和 SQL 证据正确率全部为 0。

主要问题不是 HCCL、显存或 optimizer，而是训练目标本身：

1. 正式 V2 数据只验证了 SQL 可执行与标签存在，没有验证用户 instruction 是否唯一指向 hidden gold。200 条中 191 条被确定性规则标记为需要人工复核。
2. 训练时 800 条轨迹只有 24 条最终答案正确、19 条 SQL 证据正确、4 条严格正确；大量 reward 来自“用了必需表”和“给了最终回答”，不是答对。
3. 200 个 GRPO group 中 43 个组奖励完全相同；同一 prompt 平均只训练 1.25 个 group，正确行为既少又没有足够重复。
4. 正式 V2 没有保留老板源 system prompt，使用的是项目短 fallback；模型不知道唯一数据库已经在 `/workspace`，375/800 条轨迹尝试了不安全命令，主要是枚举容器根目录和其他沙箱。
5. 学习率 `1e-7`、平均 PPO KL 约 `5.31e-4`、clipfrac 约 `0.00142`，更新很保守。在稀疏且部分错位的 reward 下，盲目增加 step 不会可靠解决问题。

因此本轮的正确下一步不是继续扩展 V2 训练，也不是先调吞吐，而是重建 instruction/gold 对齐的 V3 数据，补 source system prompt 和沙箱门禁，再做小规模 reward/学习率 A/B。

## 1. 运行范围与证据

| 项目 | 结果 |
| --- | --- |
| 运行名 | `llin-pi-formal-grpo-4of4-50step-20260803-03` |
| 时间 | 2026-08-03 12:20:05 UTC 至 2026-08-04 00:41:49 UTC |
| 总时长 | 12h 21m 44s |
| 训练 | 50/50，退出码 0 |
| rollout | 50 文件 × 16 条 = 800 条 |
| prompt | train 160 个唯一 prompt |
| GRPO group | 200 个，每组 4 条，组大小全部正确 |
| validation | step 10/20/30/40/50，各 20 条、贪心 n=1 |
| checkpoint | `global_step_50`，约 48 GiB，model+extra |
| verifier | 0 异常；线上 score 与奖励公式重放 0 差异 |

机器可读分析由以下只读脚本生成：

- `scripts/analyze_formal_grpo_50step.py`
- `scripts/audit_formal_instruction_gold_alignment.py`

原始 rollout、driver log、数据和 checkpoint 仍只保留在 `/data3/llin`，不进入 Git。

## 2. 当前奖励函数实际在奖励什么

当前 V2 奖励定义：

```text
score = 0.60 × 最终答案正确
      + 0.25 × SQL 证据结果与 gold SQL 完全一致且 bash 成功
      + 0.10 × 使用了必需表
      + 0.05 × 有最终可见回答
```

只要出现 unsafe 命令或工具协议非法，整条轨迹为 0。严格准确率要求最终答案、SQL 证据、必需表、bash 成功和安全全部成立。

800 条训练轨迹的实测分解：

| 组件 | 条数 | 比例 |
| --- | ---: | ---: |
| 严格正确 `acc` | 4 | 0.50% |
| 最终答案正确 | 24 | 3.00% |
| SQL 证据正确 | 19 | 2.38% |
| 使用必需表 | 542 | 67.75% |
| 有最终回答 | 315 | 39.38% |
| bash 成功 | 800 | 100% |
| 协议有效 | 800 | 100% |
| 安全 | 425 | 53.13% |

score 分布：

| score | 条数 | 主要含义 |
| ---: | ---: | --- |
| 0.00 | 436 | unsafe 或无有效得分组件 |
| 0.05 | 57 | 只有最终回答 |
| 0.10 | 148 | 只用了必需表 |
| 0.15 | 135 | 必需表 + 最终回答 |
| 0.35 | 6 | SQL 证据 + 必需表，未答对/未收尾 |
| 0.65 | 1 | 最终答对 + 最终回答，证据/表不全 |
| 0.75 | 13 | 最终答对 + 必需表 + 最终回答，SQL 证据不匹配 |
| 1.00 | 4 | 全部严格成立 |

平均 score 为 `0.068`，中位数为 `0`。这说明训练的大多数比较信号不是“答案对错”，而是安全、表名和是否收尾。

## 3. 训练有没有学到东西

### 3.1 训练 rollout 有弱改善，但不是稳定正确能力

| 窗口 | 平均 reward | 最终答对 | SQL 证据正确 | 严格正确 | 安全 | 有最终回答 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step 1–10 | 0.0541 | 1.25% | 3.13% | 0% | 44.38% | 33.13% |
| step 11–20 | 0.0650 | 2.50% | 2.50% | 0% | 42.50% | 33.75% |
| step 21–30 | 0.0647 | 3.13% | 2.50% | 1.25% | 50.63% | 33.75% |
| step 31–40 | 0.0584 | 3.13% | 0% | 0% | 58.13% | 43.75% |
| step 41–50 | 0.0978 | 5.00% | 3.75% | 1.25% | 70.00% | 52.50% |

最后十步确实更安全、更经常给最终回答，平均 reward 也提高；但严格正确仍只有 2/160。可以说模型学到了一部分行为规范，不能说任务正确率已经改善。

### 3.2 validation 没有出现严格能力提升

| step | reward | strict acc | 最终答对 | SQL 证据 | 必需表 | 安全 | 最终回答 | 平均 turns |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.0800 | 0 | 0 | 0 | 90% | 75% | 35% | 48.6 |
| 20 | 0.0700 | 0 | 0 | 0 | 95% | 60% | 35% | 47.6 |
| 30 | 0.0850 | 0 | 0 | 0 | 95% | 75% | 35% | 49.9 |
| 40 | 0.0925 | 0 | 0 | 0 | 95% | 80% | 40% | 48.8 |
| 50 | 0.0925 | 0 | 0 | 0 | 90% | 80% | 40% | 48.4 |

reward 上升来自安全率和最终回答率，不来自答案或 SQL 正确。只有 5 个验证点，且中间逐条轨迹被文件覆盖，所以不绘制趋势图，避免对极小样本制造视觉确定性。

## 4. 根因一：instruction 与 hidden gold 大量不对齐

V2 builder 的旧门禁只做：数据 split 隔离、gold SQL 可执行、gold value 能在 SQL 结果中找到。它没有判断用户问题是否足够明确地要求该 SQL 结果。

对 200 条 manifest 的确定性复核触发器：

| 问题 | 条数 |
| --- | ---: |
| 广泛分析/原因/建议问题，却用唯一 numeric/table hidden target 打分 | 99 |
| instruction 写“最新/最近/本期”，SQL 没有时间字段、时间过滤或时间排序 | 161 |
| SQL 使用 `LIMIT` 但没有 `ORDER BY`，返回类别没有明确业务顺序 | 71 |
| 问原因/问题/建议，gold 只是一条 `COUNT(*)` | 23 |
| numeric gold 没有聚合，只取某列最大/首行 | 17 |
| 至少命中一项复核触发器 | 191/200（95.5%） |

代表性例子：

- “冷链仓储中心最近这期数据暴露出哪些问题，应该如何处理？”的 hidden gold 是 `SELECT COUNT(*) FROM fact_warehouse`。
- “请根据最新的温度数据记录，撰写一份分析报告。”的 hidden gold 是按发货人汇总湿度并 `LIMIT 3`，但没有 `ORDER BY`。
- “最新一期数据里，冷链线路出现了哪些异常或风险点？”的 hidden gold 是取 `avg_transit_hours` 最大的一行。

模型做更丰富的分析并不等于能猜到这些隐藏口径。这个问题会同时压低最终答案正确和 exact SQL evidence，并让 GRPO 把合理轨迹当作失败样本。

这些规则是复核触发器，不是对 191 条的最终人工判死；但 95.5% 的触发率已经足以阻止继续正式训练。

## 5. 根因二：system prompt 和工作区并未真正与老板一致

正式 V2 builder 统一使用项目 fallback system prompt；source manifest 没携带老板原始 system message。因此此前“完整 PI 四工具”只代表工具名和多轮框架接近，不能代表 system prompt、工具说明和运行条件完全一致。

旧 prompt 没告诉模型数据库固定在当前 workspace。首条训练样本就先查询不存在的 `logistics.db`，随后执行 `find /`，枚举出 `/pi_sandbox/dev/...` 的其他数据库，最后才发现当前目录 `logistics.sqlite`。

完整 800 条行为审计：

- 17,334 个 bash 调用；
- 375 条轨迹含至少一个 unsafe 命令，与线上 `safe=0` 完全一致；
- 1,009 次 host-path escape 命中，另有 7 次网络命令、1 次 destructive 命令；
- write/edit 实际使用为 0，说明虽然提供四工具，模型几乎只走 bash。

这既浪费工具轮次，又造成跨环境枚举风险。它是运行时隔离缺陷，不应仅靠 reward 事后惩罚。

## 6. 根因三：GRPO 的有效比较信号太稀疏

- 200 个完整 n=4 group 中，43 个（21.5%）四条 reward 完全相同，组内 advantage 无法区分好坏。
- 只有 15 个 group 至少出现一次最终答对，16 个至少出现一次 SQL 证据正确，4 个至少出现一次严格正确。
- 160 个唯一训练 prompt 只产生 200 个 group；每个 prompt 平均 1.25 次、最多 2 次 group exposure。
- 每条轨迹工具调用中位数和 p90 都是 26，接近工具/assistant 轮数边界；平均训练 `num_turns` 为 45.43，说明大量样本长时间探索但没有收敛到评分目标。

GRPO 需要同 prompt 多个 response 之间存在可学习的质量差。当前大多数差异发生在安全/收尾，而核心正确行为太少。

## 7. 根因四：更新幅度保守，但不是第一优先级

49 个完整 driver metric 行的平均值：

| 指标 | 平均 | 解释 |
| --- | ---: | --- |
| PPO KL | 0.000531 | policy 相对参考变化很小 |
| clipfrac | 0.001415 | 几乎没有样本触及 PPO clip |
| grad norm | 1.157 | 有梯度，不是训练断路 |
| actor update | 160.01s | 前反向/更新正常执行 |
| param sync | 8.96s | 权重同步不是主瓶颈 |

`1e-7` 很保守，后续可以对比 `3e-7`；但在数据 target 错位、核心 reward 极稀疏时，先提高学习率可能只是更快学会错误代理目标，因此不应现在直接改正式默认值。

## 8. 效率结果

50 个 fully-async 阶段记录：

| 阶段 | 平均 |
| --- | ---: |
| 完整 step | 655.08s |
| 等待完整 group | 486.55s |
| actor 更新 | 159.41s |
| 反序列化 | 0.017s |
| batch assemble | 0.014s |
| advantage 计算 | 0.0015s |

队列等待占完整 step 的 `74.27%`。说明 rollout 仍是长期供给瓶颈；但在训练目标不可信时继续做 Fastest-K 或更深 backlog 只会更快地产生低质量样本。当前优先级应从“吞吐优化”切换为“数据与奖励有效性”。

## 9. 本次已经修改的内容

### 9.1 validation 轨迹不再覆盖

问题：trainer 在 10/20/30/40/50 触发验证，但 rollouter 用自身数据计数命名文件，最终只留下 `177.jsonl` 和反复覆盖的 `200.jsonl`。

修复：

- trainer RPC 传 `current_param_version`；
- rollouter 在验证期间临时使用该值，结束后恢复原计数；
- 两台 Ray 启动程序在 worker 注册前幂等应用；
- 下一次验证应产生 `10/20/30/40/50.jsonl`。

### 9.2 禁止根目录枚举

安全契约新增 `find/ls/du/tree /` 根目录扫描拦截；`/workspace/...` 仍允许。它避免模型再次枚举其他沙箱，且不会削弱只读数据库查询。

### 9.3 system prompt 来源显式化

正式数据构造现在：

- manifest 有 `system_prompt` 时原样保留，并标记 `system_prompt_source=source`；
- 没有时才使用 fallback，并明确数据库固定为 `/workspace/logistics.sqlite`、禁止扫描其他环境、要求用尽量少的 SQL 后直接给出可见答案。

这不会把 fallback 冒充成老板原始 system prompt；V3 数据必须实际补齐 source system。

### 9.4 增加可复现审计

- 50-step 分析器验证文件数、行数、奖励公式、组件、group 方差、prompt exposure、工具行为和训练指标。
- instruction/gold 审计器给出语义复核触发器，并支持 `--fail-on-flags` 阻止不合格数据进入训练。

## 10. 本次刻意没有直接修改奖励权重

当前 V2 reward 实现与线上记录完全一致，0 verifier error、0 重放差异。它的问题是学习信号稀疏，且数据 hidden target 本身不可靠；直接改权重会把两个变量混在一起。

建议在 V3 对齐数据上比较两套 reward：

1. V2 原配方作为控制组。
2. V3 候选：最终答案 0.50、精确 SQL 0.20、查询结果包含 gold 证据 0.15、必需表 0.10、最终回答 0.05；unsafe/非法协议仍硬归零。

“查询结果包含 gold”必须限制行数、要求必需表且防止 `SELECT *` 套中答案，避免 reward hacking。没有 A/B 前不把该候选设成正式默认。

## 11. 下一轮推荐顺序

### 阶段 A：重建 formal PI V3

1. 从老板原始轨迹回填 source system prompt，不允许 silent fallback 进入正式集。
2. instruction 必须明确查询对象、指标、聚合方式、过滤条件、排序和输出行数；或者把 broad analysis 改为有结构化 rubric 的多目标 verifier。
3. 所有 `LIMIT` 必须有业务确定的 `ORDER BY`；所有“最新/最近”必须有可验证的时间条件。
4. 重新执行 gold SQL，并做 stratified 人工复核：numeric/table、三套环境、简单/复杂、多表 join 均覆盖。
5. `--fail-on-flags` 和人工签字都通过后再生成 train/val/test。

### 阶段 B：先做监督暖启动

从老板已成功 rollout 的轨迹构造清洗后的 SFT/behavior-cloning 集，只训练 assistant token，工具结果 mask；先让模型学会在真实 PI prompt 下定位 `/workspace`、少量查询并收尾。GRPO 不适合从 0.5% 严格正确率直接学习极稀疏 outcome。

### 阶段 C：小规模 GRPO A/B

- 32 个已人工确认的训练 prompt、固定 20 个 validation prompt；
- 每个 prompt 至少 4–8 次 group exposure，而不是 1–2 次；
- 对比 reward V2/V3 和学习率 `1e-7/3e-7`；
- 先跑 20 step，门禁看 strict acc、final correct、SQL evidence、zero-variance group、KL 和 unsafe rate；
- 只有 validation 核心指标出现可重复提升，才扩大到 100+ step。

### 阶段 D：最后恢复吞吐优化

在质量门禁通过后再比较 `4→4` 与 Fastest-K；必须同时报告有效正确样本/小时、strict acc、reward、超时与选择偏差，不能只看 step time。

## 12. 证据边界

- instruction/gold 的 191/200 是自动复核触发器，不等同于 191 条全部人工判错；但足以暂停训练。
- 训练 rollout 的最后十步 reward 上升是真实观测，但没有转化为 validation 的最终答案或 SQL 正确。
- 中间 validation 逐条样本被覆盖，无法恢复；只能使用 driver 中的五次聚合指标。
- 50-step checkpoint 证明工程流程稳定，不证明模型收敛或优于冻结基线。
