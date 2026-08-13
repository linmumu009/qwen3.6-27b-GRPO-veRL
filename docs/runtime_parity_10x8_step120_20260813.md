# Step 120：PI-Agent 与 veRL rollout 运行时对照

日期：2026-08-13

## 结论

本次 `10 题 × 8 条 × 2 个运行时` 工程门禁未通过，当前不能把 PI-Agent 与 veRL rollout 视为等价执行环境，也不能据此启动 group 筛选或训练。

两臂的纯最终结果正确率都是 `0/80`，10 个题组也都是全错。这只能说明两边在本次困难样本上同时触底，不能证明分布一致；与此同时，最终回答完成率相差 `16.25` 个百分点，PI 首轮还有 `12/80` 条超过 30 分钟，运行时行为存在实质差异。

2026-08-13 的追加审计确认：**本次 temperature 不是 0，而是两臂均为 1.0；但“其他配置全部一样”不成立。**因此本实验的有效解释是“两个现网执行路径不兼容”，不能解释成“两个严格相同配置只因 PI/veRL 框架不同而分叉”。

## Temperature 实际为 1.0，不是 0

veRL 配置中确实存在 `val_kwargs.temperature=0`，但它是普通确定性验证的默认值；GRPO 训练 rollout 的独立字段 `actor_rollout_ref.rollout.temperature` 默认是 `1.0`。

本次专用 parity 入口没有使用普通验证默认值：

- standalone batch 明确设置 `validate=true`、`do_sample=true`；
- 安装中的 `AgentLoopManager.generate_sequences` 在 `validate=true` 时读取 `val_kwargs`；
- 本次覆盖值与最终运行合同均为 `temperature=1.0 / top_p=0.95 / top_k=20`；
- veRL driver 日志也记录了 temperature 1.0，未出现本次请求被改为 0 的证据。

PI 臂的 runner 和 PI 客户端没有显式传 temperature/top-p/top-k。PI 的 OpenAI-compatible request builder 只在 option 有值时才发送 temperature；本次最终由 Step 120 vLLM 服务加载的 `generation_config.json` 提供 `temperature=1.0 / top_p=0.95 / top_k=20 / do_sample=true`。因此两臂的有效采样参数一致。

## 80 条是怎样生成的

两臂都不是把一条结果复制 8 次：

- PI-Agent：10 个题目各展开为 8 个独立 sample key；每个 sample 启动一个独立 native PI session，最终保留 80 个位置各一条结果。首轮并发上限为 32，超时恢复只补未完成位置，不增加第九条候选。
- veRL：先加载 10 行数据，再用 `repeat_times=8, interleave=true` 扩成 80 行；AgentLoopManager 为每一行创建独立逻辑 request ID 和完整 agent trajectory。这里的 80 来自显式 batch repeat，不是一次 HTTP 请求的 `n=8`。
- 两臂均关闭 Fastest-K，80 条全部保留。

服务器内对完整 assistant 轨迹做精确去重后，两臂的 10 个题组全部都是 `8/8` 条互不相同：共 20 个题组中没有一个“8条完全相同”，组内两两重复比例均为 0。这个结果排除了复制候选；temperature 的最终判定仍以来自实际调用链和日志的证据为准，而不是仅凭输出不同反推。

## 对齐的是核心输入，不是全部运行时配置

- 模型：相同 Step 120 权重，跨机导出文件校验一致。
- 数据：冻结 val20 中确定性分层选取 5 条 numeric、5 条 table；仅用于评测，禁止进入训练。
- 采样：每题每臂 8 条，`temperature=1.0`、`top_p=0.95`、`top_k=20`，最大上下文 49,152 tokens。
- 工具合同：四工具 schema、Qwen3 Coder parser 和沙箱任务资产对齐；不启用 Fastest-K，全部候选均保留。
- 评分：仅重新计算最终结果正确性；不使用过程分、SQL 形态、工具调用数量或旧 reward。
- 资源拓扑：两臂各用单机 16 张 NPU、TP8×DP2、max batched tokens 16,384 和 max sequences 24；实验结束后两机活动 vLLM worker 均为 0。

以下配置或运行策略没有对齐：

| 项目 | PI-Agent | veRL rollout |
| --- | --- | --- |
| 单次 assistant 生成上限 | 每次最多 8,192 tokens | 动态使用剩余上下文，最高可到45,056响应预算 |
| 上下文管理 | PI 原生压缩并重试 | continuous-token 累积，不是同一压缩器 |
| 终止条件 | native PI loop + 外部单轨迹墙钟超时 | 明确26个assistant/25个user回合，无相同单轨迹墙钟超时 |
| 并发 | 首轮32个PI客户端 | 16个agent workers |
| 工具实现 | native PI sandbox tools | 项目 veRL tool wrappers；schema相同但实现不同 |
| vLLM显存比例 | 0.85 | 0.80 |
| 随机数 | 独立随机请求 | 独立随机请求；未与PI逐条绑定相同seed |

这些差异正是部署路径的一部分，所以仍可以用来回答“实际 PI 与训练 rollout 是否可互换”；但它们阻止我们把差异归因到某一个框架变量。

## 安全聚合结果

| 指标 | PI-Agent | veRL rollout | 判定 |
| --- | ---: | ---: | --- |
| 轨迹数 | 80 | 80 | 结构通过 |
| 题组数 | 10 | 10 | 结构通过 |
| 最终结果正确 | 0/80 | 0/80 | 同时触底，信息不足 |
| 有最终回答 | 69/80（86.25%） | 56/80（70.00%） | 差 16.25pp，失败 |
| 最终超时 | 1/80 | 0/80 | 零容忍门失败 |
| 终止运行错误 | 17/80 | 0/80 | 零容忍门失败 |
| 全错题组 | 10/10 | 10/10 | 仅观察，不开放筛选 |

PI 第一轮给全部 80 条各 30 分钟上限时，68 条在期限内结束、12 条超时；恢复执行不能抹去这 12 个首次超时。剩余长尾最终仍有 1 条达到 60 分钟上限。

准确率绝对差、逐题平均准确率差和 Bernoulli JS 均为 0，只是因为两臂都没有正确样本。完成率门、超时门和运行错误门失败，故总门禁为失败。

## 对 group 筛选的影响

本次观察到 10 个全错组，但可执行的筛选数量为 0：

1. 运行时 parity 没有通过，不能把 veRL 的 bucket 当作 PI 实际分布的可靠替代。
2. 这 10 题来自 evaluation-only 的 val20，本身禁止用于 GRPO 更新。
3. 这些题尚未通过后续更严格的题意—gold—SQL 语义审核。

即使未来 parity 通过，也只能在另一份严格语义批准的训练池上重新生成 fresh groups：mixed groups 可进入当次更新；全对和全错只从当次 optimizer update 排除，不永久删除，不挑选单条轨迹。

## 与纯最终结果离线重放方案的关系

纯最终结果 scorer 与离线重算已经在本实验中投入使用，成功避免过程奖励掩盖错误答案。但本批 160 条结果全部错误，无法提供 mixed-group 训练信号。继续在这 10 题上离线重放不会改变训练决策。

## 下一步最高价值动作

保持训练关闭，先把下一轮目标明确为“严格配置归因”，修复最影响可比性的三项：veRL 每次 assistant 生成也上限 8,192 tokens、两侧使用相同单轨迹墙钟、报告分别记录 compaction 与终止原因。然后只选 3 个代表题、每题 2 条做小复验，不立即重跑完整 `10×8`。

现有敏感轨迹仍可在服务器内、无 NPU 地比较以下节点：

- 最后一次有效工具结果后是否继续查询；
- 是否进入最终回答，以及终止原因；
- PI 上下文压缩/重试与 veRL 49K 截断的对应关系；
- 同题两臂 assistant 回合和工具调用数量分布。

三个代表题分别来自“两边都完成、仅 PI 完成、两边都未完成”。小复验先证明温度、单次 token cap、墙钟和终止统计都在实际请求层生效；通过后才重复完整 parity。只有完整门通过，才把相同筛选方法应用到独立、严格语义批准的训练池。

机器可读的无任务标识结果见 [`runtime_parity_10x8_step120_20260813_summary.json`](runtime_parity_10x8_step120_20260813_summary.json)；采样与配置追加审计见 [`runtime_parity_sampling_config_audit_20260813_summary.json`](runtime_parity_sampling_config_audit_20260813_summary.json)。原始 prompt、gold、SQL、轨迹、任务标识和逐题比较均只保留在服务器权限受限目录中。
