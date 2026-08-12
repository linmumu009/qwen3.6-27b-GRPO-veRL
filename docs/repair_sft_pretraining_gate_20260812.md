# 纠错 SFT 训练前门禁（2026-08-12）

## 结论

现有两份 16 题自由回放的第一条 SQL 均没有产生可验证的正确证据，也没有任何一条与教师 SQL 的执行结果机械等价。NPU 上的 exact token-rank 门禁进一步确认：通用 SFT 虽把教师 SQL token 的 greedy 率从 `50.46%` 提高到 `70.21%`，但两份模型仍都是 `0/16` 题整段教师 SQL 全部 greedy。因此，当前瓶颈不是“SQL 字符串不同但结果等价”，而是首条证据获取本身失败。

下一次训练固定为从 Step 120 开始的一步单变量 SQL 加权金丝雀：工具结构权重 `0.25`、SQL payload 权重 `8.0`、最终答案权重 `1.0`。第一步不加入模型自身错误状态纠正样本，避免同时改变两个变量。

本次只在 NPU 上启动了两个 forward-only 模型任务；两者都没有初始化 optimizer、没有训练，也没有保存 checkpoint。

## CPU 首条查询语义门禁

| 模型 | gold 支持 | 教师结果机械等价 | 错误/不足证据 | 空结果 | 执行错误 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step 120 | 0/16 | 0/16 | 13 | 3 | 0 |
| 通用 SFT Step 5 | 0/16 | 0/16 | 13 | 2 | 1 |

门禁使用只读、immutable SQLite 连接，只接受 `SELECT/WITH`，同时限制单条查询的执行时间与最大返回行数。输出只保留 task id、查询哈希、分类和聚合计数，不保存原始 SQL。

## NPU exact token-rank 门禁

| 模型 | 教师 SQL token | greedy | top-5 | 平均 rank | 最大 rank | 整段全 greedy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Step 120 | 329 | 166（50.46%） | 254（77.20%） | 56.59 | 7547 | 0/16 |
| 通用 SFT Step 5 | 329 | 231（70.21%） | 302（91.79%） | 17.53 | 2738 | 0/16 |

两份运行的数据哈希和 task id 完全一致；核心前向合计 `85.38s`，含两次模型加载和分布式释放的端到端墙钟约 `253.3s`。三次工程失败分别在嵌套词表配置、BSHD 统一填充和 veRL 元数据 API 契约处 fail closed，均未写 checkpoint；修复后的隔离运行成功退出。

## 已完成的工程准备

- 新增首条 SQL 只读执行、gold 支持和结果等价分类器。
- teacher-forced 纯前向诊断现可在 TP4 下跨词表分片计算精确目标 token rank，并记录首个非 greedy SQL token 的相对位置、rank、token id 与概率。
- 新增 SQL 加权数据集、CPU mask 门禁和一步金丝雀启动器；仍使用官方 veRL SFT loss，通过加权 `loss_mask` 改变组件质量占比。
- 16 行 SQL 加权 mask CPU 门禁已经通过：组件 mask 全部非空、权重逐行精确匹配，SQL 占加权 loss mass 的均值为 `78.05%`。
- 金丝雀只保存最终 `model,extra`，不保存 optimizer；仍为 train236 同题开发门禁，不允许作 held-out 提升声明。

## 后续执行顺序

1. 已完成 Step 120 与通用 SFT Step 5 的 forward-only token-rank 门禁，不初始化 optimizer、不保存 checkpoint。
2. 已完成 16 行 SQL 加权 mask CPU 门禁，产物与当前代码版本一致。
3. 从 Step 120 启动一步 SQL-only 金丝雀。
4. 训练后先比较 SQL rank；rank 不动则立即停止。rank 移动后再做首条查询语义回放，仍不直接扩到完整老板回放。

安全聚合数据见 [`repair_sft_pretraining_gate_20260812_summary.json`](repair_sft_pretraining_gate_20260812_summary.json)。
