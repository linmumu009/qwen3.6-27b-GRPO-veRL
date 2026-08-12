# Step 120 task-specific schema-oracle 首动作门禁

日期：2026-08-13
结论：两个预注册门禁均失败；不做 runtime schema 晋级，不构造 pair，不启动 optimizer。

## 为什么做这个实验

原生模型与 Step 120 的完整 64 题公平对照已经证明，高过程分错答在原生模型中就存在，训练没有制造或放大该类代理错配；Step 120 新暴露的主要代价是 SQL 覆盖和完整收尾下降。随后两轮通用 prompt 干预也都失败：查询启动约束没有带来带结果查询，结构化 `path→schema→SELECT/WITH` 工作流仍为 `0/41`。

本轮给模型一个有意偏强的诊断上界：对完整 64 个与冻结 16、val20、test20 不重叠的严格任务，只注入隐藏验证 SQL 所涉及表的 SQLite metadata，再要求第一动作直接执行非交互只读查询。它回答两个问题：

1. 已知相关表及字段时，Step 120 能否把自然语言任务转成正确查询？
2. 若仍不正确，能否至少稳定产生足够多带真实结果的错误查询，用于后续 correct-vs-actual-wrong pair 构建？

这里的相关表选择使用隐藏 gold，只能作为能力上界，不能视为可部署输入，也不能用于晋级声明。

## 数据与防泄漏

- 64/64 唯一任务；54 个 numeric、10 个 table。
- 每题 gold SQL 的 `FROM/JOIN` 都只涉及一张表；注入 8–25 个字段的名称、类型、not-null、主键位置及选中表间外键。
- schema 从 immutable/read-only SQLite `table_info` 与 `foreign_key_list` 提取。
- prompt 不含数据库行、工具结果、expected value、答案或 gold SQL；原 hidden verifier 保持不变。
- 只允许 Step 120、greedy `n=1`、2 个助手回合和 1 次工具反馈；第一回合查询执行后，第二回合禁止再次调用工具。
- 训练、pair 构建和 promotion 在数据合同中默认关闭。

## 运行完整性

有效运行 `llin-schema-oracle-action-step120-20260813-02`：

- 强制出现 checkpoint→rollout 权重同步标记，排除误用原生模型初始态。
- `exit 0`，验证文件 64/64 行，耗时 `477.15s`。
- 纯前向；没有 optimizer、backward 或新 checkpoint。
- 共解析 127 次 bash 调用：64 次第一动作都有真实工具响应，第二回合仍有 63 次违反约束的终端调用但未执行，不能进入首查询证据。
- 一条第一动作的 Qwen tool block 发生 token 截断；在线 runtime 接受后返回真实 parser error。转换器原样保留调用与响应并单独审计，没有补造结果。

首个 `-01` 运行把 `max_assistant_turns` 误设为 1，虽然 64/64 都生成工具调用，却在工具执行前停止，导致 64 条第一响应缺失。该运行已判无效，不进入任何门禁统计；合同与适配器随后均增加回归测试。

## 结果

| 首查询结果 | 条数 | 比例 |
| --- | ---: | ---: |
| 正确或结果等价 | 4 | 6.25% |
| 有真实结果但错误 | 35 | 54.69% |
| 未产生可识别只读查询 | 25 | 39.06% |

35 条错误首查询进一步分为：

- 23 条可执行但证据错误或不足；
- 10 条可执行但结果为空；
- 2 条 schema、语法或执行错误。

工具侧 64/64 都调用 bash/SQLite，但只有 39/64 的第一动作包含可识别只读查询。整段轨迹包含 40 个重复 bash 调用；63/64 在拿到第一次工具结果后仍尝试第二次工具调用，说明动作停止控制也没有服从冻结约束。

## 预注册门禁

| 门禁 | 观测 | 要求 | 结果 |
| --- | ---: | ---: | --- |
| 正确/等价首查询，开放无 gold 表选择的 runtime schema 验证 | 4 | 32 | 失败 |
| 带结果错误首查询，开放 pair 构建与 CPU 审计 | 35 | 48 | 失败 |

因此：

- 不把 schema 注入当成 runtime 修复；
- 不从 35 条样本选择性凑 pair，避免任务和失败类型选择偏差；
- 不启动 optimizer，不保存候选 checkpoint，不做 promotion。

## 解释与下一步

相关表 schema 确实把“完全不查询”部分转成了可执行 SQL，但它没有解决语义规划到 SQL 的映射：39 条真实首查询只有 4 条正确。与此同时，25 条在有相关表、字段和强制单动作要求时仍不发查询，63 条在获得结果后仍继续调用工具。这说明剩余瓶颈至少同时包含首动作控制和 SQL 语义实现，而不是单纯缺少 schema。

下一步转为 chosen-only schema-conditioned first-action supervision：为完整 64 题生成唯一、机械可执行且结果与 verifier 一致的正确 bash/SQLite 首动作，只监督该助手动作，不加入工具结果和最终答案。先完成 CPU 防泄漏、SQL 可执行/等价、tokenization、assistant-only loss mask 和 48/16 内部分割门禁；全部通过后再做 Step 120 teacher-forced 基线，仍不直接训练。

该阶段继续把 gold 选表 schema 标记为诊断输入。数据库全库包含 54 张表、636 个字段，不能把“注入全部 schema”当作一个无成本修复；若 chosen-only 动作信号可学，后续还需单独验证不依赖 gold 的 schema selector 或受控检索器。
