# Step 120 查询启动干预门禁

日期：2026-08-13

## 结论

在完整 25 回合仍未发起可识别只读查询的 41 个 Step 120 任务上，追加不含答案、表名、字段名、查询正文或任务字面量的通用查询启动约束，并只给 3 个助手回合和 3 次工具反馈。最终带真实工具结果的只读查询恢复为 `0/41`，另有 `2/41` 发出了可识别查询但结果未观测，远低于预注册的 `31/41` 通过线。

因此，“只在运行时补一句先查询”不足以修复当前失败；训练和 checkpoint 晋级继续关闭。下一目标应从笼统的 query initiation 收窄到非交互 SQLite 工作流、路径/CLI 去循环和 schema-to-query realization。

## 合同与运行有效性

- 任务：只选择同一 64 题完整 25 回合 Step 120 轨迹中的 `41` 个 `no_readonly_query` 任务。
- 干预：任务无关，不提供答案、相关表/字段、gold SQL、工具结果或任务字面量。
- 推理：Step 120 dist checkpoint，greedy n1，`3 assistant / 3 tool-result`，49,152-token 上下文。
- 权重：日志明确出现强制 actor→vLLM 权重同步标记。
- 产物：`exit 0`，`41/41` validation 行；forward-only、val-only，无 optimizer、无 checkpoint。

## 结果

| 指标 | 数量 |
| --- | ---: |
| 带真实结果的可识别只读查询 | 0 |
| 可识别只读查询但结果未观测 | 2 |
| 无可识别只读查询 | 39 |
| 任意 SQLite 命令覆盖 | 41 |
| schema discovery 覆盖 | 18 |
| 总工具调用 | 172 |
| Bash / read | 170 / 2 |
| 已观测 / 未观测工具调用 | 109 / 63 |
| 重复 Bash | 112 |

SQLite 命令族进一步分为：路径或 CLI-only `91`、`.tables/.databases/.dbinfo` 类 `13`、`.schema` 类 `15`、可识别 SELECT/WITH `2`。这排除了“首查询解析器漏掉大量 SELECT/WITH”的解释：主要预算确实消耗在数据库路径、CLI 启动、schema 探索和重复调用，而不是查询正文。

## 归因边界

该干预让 `41/41` 任务都进入 SQLite 命令族，说明模型会响应工具方向提示；但它没有在冻结的三回合预算内把探索转成带结果查询。因此更准确的归因是：

1. 不是完全拒绝工具或完全忽略指令；
2. 不是现有首查询解析器大量漏识别 SELECT/WITH；
3. 当前瓶颈是非交互 sqlite3 调用方式、路径/schema 探索去循环，以及 schema 后的查询实现；
4. 本门禁不证明更长干预预算或 task-specific schema oracle 也会失败。

## 下一门

下一项最高信息价值测试仍不训练：在同一 41 题上冻结一个更结构化、仍不提供 task-specific schema 或答案的三回合工作流——第一回合最多定位一次数据库，第二回合最多做一次 schema 检查，第三回合必须发出一条非交互 SELECT/WITH，并禁止交互 shell 与重复命令。

- 若带真实结果的只读查询达到 `31/41`：优先采用运行时工作流约束，再在完整 64 题与冻结 val20 上验证，不更新权重。
- 若仍低于 `31/41`：进入 task-specific schema grounding / action supervision 数据门禁；只有机械验证、不重叠的训练对达到至少 48 条，才允许初始化 optimizer。

