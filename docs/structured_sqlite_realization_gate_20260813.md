# Step 120 结构化 SQLite realization 门禁

日期：2026-08-13

## 结论

在同一 41 个 Step 120 完整预算无查询任务上，把通用“先查询”提示替换为严格的三回合非交互工作流：第一回合最多定位一次数据库，第二回合最多检查一次 schema，第三回合必须执行 SELECT/WITH，并禁止交互 shell 与重复命令。结果仍为 `0/41` 条可识别只读查询，低于预注册 `31/41`。

结构化约束减少了总工具调用和重复命令，但没有把路径/schema 探索转成查询正文。因此不能再把下一步押在更强的通用运行时提示；下一门应提供机械提取、task-specific 但不含答案或 gold SQL 的 schema oracle，测试 schema grounding 是否足以触发查询，并采集同状态 rejected actions。

## 有效性

- 同一 41 题、同一 hidden verifier、同一 Step 120 dist checkpoint；只替换干预文本。
- greedy n1，`3 assistant / 3 tool-result`，49,152-token 上下文。
- actor→vLLM 强制同步标记存在，`exit 0`，`41/41` validation 行。
- forward-only、val-only，无 optimizer、无 checkpoint。

## 与通用查询启动干预比较

| 指标 | 通用提示 | 结构化工作流 |
| --- | ---: | ---: |
| 带结果可识别只读查询 | 0 | 0 |
| 未观测可识别只读查询 | 2 | 0 |
| 无可识别只读查询 | 39 | 41 |
| 总工具调用 | 172 | 132 |
| 任意 SQLite 覆盖 | 41 | 41 |
| schema discovery 覆盖 | 18 | 15 |
| 路径/CLI-only 调用 | 91 | 81 |
| schema catalog / definition | 13 / 15 | 15 / 7 |
| 重复 Bash | 112 | 107 |
| 未观测工具调用 | 63 | 48 |

结构化提示确实压缩了部分探索预算，但 `41/41` 仍未形成 SELECT/WITH。首查询机械分类和命令族细分一致，不是工具结果等价判定造成的假失败。

## 下一门

在完整 64 个不重叠 current-definition 严格任务上构造单回合 schema oracle：

1. 只提供从只读 SQLite 机械提取的相关表结构，不提供 expected value、gold SQL、工具结果或答案；
2. 提供任务无关的动态数据库定位 + 非交互 sqlite3 命令模板，模型只需补全一条 SELECT/WITH；
3. 用真实工具结果机械分类首查询；
4. 若至少 32/64 首查询正确或等价，先验证 runtime schema injection，不训练；
5. 否则只有至少 48 个不同任务产生带结果的错误查询，才允许构造同状态 correct-vs-actual-wrong pairs；数量不足继续关闭 optimizer。
