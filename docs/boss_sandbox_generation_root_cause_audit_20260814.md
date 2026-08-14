# 老板沙箱生成与 DWH 语义错位根因审计（2026-08-14）

## 结论

当前 DWH 训练数据的主要问题已经定位到任务生成器，而不是 SQLite 数据生成、服务器同步或 7 月 23 日的 instruction variant 统一操作。

生成器先根据 `task_type` 模板写 `natural_language_instruction`，随后才从真实 SQLite 数据反向构造 `gold_answer + verification_sql`。后一步允许在多种查询模式间回退，最后还能退化到 `SUM`、`AVG`、`COUNT(*)` 或 `SELECT * LIMIT 1`，但查询计划变化后不会重新生成或校正题面。最直接的错误路径是：`single_metric_query` 的题面只要求“查询/给我数据”，后端却固定优先生成 `SUM(...)` 作为唯一 gold。

现有 QA 不能阻止该问题：它验证字段存在、SQL 可执行和结果非空，却不验证题面所要求的操作、过滤、分组、时间口径和输出形态是否与 SQL 相同；领域配置还把失败策略设为 `on_fail=tag`，即失败样本默认仍保留。

因此，SQLite 环境和大部分 SQL 资产仍可复用，但旧的 `instruction—gold—SQL` 三元组不能未经修复直接用于 GRPO。

## 源码与产物流

实际生成链路为：

1. `step0_prd`：生成业务需求背景。
2. `step1_factor`：生成实体、状态和动作因子。
3. `step2_taxonomy`：生成任务分类与覆盖约束。
4. `step3.1_schema`：生成 DWH schema。
5. `step4.1_data`：按 seed 合成表数据，做外键/业务规则对齐，可选注入异常，同时写 JSONL 和 `database/*.sqlite`。
6. `step5.1_tasks`：先渲染自然语言模板，再执行 backward generation，从 SQLite 查询结果构造 hidden gold 和 verification SQL。
7. 可选扩展：task tree、adversarial mutation、selector、QA gate。
8. `Step 5b`：调用多个 LLM 生成 instruction rewrites，只要求“语义相同”，没有用 SQL AST 或结构化 query plan 逐条验收改写。
9. 老板服务器导入：Mac 上的 `sf_my_sandbox/output/<version>` 被整理到 `huawei_train/datasets/sandboxes/raw/<group>/<version>`，随后可统一 instruction variants、合并 hybrid gold。
10. 运行时沙箱：只复制 SQLite、documents 和字典到 runner，显式删除 task JSONL，避免向 agent 泄漏 hidden gold。

服务器上的 `huawei_train` 不是 Git 仓库，而且注释明确指出原始 629-version 生成产物位于老板 Mac 的 `/Users/renjunxiang/coding/sf_my_sandbox/output`。5 号机只保留导入后的资产和后处理代码；一份较早的生成器源码副本原已存在于 `/data/liulin/llin-rl-dpo-p2/native_sf_my_sandbox`。

## 代码级根因

### 1. 题面与查询计划不是同一个对象

`generate_tasks.py::_generate_answerable()` 的执行顺序是：

1. `_build_instruction(...)` 生成题面；
2. 创建 `gold_answer=None` 的 task；
3. `_try_backward_generation(...)` 再选择和执行 SQL；
4. 查询非空即接受。

题面并不保存一个可供后续 SQL 构造复用的结构化计划。因此，“指标、时间、过滤、聚合、分组、输出形态”只能靠两个独立函数碰巧一致。

### 2. `single_metric_query` 与 `SUM` 被直接混用

`single_metric_query` 模板主要是“查询某日指标”“给我数据”“调一下记录”等明细/单值表达；但 `_try_backward_generation()` 把它与 `aggregate_query` 放在同一分支，第一选择都是 `SELECT SUM(...)`。这正好复现当前人工审查里最常见的拒绝原因：题面未要求加总，但 gold 只接受加总值。

### 3. fallback 会改变任务语义

当首选查询失败时，生成器可能依次尝试：去掉日期条件、改成其他聚合、强制 JOIN、`SUM`、`AVG`、最大值、`COUNT(*)`、最终 `SELECT * LIMIT 1`。这些 fallback 只改 SQL/gold，不改已经生成的题面。一个数据库层“可回答”的任务因此可能在语义上已经变成另一道题。

### 4. 分析型题目只有窄数值 gold

`diagnostic_recommendation`、`attribution_analysis`、`cross_domain_analysis` 等题面要求原因、影响或建议，但生成器可以只用一个极值或分组汇总作为 gold。SQL 结果能支持某个数字，不代表完整回答了分析请求。

### 5. QA 只验证可执行性，不验证蕴含关系

`TaskValidator` 对 answerable 任务主要检查 gold 是否存在、SQL 是否能执行、结果是否非空。`QAQualityGate` 进一步检查标签、字段和 SQL 执行，但没有比较题面操作与 SQL AST；SQL 执行时甚至只调用 `execute`，没有逐字段复核 gold 与结果。配置的 `on_fail=tag` 还会保留失败任务。

### 6. LLM 改写没有语义闭环

改写器只把模型输出附加到 `instruction_variants`，清洗重点是格式，不是查询语义。后续 `unify_instruction_variants.py` 又会根据历史轨迹锁定一个改写版本。虽然本轮 18 个 mixed 的问题不是这一步造成的，但该设计仍可能为其他数据引入新的语义漂移。

## 数量证据

以下统计不包含题面、SQL、gold 值或 task ID。

| 范围 | 总数 | SQL 使用 SUM | 题面无明确聚合词但 SQL 使用 SUM | 其中 `single_metric_query + SUM` |
| --- | ---: | ---: | ---: | ---: |
| SFT 全量 DWH | 9,500 | 3,671 | 2,411 | 263 |
| 最终 rollout 281 池 | 281 | 234 | 97 | 87 |

`97/281 = 34.5%` 是基于显式聚合词的确定性风险标记，不等于完整人工审核；它证明筛选器此前的“零语义预警”规则漏掉了最核心的 operator mismatch。`87/113 = 77.0%` 的 final-pool `single_metric_query` 同时使用 SUM 且题面未明确要求聚合。

截至本次来源核验时，已有人审结的 18 个 mixed 中：

- 18/18 的 instruction hash 精确匹配老板 raw manifest 的 `natural_language_instruction`；
- 18/18 与 7 月 23 日 instruction 统一前的 `.bak` 原始题面相同；
- 3 个批准项都是 `aggregate_query + SUM + 明确聚合题面`；
- 15 个拒绝项中，14 个是 `single_metric_query + SUM` 且题面没有明确聚合词，另 1 个 aggregate 题存在日期/答案路由问题；
- 人工原因码累计包括 13 个 `aggregate_not_requested`、1 个日期口径问题和 1 个“影响分析未被 verifier 覆盖”。

这排除了“运行时选错 rewrite”作为当前主因，主因发生在原始 step5 任务生成阶段。

## 已复制的代码快照

按用户指定，相关代码已复制到 5 号机：

`/data3/llin/qwen3.6-27b-verl-grpo/source_snapshots/rjx_sandbox_pipeline_20260814`

快照内容：

- `generator_source/`：DWH schema/data/task 生成器、pipeline runtime 和必要领域生成器代码；
- `server_import_postprocess/`：老板 `huawei_train` 中的 data 导入、runner、manifest、variant、judge/reward 相关脚本；
- `SHA256SUMS`：快照内文件的源拷贝哈希清单。

快照共 263 个文件、约 2.75 MB；`SHA256SUMS` 自身 SHA-256 为 `134b1ff4f19c00e1b6fa7260e080dab676a9119922a32c65f27d97bec7341b08`。目录已去除组/其他用户写权限。

未复制数据库、task JSONL、轨迹、模型、运行输出、Git 元数据、个人配置或凭据。原生成器领域目录中的 `chat_api_config.json` 存在内嵌 API key，已明确排除；快照中长凭据字面量扫描为 0。

## 不浪费现有数据的修复方案

### A. 保留环境，重建任务三元组

19 个 SQLite 环境、schema、表关系和可执行 verification SQL 都是有价值的。不要删除整批数据；把旧 task 标为 `semantic_misaligned_legacy`，以新 task ID 生成修复版本，并记录 `repair_of`、旧/新 instruction hash、query-plan hash 和修复原因。

### B. 对明显错位任务分两种修复

1. 如果真实目标就是汇总：从 SQL AST 生成最小、明确的题面，写清指标、SUM、过滤、日期、分组和输出形态。
2. 如果题面真实目标是“给数据/记录”：保留题面，重做 SQL/gold 为与该请求一致的表格或明细答案，不能擅自增加 SUM。

不能一律给题面补“合计”来迁就旧 gold；`single_metric_query` 的原始任务类型说明不少任务更可能应该修 verifier，而不是修措辞。

### C. 改为 SQL-plan-first 原子生成

新生成器应先产生并冻结结构化 `QueryPlan`：

`metric + aggregation + filters + time_field/range + group_by + ordering/limit + output_shape`

随后：

1. 从 QueryPlan 编译 verification SQL；
2. 在只读 SQLite 中执行并得到 gold；
3. 从同一个 QueryPlan 渲染 canonical instruction；
4. 用 SQL AST 反解析结果与 QueryPlan 逐字段比对；
5. 任一不一致就丢弃本次生成并重新采样，禁止改变语义的 fallback。

### D. 把语义门改成 fail closed

- QA 默认 `on_fail=filter`；未知类别也拒绝。
- 对每条任务机械核对 instruction operator 与 SQL AST：SUM/AVG/COUNT/MAX/MIN、group by、日期字段、过滤值、排序/limit、answer shape 必须一致。
- 分析/建议/归因类任务不能只靠 scalar/table SQL verifier 进入 GRPO；要么配套覆盖完整请求的 rubric/judge，要么只进入 SFT/reference。
- 每个 LLM rewrite 都重新过同一 QueryPlan entailment gate，不合格 variant 单独删除。

### E. 修复后必须 fresh rollout

题面或 verifier 任一改变，旧轨迹和旧 mixed 分桶都失效。修复任务通过 CPU 语义门后，应重新做 `n=8` rollout；不能把旧轨迹换标签后直接训练。

## 建议执行顺序

1. 立即把最终 281 池中的 97 个明确 operator mismatch 隔离，其他 184 个仍保持待审，不能自动视为安全。
2. 先修 32 条：优先 16 条改题面、16 条改 verifier，比较哪种修复保留业务意图和 mixed 产率更好。
3. 逐条通过 QueryPlan、SQL 执行、gold 支持和人工语义四门后，做 fresh `32×8` rollout。
4. 修复产率稳定后再批量处理 9,500 池；数据库与 schema 不重造，只重建 task layer。
5. 同步修改生成器，避免继续生产同类坏样本；在新代码通过合成反例测试前不再扩充旧模板数据。
