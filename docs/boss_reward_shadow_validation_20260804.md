# 老板 KB/DWH 评测逻辑复用与影子奖励验证

日期：2026-08-04
状态：影子验证完成，未切换正式训练奖励

## 1. 目标与安全边界

本次工作评估老板的 KB/DWH 评测框架能否直接作为 veRL 在线 GRPO 奖励，并按以下边界实施第一阶段：

1. 只移植确定性、可审计的判定内核，不直接导入带全局路径和可选外部 LLM 的脚本。
2. 新奖励保持 `shadow_only`，现有训练入口继续使用 `llin_verl/pi_reward.py`。
3. DWH 只有 gold SQL 与 gold value 自洽的 numeric/table 任务可以成为在线候选。
4. KB 在没有经过校准的语义评测器前一律不能成为在线主奖励。
5. 历史回放只按唯一 `task_id` 连接，不允许按 instruction 模糊匹配或重复轮转。
6. 原始轨迹、数据库、评测输出和文档内容不进入 Git。

老板侧审计源码：

- `/data/renjunxiang/coding/huawei_train/scripts/data/judge_trajectory.py`
- `/data/renjunxiang/coding/huawei_train/scripts/data/judge_trajectory_openai.py`
- `/data/renjunxiang/coding/huawei_train/scripts/data/reward_judge.py`

## 2. 老板框架中可复用与不可复用的部分

### 2.1 DWH

可复用判定内核：

- 执行 `verification_sql` 得到标准结果；
- 提取并执行 Agent 的所有 SELECT；
- 任意 Agent SQL 结果与标准结果相等即视为 SQL 证据命中；
- 检查最终答案数值、必需表和完整性。

不能直接复用的部分：

- numeric/table 在 Agent SQL 未命中时仍可仅凭最终答案包含 gold 数字判 correct；
- 报告型只检查完成、超过 200 字和表命中，默认没有语义正确性；
- OpenAI 入口与 raw 入口的执行安全、解析细节并不完全一致；
- 总奖励允许错误答案凭过程和效率获得过高分数。

### 2.2 KB

可复用子信号：

- 数字容差匹配；
- 来源文档列表；
- 完整性、工具类型和效率统计；
- 文档名、版本、状态等文本锚点只能作为弱信号。

不能直接复用的部分：

- `judge_kb` 不要求模型实际访问 `source_documents`；
- `reward_judge.score_docs_hit` 允许答案文本自称 `doc_003`，不能证明读取；
- 100 条 unanswerable 的 `gold_answer=None` 会让无数字检查天然通过；
- raw 与 OpenAI 两个 KB 入口存在关键词和 `answer_ok/tables_ok` 定义漂移；
- 超过 50 字不能代表 KB 答案正确。

## 3. 新影子奖励设计

实现文件：`llin_verl/boss_reward_shadow.py`。

### 3.1 DWH 候选奖励

先执行硬门禁：

- task 必须是 DWH numeric/table；
- `verification_sql` 必须存在、只读可执行；
- SQL 结果必须支持 gold value；
- PI 工具协议有效；
- bash 命令通过安全检查。

分数采用结果门控：

| 条件 | 分数 |
| --- | ---: |
| 答案、SQL、必需表、成功 bash、安全协议全部正确 | 1.00 |
| 答案和 SQL 正确，但过程要求不完整 | 0.85 |
| SQL 正确、最终答案错误 | 0.20 |
| 只有最终答案碰到 gold | 0.15 |
| 只有安全的终态答案 | 0.05 |
| gold 不自洽、协议无效或不安全 | 0.00 |

因此“在长答案里罗列大量数字碰撞 gold”最多得到 0.15，不再获得旧奖励的 0.75 或老板 verdict 的 correct。

### 3.2 KB 影子信号

KB 只记录以下确定性组件：

- 成功 `read` 或 `cat/grep/sed/awk/head/tail` 内容读取事件是否访问全部 `source_documents`；仅 `find/ls` 发现文件名不算读取；
- gold 数字是否出现；
- gold 文档名/版本/状态锚点是否出现；
- unanswerable 是否明确拒答；
- 最终答案、协议和安全状态。

答案文本里单独写 `doc_003` 不算文档访问。KB 分数上限固定为 0.25，`acc` 和 `online_eligible` 永远为 0，并输出 `requires_semantic_judge=1`。

## 4. 历史回放数据

任务清单：

- `boss_v15_tasks.jsonl`：1,500 条，DWH/KB/Hybrid 各 500 条。

轨迹：

- v0.24 从老板原始 Qwen3.6 v15 事件文件导出的 `qwen36_v15_original_with_task_id.jsonl`；
- 1,500 条、1,500 个唯一 task_id。

离线对照：

- 老板 `qwen3.6-27B_v15_fixed.jsonl` verdict。

第一次误用了旧 OpenAI converted 文件，门禁立即发现其 1,500 行只有 1,499 个唯一 task_id，并因重复 task_id 停止，没有产生评分。最终回放只使用 task_id 唯一的同源文件。

## 5. 1000 条 KB/DWH 影子回放结果

回放覆盖 500 条 DWH 和 500 条 KB；Hybrid 不在本轮范围。

### 5.1 DWH

| 指标 | 数量 |
| --- | ---: |
| DWH 总任务 | 500 |
| numeric/table 且含 verifier | 280 |
| gold SQL 与 gold value 自洽 | 277 |
| gold 不一致、拒绝在线奖励 | 3 |
| 报告/非结构化、需要语义评测 | 220 |
| 最终答案数字正确（277 内） | 31 |
| Agent SQL 结果正确（277 内） | 14 |
| 严格全部正确 | 6 |

候选分数分布：

- 0.00：281
- 0.05：180
- 0.15：25
- 0.20：8
- 1.00：6

老板在 500 条 DWH 中判 correct 47 条，但只有 1 条同时被候选严格规则判 correct。另有 5 条候选严格正确却被老板判 partial/incorrect，说明解析入口之间仍有差异；这 5 条必须人工复核后才能讨论正式切换。

### 5.2 KB

| 指标 | 数量 |
| --- | ---: |
| KB 总任务 | 500 |
| answerable | 400 |
| unanswerable | 100 |
| 历史轨迹真实访问全部来源文档 | 400 |
| gold 数字匹配 | 88 |
| gold 文本锚点匹配 | 337 |
| 检测到拒答 | 37 |

老板把 100 条 unanswerable 中的 99 条判为 correct；其中 89 条没有检测到任何拒答表达。这不能证明 89 条必然错误，但证明当前规则没有区分“正确拒答”和“完整长回答”的能力。

KB 候选分数分布：

- 0.00：21
- 0.05：70
- 0.10：32
- 0.15：169
- 0.25：208

所有 KB 仍为 shadow-only，没有任何一条获得在线 eligibility 或 strict acc。

## 6. 回放中修复的工程问题

### 6.1 `/workspace/` 被误判为宿主根目录

旧 `command_is_safe` 直接删除字符串 `/workspace`：

```text
ls /workspace/  ->  ls /
```

随后根目录扫描规则会把合法工作区命令判为不安全。现在改为将 `/workspace` 映射成安全的相对标记，`ls/find /workspace` 被允许，而真实 `/`、`/etc`、`/data3` 等路径仍被禁止。

### 6.2 旧 converted 文件重复 task_id

base 和 table 两套旧 OpenAI converted 文件各有一个重复 task_id。影子回放器对 manifest、trajectory、boss verdict 都执行唯一 task_id 门禁，禁止静默覆盖或轮转连接。

### 6.3 沙箱路径漂移

老板评测脚本默认路径与当前数据部署路径不同。最终明确使用只读 raw 数据库根目录：

`datasets/sandboxes/raw/sft/20260628_v15/logistics.sqlite`

raw 与 runner 数据库 SHA256 一致。路径错误的第一次回放将全部 DWH 标为 verifier error，没有误报 eligibility。

## 7. 正式接入前的剩余门槛

1. 人工复核 6 条候选 strict DWH，尤其是与老板 verdict 不一致的 5 条。
2. 从 DWH 的 0.15/0.20/1.00 各抽样，确认 SQL 解析、行顺序和最终答案语义没有误判。
3. 为 KB 按五个 subtype 各抽样建立人工标签，校准语义 judge；不可回答任务必须单独标注正确拒答与编造。
4. 语义 judge 必须固定模型、prompt、版本和缓存，先用于离线筛选；没有稳定性证据前不进入在线 rollout 热路径。
5. 通过上述门槛后，先在 20-step 运行中同时记录旧 reward 和候选 reward，不改变优化目标，再决定是否把 DWH V3 切成正式奖励。

## 8. 当前结论

- DWH：277 条可以进入严格候选池，但正式替换前仍需差异样本人工复核。
- KB：老板规则不能直接作为在线奖励；当前只完成可审计子信号和影子回放。
- 正式训练脚本没有切换 reward，当前没有运行训练，也没有占用 NPU。
- 本地全项目回归测试为 `92 passed`。
