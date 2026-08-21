# qwen3.6-27b-GRPO-veRL

Qwen3.6 27B 的 GRPO / veRL 训练项目。

## 当前方案

- 5 号机负责 Megatron 全参训练，拓扑为 TP=4、PP=2、CP=2，Ray 自定义资源名为 `llin_trainer`。
- 6 号机负责 vLLM 异步轨迹推理，拓扑为 TP=8、DP=2，Ray 自定义资源名为 `llin_rollout`。
- 两台机器通过内网 Ray 集群通信；训练权重使用 veRL 的 `nccl` 检查点后端，在昇腾环境中实际注册为 HCCL 广播。
- HCCL 固定使用 `eno0` 和 `192.168.202.0/24` 内网，并为 host/NPU socket 分配互不重叠的端口范围，避免自动选中不可达的管理网卡。
- actor 模型参数常驻 NPU，不做参数卸载；Adam 优化器与梯度卸载到 5 号机内存，并开启全量激活重计算。LoRA 已关闭（`lora_rank=0`）。
- rollout 开启 Continuous Token、prefix caching 和两路 vLLM cache 计数；新的正式 100-step/12-group 入口在第 1–99 步均不验证、不保存，只在第 100 步验证一次并保存完整 `model,optimizer,extra`，确保后续可恢复 Adam 动量、方差和学习率调度器状态。
- 长上下文配置将最大上下文设为 `49,152` tokens（默认初始 prompt `4,096` + 多轮 response `45,056`）；正式 PI 配置将允许最多 25 次工具反馈和随后 1 次最终回答，单轮最多 4 个并行工具调用，单次工具返回放宽到 `32,768` 字符；100-step/12-group 入口将 vLLM cache 预算设为 `0.80`、chunked prefill batch 设为 `16,384` tokens、每个 TP8 副本最多 24 个活跃序列，并用可容纳最大 embedding 张量的 `2560 MiB` 权重同步 bucket。
- 新正式数据入口改为 `boss-pi-aligned-grpo-v1`：system 与四工具 schema 直接冻结自老板 `pi_to_openai.py` 并校验 SHA256，不再存在项目 fallback；旧 formal V2 已被正式启动器硬拒绝。
- 20-step One-Step-Off-Policy 的长尾切换判据已触发；后续推荐使用 bounded fully-async：一个 prompt 的 4 条 GRPO 轨迹作为不可拆分 group。正式 100-step 配置每次更新消费 4 groups，queued-token 上限为 `1,572,864` tokens（等价于 8 个全部跑满 48K 的 groups），以 `staleness=2` 将在途上限控制为 12 groups，满载时背压而不是丢弃旧样本。
- bounded fully-async 已支持 Fastest-K 过量采样：默认物理生成 6 条候选、最先完成的 4 条组成完整 GRPO group，剩余候选取消；可用 `OVERSAMPLE_CANDIDATES=4` 恢复无过量采样的 baseline。该能力已验证吞吐收益，但仍需多步质量 A/B 后才能作为正式训练默认策略。
- Fastest-K 的逐请求取消已改为 vLLM 0.18 的公开 external-request API；V4 门禁实测 8/8 个落后候选完成物理取消，且不清空 prefix cache。正式入口保持无过量采样的 `4→4` group 内采样，每次更新消费 4 个 group；新的 12-group 在途深度对应 48 条轨迹，与两个 TP8 副本各 24 个序列槽位对齐。
- 老板 KB/DWH 评测逻辑已完成 1,000 条历史影子回放：原 277 条 DWH 通过 gold SQL 自洽门禁；进一步来源复核发现唯一一组相同 prompt 绑定冲突 gold，现保留相对更贴近题意的 `task_000147`、剔除 `task_000033`，未来正式资产为 `236/20/20`。KB 因缺少已校准语义 judge 全部保持 shadow-only；DWH 在线奖励继续使用 `0.7 × boss_reward + 0.3 × strict evidence`。
- 连续最终答案正确性奖励已通过既有 3,200 条轨迹离线门禁，并以 `PI_DENSE_CORRECTNESS_WEIGHT` 作为默认关闭的可选训练分量；首轮试验固定为 `30%`，其余安全硬门控与正式拓扑保持不变。
- Step 100→120 的 20-step dense30 试验已完成并保存完整 `model,optimizer,extra`。老板原版 val20 总奖励方向性升至 `0.563745`，但 dense30 同口径复算几乎不变、最终数值正确未提高；当前保留 Step 120，暂停直接续训，先扩大密封评测并增强组内正确性信号。
- Step 120 的 48K 强制收尾门禁已完成：第 22 个助手回合触发时救回 4 道未收尾题中的 3 道，老板原版六题平均奖励由 `0.2750` 升至 `0.5458`，但最终数值正确仍为 0；对剩余 `task_000196` 提前到第 14 回合后可收尾并获 `0.5625`，判定仍为 `result_wrong_process_ok`。因此当前不直接扩到 64K/96K或续训100步，先做预算感知的工具调用拦截、纠正监督和同运行配对门禁。
- Step 125 的 `2 groups × 8 responses` 五步金丝雀未通过老板原版门禁：相对 Step 120，val20 数值正确由 `2/20` 降至 `1/20`、完整收尾由 `16/20` 降至 `15/20`。五步训练中正确轨迹平均奖励 `0.7758`、错误轨迹 `0.1600`，4/4 个 mixed-correct groups 均严格正确排序，但 `6/10` 个 prompt 仍为全错；下一步停止同配方续训，先做 16 条机械验证纠错 SFT 冒烟，再扩至 48–64 条并只用 mixed-correct groups 做短 GRPO 金丝雀。
- 两台服务器下的纠错实验按角色流水线执行：5 号机保留 16 卡 Megatron 训练，6 号机负责数据机械核验、回放和 Agent 评测，不做低样本 32 卡跨机 SFT。首个 16 条 go/no-go 预计 `4–6h`；全部门禁通过时，48–64 条纠错 SFT、两次有效 GRPO 更新和一次密封 test20 预计累计 `12–16h`。
- veRL 官方 `verl.trainer.sft_trainer` 的 Step 120 模型态初始化与单步全参 SFT 已在 5 号机实跑通过：TP4/PP2/CP2、Qwen3.6 完整工具模板、assistant-only loss mask、全新 CPU-offload Adam 均可工作；成功步 loss `0.9603`、grad norm `141.10`、单卡峰值 `26.27 GiB`、整机 CPU 内存 `821.63 GiB`，退出码 `0`。该运行仅为一条合成数据的不可晋升工程门禁，下一步才进入 16 条真实纠错数据机械核验。
- 16 条真实 train236 纠错轨迹已通过机械核验并完成 5 步 veRL 官方全参 SFT：loss 从 `1.8738` 降至 `0.5764`，墙钟 `11m04s`，最终 model-only Megatron checkpoint 的 32 个分片完整（`54.72 GB`）。但相同 16 题的老板原始评分器门禁未通过：正确数保持 `2/16`，平均奖励 `0.7000 → 0.6063`，完整收尾 `15/16 → 13/16`，因此不扩到 48–64 条，也不作 held-out 泛化声明。
- Step 120 的一步单变量 SQL 加权金丝雀已完成：工具结构/SQL/最终答案为 `0.25/8/1`，SQL NLL `2.4484 → 2.0612` 且 `16/16` 改善，教师 SQL greedy token `166 → 173`、平均 rank `56.59 → 41.35`。但逐题 SQL 概率超过 0.5 仍为 `0/16`，首条 SQL gold 支持和教师结果等价均仍为 `0/16`；48K 自由回放耗时约 `55m23s`，终止回答仅 `13/16`。候选不晋级、不续训；下一训练目标改为模型首错查询/工具结果条件下的 SQL 恢复监督，不再单独提高 SQL 权重。
- 状态条件化纠错入口已按严格单变量设计补齐：复用 Step 120、相同 16 题和 `0.25/8/1` 目标权重，仅把模型首个错误 SQL 及实际工具结果加入历史；错误 assistant 回合由选择性 mask 保证 loss 为 0，只监督纠正 SQL 与最终答案。新增全查询只读语义审计，在第 1/2/3 条和任意后续查询上定位首次正确或等价证据；两项 CPU 门禁通过前不占用 NPU。
- 状态条件化一步金丝雀已完成：纠正 SQL NLL `1.6815 → 1.4118` 且 `16/16` 改善，greedy token `221 → 225`、top-5 `284 → 298`、平均 rank `20.12 → 16.72`；但逐题概率超过 0.5 仍仅 `1/16`（要求 `12/16`），整段全 greedy 仍为 `0/16`。按门禁停止，不跑短/完整回放、不续训；下一目标改为首错语义分类与 critical-token 对比纠错数据。
- Semantic critical-token 一步金丝雀已完成：把每题首个 semantic non-greedy SQL token 从 `8×` 提到 `32×` 后，SQL NLL `1.2929 → 1.1435` 且 `16/16` 改善，greedy/top-5 各增加 4 个；query-start `3/3` 转为 greedy，但 aggregation `9/9` 仍是首分叉，完整 SQL 概率 `>0.5` 仍为 `2/16`（门槛 `12/16`）。后续训练暂缓，先以同一 16 条首错状态执行 Control / operator oracle / full semantic plan 三臂一次生成门禁，区分 plan selection、schema grounding 与 plan-to-SQL realization。
- 三臂 semantic-plan 门禁与 correct-vs-actual-wrong margin 门禁均已完成：Control/operator/full plan 分别恢复 `1/16`、`1/16`、`2/16`，两个 oracle 均未过线；正确 SQL 的 semantic-delta 在 Step 120 下 `0/16` 占优，平均 margin `-1.1877`，且 aggregation/query-start/identifier/clause 全部偏向实际首错 SQL。下一步锁定为一次 pairwise plan-to-SQL 金丝雀；训练后须达到正确 delta `≥12/16` 占优、`≥12/16` margin 改善且无更早分叉回退，才允许短回放。
- 一步 pairwise plan-to-SQL 金丝雀已完成：`16/16` 逐题 margin 改善，平均 margin `-1.1877 → -0.7646`，但正确候选仅 `3/16` 占优，未达到 `12/16`；更早分叉回退和冻结 target 非法均为 0。按冻结规则不做短回放、不追加同 16 对训练、不晋级 checkpoint；下一步把这 16 对冻结为评价集，先获取不重叠且机械验证的分层训练 pairs。
- 原生 Qwen3.6-27B 与 Step 120 的相同首错状态概率归因仍有效：两者均为正确 semantic delta `0/16` 占优，平均 margin 为 `-1.2057/-1.1877`，核心 SQL misranking 在原生模型中已经存在。完整 64 题、25 工具反馈公平对照进一步确认，高过程分错答为原生 `13`、Step 120 `8`，训练没有新制造或放大代理错配；但 Step 120 的只读 SQL 覆盖 `30→23`、完整回答 `40→35`、平均总奖励 `0.2858→0.2612`，同时 numeric correct `5→7`，暴露出正确性与覆盖/完成的权衡。
- chosen-only schema-conditioned 首动作一步金丝雀已完成：train48 的 `0.25/8` 加权更新使 calibration16 SQL NLL `1.2913→1.1267`、`16/16` 逐题改善，top-5 `344→348`、mean rank `18.78→15.87`，且结构 NLL 未退化；但 greedy token 仅 `277→282`（`+5 < +12`），完整 SQL 全 greedy 仍为 `0/16`，未过预注册门。14/16 的首个 non-greedy 边界未移动，其中原有 aggregation `9/9` 全部保留；按合同不跑自由回放、不追加训练、不晋级 checkpoint。
- 新一轮训练前先执行 current-definition 数据池审计：正式 train236 在旧严格筛选下仅有 25 个候选，扣除冻结 16 题后只剩 9 个，禁止直接启动 pairwise 训练。新增审计从老板当前权威任务定义重建漂移 instruction/gold 身份，逐条执行只读 SQL、核对 gold 支持，并同时隔离冻结 16 题、val20、test20 的 task/instruction/SQL 哈希；只有严格可用新任务达到至少 48 条才允许进入 Step 120 首错采集。
- 新增的 22 条真实首错状态已冻结为 evaluation-only，并完成 Step 120 与现有 chosen-only 一步 checkpoint 的同状态纯前向对照：候选有 `18/22` 逐题 semantic-delta margin 改善，平均值 `-1.1454→-1.0843`，但正确候选占优仍为 `3/22`，full-SQL 正确占优仍为 `1/22`；预注册的 `17/22` 决策边界门失败，因此不跑 full64、不追加训练、不晋级。该 22 条足以拒绝当前候选，但不用于训练或总体泛化估计；下一步仍是另建至少 48 条不重叠真实首错训练 pair。
- 冻结 eval22 后的训练供给已按源任务身份重新核验：原生 full25 的27个已观测首错中16个命中eval22，剩余11个里又有4个命中chosen-only calibration16，旧 frozen16、val20、test20无额外重叠，最终只保留7对。7对均通过机械、token/mask和Step120纯前向门；正确 semantic-delta/full-SQL均为 `0/7` 占优，平均margin为 `-1.7816/-0.9334`，说明这些原生真实首错仍被Step120系统性误排。训练硬门继续保持48对，当前缺口41；review138即使全部批准，按历史 `22/64` 产率补足缺口的预测概率约 `76.5%`，90%/95%把握需158/172个已批准候选。
- 42条最低机械风险的review-required任务已完成逐条题意—gold—SQL语义审核：42/42机械可执行、顺序扰动稳定且期望值被查询结果支持，但题意无歧义蕴含gold和SQL完整回答题意均为`0/42`，最终语义批准`0/42`。这证明当前问题是高严重度标签语义失配，不是SQL不可执行；停止审核同队列剩余96条，也不为其消耗NPU。
- 用`0/42`批准率与历史`22/64` pair产率做两阶段Jeffreys后验压力测试，剩余96条预计仅批准约`1.12`条、形成约`0.39`对pair，补齐41对缺口的预测概率约`7.1×10^-18`。下一动作改为先新建32条SQL-first训练候选和3条parity-only哨兵：SQL只负责生成/验收hidden gold，rollout仍只用题面与最终结果；CPU语义门通过后，先在哨兵上做`3×2×双臂`请求级复验，再按32条一批采集真实首错状态。
- 多沙箱DWH只读筛选已实跑：19个版本、每版500题和独立数据库，共9,500题，CPU机械门仅耗时1.919秒；5,099条gold与SQL结果一致，459条通过高精度门。排除v15、跨版本重复题面和高语义风险任务类型后，得到18个非v15沙箱上的281条直接查询池（234 numeric / 47 table），仍需显式语义审核。推荐先做64题分层pilot，预计从审核到`64×8`双机rollout出结果约5–8小时；不直接投入16–24小时跑完281题。
- 281题双机筛选采用逐shard累计门禁：每题必须恰有8条完整可用轨迹、无runtime error和超时，并且纯最终结果正确数为`1–7/8`才进入mixed审视队列；`0/8`和`8/8`仅从本次GRPO更新排除，不从源数据永久删除。mixed仍只是候选，须继续人工核对“题意无歧义蕴含gold、SQL完整回答题意、SQL结果支持期望值、最终结果路由可信”，审核前`training_allowed=false`。
- 首批双机各32题完成后共有7个mixed候选，逐题语义复核仅批准1个、拒绝6个：5个把“查看/给出数据”擅自收窄为求和，1个日期字段口径含混且出现“正文提到gold但最终结论选择另一数值”仍被纯数值包含式评分判对。首批说明mixed难度只是必要条件，不是训练数据质量保证；批准候选在全281题收尾和合并复核前仍保持`training_allowed=false`。
- 双机各48题时累计18个mixed候选，语义批准增至3个、拒绝15个；第三个shard新增11个mixed中仅2个通过，9个拒绝项包括8个未明确要求求和却使用SUM gold的任务，以及1个要求评估维表变动影响但SQL/reward只覆盖单一金额汇总的多意图任务。当前优质mixed的累计产率为`3/96=3.125%`（按已完成题计），后续继续逐shard审视，不因高`7/8`正确数自动放宽语义标准。
- 新 DWH 数据不再复用老板的 instruction-first/backward-SQL 链：单个物流 SQLite 沙箱内以结构化 QueryPlan 同源生成只读 SQL 和 hidden gold，再把仅含合成业务语义的约束交给用户私有 OpenAI-compatible API 改写为自然题面；URL、模型名和 key 只从 5 号机 0600 私有配置读取，SQL、gold、数据库行和凭据均不进入请求。最终 `20260814_llin_dwh_planfirst_api_v3` 固定 300 条、每 50 条提升一档，共 6 档，覆盖老板、财务、分析、运营、仓储、区域、采购、客服、计划、销售和普通员工 11 类角色；300/300 SQL/gold 精确重放、300/300 语义保真、零技术词、零重复。原生成清单中的 `4` 次小校准已被同实际 GRPO 一致的每题 `8` 次双模型比较取代，训练继续关闭。
- 新的8沙箱开放业务生成以旧 Band 5 作为 Level 1 起点，不再按表面JOIN数量定难度；Level 1–5依次覆盖分组排名、跨期比较、流程诊断、整体基线归因和经营优先级，每沙箱固定500题、每级100题。v15/v20/v21/v22/v23/v24/v25/v26的最终一致离线批次共4,000题，4,000/4,000 SQL/gold重放和语义锚点门禁通过，所有连接均为必要连接、开放解释不设隐藏唯一答案，正式`8×`实测校准前继续禁训。
- v3 不再直接进入五步训练，先执行基座与 Step 120 的同题概率地图：两模型均对 300 题各生成 8 条独立 PI-Agent 轨迹，共 4,800 条；前 60 题按六档各 10 题冻结为 evaluation-only，其中首个 48 题分片固定为每档 8 题的并发/评分先导，剩余 240 题才是候选供给。模型可见运行时投影严格只包含 `logistics.sqlite`、`schema_dictionary.md` 和空 `documents/`，含 hidden gold/SQL 的 `dwh_tasks.jsonl` 永不复制进工作区；全流程只做最终 numeric/table 结果评分并保持 `training_allowed=false`。
- PI-Agent 与 veRL rollout 的 `10题×每题8条` 部署路径门禁已完成但未通过：追加调用链审计确认两臂有效采样均为 `temperature=1.0 / top_p=0.95 / top_k=20`，且每个题组的8条轨迹全部互异，不是temperature 0或复制候选；但单次token cap、compaction、墙钟、并发和工具实现并不相同，因此只能结论为“现网路径不兼容”，不能称为严格同配置A/B。当前不开放 bucket 筛选或训练；10条 val-only 题始终禁止进入训练，全对/全错也不得永久删除。
- Step 120 的单请求长上下文阶梯已在空闲 6 号机完成：并发1、TP8、固定生成256 tokens、关闭prefix cache时，2K/40K/48K解码分别为`9.331/9.121/9.190 tokens/s`，40K/48K仍保留2K的`97.75%/98.48%`，未出现“超过40K只剩1/10”；TTFT由`0.300s`增至`3.625/4.318s`，64K因当前`max_model_len=49,152`未执行。该结果只代表单序列速度，不等同于高并发服务总吞吐。
- 所有新增镜像、容器、工作目录和实验名均以 `llin` 开头，不复用或修改其他人的环境。

当前服务器部署：

| 项目 | 5 号机 | 6 号机 |
| --- | --- | --- |
| 角色 | 训练 | rollout 推理 |
| 工作目录 | `/data3/llin/qwen3.6-27b-verl-grpo` | `/data3/llin/qwen3.6-27b-verl-grpo` |
| 容器 | `llin-verl-trainer-m05-20260730` | `llin-verl-rollout-m06-20260730` |
| 镜像 | `llin-verl-a3:20260730` | `llin-verl-a3:20260730` |
| 容器权限 | 特权模式（仅重建上述 `llin` 容器） | 特权模式（仅重建上述 `llin` 容器） |
| 当前实验 NPU | Ray trainer 服务在线；teacher-forced 纯前向诊断已完成，无活动训练 | Ray rollout 服务在线；当前无活动回放 |

## 数据结论

老板原有的 `trajectories_v15_27B_table.tar.gz` 是 PI agent 事件轨迹，共包含 1,500 个 JSONL 文件。它包含提示、模型消息、工具调用和工具输出，但没有可直接供 GRPO 使用的显式 reward，因此不能原样传给 veRL。

早期 smoke 采用只读 `query_sqlite`；当前 boss-aligned 数据阶段使用以下严格 PI 契约：

1. `bash/read/edit/write` 四工具 schema 与老板 `DEFAULT_TOOLS` 做规范化精确比较；system 固定为老板 `DEFAULT_SYSTEM`，二者均记录源文件与内容哈希。
2. 每条轨迹从对应 `sft/<version>` 环境复制独立可写工作区；四个工具在整条轨迹中共享状态，结束后统一清理。
3. 昇腾 veRL 镜像缺少 `sqlite3` CLI，项目提供只读兼容代理，使模型可按当前数据中实际携带的 system prompt 在 Bash 中调用 `sqlite3`。
4. GRPO 只接收真实 source task 的 numeric/table verifier，并要求 instruction/gold 哈希经过显式 alignment review；未审核、报告型、KB/Hybrid 和无严格 verifier 的样本只进入 SFT/reference。
5. 旧 200-task V2 混用了 Qwen3.7-Max manifest 与 Qwen3.6 conversation，已废止。当前从 v15 原始 Qwen3.6 事件文件按 `task_id` 连接同源 sandbox task：1,500 条完整轨迹中，1,000 条 KB/Hybrid 和 220 条无严格 verifier 样本进入 SFT，3 条 gold 不一致被拒绝，277 条进入待审核队列；审核完成前不生成正式 GRPO Parquet。

来源复核进一步确认：归档事件中的生成模型字段为 `provider=my-local, model=Qwen3.6-27B`，它的历史回答只进入 SFT/reference，不会放入 GRPO prompt；GRPO 的 hidden label 来自老板 v15 task manifest 的 `verification_sql + gold_answer`。修正后 `276/276` 条 SQL 均可执行、非空且与 expected value 一致，但 `271/276` 条仍命中至少一个语义预警，因此只能称为“数据库层机械自洽”，不能称为“已完成人工逐题语义确认”。

原始轨迹、验证清单、Parquet、模型、checkpoint 和运行日志均不会提交到 Git。

## 目录

- [`docs/training_experiment_report_20260731.md`](docs/training_experiment_report_20260731.md)：从初始环境、数据改造、十余次关键尝试到最终 One-Step 与 bounded fully-async 跑通的完整复盘报告。
- [`docs/trajectory_rollout_investigation_20260731.html`](docs/trajectory_rollout_investigation_20260731.html)：同 prompt 轨迹长度对比、长尾 rollout 超时、完整 GRPO group 队列与 vLLM 真取消方案的可交互调查报告。
- [`docs/context_48k_tool_turn_validation_20260731.md`](docs/context_48k_tool_turn_validation_20260731.md)：8K/16K/32K/48K 阶梯实跑、显存峰值、system prompt 血缘和工具调用轮次对齐报告。
- [`docs/long_context_decode_benchmark_step120_20260814.md`](docs/long_context_decode_benchmark_step120_20260814.md)：Step 120 在2K/4K/8K/16K/32K/40K/48K的单请求TTFT、解码速度和端到端墙钟阶梯；64K记录为当前49,152上限的容量边界。
- [`docs/fastest_k_oversampling_validation_20260731.md`](docs/fastest_k_oversampling_validation_20260731.md)：`4→4` 与 `6→最快4` 的严格单步 A/B、吞吐收益、质量选择偏差和物理 vLLM 取消证据边界。
- [`docs/fastest_k_efficiency_20step_20260731.html`](docs/fastest_k_efficiency_20step_20260731.html)：五组拓扑/过量采样矩阵、8-group 预热的 20-step fully-async 时序、奖励泄漏复核和下一步效率实验的自包含技术报告。
- [`docs/fastest_k_abort_debug_20260801.html`](docs/fastest_k_abort_debug_20260801.html)：严格奖励在线门禁、Fastest-K V2–V4 假取消故障链、external/internal request ID 根因、最终 8/8 物理取消和显存释放的完整技术复盘。
- [`docs/step_efficiency_investigation_20260804.html`](docs/step_efficiency_investigation_20260804.html)：48K v15 五步队列等待诊断、2-group batch 容量估算、纯 Fastest-K 质量偏差与延迟备用方案的技术报告。
- [`docs/frozen_model_baseline_20260803.md`](docs/frozen_model_baseline_20260803.md)：完整 PI Agent、48K 上下文和 200 条正式任务的冻结模型基线，以及四次启动的故障与修复复盘。
- [`docs/formal_pi_failure_reproduction_20260803.md`](docs/formal_pi_failure_reproduction_20260803.md)：冻结基线 `-01～-04` 与正式 50-step `-01～-03` 的逐次配置、原始报错、根因、修复、验证和复现排障手册。
- [`docs/formal_grpo_50step_quality_diagnosis_20260804.md`](docs/formal_grpo_50step_quality_diagnosis_20260804.md)：正式 50-step 完成结果、800 条奖励分解、GRPO 组内方差、instruction/gold 对齐、system prompt 与沙箱隔离问题及 V3 训练建议。
- [`docs/boss_data_alignment_correction_20260804.md`](docs/boss_data_alignment_correction_20260804.md)：逐项记录固定 200 条、fallback system、工具 schema/runtime、Qwen3.7/Qwen3.6 manifest 混用、hidden reward 与 GRPO/SFT 分流的根因和更正。
- [`docs/boss_reward_shadow_validation_20260804.md`](docs/boss_reward_shadow_validation_20260804.md)：老板 KB/DWH 评测逻辑复用边界、1000 条 task-id 精确影子回放、奖励防投机设计和正式接入门槛。
- [`docs/dwh_kb_reward_divergence_examples_20260804.html`](docs/dwh_kb_reward_divergence_examples_20260804.html)：从老板 v15 原始任务和完整 PI 轨迹中各选一个 DWH/KB 高分差案例，逐项对照老板奖励、本项目影子奖励、原始证据、误判来源和修正建议。
- [`docs/v15_dwh_full277_reward_alignment_20260804.html`](docs/v15_dwh_full277_reward_alignment_20260804.html)：277 条老板 v15 DWH 的全量使用、237/20/20 防泄漏分割、语义预警、老板主奖励与严格证据护栏审计。
- [`docs/training_data_provenance_quality_audit_20260806.html`](docs/training_data_provenance_quality_audit_20260806.html)：追溯任务、Qwen3.6 源轨迹、hidden label 与 GRPO 输入的区别，比较重复 prompt 的两条冲突 gold，并记录修正后的 236/20/20 资产和正确性证据边界。
- [`docs/v15_dwh_frozen_baseline_20260804.md`](docs/v15_dwh_frozen_baseline_20260804.md)：固定 val20 冻结模型指标、`None` 聚合故障、安全硬归零观测补强、主动中止的 step0 运行和最终 5-step 门禁。
- [`docs/v15_dwh_bossreward_5step_20260804.md`](docs/v15_dwh_bossreward_5step_20260804.md)：真实 DWH 5-step 的逐步耗时、80 条训练轨迹、冻结基线对比、长尾队列、非致命日志问题和 PP=2 checkpoint 缺层复盘。
- [`docs/boss_exact_pre_post_100step_20260806.md`](docs/boss_exact_pre_post_100step_20260806.md)：同一固定 val20 上直接调用老板原始 manifest、数据库和三份评分脚本，对比冻结模型与 step-100 的总奖励、正确性、过程质量和完整收尾。
- [`docs/step100_checkpoint_hf_export_20260806.md`](docs/step100_checkpoint_hf_export_20260806.md)：step-100 可续训 Megatron checkpoint 与独立 HF 导出的路径、1199-tensor 完整性、MTP 继承边界和 TP8 vLLM 最小生成验收。
- [`docs/boss_exact_pre_post_100step_20260806_external.md`](docs/boss_exact_pre_post_100step_20260806_external.md)：面向外部汇报的精简版，保留核心结论和聚合指标，移除评测集规模、逐题标识、内部路径、文件哈希等内部信息。
- [`docs/boss_exact_step100_step200_20260807.html`](docs/boss_exact_step100_step200_20260807.html)：同一固定 val20 上直接调用老板原版评分器的 Step 100/200 配对复评；包含核心分数、胜负平、输入一致性审计和下一步建议。
- [`docs/step120_dense_trial_20260810.html`](docs/step120_dense_trial_20260810.html)：Step 100/120/200 同题老板原版评分、dense30 同公式复算、配对不确定性与结束阶段耗时诊断的自包含技术报告。
- [`docs/step120_dense_trial_20260810_summary.json`](docs/step120_dense_trial_20260810_summary.json)：不含原始轨迹与机器绝对路径的聚合分析、逐题配对结果、输入一致性和运行时审计。
- [`notebooks/step120_dense_trial_analysis_20260810.ipynb`](notebooks/step120_dense_trial_analysis_20260810.ipynb)：从汇总 JSON 从头执行通过的可复现分析与图表 notebook。
- [`docs/next_experiment_strategy_20260810.html`](docs/next_experiment_strategy_20260810.html)：Step 120未收尾根因、64K/80K/96K训练与rollout容量、100步成本和分级快速实验门禁的自包含技术报告。
- [`docs/next_experiment_strategy_20260810_summary.json`](docs/next_experiment_strategy_20260810_summary.json)：不含原始轨迹与机器路径的回合边界、显存规划、墙钟成本与实验优先级聚合。
- [`notebooks/next_experiment_strategy_20260810.ipynb`](notebooks/next_experiment_strategy_20260810.ipynb)：从头执行通过的96K容量、并发增量和快速实验成本分析 notebook。
- [`docs/force_final_sentinel_20260810.md`](docs/force_final_sentinel_20260810.md)：Step 120 的 48K 强制收尾 sentinel6 与单题提前收口实跑、老板原版评分、失败归因和进入训练前门槛。
- [`docs/semantic_plan_and_delta_pretraining_gate_20260812.md`](docs/semantic_plan_and_delta_pretraining_gate_20260812.md)：Step 120 三臂 semantic-plan 一次生成、工具协议合规、correct-vs-actual-wrong semantic-delta margin 与下一步 pairwise 金丝雀停止门槛。
- [`docs/semantic_plan_and_delta_pretraining_gate_20260812_summary.json`](docs/semantic_plan_and_delta_pretraining_gate_20260812_summary.json)：不含原始问题、SQL、答案和服务器路径的安全聚合结果。
- [`docs/query_initiation_oracle_gate_20260813.md`](docs/query_initiation_oracle_gate_20260813.md)：Step 120 完整预算无查询任务的通用查询启动干预、SQLite 命令族细分、`0/41` 门禁结论与下一结构化工作流测试。
- [`docs/query_initiation_oracle_gate_20260813_summary.json`](docs/query_initiation_oracle_gate_20260813_summary.json)：不含原始命令、prompt、SQL、答案、task ID、工具结果或服务器路径的查询启动安全汇总。
- [`docs/structured_sqlite_realization_gate_20260813.md`](docs/structured_sqlite_realization_gate_20260813.md)：同 41 题结构化非交互 `path→schema→SELECT/WITH` 运行、`0/41` 结论、与通用干预比较及下一 schema-oracle 数据门。
- [`docs/structured_sqlite_realization_gate_20260813_summary.json`](docs/structured_sqlite_realization_gate_20260813_summary.json)：不含原始命令、prompt、SQL、答案、task ID、工具结果或服务器路径的结构化门禁安全汇总。
- [`docs/schema_oracle_action_gate_20260813.md`](docs/schema_oracle_action_gate_20260813.md)：完整不重叠 64 题的相关表 schema 上界诊断、有效 `2/1` 工具反馈协议、`4` 条正确/`35` 条带结果错误/`25` 条无查询结论及 chosen-only 下一步。
- [`docs/schema_oracle_action_gate_20260813_summary.json`](docs/schema_oracle_action_gate_20260813_summary.json)：不含原始命令、prompt、schema、SQL、答案、task ID、工具结果或服务器路径的 schema-oracle 安全汇总。
- [`docs/chosen_only_first_action_baseline_20260813_summary.json`](docs/chosen_only_first_action_baseline_20260813_summary.json)：chosen-only 64 条 CPU mask 门、calibration16 Step 120 聚合 NLL/rank 与一步 train48 金丝雀预注册阈值的安全汇总。
- [`docs/chosen_only_first_action_canary_20260813.md`](docs/chosen_only_first_action_canary_20260813.md)：train48 一步加权 SFT、model-only checkpoint、同一 calibration16 纯前向、7 项预注册门和 aggregation 边界诊断的完整结论。
- [`docs/chosen_only_first_action_canary_20260813_summary.json`](docs/chosen_only_first_action_canary_20260813_summary.json)：不含 prompt、SQL、答案、task ID、工具结果或服务器路径的一步 chosen-only 金丝雀安全聚合。
- [`docs/next_highest_value_action_review_20260813.html`](docs/next_highest_value_action_review_20260813.html)：综合原生/Step 120 归因、chosen-only/pairwise 金丝雀、真实首错数量门和 current-definition 池容量的下一步优先级复核；把第一动作收敛为 CPU 语义裁决扩池，再分批获取至少 48 个真实部署态首错 pair。
- [`docs/next_highest_value_action_review_20260813_summary.json`](docs/next_highest_value_action_review_20260813_summary.json)：上述决策的安全聚合、`22/64` pair 产率推导、80 条新增审核规划值、逐层停止门和来源哈希。
- [`docs/next_highest_value_action_review_20260813.artifact.json`](docs/next_highest_value_action_review_20260813.artifact.json)：上述复核报告的 canonical Data Analytics artifact、数据集、图表、来源与门禁定义。
- [`docs/disjoint_real_state_eval22_20260813.html`](docs/disjoint_real_state_eval22_20260813.html)：22 条冻结真实首错状态上的 Step 120—chosen-only 候选同状态对照、家族归因、预注册门禁与下一动作的自包含报告。
- [`docs/disjoint_real_state_eval22_20260813_summary.json`](docs/disjoint_real_state_eval22_20260813_summary.json)：不含 prompt、SQL、答案、task ID、工具结果或服务器路径的基线—候选安全汇总、逐题改善计数与来源哈希。
- [`docs/disjoint_real_state_eval22_20260813.artifact.json`](docs/disjoint_real_state_eval22_20260813.artifact.json)：上述报告的 canonical Data Analytics artifact、快照数据、图表、表格、来源和指标定义。
- [`docs/next_training_supply_strategy_20260813_summary.json`](docs/next_training_supply_strategy_20260813_summary.json)：v3安全汇总；按完整冻结集合和已完成的7对Step120纯前向结果重算41对缺口、review138成功概率与下一动作。旧同名HTML/artifact保留历史v2快照，不再作为当前决策来源。
- [`docs/native_outside_eval22_supply_20260813_summary.json`](docs/native_outside_eval22_supply_20260813_summary.json)：原生full25已观测首错与eval22、chosen-only calibration16、旧frozen16、val20、test20的服务器侧聚合差集，最终保留7个唯一状态。
- [`docs/native_disjoint7_step120_margin_20260813_summary.json`](docs/native_disjoint7_step120_margin_20260813_summary.json)：7对原生真实首错候选的机械、token/mask、Step120纯前向margin、无checkpoint和NPU释放安全摘要。
- [`docs/disjoint_pair_review_pilot42_20260813_summary.json`](docs/disjoint_pair_review_pilot42_20260813_summary.json)：42条最低机械风险审核试点的分层、SQLite反向扫描稳定性、容量用途与fail-closed状态；敏感审核包只留服务器。
- [`docs/disjoint_pair_semantic_review_supply_rebase_20260813_summary.json`](docs/disjoint_pair_semantic_review_supply_rebase_20260813_summary.json)：42条逐项语义审核的去标识聚合、剩余96条的两阶段Jeffreys后验压力测试、41对缺口和新任务供给目标。
- [`docs/disjoint_pair_semantic_review42_20260813.artifact.json`](docs/disjoint_pair_semantic_review42_20260813.artifact.json)、[`docs/disjoint_pair_semantic_review42_20260813_report_notes.json`](docs/disjoint_pair_semantic_review42_20260813_report_notes.json)：技术报告的canonical artifact与受众/结构/图表/来源说明；当前portable reader在本次和既有对照artifact上均停留fallback并超时，因此未发布未完成QA的HTML。
- [`docs/runtime_parity_10x8_step120_20260813.md`](docs/runtime_parity_10x8_step120_20260813.md)：Step 120 在 PI-Agent 与 veRL rollout 上各80条的部署路径对照；追加审计区分已对齐的模型/任务/有效采样与未对齐的token、compaction、终止和并发策略。
- [`docs/runtime_parity_10x8_step120_20260813_summary.json`](docs/runtime_parity_10x8_step120_20260813_summary.json)：不含 prompt、gold、SQL、轨迹、任务标识、逐题哈希或服务器路径的双臂安全聚合。
- [`docs/runtime_parity_sampling_config_audit_20260813_summary.json`](docs/runtime_parity_sampling_config_audit_20260813_summary.json)：区分GRPO训练temperature 1.0、普通验证默认temperature 0和本次parity显式temperature 1.0；记录两臂80条的生成方式、逐组精确去重与未对齐运行时配置。
- [`docs/next_action_priority_after_runtime_parity_20260813.artifact.json`](docs/next_action_priority_after_runtime_parity_20260813.artifact.json)、[`docs/next_action_priority_after_runtime_parity_20260813_summary.json`](docs/next_action_priority_after_runtime_parity_20260813_summary.json)、[`docs/next_action_priority_after_runtime_parity_20260813_report_notes.json`](docs/next_action_priority_after_runtime_parity_20260813_report_notes.json)：运行时审计后的整体优先级复核；把32条可信训练候选定为第一动作，并把小规模parity复验迁移到3条独立哨兵题上。canonical artifact已通过到官方构建器静态图阶段，但portable reader在5秒/30秒预算下均停留fallback，因此未发布未完成QA的HTML。
- [`docs/boss_multisandbox_dwh_screening_20260813_summary.json`](docs/boss_multisandbox_dwh_screening_20260813_summary.json)：19版9,500条DWH的CPU只读SQL/gold/语义预警/重复筛选、281条非v15直接查询池，以及64题pilot与全281题双机`n=8`耗时模型。
- [`docs/semantic_delta_pairwise_canary_20260812.md`](docs/semantic_delta_pairwise_canary_20260812.md)：一次 reference-free pairwise 更新、工程失败修复、训练资源、同数据概率前后门禁与 fail-closed 决策。
- [`docs/semantic_delta_pairwise_canary_20260812_summary.json`](docs/semantic_delta_pairwise_canary_20260812_summary.json)：不含原始问题、SQL、答案和服务器路径的一步 pairwise 安全聚合结果。
- [`docs/native_vs_step120_reward_behavior_attribution_20260812.md`](docs/native_vs_step120_reward_behavior_attribution_20260812.md)：原生模型与 Step 120 的相同错误状态概率对照、同题老板原版自由回放和奖励代理错配归因。
- [`docs/native_vs_step120_reward_behavior_attribution_20260812_summary.json`](docs/native_vs_step120_reward_behavior_attribution_20260812_summary.json)：不含原始问题、SQL、答案、逐题标识和服务器路径的安全归因汇总。
- [`docs/accuracy_improvement_strategy_20260810.html`](docs/accuracy_improvement_strategy_20260810.html)：结合 Step 100/120/200、前后两个100步组内信号和强制收尾实验的准确率瓶颈诊断；给出 oracle 梯度、纠错 SFT、奖励分层及 `2 groups × 8 responses` 金丝雀路线。
- [`docs/accuracy_improvement_post_step125_20260811.html`](docs/accuracy_improvement_post_step125_20260811.html)：结合 Step 125 金丝雀、同题老板原版评分、oracle 梯度与组内奖励排序的准确率复盘；给出纠错 SFT、mixed-only GRPO 和密封 test20 的分阶段门禁。
- [`docs/accuracy_improvement_post_step125_20260811_summary.json`](docs/accuracy_improvement_post_step125_20260811_summary.json)：不含原始轨迹与服务器绝对路径的 Step 125 组内信号、checkpoint 对比、oracle 结果和下一轮实验门槛聚合。
- [`docs/repair_sft_two_server_time_estimate_20260811.html`](docs/repair_sft_two_server_time_estimate_20260811.html)：用既有 oracle、Step 125 金丝雀、val20 和 checkpoint 实测墙钟，估算两台服务器并行下的纠错 SFT 首个决策点与完整门禁关键路径。
- [`docs/repair_sft_two_server_time_estimate_20260811_summary.json`](docs/repair_sft_two_server_time_estimate_20260811_summary.json)：两机角色、实测耗时基线、累计里程碑、关键路径区间和 18–24 小时下行情形的安全聚合。
- [`docs/verl_repair_sft_smoke_20260811.md`](docs/verl_repair_sft_smoke_20260811.md)：veRL 官方 SFT trainer 从 Step 120 分布式模型权重初始化、Qwen3.6 完整工具模板 assistant-only mask、四次隔离启动与最终单步前反向成功证据。
- [`docs/repair_sft_train236_overfit_20260811.md`](docs/repair_sft_train236_overfit_20260811.md)：16 条 train236 真实纠错轨迹的来源、机械门禁、5 步 veRL 全参 SFT 指标、资源峰值、checkpoint 完整性与同题老板评分器回放边界。
- [`docs/repair_sft_teacher_forced_diagnosis_20260811.html`](docs/repair_sft_teacher_forced_diagnosis_20260811.html)：Step 120/SFT Step 5 的 teacher-forced 分项概率、老板自由回放、首条 SQL 分叉和下一轮 SQL-focused 门禁的自包含技术报告。
- [`docs/repair_sft_teacher_forced_diagnosis_20260811_summary.json`](docs/repair_sft_teacher_forced_diagnosis_20260811_summary.json)：不含原始问题、SQL、答案与服务器绝对路径的安全聚合指标、运行资源和证据链。
- [`docs/repair_sft_teacher_forced_diagnosis_20260811_artifact.json`](docs/repair_sft_teacher_forced_diagnosis_20260811_artifact.json)：上述报告的 canonical Data Analytics artifact、数据集、图表、来源与技术结论定义。
- [`docs/repair_sft_pretraining_gate_20260812.md`](docs/repair_sft_pretraining_gate_20260812.md)：首条 SQL 的只读执行、gold 支持与机械等价门禁，以及据此冻结的一步 SQL-only 金丝雀配方。
- [`docs/repair_sft_pretraining_gate_20260812_summary.json`](docs/repair_sft_pretraining_gate_20260812_summary.json)：不含原始问题、SQL、答案和服务器路径的安全聚合门禁结果，包括已完成的 exact token-rank 对比。
- [`docs/repair_sft_sql_weighted_canary_20260812.md`](docs/repair_sft_sql_weighted_canary_20260812.md)：一步 SQL 加权训练、checkpoint、rank、首条 SQL 语义和 48K 自由回放的完整门控结论。
- [`docs/repair_sft_sql_weighted_canary_20260812_summary.json`](docs/repair_sft_sql_weighted_canary_20260812_summary.json)：不含原始问题、SQL、答案和服务器路径的金丝雀安全聚合与 fail-closed 决策。
- [`docs/repair_sft_state_conditioned_canary_20260812.md`](docs/repair_sft_state_conditioned_canary_20260812.md)：全查询语义基线、首错零-loss 数据、一步状态条件化训练和训练后概率/rank 门禁结论。
- [`docs/repair_sft_state_conditioned_canary_20260812_summary.json`](docs/repair_sft_state_conditioned_canary_20260812_summary.json)：不含原始问题、SQL、答案和服务器路径的状态条件化金丝雀安全聚合与停止决策。
- [`docs/repair_sft_critical_token_canary_20260812.md`](docs/repair_sft_critical_token_canary_20260812.md)：semantic critical-token 单变量数据门禁、一步训练、同数据前后概率/rank、原 token 恢复归因和停止决策。
- [`docs/repair_sft_critical_token_canary_20260812_summary.json`](docs/repair_sft_critical_token_canary_20260812_summary.json)：不含原始问题、SQL、答案和服务器路径的 critical-token 金丝雀安全聚合与证据哈希。
- [`docs/repeated_sql_causal_diagnosis_20260812.html`](docs/repeated_sql_causal_diagnosis_20260812.html)：把首条 SQL 语义门禁、同题自由回放、48K 强制收尾和正确证据 oracle 串成因果链，区分重复查询对准确率、完成率和墙钟的不同作用。
- [`docs/repeated_sql_causal_diagnosis_20260812.artifact.json`](docs/repeated_sql_causal_diagnosis_20260812.artifact.json)：上述重复 SQL 因果诊断的 canonical report artifact、聚合数据、图表和来源定义。
- [`docs/leadership_experiment_update_methodology_20260806.md`](docs/leadership_experiment_update_methodology_20260806.md)：从多轮实际修订中提炼的领导汇报方法论，固化四段结构、数字精度、口径边界、抗奖励投机表述、行动项口吻和自检清单。
- `llin_verl/pi_sqlite_tool.py`：只读 SQLite 轨迹工具。
- `llin_verl/pi_workspace_tools.py`、`llin_verl/pi_agent_loop.py`：完整 PI 四工具、轨迹级共享沙箱、事件审计和统一清理。
- `llin_verl/trajectory_telemetry.py`：异步上下文隔离的逐轨迹遥测，分别记录调度排队、模型生成、工具执行、执行/总耗时，并在超时物理终止前只保留已生成 token 数量而不复制内容。
- `llin_verl/pi_sqlite_cli.py`：为官方昇腾镜像补齐的受限只读 sqlite3 CLI 兼容层。
- `llin_verl/pi_reward.py`：最终答案、可执行 SQL 证据、必需表和安全协议联合奖励 V2。
- `llin_verl/boss_reward_shadow.py`、`scripts/replay_boss_reward_shadow.py`：DWH 结果门控候选奖励、KB 文档/拒答影子信号和老板历史 verdict 并行回放；当前不接训练入口。
- `runtime/sitecustomize.py`：将训练池固定到 5 号机、rollout 池固定到 6 号机。
- `scripts/prepare_pi_dataset.py`：验证轨迹到 veRL Parquet 的转换程序。
- `scripts/prepare_pi_formal_dataset.py`：只保留旧 V2 历史复现，默认阻断，必须显式传 `--allow-legacy-v2`。
- `scripts/prepare_boss_aligned_dataset.py`、`scripts/export_boss_task_manifest.py`、`scripts/check_boss_alignment_contract.py`：按真实 task_id 连接老板轨迹与 task、全量/显式 pilot 分流、review queue、GRPO/SFT 隔离及正式启动硬门禁。
- `scripts/select_v15_dwh_batch.py`、`scripts/check_boss_reward_dataset.py`：按老板 v15 原始任务契约审核 277 条 DWH、分层切分并防止重复 prompt 跨 split 泄漏，随后在真实 Parquet 上验证奖励字段与任务族。
- `scripts/analyze_boss_validation.py`：不重跑 rollout，直接汇总老板主奖励 validation JSONL，并检查空值、混合奖励公式、分类型正确率和安全命令重放。
- `scripts/prepare_boss_exact_evaluation.py`：把 veRL 保存的 Qwen 多轮文本轨迹无损还原为老板原版 OpenAI messages，严格按 task_id 复制原始 task manifest，并审计并行工具调用、缺失响应、最终回答和输入哈希。
- `scripts/verify_checkpoint_integrity.py`：在正式启动器发布成功退出码前检查 HF tensor key/分片或 Megatron distributed checkpoint 元数据与分片，缺失时 fail closed。
- `scripts/configure_live_optimizer_checkpoint.py`：对运行中的 veRL Ray WorkerDict 逐 rank 检查或在线切换最终 checkpoint 内容，用于在不中断训练的前提下补启用 `model,optimizer,extra`，并要求所有预期 worker 回读一致。
- `scripts/analyze_formal_grpo_50step.py`、`scripts/audit_formal_instruction_gold_alignment.py`：完整 50-step 训练信号、奖励组件、GRPO group 方差、工具行为及 instruction/gold 语义复核触发器。
- `scripts/analyze_canary_rollout_signal.py`：只读汇总指定 rollout 文件窗口的 mixed/all-wrong group、正确/错误奖励分离、组内排序和 SQL 证据率；默认只输出聚合 JSON，不复制原始轨迹。
- `scripts/start_ray_m05.sh`、`scripts/start_ray_m06.sh`：两机 Ray 启动程序。
- `scripts/check_ray_roles.py`：跨机角色落点验证。
- `scripts/check_hccl.py`：两机基础 HCCL all-reduce 验证。
- `scripts/check_hccl_fanout.py`：1 个训练 rank 到 16 个 rollout rank 的权重广播拓扑验证。
- `scripts/run_pi_grpo_smoke.sh`：Qwen3.6-27B 单步轨迹 GRPO 冒烟实验。
- `scripts/launch_pi_grpo_smoke.sh`：带退出码、起止时间和完整日志的后台实验启动器。
- `scripts/run_pi_grpo_megatron_tp4_pp2_cp2.sh`：16-NPU Megatron TP4/PP2/CP2 全参轨迹 GRPO 配置。
- `scripts/launch_pi_grpo_megatron_smoke.sh`：Megatron 单步实验的日志、时间和退出码启动器。
- `scripts/run_pi_formal_50step.sh`、`scripts/launch_pi_formal_50step.sh`：保留原 50-step 正式入口。
- `scripts/run_pi_formal_100step_12groups.sh`、`scripts/launch_pi_formal_100step_12groups.sh`：固定 `4 groups/update × 4 responses`、12 个在途 groups、100 步、仅第 100 步验证与保存；同样只接受 full、已审核、哈希完整的 boss-aligned train/val。
- `scripts/prepare_pi_step100_resume_view.sh`、`scripts/run_pi_formal_step100_to_step200_12groups.sh`、`scripts/launch_pi_formal_step100_to_step200_12groups.sh`：从现有 step-100 完整模型/RNG 恢复到累计 step-200，保持 12-group 正式配置，新增恰好 100 次更新；因原 checkpoint 未保存 Adam 状态而显式重置 optimizer，并因 train237 修正为 train236 而丢弃旧 dataloader 游标。
- `scripts/replay_dense_correctness_gate.py`、`scripts/run_pi_dense_correctness_step100_to_step120.sh`、`scripts/launch_pi_dense_correctness_step100_to_step120.sh`：在前后200步的3,200条轨迹上审计连续正确性组内排序，并从Step 100执行20步、30%候选奖励、仅末步验证/保存的隔离试验。
- `scripts/run_unattended_accuracy_pipeline_host.sh`：从Step 120自动执行12题三条件oracle诊断、3,200条分层奖励回放、`2 groups × 8 responses`五步金丝雀，并且只在老板准确率、mixed-correct、完成率、过程分与checkpoint门禁全部通过时续跑20步；失败即停止并释放Ray资源。
- `scripts/run_pi_banded_2x8_resume.sh`、`llin_verl/pi_reward.py::compute_score_banded_v1`：把无答案、错误答案、SQL正确但综合错误、最终答案正确划入不重叠奖励区间；8条同prompt候选全部用于GRPO，不做最快样本选择。
- `scripts/launch_v15_dwh_gate_after_baseline.sh`：等待冻结 val20 成功退出后自动启动 5-step GRPO；基线失败时阻断训练并记录监督状态。
- `scripts/check_formal_data_on_ray.py`：正式运行前分别在 `llin_trainer` 和 `llin_rollout` Ray 节点计算 train/val 文件大小与 SHA256，任一节点缺失或内容不一致即在模型加载前失败。
- `scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh`：TP4/PP2/CP2 训练、TP8/DP2 rollout 的 bounded fully-async 配置，按完整 GRPO group 入队并以 queued tokens 做背压。
- `scripts/patch_verl_fastest_k_oversampling.py`：给 fully-async AgentLoop 增加可配置候选过量采样、最快 K quorum、完整 GRPO group 选择和逐请求 vLLM 取消链路。
- `scripts/patch_verl_fastest_k_abort_observability.py`、`scripts/patch_verl_fastest_k_abort_retry.py`：区分无活跃请求、服务端确认、自然完成、重试耗尽与取消失败，并关闭 Fastest-K 取消注册竞争。
- `scripts/patch_verl_vllm_abort_api.py`、`scripts/patch_verl_abort_partial_tokens.py`：修复 vLLM 0.18 external/internal request ID 混用，物理终止前读取无内容的部分 token 计数，再汇总到逻辑轨迹。
- `scripts/monitor_npu_utilization.py`、`scripts/monitor_vllm_cache_metrics.py`：两机 NPU 稳态利用率与两路 vLLM prefix-cache 计数采样。
- `scripts/analyze_grpo_steady_state.py`：汇总 20-step 稳态耗时、长尾、NPU 利用率和 cache 命中率，并输出 fully-async 切换判据。
- `scripts/analyze_trajectory_comparison.py`：只读扫描老板轨迹、同源 converted 轨迹、本次 320 条 rollout 与 20-step 日志，输出可复查的长度和超时统计。
- `scripts/analyze_fastest_k_efficiency.py`：解析预热、队列等待、actor 更新、Fastest-K quorum/丢弃/abort、staleness 和严格奖励 replay。
- `scripts/build_fastest_k_efficiency_report.py`：从聚合摘要生成 canonical report artifact，原始轨迹、日志与 checkpoint 不进入报告载荷。
- `scripts/estimate_48k_capacity.py`：依据已验证的 6K 实测峰值、Qwen3.6 64 层混合 GDN/全注意力结构及 TP/PP/CP 切分，估算 48K 训练激活和 rollout KV/GDN cache 容量。
- `llin_verl/megatron_bridge_compat.py`、`scripts/patch_verl_megatron_bridge_compat.py`：为昇腾验证版 Megatron-Bridge 补齐当前 veRL 所需的最小兼容接口。
- `scripts/prepare_repair_sft_smoke_dataset.py`、`scripts/check_repair_sft_dataset.py`：生成不可晋升的确定性工具调用合成样本，并在占用 NPU 前检查 assistant loss 与非 assistant 上下文遮罩。
- `scripts/qwen36_assistant_mask_sft_dataset.py`：通过 veRL 官方 `data.custom_cls` 扩展点用 Qwen3.6 完整对话模板构造 assistant-only SFT loss mask。
- `scripts/run_repair_sft_megatron_smoke.sh`、`scripts/launch_repair_sft_megatron_smoke.sh`：固定 Step 120 模型态初始化、TP4/PP2/CP2 和 extra-only 保存的一步官方 veRL SFT 冒烟入口。
- `scripts/prepare_repair_sft_dataset.py`、`scripts/run_repair_sft_train236_overfit.sh`：从 train236 审核资产中构建并机械核验 16 条单次 SQL 纠错轨迹，再从 Step 120 执行 5 步官方 veRL SFT。
- `scripts/run_repair_sft_replay.sh`、`scripts/run_repair_sft_prepost_pipeline_host.sh`：以完全相同的老板四工具、48K/25 工具回合分别回放 Step 120 与 SFT Step 5，自动回收结果、调用老板原始评分器并写入配对门禁。
- `scripts/teacher_forced_component_masks.py`、`scripts/qwen36_teacher_forced_diagnostic_dataset.py`、`scripts/run_teacher_forced_component_diagnostic.py`：把 assistant 监督严格拆为工具结构、SQL shell payload 与最终答案，在 veRL/Megatron forward-only 模式中输出逐题 NLL 和目标概率，不初始化 optimizer。
- `scripts/run_repair_sft_teacher_forced_eval.sh`、`scripts/run_repair_sft_teacher_forced_prepost_host.sh`：在相同数据、TP4/PP2/CP2 下自动比较 Step 120 与 SFT Step 5，执行 16/16 token mask 重建门禁并仅回收聚合结果。
- `scripts/analyze_repair_sft_free_run_divergence.py`：在服务器侧离线对齐教师轨迹与老板原始自由回放，统计第一条 SQL 分叉、目标 SQL 后续命中和正确证据后的继续查询，不输出原始敏感内容。
- `scripts/analyze_repair_sft_first_query_semantics.py`：只读执行两份自由回放的首条 SQL，区分 gold 支持、空结果、执行失败和错误/不足证据，并用教师查询结果排除机械等价 SQL。
- `scripts/analyze_repair_sft_all_query_semantics.py`：只读执行回放中的全部 SQL，分别统计前 1/2/3 条及任意位置首次获得 gold 支持或教师结果等价证据的任务数。
- `scripts/analyze_state_recovery_semantics.py`：在服务器侧比较首错/纠正 SQL 的表、join、时间、聚合、过滤、select 与排序差异，并与 semantic-mask v3 的首个非 greedy token 家族交叉；仅输出查询哈希和类别。
- `scripts/teacher_forced_token_ranks.py`：在 TP 词表分片上计算教师 SQL token 的精确 rank，并定位首个非 greedy 关键 token，不收集完整 logits。
- `scripts/qwen36_sql_weighted_sft_dataset.py`、`scripts/check_sql_weighted_sft_dataset.py`、`scripts/run_repair_sft_sql_weighted_canary.sh`：构造和 CPU 核验 SQL 加权 loss mask，并从 Step 120 启动仅一步、单变量、只保存最终模型的金丝雀。
- `scripts/prepare_state_conditioned_repair_sft.py`、`scripts/check_state_conditioned_sft_dataset.py`、`scripts/run_repair_sft_state_conditioned_canary.sh`：从 Step 120 首错 SQL 和真实工具结果构造零-loss 上下文，机械核验纠正查询并执行一步状态条件化金丝雀。
- `scripts/prepare_critical_token_recovery_sft.py`、`scripts/qwen36_critical_token_sft_dataset.py`、`scripts/check_critical_token_sft_dataset.py`、`scripts/run_repair_sft_critical_token_canary.sh`：冻结 semantic-mask v3 的逐题首个非 greedy SQL token，核对 token ID/offset 并只把该 token 从 `8×` 提到 `32×`。
- `scripts/analyze_critical_token_canary.py`：在哈希一致的 Step 120/训练后 forward-only 结果中复核冻结 token 是转为 greedy、仍为首个非 greedy，还是被更早的新分叉阻断归因，并将整条 SQL 概率门禁作为唯一 replay 开关。
- `scripts/prepare_semantic_plan_sufficiency_gate.py`、`scripts/check_semantic_plan_sufficiency_gate.py`：将同一 16 条 Step 120 首错 SQL 与真实工具结果复制为 Control、operator oracle、full semantic plan 三臂；CPU 门禁逐题核对相同基态、bash-only、只读首错 SQL，以及 oracle 不泄漏原 SQL、结果、答案或字面量。
- `scripts/run_semantic_plan_sufficiency_gate.sh`、`scripts/prepare_semantic_plan_gate_outputs.py`、`scripts/analyze_semantic_plan_sufficiency_gate.py`：一次加载 Step 120 后对 48 行做贪心单助手回合生成，在工具执行前停止；随后只读执行生成 SQL，以 gold 支持或教师结果等价判定恢复，并按冻结阈值自动选择下一训练目标。
- `scripts/prepare_semantic_delta_margin_gate.py`、`scripts/qwen36_semantic_delta_margin_dataset.py`、`scripts/run_semantic_delta_margin_gate.sh`：在相同首错状态下配对机械正确 correction SQL 与模型实际首错 SQL，只对 token edit span 做 Step 120 forward-only 概率 margin；同时精确重建冻结的首个 non-greedy token，作为 chosen-vs-rejected 训练前的无回退门禁。
- `scripts/run_semantic_delta_pairwise_training.py`、`scripts/run_semantic_delta_pairwise_canary.sh`、`scripts/run_semantic_delta_pairwise_pipeline.sh`：在固定顺序、每个 microbatch 一对候选的条件下，对 semantic-delta token 的长度归一化 log-probability 施加 reference-free logistic ranking loss；只做一步全参更新，随后自动重跑同一 margin 并执行 `12/16 + 12/16 + 零更早回退` 门禁。
- `scripts/run_native_repair_replay.sh`、`scripts/analyze_native_training_attribution.py`：以原生 HF 权重在同 16 题上执行 48K 老板四工具自由回放，并把相同状态 margin 与老板原版评分拆成“训练前已存在”和“训练后是否放大”两层归因。
- `scripts/analyze_disjoint_pair_candidate_pool.py`：从老板当前权威任务 manifest 重建 train236 的当前 instruction/gold 身份，在 immutable SQLite 上机械核验，并按 strict/review-required/blocked/forbidden-overlap 分桶；安全汇总不输出原始 prompt、SQL、gold 或工具结果。
- `scripts/prepare_disjoint_pair_rollout_candidates.py`：只接受上述审计中 strict-available 的 48–64 条身份，把当前权威 instruction/gold 重建回老板 system/tool 格式的推理 Parquet；输出仅供 Step 120 首错采集，显式禁止直接训练或模型晋级。
- `scripts/prepare_disjoint_first_error_pairs.py`：只把 Step 120 实际生成且机械错误/不足的首条只读 SQL 连同真实工具结果作为零-loss 状态，配对当前定义下已验证的 chosen SQL；正确/等价首查和无 SQL 题被排除，至少 48 对前继续禁止训练。
- `scripts/check_disjoint_first_error_pairs.py`、`scripts/run_disjoint_pair_margin_gate.sh`、`scripts/analyze_disjoint_pair_margin.py`：对实际 48–64 对数据动态核验 chosen/rejected 邻接、delta mask、候选符号与序号，并以 Step 120 forward-only 统计正确 SQL margin、75% 偏好阈值及首个非 greedy token 家族；全程无 optimizer/checkpoint。
- `scripts/prepare_disjoint_first_error_evaluation.py`、`scripts/run_disjoint_real_state_eval22_margin.sh`、`scripts/launch_disjoint_real_state_eval22_margin.sh`：把不足训练门槛但已机械验证的 22 对状态单独冻结为 evaluation-only，强制禁止训练与 promotion，并可对任意指定 Megatron checkpoint 做相同数据、相同 mask 的 TP4/PP2/CP2 纯前向评分。
- `scripts/analyze_disjoint_real_state_evaluation.py`、`scripts/compare_disjoint_real_state_evaluation.py`：复算 semantic-delta/full-SQL margin、Wilson 区间与 token 家族聚合，并按预注册的 `17/22 + 18/22 + 零回退` 比较 Step 120 和未来候选；失败只允许停止，不能开放训练或 full64。
- `scripts/validate_disjoint_real_state_eval22_artifacts.py`、`scripts/validate_disjoint_real_state_eval22_candidate.py`：从原始 Parquet、CPU token gate、两次 NPU 诊断、实验合同和比较器独立复核行粒度、哈希、指标与执行安全，再输出不含敏感样本内容的 Git 安全汇总。
- `scripts/prepare_native_disjoint_first_error_candidates.py`、`scripts/run_native_disjoint_pair_margin.sh`、`scripts/validate_native_disjoint_pair_artifacts.py`：按源任务身份排除五类冻结资产，构造原生真实首错候选，并以Step120做无optimizer/无checkpoint纯前向筛查和独立安全验收。
- `scripts/prepare_disjoint_pair_review_pilot.py`、`scripts/audit_review_pilot_query_stability.py`：在服务器内生成权限`0600`的42条语义审核包，先做SQLite反向无序扫描稳定性探针；机械稳定不等于语义批准，训练和rollout始终保持关闭。
- `scripts/analyze_disjoint_pair_review_pilot.py`：在服务器侧把去标识人工决策与原始packet、隐藏task identity和SQLite稳定性证据逐项联结，敏感裁决保持`0600`，只输出不含题意、SQL、gold值或task ID的安全聚合。
- `scripts/analyze_disjoint_pair_review_supply_rebase.py`：从去标识42条决策、本地pilot汇总和v3供给策略重算语义批准率；用批准率与条件pair产率的复合beta-binomial预测判断是否继续旧队列，并fail-closed输出新任务容量目标。
- `scripts/analyze_native_disjoint_pair_supply.py`：在服务器侧重建原生full25逐题首查结果，只输出排除eval22后的真实错误状态数量和类别；不输出身份或样本内容，也不把候选状态自动视为训练pair。
- `scripts/analyze_disjoint_training_supply_strategy.py`：在eval22训练禁用前提下重算48对供给缺口，以 `22/64` 观测产率的Jeffreys后验beta-binomial预测量化review队列成功概率，并将下一动作锁定为原生状态回收、纯前向margin和42条低风险语义审核pilot。
- `scripts/run_disjoint_pairwise_canary.sh`：只有不重叠 pair 数、CPU token gate 和 Step 120 margin 三门均通过时，才把实际 48–64 对作为一个完整 global batch 做一次 reference-free pairwise 更新；只保存 model+extra，随后必须回到原冻结 16 题做概率门禁。
- `scripts/analyze_rollout_command_families.py`：以不输出命令、SQL、prompt 或工具结果的方式统计工具类型、Bash 命令族、重复调用与真实工具响应覆盖，用于区分模型工具策略问题和 SQL 解析器漏识别。
- `scripts/prepare_query_initiation_oracle_candidates.py`、`scripts/analyze_query_initiation_oracle_outcomes.py`、`scripts/analyze_query_initiation_oracle_gate.py`：只抽取 Step 120 完整 25 回合仍未发起只读查询的题目，追加不含答案、表名、字段名、SQL 或字面量的通用查询启动约束；专用适配器核对 Parquet 哈希、3/3 回合和训练关闭合同，再以带真实工具结果的只读查询恢复数区分策略路由与 schema/工具实现问题。
- `scripts/prepare_structured_sqlite_realization_gate.py`、`scripts/analyze_structured_sqlite_realization_gate.py`：把同一 41 题通用干预替换为冻结的三回合非交互 `path→schema→SELECT/WITH` 工作流，禁止交互 shell 和重复命令；仍不提供 task-specific schema、query 或答案，并以 `31/41` 带结果查询决定运行时约束是否足够。
- `scripts/prepare_schema_oracle_action_gate.py`、`scripts/analyze_schema_oracle_action_gate.py`：对完整 64 个不重叠严格任务，从 immutable SQLite metadata 提取 gold SQL 涉及表的列类型与选中表间外键，不提供数据行、工具结果、expected value、gold SQL 或答案；第一回合动态定位数据库并执行非交互 SELECT/WITH，真实结果返回后第二回合禁止再调用工具，以 `32` 条正确/等价或 `48` 条带结果错误查询分别决定无 gold 表选择的全库 schema 验证与同状态 pair 构建。
- `scripts/prepare_chosen_only_schema_action_sft.py`：在 schema-oracle 两门失败后，把同一完整不重叠 64 题转换为仅监督一个正确 bash/SQLite 首动作的 chosen-only 数据；gold SQL 只进入 assistant 标签，prompt 不含 SQL、答案、expected value、数据库行或工具结果，并固定拆分为 48 条拟训练和 16 条校准，CPU tokenization/loss-mask 门通过前禁止 teacher-forced NPU 基线和训练。
- `scripts/qwen36_first_action_diagnostic_dataset.py`、`scripts/check_chosen_only_schema_action_sft.py`：用 Qwen3.6 完整工具 chat template 对 chosen-only 样本做 CPU fail-closed 门禁；要求 loss 恰好覆盖唯一 assistant tool action，system/user 全部为 0，并把 action 无重叠地拆成 tool structure 与解码 SQL 内容两部分。
- `scripts/run_chosen_only_first_action_teacher_forced.sh`、`scripts/launch_chosen_only_first_action_teacher_forced.sh`：只在 CPU 门通过后，对 calibration16 执行 Step 120 的 TP4/PP2/CP2 teacher-forced 纯前向基线；复用精确词表并行 SQL token rank，不初始化 optimizer、不保存 checkpoint，也不读取 train48。
- `scripts/analyze_chosen_only_first_action_baseline.py`：联合核对 64 条数据合同、CPU tokenization gate、calibration16 Parquet 哈希和 Step 120 forward-only 结果，只在正确 SQL 仍存在明确 token 排序缺口时开放“一步、train48-only、新 Adam、model-only checkpoint”金丝雀，并冻结 post-canary NLL/rank/退化阈值。
- `scripts/run_chosen_only_first_action_one_step.sh`、`scripts/launch_chosen_only_first_action_one_step.sh`、`scripts/analyze_chosen_only_first_action_post_canary.py`：严格执行获准的一步 train48 `0.25/8` 全参 SFT，并在相同 calibration16 上按预注册的 aggregate/per-task NLL、greedy/top-5、mean rank、tool structure 和更早分叉门自动决定是否只开放一次自由回放；无论结果如何都禁止追加训练和 promotion。

## 已验证状态

### v1.12.3 — 2026-08-21

- Qwen3.8 Step70 mixed27 四次曝光正式训练完成：108 个 GRPO 组、864 条新鲜轨迹和 54/54 次更新全部完成，监督退出码为 0，总墙钟为 9 小时 31 分 58 秒；训练端与 rollout 端结束后均已释放。
- 唯一 `global_step_54` 通过最终 checkpoint 门禁：Megatron distributed checkpoint 含 32 个模型分片且元数据完整；未保存 optimizer，6 道 sealed 数据既未进入训练也未在本次运行中评测，因此继续禁止自动晋级。
- 新增安全完成报告与机器可读结果，仅记录聚合配置、时长、末步观测和 checkpoint 完整性，不包含题面、gold SQL、任务身份、工具输出或服务器路径。

### v1.12.2 — 2026-08-20

- 将Step70 HF来源门从多层SSH下容易丢失引号的内联Python改为独立失败关闭程序，明确验证导出manifest、`verification.valid`、期望policy step、config与权重索引；双机55.56GB/1,199张量兼容扫描已通过，旧入口在Ray与NPU启动前失败。
- 双机监督器清理钩子现在会为任意数据、模型、资源、Ray或HCCL预检失败统一写入`failed`、真实`exit_code`和结束时间，不再让失败运行的状态停留在旧阶段。

### v1.12.1 — 2026-08-20

- 修复Step70 mixed27宿主包装器误用容器内项目路径的问题：包装器现在从`HOST_PROJECT_ROOT`执行双机监督器，同时继续把`CONTAINER_PROJECT_ROOT`传给容器内数据、模型和训练入口。首次启动在NPU、Ray、模型加载和rollout之前失败，未产生训练状态或残缺checkpoint。

### v1.12.0 — 2026-08-20

- 将Step70同策略严格mixed池冻结为`27 train + 6 sealed`：来源为原70题严格复测15道与1,430道困难留出集18道；密封集按v15/v20/v21固定为`2/3/1`，与训练身份零重叠且始终`training_allowed=false`。
- 正式续训采用每题4次曝光、每组8条、每步2组，共108组、864条新鲜on-policy轨迹和54次更新；拒绝每题10次的2,160条/135步方案，避免在27道小池上增加约1.93倍于上一轮的采样量和记忆风险。
- 新增Step70 HF来源门、27×4私有数据冻结器、可参数化双机启动器和最终Step54唯一checkpoint门；保持TP4/PP2/CP2训练、TP4/DP4 rollout、53,248上下文、30分钟完整轨迹超时、`reasoning_effort=medium`、`banded-v2-strict-table-v1`与只保存最终模型。

### v1.11.99 — 2026-08-20

- Qwen3.8 Step0新鲜数据正式运行`-06`已进入三机采样：v22整版500题只冻结评测，v23–v26共2,000题按`727/727/546`分给5/6/0号机；拓扑为`TP4×DP4 + TP4×DP4 + TP4×DP3`，共44张NPU、176个物理序列槽位。
- pilot100已按`35/35/30`下发，各机Level 1–5完全均衡；11个DP引擎均出现真实prompt与generation吞吐，16/16/12个VLLM worker在位，启动检查未见placement-group竞态、OOM或traceback。严格`2+2+2`、候选额外2条确认、`banded-v2-strict-table-v1`和排队时间不计入30分钟轨迹超时保持不变。
- 启动前空闲门禁发现并阻断上一失败轮遗留的5/6号机各16个VLLM孤儿worker，按专用容器和核验PID完成清理后才启动本轮。正式清理链新增Ray停止后的`VLLM::`孤儿二次回收，并为正在运行的`-06`配置独立收尾守护，避免成功或失败退出后依赖人工释放NPU。
- 安全启动证据写入`docs/qwen38_fresh_v23_v26_launch_20260820.safe.json`；只包含拓扑、任务计数、吞吐范围和布尔门禁，不包含题面、gold、SQL、任务身份、哈希、工具输出或服务器路径。

### v1.11.98 — 2026-08-20

- 修复Qwen3.8在线早停仍沿用legacy数值包含判定的问题：v23–v26的每一波现在强制使用`banded-v2-strict-table-v1`，表格行序、类别和值必须精确绑定；结果文件若缺少严格合同会fail closed，超时不再计作完整错误答案。
- 新增初步mixed候选的两条独立确认轨迹，只有累计至少2条严格正确、2条完整严格错误且runtime error为0的题进入robust候选池；所有候选继续保持`training_allowed=false`，达到24题后才允许12步金丝雀的数据门继续评估。
- 新增v22整版500题冻结、v23 pilot100与v23余下400题/v24–v26共2,000题三机均衡分片器，以及5/6/0号机各16卡`TP4×DP4`无人值守总控。总控会校验私有分片和运行环境哈希、记录pilot100实测速率/ETA、汇总难度分布，并在失败或结束时清理专用队列与Ray。
- 回归门确认30分钟超时从Agent实际执行开始，准入排队单独记账；针对严格奖励、候选确认、三机均衡分片和总控合同的29项测试全部通过。
- 首次总控在占用NPU前被宿主Python 3.13/pyarrow 25写出的损坏嵌套Parquet阻断；数据准备改到已验证的Qwen3.8容器Python 3.11/pyarrow 24执行，宿主总控只做哈希校验和编排，失败产物不复用。
- 容器内准备进一步避开只读`/pi_sandbox`挂载：五版运行投影先生成到项目内可写暂存区，再由宿主逐文件哈希同步到5/6/0号机沙箱根；只读挂载故障同样发生在模型加载前，未产生采样轨迹。
- 0号机16卡旧rollout容器没有Qwen3.8挂载，正确的专用Qwen3.8容器暴露12张空闲卡；最终效率拓扑修正为5/6号机各`TP4×DP4`、0号机`TP4×DP3`，按物理序列容量`64:64:48`把2,000题重分为`727/727/546`，避免12卡节点成为等量分片尾部瓶颈。
- 修复standalone DP副本并发创建placement group时的资源检查竞态：每个TP4副本各自检查“当前空闲NPU”会被其他副本刚完成的预留干扰，5/6号机因此误报`available=0`，而0号机因调度时序不同已实际生成。运行入口已有可见NPU数与`TP×DP`硬门，角色绑定层改由Ray placement group做原子总容量门，不再进行逐副本瞬时空闲检查。

### v1.11.97 — 2026-08-20

- 综合严格奖励回放、原70题复测和1,430题留出评测，冻结Qwen3.8 Step70为取证模型，禁止从其续训或把70+18滚入训练；下一轮从Qwen3.8原生Step0启动24题/12步金丝雀。
- v22整版500题冻结为新同seed评测集，v23–v26共2,000题用于三机严格`2+2+2`筛选；初步候选额外补2条，只有至少2条严格正确和2条完整严格错误、且runtime error为0的题可进入金丝雀。排队等待与30分钟实际轨迹超时彻底分离，超时不得冒充错误奖励制造方差。
- 模型质量主指标改为冻结题集严格task pass@K和“严格正确/全部请求”，mixed题数只用于训练前数据选择；5号机训练、6号机rollout、0号机并行采样/评测，廉价100题门通过后才跑v22全500题，全部通过后才扩到32题以上。
- 新增可复现计划聚合器、3项回归测试、安全决策记录、可读计划报告和经数据合同校验的报告源；便携HTML渲染因本地共享reader对新旧artifact均复现`reader_timeout`而未发布，Markdown报告不受影响。

### v1.11.96 — 2026-08-20

- 用相同`banded-v2-strict-table-v1`对原生Qwen3.8历史70题和Step70新采样做私有身份交集：原生严格mixed基线实际为20题而非70题；Step70保留其中10题、丢失10题，并从原先非严格mixed中新增5题，最终为15题，原生严格集合保留率50%。
- 原生历史轨迹为286条观测/234条完整/22条严格正确（9.40%），Step70为390/366/15（4.10%）；Step70在完整试次更多时严格正确条数仍下降，正确率点估计减少5.30个百分点、相对减少56.41%。由于两轮不是逐请求同seed配对，结论限定为“当前点估计明显变差”，禁止据此继续滚动训练。
- 新增不泄露身份的三机集合比较器、合成门禁测试和安全结果`docs/qwen38_native_vs_step70_original70_strict_20260820.safe.json`；私有70题、20/15行严格候选及轨迹仍不进入Git。

### v1.11.95 — 2026-08-20

- 原70题Step70三机`2+2+2`复测已完成390条轨迹：旧口径mixed为21题，按2/4/6条分别新增`4/7/10`题，剩49题未形成方差；难度分布为Level 1–5分别`1/6/9/3/2`，三机runtime error均为0。
- 严格`banded-v2`离线复算器补齐容器内`PYTHONPATH`，修复主采样成功后奖励模块无法导入导致的独立收尾失败；新增命令合同测试，确保三机严格复算都从冻结项目树加载`llin_verl`，不会依赖容器当前目录。
- `banded-v2-strict-table-v1`覆盖70/70题与390/390条观测轨迹，366条完整可评分；宽松legacy包含式启发有37道mixed，严格表格绑定仅保留15道（v15/v20/v21=`4/9/2`，难度2/3/4/5=`2/8/3/2`），22道被严格判定过滤。安全结果已固化到`docs/qwen38_step70_original70_replay_20260820.safe.json`，15行私有候选继续保持禁训、禁晋级。

### v1.11.94 — 2026-08-20

- 原生Qwen3.8 Step0与GRPO Step70已在完全相同的v15 `488`题和v20 `461`题上完成等口径复算；三组三机无序身份指纹逐版一致，采样均为`2+2+2`、最多6条、`reasoning_effort=medium`，两轮runtime error均为0。
- 最多6条的任务pass率：v15由`0/488`升至`5/488`（`0→1.02%`），v20由`2/461`升至`11/461`（`0.43%→2.39%`）；合计由`2/949`升至`16/949`（`0.21%→1.69%`，8倍）。固定首2条仅由`2/1,898`升至`3/1,898`，说明主要增益出现在第3–6条采样，绝对正确率仍低。
- 这949题是排除原生12道v15和39道v20 mixed训练候选后的原生难题留出集，因此结论是“同一困难集上的训练后恢复”，不是完整500题总体准确率。新增安全对比`docs/qwen38_native_vs_step70_heldout_v15_v20_20260820.safe.json`与可复现只读聚合器`scripts/compare_qwen38_native_step70_heldout.py`；均不含题面、gold SQL、逐题身份、工具输出或服务器路径。

### v1.11.93 — 2026-08-20

- 修复后的三机`-05`评测持续健康，0/5/6号机退出码均为空；三机v15与v20均已完成，当前全部进入v21第一波每题2条采样。48张NPU保持实际解码负载，无昨夜启动故障复发。
- v15的488题产生2,916条轨迹、5题进入mixed方差候选，较固定6条节省12条；v20的461题产生2,756条轨迹、11题进入候选，节省10条。已完成949题合计5,672条轨迹、16题候选；这些只是评测难度/方差信号，仍保持`training_allowed=false`，不得自动并入训练。
- 新增`docs/qwen38_step70_heldout_progress_20260820.safe.json`，仅记录三机拓扑、聚合数量和安全状态，不含prompt、gold SQL、任务身份、工具输出或服务器路径。

### v1.11.92 — 2026-08-19

- `-04`失败清理进一步暴露生命周期缺口：队列由`docker exec -d`独立启动，不属于Ray进程树；旧监督器只执行`ray stop`，队列内置重试仍可重新创延actor，导致监督器退出后三机残留vLLM并占用约54GiB/卡。三个专用`llin`容器已重启，主机挂载的模型、数据和失败审计目录均保留。
- 监督器最终清理改为先按精确评测名终止三机队列及其子进程，再停止Ray；匹配模式不会命中清理命令自身。无论模型校验、运行包同步、启动、队列或聚合在哪一阶段失败都会执行队列清理，避免重试与Ray回收竞态。

### v1.11.91 — 2026-08-19

- 冻结运行包对齐后，三机均成功进入v15 wave2；5/6号机首次独立TP4×DP4建池暴露veRL资源检查竞态：`RayResourcePool`先立即预留16张NPU，随后`_check_resource_available`按剩余资源读取到0并误报容量不足，自动重试又被首轮遗留placement group阻塞；0号机因调度时序不同已进入模型加载。
- 角色固定补丁改为建池前检查总容量、检查通过后再创建placement group，实际调度仍由Ray作为最终资源门；冻结运行包扩展为21文件并包含`runtime/sitecustomize.py`，防止只同步队列脚本而遗漏资源补丁。失败运行保留审计，新运行使用全新目录和干净Ray，禁止混合半成品。

### v1.11.90 — 2026-08-19

- 0号机模型校验修复后，三台Ray均成功启动，5/6号机进入v15第一波；0号机仍因三级采样脚本版本落后、不识别Step70模型元数据而在生成前退出，监督器检测到单边失败后按合同停止三机，避免继续产生版本不一致轨迹。
- 三机接力新增20文件冻结运行包：在启动Ray前把队列、三级采样、独立rollout、分析、监控和veRL补丁脚本精确同步到6/0号机，并逐文件SHA256对齐后才开放启动；远端部署树是否为Git仓库或是否预装最新版脚本不再影响评测合同。

### v1.11.89 — 2026-08-19

- 三机Step70留出接力在0号机模型复制完成后、任何评测轨迹生成前失败：0号机部署树缺少新加入的模型传输校验脚本，旧实现错误地假设每台rollout主机都具有与5号机同步的工程脚本。
- 模型传输现在把校验器与安全清单一起原子式投递到每台远端主机的监督目录，并只从该冻结副本执行逐文件SHA256校验，不再依赖远端工程目录版本；已复制的53GiB模型通过后可直接续跑，无需重新传输或生成轨迹。

### v1.11.88 — 2026-08-19

- 首次两机Step70留出队列在任何题完成前因子进程未注入项目根目录、无法导入`scripts.adaptive_dwh_wave_earlystop`而双边退出；模型导出及5→6号机56,504,635,235字节/27文件传输均已通过，失败未产生评测轨迹。
- 用户授权加入0号机后，实查三机共48张NPU全部空闲；评测改为三台独立`TP4×DP4`、每副本并发16，总副本由8增至12。1,430道留出题从权威1,500题重新机械排除70道训练题，训练重叠为0，并按难度平衡为5/6/0号机`477/477/476`。
- 三机9个私有分片已跨机逐文件SHA256一致；队列入口显式注入项目根目录，并在每次启动时仅清除自身旧`exit_code`。新接力器复用已验证Step70导出，当前正在向0号机复制并校验模型，随后自动启动三机严格`2+2+2`评测。

### v1.11.87 — 2026-08-19

- Qwen3.8正式70步训练已完成，监督器退出码为0且唯一保存的`global_step_70`通过Megatron distributed checkpoint完整性门禁。
- 训练后评测先后暴露两个独立启动配置问题：新评测运行名下未携带已冻结的1,430题数据目录，以及导出子进程未继承固定Megatron Bridge源码路径；两者都在模型转换/评测前fail-closed，没有生成可用性未验证的模型或轨迹。
- 已从8月18日冻结资产原样恢复数据，双机六个分片与安全清单SHA256逐项一致、训练交集仍为0；模型导出显式注入训练同源的Bridge/runtime路径。监督器还会在等待训练前预检冻结数据并在训练完成后复检，缺数据或跨机哈希不一致将立即失败。

### v1.11.86 — 2026-08-19

- Qwen3.8正式训练`-06`完成5/70步后，取消/超时轨迹同时缺少`min_global_steps`与`max_global_steps`，批次组装在计算参数版本跨度时触发`None-None`并失败；Step70门禁正确阻止残缺模型和留出评测继续运行。
- 失败轨迹合同扩展为全元数据fail-closed：缺少rollout log-prob或任一参数版本时统一保留组形状但将response mask清零；批次统计把缺失版本成对归一到同批最近有效版本，避免统计与陈旧轨迹计数崩溃，同时不赋予失败轨迹任何策略梯度。
- 5/6号机Ray启动入口均强制应用agent-loop与detach-utils双补丁；新增V2原位升级、单边/双边版本缺失和全缺失批次回归测试。修复后从Qwen3.8原生基座干净重启，不恢复`-06`的未保存在线状态。

### v1.11.85 — 2026-08-18

- Qwen3.8正式训练`-03`在第4步遇到一条超时轨迹的`response_logprobs=None`；旧异步队列把生成任务异常误转为正常终止并返回0，实际仅写出`global_step_4`，不得当作70步成品。新增缺失log-prob轨迹的零response-mask/零loss占位，保持8条group形状但不伪造行为策略概率；`-04`在进入优化前因补充完整checkpoint格式而主动停止，最终修复后的`-05`从原生基座重新启动。
- 正式训练强制保存Megatron distributed model checkpoint，避开PP2在线HF导出缺张量；监督器新增硬性Step70完成门：只有退出码0、checkpoint目录仅含完整`global_step_70/actor`、model格式正确且metadata/shard存在时才可写成功。训练后评测监督器独立重复该门，并要求Ray清理完成和训练监督器进程退出。
- 冻结v15/v20/v21中未参与训练的`488/461/481=1,430`题，70个训练identity全部精确排除且交集为0；按难度平衡为5/6号机各715题。Step70严格导出和逐文件SHA256跨机复制后，两机各以TP4×DP4、每副本并发16执行`2+2+2`，`reasoning_effort=medium`、94,208上下文和30分钟全轨迹超时保持不变；训练失败或模型/数据校验失败时禁止启动评测。
- 服务器侧无人值守链已部署并跟随修复后的`-05`：当前评测状态为`waiting_for_training`，不会读取`-03`的Step4残留或已停止的`-04`。新增[训练后留出评测安排](docs/qwen38_step70_heldout_eval_20260818.md)和[安全状态](docs/qwen38_step70_heldout_eval_20260818.safe.json)。

### v1.11.84 — 2026-08-18

- Qwen3.8正式训练`-03`通过两机运行环境/奖励入口、Ray和1→16 HCCL门禁；首次`2,560 MiB`参数同步14.19秒，4个完整计分预热组含302,909 tokens、727.59秒完成，证明Agent工具链与`banded-v2`奖励端到端可用。
- 第1个优化步已完成：actor更新187.34秒、整步196.88秒；参数版本1再次同步仅9.20秒，rollout资源利用率78.8%，并已进入第2步且队列仍有1组。正式70步运行现已越过模型、推理、奖励、队列、反向更新和更新后同步全链路门禁。

### v1.11.83 — 2026-08-18

- Qwen3.8正式训练`-02`以`2,560 MiB`桶在13.3秒内完成首次参数同步，证明上一轮embedding故障已修复；随后6号机过期的奖励模块缺少`compute_score_banded_v2`，6条预热请求在计分前安全退出，消息队列保持0且没有执行优化步。
- 运行时门禁扩展为在5/6号机分别导入精确奖励模块并验证目标入口可调用，失败发生在模型加载和Ray启动前；同步奖励实现后的全新重试使用`-03`，不从`-01/-02`恢复。

### v1.11.82 — 2026-08-18

- 检出Qwen3.8正式训练首次尝试在首轮参数同步前安全退出：数据、沙箱、模型、Ray/HCCL、训练器和4个TP4推理副本均已通过，但`512 MiB`同步桶无法容纳不可拆分的`2,425 MiB` BF16词嵌入张量；失败发生在第一个rollout和第一个优化步之前，因此没有轨迹或参数更新可被误用。
- 正式包装器恢复`2,560 MiB`权重同步桶，并新增启动前下限门禁，任何小于`2,560 MiB`的覆盖都会在分布式资源启动前拒绝。失败的`-01`目录保留为审计记录，干净重试使用`-02`且禁止从失败目录恢复。

### v1.11.81 — 2026-08-18

- 按所有者明确授权冻结Qwen3.8原生70题正式GRPO：每题两次、每组8条全保留、2组/步，共140组和70次更新；使用`banded-v2-strict-table-v1`，发布权限继续关闭。
- 三机286条历史轨迹的实际最大prompt为1,289 tokens，234条正常完成轨迹的最长response为43,401；正式上下文锁定为`4,096 + 49,152 = 53,248`，比最长完成轨迹留13.25%余量，并保持30分钟全PI-Agent运行超时、`reasoning_effort=medium`和`temperature/top_p/top_k=1/.95/20`。
- 5号机采用16卡TP4/PP2/CP2训练，6号机采用16卡TP4×DP4推理、每副本并发16；两机均通过70题哈希、精确两次曝光和3套只读沙箱环境预检。仅保存第70步最终`model,extra`，保留1份且不保存optimizer。
- 修复审核应用器沿用旧Arrow schema时丢弃新嵌套审核结论字段的问题；既有批准文件以显式审核、禁训和禁发布组合兼容验证，新生成文件完整持久化审核字段。新增[正式运行合同](docs/qwen38_train70_launch_20260818.md)和[安全证据](docs/qwen38_train70_launch_20260818.safe.json)。

### v1.11.80 — 2026-08-18

- 新增`banded-v2-strict-table-v1`奖励入口：table结果必须结构化解析并逐行核对类别—数值绑定、顺序、基数和唯一值列；交换配对/排名、重复类别、额外行/数字及短标签子串均不能进入正确档，原四档非重叠奖励范围保持不变。
- 奖励SQL重放新增gold结果缓存、最多32条唯一候选SELECT和5秒SQLite真实执行截止时间；真实234条完成轨迹的严格表解析p50约0.25ms、三机p95均低于0.47ms。
- 三机自动回放Qwen3.8原70道候选：旧包含式口径`70/70` mixed，但严格口径仅`20/70`保留方差（0/5/6号机为`6/4/10`），v15/v20/v21为`5/11/4`、难度1–5为`0/2/7/6/5`；其余50道自动隔离，所有私有输出继续禁训。新增[回放报告](docs/qwen38_banded_v2_strict_reward_replay_20260818.md)和[安全证据](docs/qwen38_banded_v2_strict_reward_replay_20260818.safe.json)。

### v1.11.79 — 2026-08-18

- 完成Qwen3.8原生Step0在v15/v20/v21的70道GRPO方差候选审核：70/70通过跨主机/跨沙箱身份唯一性、实际提示路由、reward/verifier结构、只读verification SQL重放、hidden gold/result hash和最终结果路由门；v15/v20/v21批准数为`12/39/19`，拒绝与待复核均为0。
- 双向语义审核完全留在5号机：第一轮从题面核对任务族、指标、分组、时间、筛选、门槛、权重、降序和Top5合同，第二轮从SQL反向核对业务字面量、聚合公式、最低样本、排序和输出形态。初次`69+1`中的唯一待复核由“解释/判断”与“说明”等价措辞规则统一修正后全量重跑为70/70，没有逐行硬编码或外部API数据出境。
- 0/5/6号机分别冻结`21/20/29`行私有批准候选Parquet，70行均为唯一身份、`explicit_semantic_reviewed=true`、权限`0600`；`training_allowed=false`与`promotion_allowed=false`继续保持。新增[审核报告](docs/qwen38_grpo_candidate_audit_20260818.md)和[安全证据](docs/qwen38_grpo_candidate_audit_20260818.safe.json)。

### v1.11.78 — 2026-08-18

- Qwen3.8原生Step0在v15/v20/v21三个沙箱共筛1,500题，得到`12+39+19=70`条GRPO奖励方差候选记录，候选率`4.67%`；难度1–5合计为`2/25/17/16/10`，对应`2.9%/35.7%/24.3%/22.9%/14.3%`。
- 九份最终安全汇总均为`complete_after_six_samples`且`training_allowed=false`。因此70是“可进入语义/gold审核的mixed候选”，不是现在即可开训的已批准池；当前严格可训练数仍为0。
- 70条为三个沙箱候选记录直接求和，尚未做跨沙箱instruction身份去重；正式合池前必须复用现有碰撞门、语义审核和训练许可门。已同步更新[对比报告](docs/qwen38_vs_qwen36_v15_v20_20260818.html)与[安全证据](docs/qwen38_vs_qwen36_v15_v20_20260818.safe.json)。

### v1.11.77 — 2026-08-18

- Qwen3.8原生Step0三机严格`2+2+2`已完成v21：2/4/6条阶段分别发现`7/3/9`道，最终`19/500=3.8%` mixed候选；实际采样2,966条，比固定6条节省34条，三台候选分布为0/5/6号机`3/4/12`，完成波次runtime error为0。
- 同一v21 500题历史Qwen3.6 Step120结果为`34/500=6.8%`；Qwen3.8少15道、低3.0个百分点、相对减少44.1%。该对比衡量当前候选生产能力，不作原生模型架构优劣结论，因为训练阶段分别为Step0和Step120。
- [Qwen3.8与Qwen3.6对比报告](docs/qwen38_vs_qwen36_v15_v20_20260818.html)和[安全证据](docs/qwen38_vs_qwen36_v15_v20_20260818.safe.json)已扩展到v15/v20/v21，三批Qwen3.8合计得到`70/1500=4.67%`候选；v20+v21同题可比范围内，Qwen3.8为`58/1000=5.8%`、Qwen3.6 Step120为`102/1000=10.2%`。

### v1.11.76 — 2026-08-18

- 进一步展开v15报告中的题面生成机制：原生流程由`task_type`模板先独立产出题面，SQL/gold随后从SQLite反向生成并可语义fallback，Step 5b只围绕原题面生成variants且不按查询计划复验。
- 明确新流程由真实值域构建EvidencePlan/QueryPlan，从同一计划生成标准题面、SQL和完整gold；外部API只自然化角色表达，返回后必须通过日期、指标、筛选、分组、Top-K、排序和输出形态锚点复验。

### v1.11.75 — 2026-08-18

- 将v15语义失配报告中的EvidencePlan/QueryPlan改写为“业务证据清单/数据查询方案”，并把术语设为可点击的站内跳转。
- 报告末尾新增术语注释模块，以`task_000257`货损趋势题说明“业务证据清单 → 数据查询方案 → 题面、SQL与gold”的生成关系，以及无数据时拒绝任务而不是偷偷换题的规则。

### v1.11.74 — 2026-08-18

- 重构老板v15语义失配复盘：删除Executive Summary和指标卡，改为单栏证据链；六类问题均展示原生task句、原verification SQL、按题面重写且在v15库只读执行的SQL、逐项差距和错误原因。
- 将原full277中`task_000033/task_000147`同题异gold从“第7类题内问题”提升为独立的跨样本标签冲突；明确保留`000147`只表示相对更接近题意，不表示其标签已经合格。
- 用直白步骤还原原生“先写题面、再用非空SQL硬凑gold”的生成链，并逐项对照EvidencePlan/QueryPlan同源生成、禁止语义fallback、改写锚点复验、完整gold重放和默认禁训的新流程。

### v1.11.73 — 2026-08-18

- 新增老板 v15 轨迹数据语义失配复盘：修正后 full276 虽然 `276/276` SQL/gold 机械重放通过，但 `271/276` 命中至少一项语义预警；归纳为6类可统计预警，并单列原 full277 的1组相同 instruction 绑定冲突 gold。
- 报告为每类提供一个脱敏 v15 实例，明确根因是 instruction、源轨迹、后续任务定义和 hidden target 未被同一语义合同绑定；旧数据不得因 SQL 可执行就直接用于训练。
- 对照记录八沙箱新链路效果：4000/4000 SQL/gold 重放、语义锚点和 API 自然化门均通过，所有任务继续 `training_allowed=false`；结构门通过不替代人工语义审核或训练 canary。

### v1.11.72 — 2026-08-18

- Qwen3.8原生Step0三机严格`2+2+2`已完成v15/v20：v15得到`12/500=2.4%` mixed候选、实际2,986条轨迹；v20得到`39/500=7.8%`、实际2,914条轨迹，分别由2/4/6条阶段发现`1/5/6`与`13/17/9`道。
- 同一v20 500题对比中，Qwen3.6 Step120首2条正确轨迹率为`50/1000=5.0%`、最多6条候选率`68/500=13.6%`；Qwen3.8原生Step0分别为`23/1000=2.3%`和`39/500=7.8%`，候选少29道、候选率低5.8个百分点。该结论只用于当前候选生产选择，不作native-to-native架构优劣声明。
- Qwen3.6旧汇总有36条轨迹既未标completed也未标timeout，因此不直接横比原始timeout标签；按`1-completed`统一后，两者首2条非完成率为`19.5%/20.0%`，工程稳定性基本相当，质量差距主要来自正确轨迹更少。新增自包含[对比报告](docs/qwen38_vs_qwen36_v15_v20_20260818.html)和[安全证据](docs/qwen38_vs_qwen36_v15_v20_20260818.safe.json)，v21完成后再按相同口径追加。

### v1.11.71 — 2026-08-17

- 独立PI-Agent rollout入口新增Qwen3.8原生HF、`reasoning_effort=medium`及可配置TP/DP/NPU合同；模型身份、94,208上下文、1,800秒超时、`temperature=1/top_p=.95/top_k=20`、1.25×滚动窗口和严格`2+2+2`早停均失败关闭，禁止混用Qwen3.6 Step120导出权重。
- 同一32题×2条轨迹在5/6号机完成TP8×DP2与TP4×DP4互换交叉基准：五次运行均64/64完成且无runtime error/OOM/上下文截断；TP4两机平均生成吞吐为1,354,686 token/小时，比TP8高4.14%，超时率为5.47%对12.50%，峰值HBM为83%对84–85%。
- 满94,208 token实测容量为TP4每副本20.93条；正式锁定5/6号机`TP4×DP4,max_num_seqs=16`和0号机12卡`TP4×DP3,max_num_seqs=16`，不采用仅余4.44%容量的并发20。v15/v20/v21各500题已按`182/182/136`难度平衡、跨机互斥分片，三机均已进入v15首轮2条采样，随后自动执行v20与v21。
- 新增不含题面或隐藏验证材料的[拓扑决策报告](docs/qwen38_27b_topology_decision_20260817.html)、[安全证据](docs/qwen38_27b_topology_decision_20260817.safe.json)和[可复现notebook](docs/qwen38_27b_topology_decision_20260817.ipynb)。

### v1.11.70 — 2026-08-17

- 将0号机`/data3/models/Qwen3.8-27b`完整复制到5/6号机，与Qwen3.6原模型平行存储；三机55,586,115,141字节文件清单摘要一致，并新增架构关键字段、完整HF tensor key、Transformers config/tokenizer和chat template的失败关闭兼容门禁。
- 新增隔离双机Qwen3.8工程冒烟入口和精确Ray物理资源门禁；实机采用5号机16卡Megatron `TP4×PP2×CP2`训练、6号机16卡vLLM `TP8×DP2`推理，从Qwen3.8原始HF权重完成16条多轮工具轨迹、权重同步和1个优化步，退出码为0且不保存检查点。
- 首次运行复现Ascend host-pinned optimizer offload `207001`启动故障；主训练入口加入默认值不变的offload开关，Qwen3.8冒烟改用device-side optimizer后通过。单步生成816.20秒、actor更新296.09秒，训练最大已分配显存38.57GiB，确认当前两机效率优先分配仍应保持训练16卡、推理16卡。
- 明确禁止把Qwen3.6 Step120权重当作Qwen3.8恢复点；正式切换必须从Qwen3.8基座重新建立基线、金丝雀和训练链路。测试完成后隔离Ray与容器已停止，模型副本和审计日志保留。

### v1.11.68 — 2026-08-17

- 首次161题续训启动在任何rollout或参数更新前失败：两机Ray和数据门禁均通过，但5号机初始化HybridDeviceOptimizer时申请Ascend host-pinned内存返回`207001`；运行目录无rollout/checkpoint、训练步为0，两机NPU已回到约2.9–3.1GiB驱动基线，失败运行保留审计且不从其恢复。新监督器除双机NPU连续空闲外，还要求5号机`MemAvailable≥1.2TiB`且`Mlocked≤128GiB`连续通过，避免外部任务刚释放NPU但host-pinned内存尚未回落时立即重启。
- 128题训练集新增按Step120经验mixedness的课程顺序：在2条内出现至少一正一错的33题先训练，随后是首次在4条出现的21题、首次在6条出现的13题；其余61题来自固定8条或不可直接比较的旧筛选协议，单独作为最后的legacy8阶段，不伪装成6条难度。
- 每个阶段内仍让每题精确出现5次，并按难度×来源轮转、每轮用稳定种子改变桶内次序；生成640行私有课程表，关闭全局shuffle后执行`2→4→6→legacy8`，总量仍为640 groups/320次更新/Step120→440，测试33题和原128/33拆分保持不变。
- 新增Step120课程续训resume view：训练侧只链接原actor，rollout侧不加载与旧train236绑定的`data.pt`，从课程表第1行开始但保留累计Step120计数；原优化器继续按既定合同重置，课程表在两机按相同源哈希确定性构建并复核。

### v1.11.67 — 2026-08-17

- 新增Qwen3.8 reasoning-effort技术调研文档：基于0号机Qwen3.8-27B、PI 0.82.1与vLLM 0.23.0实机代码确认三档提示词、thinking边界token、preserved thinking和当前预算行为，并整理多种训练范式及云端Qwen3.8-Max与开源27B的差异与证据边界。

### v1.11.66 — 2026-08-17

- 双机资源门禁进一步改为失败关闭：本机`npu-smi`异常、跨机SSH失败或返回空状态时一律按“资源忙”处理并继续等待，不会把监控故障误当成空闲后抢占共享NPU。

### v1.11.65 — 2026-08-17

- 服务器容量复核确认Step120完整检查点约459GB，其中模型约51GB、优化器约408GB，而5号机仅余563GB；候选五轮续训因此保持每40步保存、最近2份，但保存载荷改为`model+extra`，不落盘优化器，预计两份约102GB并保留充足rollout余量。
- 训练拓扑、学习率、2×8组形状、banded-v1奖励和Step120起点均不变；若从中间检查点恢复，沿用此前金丝雀做法重置优化器，避免为了精确优化器恢复而把共享磁盘写满。

### v1.11.64 — 2026-08-17

- 修正双机空闲门禁对`npu-smi`进程表的匹配边界：只把带有“芯片号、物理号、PID、进程名”的真实进程行判为忙，不会把设备状态行误判为进程，也不会在存在vLLM worker时误判为空闲。

### v1.11.63 — 2026-08-17

- 修复私有候选拆分器按绝对脚本路径运行时的项目模块定位：启动时显式加入仓库根目录，因此服务器训练入口不依赖调用者预先设置`PYTHONPATH`；新增直接文件入口回归测试。

### v1.11.62 — 2026-08-17

- 新增161题私有候选池的确定性难度分层拆分：128题训练、33题密封测试；训练集难度1/2/3/4/5/未知分别为`6/30/9/41/32/10`，测试集为`1/8/2/11/8/3`，题面身份严格互斥且合计完整覆盖原池。
- 原始冻结池继续保持只读和`training_allowed=false`；派生训练集单独记录本次所有者明确授权并开放训练，测试集保持`training_allowed=false/evaluation_only=true`，两者均禁止晋级，敏感Parquet与环境清单仍为`0600`且不进入Git。
- Step120续训保持既有`2 groups/update × 8 responses/group`、banded-v1、`1e-7`、48K和双机16+16卡参数；`128×5=640 groups`精确对应320次更新并结束于Step440。单轨迹与工具超时由900秒放宽到1800秒，每40步保存一次模型+优化器并只保留最近2个完整检查点，最终测试只跑密封33题。
- 双机启动器新增三次连续空闲门禁：5号机当前被外部Qwen3.8作业占用时只等待、不终止或抢占；两机同时空闲、数据/沙箱/哈希复核通过后才启动Ray与训练，退出时自动清理本次Ray资源。

### v1.11.61 — 2026-08-17

- 在v21第6条波次尚为`0/235`完成时安全停止5号机控制器、重试启动器、runner与对应NPU监控；6号机沿用已完成的6条收尾，未完成轨迹不进入候选口径。
- 新增严格GRPO候选池冻结器：逐来源核对预期行数、可选哈希选择器、题面与hidden verifier完整性、跨来源instruction唯一性，并拒绝任何已开放训练或晋级的输入；敏感Parquet以`0600`原子写入，安全摘要只记录计数和哈希。
- 当前冻结口径为161题：281题池语义批准13、plan-first v3候选46、v20方差候选68、v21截至6条候选34；合并后继续统一保持`training_allowed=false`，未经语义审核不得直接训练。

### v1.11.60 — 2026-08-17

- v21效率优先筛选允许在全局第6条波次完成后安全收尾：已完成的2/4/6条分片原样复用，不启动第8条波次，候选判定仍要求累计至少一条正确和一条已完成错误。
- 三波finalizer新增2/4/6条互斥覆盖、实际轨迹、相对全6条基线节省量及候选难度安全汇总；候选继续保持`training_allowed=false`并等待语义审核。
- 运行控制器新增显式`--max-target-samples 6|8`合同，默认仍为8；本次v21两臂固定为6，避免无意改变其他队列。

### v1.11.59 — 2026-08-16

- 新增v21全量500题无人值守严格`2+2+2+2`筛选器：先对每题采2条，累计出现至少一条正确和一条已完成错误即在2/4/6条任一检查点停止，仅未决题进入下一轮，最多8条。
- v21逐机接力器等待对应v20分片完成后自动启动，保持Step120、1.25×逻辑Agent窗口、94K上下文、30分钟超时及原任务身份不变；5/6号机可各自完成即接力，避免快机空等慢机。
- 最终安全汇总记录2/4/6/8条停止分布、实际轨迹数、相对全500×8采样的节省量及候选难度分布；候选仍为`training_allowed=false`，必须经过语义审核后才可训练。

### v1.11.58 — 2026-08-16

- 新增严格`2+2+2`方差早停筛选器：复用全500题已有的首2条，仅对精确排除25题直接mixed和51题旧定向探针后的424题继续采样；每轮只补2条并重新判断，一旦累计出现至少一条正确和一条已完成错误就立即冻结，不再进入下一轮。
- 424题冻结集合按原双机分片保持互斥：5号机215题、6号机209题；候选只证明Step120当前策略下存在非零最终结果奖励方差，始终`training_allowed=false`，仍需后续题意、gold和verification SQL语义审核。
- 接力控制器沿用通过后的`1.25×`逻辑Agent窗口（5号机物理48/逻辑60，6号机物理64/逻辑80）、Step120、temperature/top-p/top-k、94K上下文和30分钟超时；输出4/6条停止分布、实际轨迹数、相对`424×6`节省量、难度与安全错误聚合。

### v1.11.57 — 2026-08-16

- 为工具型Agent采样加入显式、受上限保护的逻辑窗口超配：默认仍保持`1.0×`历史行为，只有启动器明确声明倍率时才允许超过vLLM物理序列容量，硬上限为`2.0×`；合同记录物理容量、逻辑窗口、实际倍率和是否超配。
- 修正滚动补位遥测的入队时间：每条轨迹在实际进入逻辑调度窗口时单独计时，不再让后续补位错误继承整轮运行开始时间，为`1.0×/1.25×`A/B提供可比的排队、生成、工具和总耗时。
- 下一步冻结Step120、数据、采样、94K上下文、30分钟超时和物理`max_num_seqs`，仅在5/6号机把逻辑Agent窗口从`48/64`提高到`60/80`，以轨迹/小时、生成tokens/小时、NPU利用率、vLLM等待和超时率决定是否保留或继续测试`1.5×`。

### v1.11.56 — 2026-08-16

- 复核当前纯采样拓扑：实际为vLLM `TP8×DP2×DCP1`，不包含CP2；`CP2`属于Megatron训练侧的`TP4×PP2×CP2`，两者不应混称。
- 结合Step120结构（64层、16层全注意力、4个KV heads）和已安装vLLM-Ascend上下文并行约束，确认`TP8×DCP2×DP2`在结构上可行并可把每卡全注意力KV缓存从`16 KiB/token`降至`8 KiB/token`，但DCP2仍为实验路径，必须做真实吞吐门禁。
- 将`TP8×DCP2×DP2`列为第一候选、`TP4×DP4`列为第二候选、`TP2×DP8`列为可选研究门；冻结94K/30分钟和采样合同，以同任务双机交叉门禁按生成tokens/小时、轨迹/小时及最终mixed候选/小时决策，不在门禁前启动剩余424题。

### v1.11.55 — 2026-08-16

- 完成当前v20 Step120双机方差筛选的效率诊断：有效恢复运行前216条完整轨迹无runtime error/OOM，但两机累计AICore仅`14.4%/17.1%`、vLLM序列填充约`67.7%/60.6%`，说明安全合同有效但端到端agent窗口仍偏保守。
- 明确下一沙箱的效率优先级：固定`+6`改为`2+2+2`逐波补采并在明确mixed后立即早停；逻辑agent窗口先以物理序列容量`1.25×`做门禁，再视vLLM等待、超时与OOM决定是否升至`1.5×`；5号机另做`24→28→32`逐档容量测试。
- 冻结Step120、采样分布、TP8×DP2、94K上下文和30分钟合同，不中断仅剩90条的当前运行；新增结构化安全汇总、canonical artifact和经过响应式验证的自包含效率报告。

### v1.11.54 — 2026-08-16

- 固化当前GRPO候选供给和训练容量口径：v20的500题首2条筛选已确认25题直接mixed，51题定向探针尚未形成有效新增；累计84题由13条语义批准、46条plan-first mixed和25条v20直接mixed组成，不再把经验候选误称为已获准训练数据。
- 按当前`2 groups/update × 8 responses/group`合同核算：84题一轮为42步/672条在线轨迹，两轮为累计84步/1,344条；推荐先做10题5步金丝雀，一轮门禁通过后才允许第二次曝光，并显式记录94K筛选与48K训练上下文的不一致风险。
- 新增可复核的结构化汇总、canonical report artifact和自包含HTML训练容量报告，不包含题面、gold、SQL、task ID或轨迹内容。

### v1.11.53 — 2026-08-16

- 收紧历史mixed匿名结构的使用方式：结构签名只作为每签名固定少量探针，不再把同一宽泛结构下的全部任务补满8条；优先补采两条全对或正确加超时等高信号不确定题，并保留每难度小额探索。
- 结构探针和难度探索均按instruction哈希确定性选择，避免运行次序改变候选；安全manifest显式记录每签名配额，使双机补采规模可审计且不再接近全池重采。

### v1.11.52 — 2026-08-16

- 自适应DWH筛题改为以“证明组内奖励方差”为停止目标：前2条已出现至少一条正确和一条明确错误的任务立即冻结为GRPO候选，不再浪费6条补采；候选仍只包含题面与gold，保持`training_allowed=false`并等待语义审核。
- 只有全对、正确加超时、历史mixed匿名结构命中或分层探索保留等尚未证明方差的任务才补6条；最终安全汇总分别记录两条即停候选、补采新增候选、节省轨迹数及合并后的匿名候选数据哈希。
- 新增直接候选与补采集合互斥、候选合并、宽松超时mixed和冻结采样合同回归测试；相关14项测试通过。

### v1.11.51 — 2026-08-16

- 自适应筛题脚本内聚 canonical JSON 与文件哈希实现，移除对数据生成链的传递依赖，保证裁剪部署的 h06 rollout 容器也能独立执行画像、补采准备和最终合并。

### v1.11.50 — 2026-08-16

- 新增开放 DWH 两阶段自适应筛题：全池先采 2 条，命中正确答案、匹配历史 mixed 匿名结构签名或进入分层探索配额的题目再补 6 条；最终按 instruction 哈希无损合并为 8 条 group，并同时产出严格 mixed 与“允许超时但至少一条明确正确、一条明确错误”的宽松候选。
- 新增单机无人值守接力器，冻结 Step120、`temperature=1/top_p=.95/top_k=20`、4096/90112/94208 token 和 1800 秒合同；h05/h06 的 6 条补采 task batch 分别限制为 8/10，使并发不超过 48/64 的真实准入容量。
- v20 第一阶段改为两机各 250 题×2 的滚动采样；停止旧 v15 补跑与原始模型采样方向，完整原子分片继续保留为历史证据。

### v1.11.49 — 2026-08-16

- 修复双机单臂超时回填 finalizer 未传入 `250×8` 形状而误用 `300×8` 默认值的问题；启动器与无人值守接力器现显式冻结题数和每题轨迹数，避免把完整的 2000 条单臂结果误报为缺少 400 条。

### v1.11.48 — 2026-08-16

- standalone rollout 与超时回填 finalizer 的后台子进程改用非登录 Bash，避免登录配置覆盖显式传入的数据集、分片和输出目录；新增合同测试，保证逐轨迹遥测的新运行与回填汇总使用调用方冻结的环境。

### v1.11.47 — 2026-08-16

- 收紧逐轨迹排队口径：从模型服务完成初始化、待处理集合进入调度队列开始计时，到对应 agent 真正开始执行为止；滚动窗口外等待不再被误记为接近零，批量与滚动模式使用同一时钟起点。

### v1.11.46 — 2026-08-16

- 为 standalone PI rollout 增加逐轨迹排队、生成、工具执行、执行/总耗时、调用/回合数与 token 计数；状态通过 `ContextVar` 隔离，避免同一 agent worker 并发轨迹相互串写。
- 超时路径在公开 vLLM abort 删除请求状态前读取已生成 token 数，只保存计数、不保存额外内容；逻辑请求层聚合多物理请求，原子 shard 与安全摘要同步输出遥测覆盖率及 p50/p95/max。
- 滚动和批量调度均在真正提交前写入 enqueue 时间，后续运行可直接判断长尾来自排队、生成还是工具执行；旧的在途进程不热改，其结果仍按旧合同处理。

### v1.11.45 — 2026-08-15

- standalone PI-Agent rollout 新增工具运行时可见性硬门禁：在加载模型或占用NPU前，从采样Parquet聚合提取`environment_id`，并在实际工具根目录`/pi_sandbox`下验证环境目录、SQLite、schema和documents完整可读；数据库必须以只读方式成功打开且包含关系对象。
- 门禁只把环境数量和组件状态写入安全合同，不输出environment ID、题面、gold、SQL或数据库内容；数据库误放在项目目录而未进入PI-Agent挂载时立即失败，避免把工具不可用误判为模型全错。

### v1.11.44 — 2026-08-15

- 双机DWH结果分析新增response token直方图及分Level直方图，finalizer据此精确重建双机合并后的均值、P50/P90/P95/P99和最大长度，不再用两个arm的分位数近似。
- 双机安全合并器同步保留Level×结果bucket与Level×0–8正确次数交叉分布；旧版缺少这些字段的安全汇总仍保持兼容，但新500题开放DWH批次要求完整输出。

### v1.11.43 — 2026-08-15

- 500题开放DWH准备器新增目标推理容器内SQLite重放门：表格行数、类别和排序必须完全一致，数值跨SQLite实现漂移只允许绝对误差不超过`0.011`；超过界限或任何结构变化继续失败关闭。
- 对有界漂移只在0600采样Parquet中使用模型实际可见数据库的重放结果作为最终表格gold，源沙箱保持字节不变；安全清单记录调整题数与最大漂移，数据仍为`training_allowed=false`。

### v1.11.42 — 2026-08-15

- 新增500题开放业务DWH的泄漏隔离采样准备器：先重放500/500 SQL/gold并验证五级各100题，再生成只含数据库、schema和空documents目录的运行投影；输出按五级均衡切成双机各250题，每题8轨迹，继续明确禁训。
- 281题通用最终结果分析器扩展为同时输出五级×0–8正确次数、五级×all-wrong/mixed/all-correct/错误状态交叉表，以及整体和分级response token长度分布；mixed仍只进入显式语义审查队列，不能自动训练或晋级。
- 500题采样合同固定Step120、`temperature=1.0/top_p=0.95/top_k=20`、prompt 4,096、response 90,112、总上下文94,208和单轨迹1,800秒；并发只允许在不改变该合同的前提下按机器容量调整。

### v1.11.41 — 2026-08-15

- 新增8沙箱开放业务DWH生成器：只读快照老板v15、v20–v26数据库，以旧Band 5为Level 1起点，按五个等级/五个任务族各100题生成每沙箱500题；难度改由必要连接、证据步骤、时间比较、派生指标和业务开放度共同定义，冗余连接必须为0。
- 新增私有API自然化适配器与通用化原子重写入口：仅发送合成角色、业务草稿、任务族和语义锚点，SQL、gold、数据库行、路径与凭据不外发；支持任意任务数、16并发、失败续跑和逐条语义保真门禁。
- 新增多沙箱只读审计器和生成策略文档：逐条重放验证SQL、比较hidden gold、检查题面锚点/技术词/重复项及五级特征单调性；最终结构等级必须再由每题8次采样的`0–8`正确次数校准，核心训练池优先取`3–5/8`，所有数据在审核与实测完成前保持禁训。

### v1.11.40 — 2026-08-14

- 新增 plan-first v3 超时槽位重跑适配器：从原生/Step120完整 `300×8` 分片中精确抽取 `trajectory_timeout=true` 的原采样位置，一条超时对应一条 `n=1` 重试行；重试集合必须与原超时集合一一相等，已完成轨迹不得进入重试。
- 重试合同同时把单轨迹上限从`900`提高到`1,800`秒、总response预算从`45,056`提高到`90,112` tokens，并保持prompt上限`4,096`，因此总上下文提高到`94,208` tokens；训练和promotion继续关闭。
- 新增原位回填与arm finalizer：重试结果只替换原`task×sample`超时槽位，保留每题恰好8条，随后重跑纯最终结果分析；安全摘要记录解决/残余超时，题面、gold、SQL、task ID和输出仍仅存在于0600服务器工件。

### v1.11.39 — 2026-08-14

- 完成 XTuner、veRL、ms-swift 与 MindSpeed 的源码级选型调研：区分端到端 RL 编排层、训练引擎、模型套件和昇腾加速层，并按 Qwen3.6-27B、Ascend A3、Megatron 并行、双机训推分离、异步长尾和多轮工具合同逐项比较。
- 将 XTuner、ms-swift、MindSpeed-RL、MindSpeed Core 和 MindSpeed-LLM 官方主干浅克隆到已忽略的 `reference/`，并把既有 veRL 引用快进到最新主干；新增可复现的远端/分支/提交清单，明确引用更新不等于服务器运行时升级。
- 选型结论保持 veRL 为当前生产框架；优先给 ms-swift 做受限 Qwen3.6/Megatron 兼容 POC，观察 XTuner Beta RL 的真实跨设备权重同步成熟度，不向已暂停新增功能集成的 MindSpeed-RL 迁移。

### v1.11.38 — 2026-08-14

- 新增 plan-first v3 双模型比较数据适配器：复验 300/300 SQL/gold 后，将首 60 题预冻结、首 48 题按六档各 8 题交错排列，其余 240 题标记为 training-candidate 但继续禁训；Parquet 统一绑定题面、最终 numeric/table gold、验证 SQL、数据库环境和老板原始 system/四工具合同。
- 新增最小权限运行时投影：每条轨迹只复制数据库、schema 和空文档目录，任何额外文件或哈希不一致均 fail closed，避免模型从源任务清单读取 hidden gold/SQL。独立 vLLM 运行器现区分原生 HF checkpoint 与经清单验证的 Step 120 HF 导出，并记录正确 policy step。
- 新增双臂配对分析和 Ray finalizer：按同题汇总 `0–8`、all-wrong/mixed/all-correct 转移、六档与 numeric/table 成功率、Step120 相对基座胜负，以及仅在 240 题候选池内的 Step120 mixed 数量；请求随机性按题级配对解释，不伪称逐请求同 seed。

### v1.11.37 — 2026-08-14

- 新增无文本落盘的vLLM长上下文单请求基准：固定并发1、256-token生成、TP8和关闭prefix cache，按2K/4K/8K/16K/32K/40K/48K各重复两次，分别记录TTFT、解码吞吐和端到端墙钟；64K超过当前49,152上限时fail closed。
- Step 120实跑显示40K/48K解码为`9.121/9.190 tokens/s`，相对2K的`9.331 tokens/s`保留`97.75%/98.48%`，没有40K后的10倍下降；TTFT由`0.300s`增至`3.625/4.318s`。结果明确限定为单序列，不外推到高并发服务合计吞吐。

### v1.11.36 — 2026-08-14

- 将 plan-first DWH 生成器升级到 v3：第 4 档明确区分“发货仓类型”与具体仓库名称，避免先锁死比较维度的歧义；API 改写器新增失败原因反馈、等价业务说法支持、仓库类型显式门禁、断点行重新校验、显式重试和连接关闭，避免隐藏重试把低并发放大。
- 新增自然语言质量审计与选择性修订器：检查 11 类角色覆盖、语义约束、技术词、重复、口语化、开头模板和长度；可在 QueryPlan/gold/角色完全未变且旧题面重新校验通过时复用已有行，只重写指定难度档或新门禁拒绝项。
- 5 号机最终部署 `/data3/llin/qwen3.6-27b-verl-grpo/sandboxes/sft/20260814_llin_dwh_planfirst_api_v3`：先复用 250 条并重写第 4 档 50 条，再只重写 1 条残余歧义；最终 300/300 SQL/gold 重放和语义门通过，300 条题面唯一、零技术词、291 种规范化开头，沙箱目录 0700、文件 0600，`training_allowed=false`。

### v1.11.35 — 2026-08-14

- 修复双机finalizer仍只接受旧版arm摘要合同的问题：合并器现支持同版本的v1或v2摘要，但拒绝双臂版本混用和未知合同；新增v2全量合并、v1兼容和混版fail-closed测试。

### v1.11.34 — 2026-08-14

- 双机281题筛选完成全量收尾：合计281题/2248轨迹，最终得到49个技术mixed候选；逐题核对题意、gold、验证SQL和8条最终答案后，13个进入语义批准候选、36个拒绝，批准项仍保持训练与promotion关闭。
- 5号机最后13题先修复104条轨迹不能被16个Agent worker等分的问题，再因重复vLLM启动耗尽host-pinned内存而迁移到空闲6号机续跑；回传分片哈希一致，既有256题未重算，最终两臂均通过完整分片与匿名裁决覆盖门禁。

### v1.11.33 — 2026-08-14

- 修复双机281题筛选的尾批等分约束：当真实轨迹数不能被16个Agent worker整除时，只在内存中复制最少尾行补齐worker分块，生成后立即裁掉补行，再按原始task/sample身份原子写入分片；已完成的256题分片保持不变。
- 5号机最后13题由104条临时补到112条执行，持久化仍严格要求104条；新增纯函数和合同测试覆盖整除、尾批补齐、非法形状与裁剪顺序，训练与promotion继续关闭。

### v1.11.32 — 2026-08-14

- 双机281题筛选完成第8个完整分片（各128题、合计256题/2048轨迹）的增量分析与匿名语义复核：累计48个mixed候选中13个批准、35个拒绝；批准项在全量收尾前继续保持训练关闭。

### v1.11.31 — 2026-08-14

- 新增确定性、无外部 API 的 plan-first 物流 DWH 生成器：一个 SQLite 沙箱生成 300 条任务，按每 50 条一个难度带逐步增加过滤、分组、连接、Top-K、HAVING 和派生比率；题面、SQL 与 gold 共享同一 QueryPlan，避免旧链的语义漂移。
- 新增精确目录验证器与测试：300/300 SQL 可执行、非空并精确重放 gold，6 档各 50 条，现有 catalog 语义审计为 0 告警；生成摘要固定记录文件哈希且不记录题面、SQL、gold 或 API key。
- PI workspace 与只读 SQLite 工具支持通过 `PI_AGENT_SANDBOX_LOWER` 指向用户私有沙箱根；新版本已部署到 5 号机 `/data3/llin/qwen3.6-27b-verl-grpo/sandboxes/sft/20260814_llin_dwh_planfirst_v1` 并以同一精确目录验证器复验通过。训练继续关闭，先执行 `6×8×4=192` 条分层 rollout 校准并排除全对/全错题。

### v1.11.30 — 2026-08-14

- 双机281题筛选完成第7个完整分片（各112题、合计224题/1792轨迹）的增量分析与匿名语义复核：累计42个mixed候选中13个批准、29个拒绝；批准项在全量收尾前继续保持训练关闭。

### v1.11.29 — 2026-08-14

- 双机281题筛选完成第6个完整分片（各96题、合计192题/1536轨迹）的增量分析与匿名语义复核：累计37个mixed候选中11个批准、26个拒绝；批准项在全量收尾前继续保持训练关闭。

### v1.11.28 — 2026-08-14

- 双机281题筛选完成第5个完整分片（各80题、合计160题/1280轨迹）的增量分析与匿名语义复核：累计29个mixed候选中8个批准、21个拒绝；批准项在全量收尾前继续保持训练关闭。

### v1.11.27 — 2026-08-14

- 完成老板沙箱生成链与 DWH 语义错位根因审计：确认 step5 先生成题面、后独立 backward-generate SQL/gold，且 fallback 改变查询语义时不回写题面；现有 QA 只验证可执行性并以 `on_fail=tag` 保留失败样本。
- 9,500 条 SFT DWH 中有 2,411 条“SUM SQL 但题面无明确聚合词”；最终 281 池中为 97 条，其中 87 条是 `single_metric_query + SUM`。已审 18 个 mixed 全部匹配 raw `natural_language_instruction` 且与 instruction 统一前备份一致，排除后续 variant 选择作为当前主因。
- 相关生成器和老板服务器后处理代码已复制到 5 号机 `/data3/llin/qwen3.6-27b-verl-grpo/source_snapshots/rjx_sandbox_pipeline_20260814`，共 263 个文件；只含源码和哈希，不含数据、轨迹、模型、Git 元数据或凭据。修复方向改为复用 SQLite/schema、以新 task ID 重建 plan-aligned instruction—gold—SQL，并在 fresh rollout 前保持训练关闭。
- 完整证据、代码路径和不浪费旧数据的修复方案见 [`docs/boss_sandbox_generation_root_cause_audit_20260814.md`](docs/boss_sandbox_generation_root_cause_audit_20260814.md)。

### v1.11.26 — 2026-08-14

- 双机281题筛选完成第4个完整分片（各64题、合计128题/1024轨迹）的增量分析与匿名语义复核：累计24个mixed候选中7个批准、17个拒绝；批准项在全量收尾前继续保持训练关闭。

### v1.11.25 — 2026-08-14

- 完成双机第三个shard的新增32题、256条轨迹审视：新增11个mixed中批准2个、拒绝9个，累计18个mixed批准3个。匿名裁决账本已扩到48题范围；批准集仍由哈希裁决机械生成，并在全量finalizer完成前保持训练与promotion关闭。

### v1.11.24 — 2026-08-14

- 完成281题池首批64题、512条轨迹的逐shard复核：技术门得到7个mixed，人工核对题意、gold、verification SQL和8条最终答案后仅1个通过。新增不含题面、SQL、gold值、最终答案或task ID的匿名哈希裁决账本与fail-closed测试；6个拒绝项中5个为未明确要求聚合、1个为日期字段与最终结论路由风险。

### v1.11.23 — 2026-08-14

- 多沙箱DWH rollout分析器新增逐完整shard累计模式和最小敏感审视队列；runtime error与超时均fail closed，任何异常轨迹都不能误入mixed候选。安全摘要记录已完成shard范围、累计`0–8`分布和筛选合同，题面、gold、SQL与八条最终答案仅保存在`0600`敏感文件中，等待显式语义裁决。

### v1.11.22 — 2026-08-14

- 修复独立rollout的混合超时批次拼接：超时占位轨迹显式保留`global_steps/min_global_steps/max_global_steps`列，避免某个agent worker整块超时后外层`DataProto.concat`得到短列；新增合同回归测试覆盖该场景。

### v1.11.21 — 2026-08-14

- 修复双机后台收尾器的环境传递：显式导出两臂路径、Ray 地址、远端资源、轮询与超时参数，确保脱离启动 shell 后仍能自动等待、回收和合并。

### v1.11.20 — 2026-08-14

- `timeout-gate8x1-04` 真实 16 卡门禁通过：8/8 轨迹在 3 秒处标记超时，运行和分析退出码均为 0，8 个题组全部进入 `timed_out`，无 runtime error、无 mixed 误选。
- 8 条中 4 条已注册物理 vLLM 请求并收到 abort 确认，其余在阈值时尚无物理请求；正式 shard 现同时保存/汇总 physical request、ack 与 error 计数，便于 900 秒无人值守运行后审计取消完整性。

### v1.11.19 — 2026-08-13

- 修复真实 `timeout-gate8x1-03` 暴露的逻辑请求 ID 重复传参：上游 manager 已注入 `__llin_request_id` 时，PI wrapper 现原地复用/覆盖同一字段，不再同时显式传入第二份。
- `-03` 已完成 16 卡模型加载，但在首个 vLLM 请求创建前退出；保留失败证据并以隔离的 `-04` 继续硬取消门禁。

### v1.11.18 — 2026-08-13

- 修复真实超时门禁发现的 veRL 严格配置缺口：现有幂等 `MultiTurnConfig` 补丁同步声明 `agent_timeout_seconds`，避免 LLM server 初始化在模型加载前拒绝新字段。
- 失败的 `timeout-gate8x1-01` 在加载模型和生成请求前退出；保留证据并使用隔离的 `-02` 复验，不将失败尝试冒充超时取消成功。

### v1.11.17 — 2026-08-13

- 为多沙箱 DWH 筛选加入单轨迹 900 秒硬超时：先通过现有逻辑/物理请求映射取消 vLLM 请求，再终止 agent 协程、清理独立沙箱，并以 `trajectory_timeout=true` 保留完整 task/sample 位置。
- 超时题组单列为 `timed_out`，不再误当全错或 mixed 训练候选；安全汇总和双机合并均报告 timeout/evaluable 计数。
- 两机独立 rollout 默认提高到每个 TP8 引擎 32 个序列，并加入三次自动分片续跑；单次故障无需人工监督即可从原子 shard 继续。
- 新增常驻双机收尾器：不占 NPU 资源地等待两臂退出，自动从 6 号机回收安全汇总与 mixed 数据，在 5 号机合并 281 题结果并写入最终状态。

### v1.11.16 — 2026-08-13

- 多沙箱standalone后台启动器默认同步启动5秒NPU/HBM采样，并以`exit_code`自动停止；每机全量rollout成功后自动执行server-only最终结果评分和mixed筛选，评分失败会把整个运行标为失败，不会只报告模型生成成功。
- 正式运行因此具备无人值守闭环：分片原子落盘/断点续跑→完整形状检查→纯最终答案评分→安全聚合与0600 mixed候选；全程不初始化optimizer/actor、不保存checkpoint、不把敏感结果写入Git。

### v1.11.15 — 2026-08-13

- standalone安全摘要补充每个vLLM引擎最近一次`Running/Waiting`值，用于区分历史触顶与当前长尾剩余请求；与最近16-chip NPU窗口组合判断运行是否仍在推进，不输出请求内容或task identity。

### v1.11.14 — 2026-08-13

- standalone安全摘要新增最近一个完整16-chip采样窗口的AICore/NPU/HBM聚合，区别“整个运行平均利用率”和“当前是否仍在实际解码”；仍不读取或输出prompt、答案、SQL、工具内容与task identity。

### v1.11.13 — 2026-08-13

- 新增双臂结果合并器：只接受两份既定outcome summary和两份mixed敏感Parquet，验证`tasks×n=trajectories`、采样数一致、mixed行数一致及verifier identity无交集后，合并正确率、完成率、0–8直方图、bucket和版本/answer type分布。
- 合并后的mixed Parquet仍为0600且训练/晋级关闭；Git只允许进入不含题面、gold、SQL、task ID、工具调用或服务器路径的安全聚合。

### v1.11.12 — 2026-08-13

- 新增281题server-only最终结果分析器：按分片中的`source_task_index×sample_index`强制验证每题恰好8条，复用现有`score_final_outcome`只看最终可见答案，不读取SQL执行过程或以工具命中代替正确。
- 每题分为`all_wrong/mixed/all_correct`并输出0–8正确数直方图、完成率、错误率及按版本/answer type聚合；per-task identity与mixed候选Parquet保持0600且不进入Git，安全summary不含题面、gold、SQL、task ID、工具调用或服务器路径。
- mixed仅作为后续语义审核和训练候选层，`training_allowed/promotion_allowed`继续为false；全对/全错不参加当次GRPO候选，但不会被永久删除。

### v1.11.11 — 2026-08-13

- 收紧standalone安全摘要的上下文错误判据：只有同一日志行同时包含`ERROR/Exception/ValueError/RuntimeError`和context/truncation信号才计入，避免把正常`data.truncation=error`启动合同误报为9个运行错误；probe原日志的Traceback和OOM计数始终为0。
- 281题双机正式Parquet CPU门禁通过：5号机141/141、6号机140/140均保留，`n=8`分别对应1,128/1,120条预期轨迹，全部prompt通过4,096-token过滤；每机规划5个32题分片，可断点续跑。

### v1.11.10 — 2026-08-13

- 为281题双机满载补齐确定性敏感数据拆分：按冻结行序交错分成5号机141题、6号机140题，逐题verifier identity保证两臂无交集且并集精确等于281；Parquet继续0600，安全manifest只记录行数、版本分布和文件hash。
- 新增standalone安全运行摘要器：只读取执行合同、分片形状、response token计数、vLLM Running/Waiting、NPU利用率/HBM和错误标记，输出不含题面、gold、SQL、task ID、工具调用或服务器路径的聚合；用于客观比较`max_num_seqs=24/32`并发探针。

### v1.11.9 — 2026-08-13

- 修正画像器后在新目录重扫时确认当前训练容器只暴露8,000条manifest：v45/v46/v50仅有数据库、manifest缺失，新扫描只能形成397条高精度候选和233条严格直接查询题，不能替代此前完整9,500题扫描冻结的459/281池。
- 新增fail-closed候选迁移器：仅接受精确459行、精确旧hash修复数、原题面hash不变且导出gold键严格为`answer_type/value/verification_sql`的既有敏感候选；只重算导出对象的自证hash并原子写入0600新文件，不读取到本地、不改变题面、gold值、SQL、task identity或筛选成员。
- 281题实验继续使用原完整池迁移后的等价候选；8,000题重扫产物只保留为环境缺失证据，不进入并发探针或全量运行。

### v1.11.8 — 2026-08-13

- 281题敏感Parquet首次构建在写文件前被gold hash完整性门主动拦截：画像器原先对未导出的source gold扩展字段做哈希，而候选包只保留`answer_type/value/verification_sql`，导致导出对象无法自证完整性；本次没有占用NPU，也没有产生可用Parquet。
- 画像器改为对实际导出的精简gold对象计算`gold_sha256`，并补充画像器→候选包→rollout构建器的精确哈希回归；服务器将以新运行目录重建459条候选和281条严格池，不修改旧证据。

### v1.11.7 — 2026-08-13

- 按全量281题筛选要求新增严格server-only数据构建器：仅接收18个非v15沙箱、跨版本唯一题面、机械SQL/gold通过且无确定性语义预警的aggregate/single-metric/comparison任务；模型输入保持老板system、guidance和`bash/read/write/edit`四工具合同，hidden gold/SQL/identity只进入0600 Parquet，训练与晋级均关闭。
- standalone veRL runner改为可配置题数、`n`、分片大小、Agent workers和vLLM序列上限，同时强制`4,096 + 45,056 = 49,152`上下文合同及`temperature=1.0 / top_p=0.95 / top_k=20`采样合同；每个分片必须精确产出`题数×8`行后才原子落盘，可安全断点续跑。
- Ray启动脚本保留正式训练默认资源名不变，并允许显式覆盖为`llin_rollout_m05/m06`，支持两台机器各自固定一套TP8×DP2服务；新增统一后台启动器与0600 Ray直传，避免敏感题面、gold和SQL经过本地工作区。
- 并发选择门固定为同一8题双机同时对照：5号机`max_num_seqs=24`基线、6号机`32`候选，其余模型、上下文、工具、回合、采样和`n=8`完全相同；只有行数、长度、错误/OOM和资源观测均通过才用更快配置启动281题全量推理。

### v1.11.6 — 2026-08-13

- 5号机已同步跨沙箱筛选器并完成9,500条DWH只读全量运行：`5,253/5,253`条verification SQL可执行且非空，`5,099/5,253`条hidden gold被查询结果支持；全流程CPU实测`1.919s`，未占NPU。
- source answerable、numeric/table、QA passed、数据库验证、SQL/gold机械一致和零确定性语义预警共同收缩到459条；再排除v15、跨版本重复题面及trend/report/anomaly/attribution等高风险类型，保留18个非v15沙箱上的281条直接查询题，其中aggregate/single-metric/comparison为`123/113/45`，numeric/table为`234/47`。
- 采用最近Step120 veRL `10题×8条`在16 NPU约60分钟的实测吞吐，双机32 NPU按约20题组/小时估算：64题pilot基线3.2小时，计入长尾按4–6小时；全部281题基线14.05小时，计划区间16–24小时。
- 下一步不直接满跑281题：先对64条分层候选做显式题意—gold—SQL审核，预计48–96分钟；通过后双机满载`64×8`，加评分/bucket后约5–8小时得到首个决策结果。CPU机械通过不等于训练批准，训练与晋级继续关闭。
- 本轮定向筛选器测试与全仓回归均通过：`4 passed`、`369 passed`。

### v1.11.5 — 2026-08-13

- 采用多沙箱供给方向：5、6号机共32张NPU均为空闲；老板原始SFT沙箱目录包含19个版本、每版500题和独立SQLite，共9,500题。先跨版本筛题比继续只围绕v15造题更能扩大供给并降低单沙箱过拟合。
- 新增只读跨沙箱画像器：逐版本统计grain、task/answer/QA/difficulty分布、任务与题面重复、answerability和source-validation覆盖；可在SQLite `mode=ro + query_only + authorizer`三重门下执行verification SQL，核对非空结果与hidden gold，并复用题意—SQL确定性语义预警。
- 安全summary只输出聚合计数；含题面、gold、SQL和task identity的候选包强制0600且默认不进入Git。候选定义是高精度初筛而不是训练批准，仍需显式语义审核、冻结集隔离和fresh rollout。
- 本地测试覆盖机械匹配与语义预警分离、跨版本题面/identity重复和SQLite变更语句拒绝，完整回归为`368 passed`；服务器执行尚未开始，因为向服务器项目目录同步新脚本需要额外的显式传输授权，未尝试绕过权限门。

### v1.11.4 — 2026-08-13

- 综合训练供给、42条语义审核、7对原生候选margin门、纯最终结果重算和PI/veRL `10×8`运行时对照后，第一优先仍是重建可信任务供给，而不是训练、重复奖励重放或再次运行旧题完整`10×8`。
- 下一批改为32条SQL-first训练候选加3条parity-only哨兵：先执行只读SQL得到hidden final result，再反向写明指标、过滤、分组、时间范围和输出形态；SQL只用于造题与语义验收，模型仍只接收题面，奖励仍只比较最终结果。
- 3条哨兵在训练集之外永久冻结；CPU语义门通过后才做`3题×2条×双臂`请求级复验，对齐temperature/top-p/top-k、每次assistant 8,192-token上限、墙钟和终止统计。通过后才对32条训练候选生成fresh `n=8` veRL groups。
- 保持48对训练硬门：当前7对只作为off-policy候选层，新增41对前不训练；mixed groups可进入当次更新，全对/全错只临时排除且不得永久删除。
- 决策证据独立复算通过；报告canonical artifact包含完整Executive Summary、pair供给图、动作表、后续问题与边界。当前桌面无MCP report renderer，portable reader在官方builder中持续fallback并超时，因此保留artifact与report notes，不发布未完成渲染QA的HTML。

### v1.11.3 — 2026-08-13

- 追踪真实采样调用链后确认：veRL普通验证默认`val_kwargs.temperature=0`，GRPO训练rollout默认`temperature=1.0`；本次standalone parity以`validate=true`读取显式覆盖的`val_kwargs.temperature=1.0/top_p=0.95/top_k=20`，没有使用temperature 0。PI runner/client不传三项覆盖，由Step120服务`generation_config.json`提供相同`1.0/0.95/20`和`do_sample=true`。
- 复核80条构造：PI是`10×8`个独立native session位置，veRL是10行batch按8次interleave扩成80个独立逻辑request；两臂各10组均为`8/8`完整assistant轨迹互异，组内两两重复比例为0，排除复制候选。新增只输出计数的服务器侧去重审计器。
- 撤回“其他配置全部相同”的过强表述：PI/veRL在每次assistant token上限、上下文压缩、墙钟/回合终止、32/16客户端并发、工具实现、vLLM显存比例和逐请求seed配对上不一致。既有结论降级为部署路径兼容性失败，不能做严格单变量框架归因；准确率共同触底和训练/bucket禁用结论不变。
- PI后续运行新增generation config硬门，temperature 0、`do_sample=false`或top-p/top-k漂移会在生成前失败；standalone veRL安全合同显式记录训练/验证temperature来源和`strict_runtime_configuration_matched=false`。本地完整回归为`365 passed`。

### v1.11.2 — 2026-08-13

- 完成 Step 120 的 PI-Agent / veRL `10题×8条×双臂` 16-NPU 并行运行。两臂结构均为80条、10组且样本序号完整；最终结果均为`0/80`、10组全错，准确率差为0但属于共同触底，不能作为运行时等价证据。
- PI/veRL 最终回答率为`86.25%/70.00%`，绝对差`16.25pp`超过10pp门；PI 首轮30分钟上限有`12/80`超时，恢复后仍有1条达到60分钟上限。完成率、零超时和零运行错误门失败，parity总门失败。
- 修正 PI 轨迹错误审计：中途 assistant error 后成功重试不再误判为终止失败；首次超时仍永久计入运行时门。安全汇总进一步区分观察到的uniform groups与实际可执行筛选，门禁或语义/数据角色不满足时optimizer排除数保持0。
- 10条诊断题继续保持evaluation-only，未开放训练、未删除全错组、未挑选单条轨迹。原始题面、gold、SQL、任务标识与轨迹只留服务器，Git仅保存合同、代码、测试、报告和无标识聚合；两机实验vLLM worker已释放为0，本地完整回归为`359 passed`。

### v1.11.1 — 2026-08-13

- 用 PI CLI 实际请求路径校正双臂采样合同：PI 不显式发送 sampling overrides，实际继承 Step120 `generation_config.json` 的 `temperature=1.0 / top_p=0.95 / top_k=20`；veRL 臂显式镜像三项，避免沿用未生效的配置文件 `temperature=0.7`。
- 增加专用 veRL `10×8` val-only 启动器和双臂敏感轨迹规范化器；纯最终结果统一重算，PI API 错误、超时和 veRL runtime error 成为零容忍结构门，失败请求不能被“进程退出码0”误计为有效轨迹。
- 首个 PI 启动因 vLLM 缺少 Qwen3 Coder 自动工具解析而产生80条API错误，已完整隔离为不可用审计目录；修正服务启用 `qwen3_coder` tool parser 后从全新目录重跑，不复用失败轨迹。

### v1.11.0 — 2026-08-13

- 增加不读取 SQL、工具证据或过程分的纯最终结果 shadow scorer，并保留二值正确性与仅诊断用 dense 分数两条独立信号。
- 增加冻结 val20 的确定性 `5 numeric + 5 table` 诊断集构造器、老板原生 PI-Agent 32-worker 运行器、`10×8` 双臂分布比较器和预注册运行合同；敏感 prompt/轨迹/答案仍只保留服务器端，Git 仅记录安全合同与聚合门禁。
- 明确 uniform group 的使用边界：全对与全错只从当次 GRPO 更新排除，不永久删除、不挑单条轨迹，筛选后训练必须重新采样完整 group。

### v1.10.0 — 2026-08-13

- 获得明确数据授权后，逐条审核服务器内42条最低机械风险任务；服务器侧完整联结验证通过，敏感题意、SQL、gold和task ID继续保留在权限`0600`文件中，Git只包含去标识review index、证据门、原因码和聚合结果。
- 42/42均通过机械与SQLite顺序扰动稳定性门，且42/42期望值受查询结果支持；但题意无歧义蕴含gold、SQL完整回答题意和最终语义批准均为`0/42`。高严重度根因类别集中于latest/时间范围/指标分组/归因/跨域题意被窄静态聚合替代，当前review-required队列不得用于rollout或训练。
- 以`Beta(0.5,42.5)`预测剩余96条语义批准、再积分历史pair产率`Beta(22.5,42.5)`：预计批准`1.12`条、形成`0.39`对pair，95%预测上界2对，补齐41对缺口的概率约`7.1×10^-18`。停止审核剩余96条；下一步先构造32条显式current-definition任务pilot，通过后扩至172个已批准候选，并按32条分批采集，新增41对即停止。
- 新增服务器侧敏感联结汇总器、本地供给重算器、逐条去标识决策清单、canonical技术报告artifact与回归测试。portable HTML打包器在本次和既有对照artifact上均因reader停留fallback而超时，未将未完成最终上下文QA的HTML写入仓库，阻断证据已记录在report notes。
- 本地完整回归为`340 passed`；新增JSON均可解析，Git安全产物未检出task编号、服务器路径或主机名，训练与promotion开关继续保持关闭。

### v1.9.0 — 2026-08-13

- 修复派生pair/calibration资产的冻结身份提取：优先使用`source_task_id`，再回退ground-truth/display identity。完整排除eval22、chosen-only calibration16、旧frozen16、val20、test20后，原生候选由旧估算11对修正为7对；额外4条全部命中chosen-only calibration16。
- 7对原生真实首错通过机械chosen、真实rejected/tool result、相同状态前缀、token/mask和Step120 TP4/PP2/CP2纯前向门。正确semantic-delta/full-SQL均为`0/7`占优，平均margin为`-1.7816/-0.9334`；运行166秒、无optimizer、无checkpoint、结束后NPU进程为0。
- 训练门继续固定为48对，保留7对后缺口为41；review138即使全部语义批准，按历史pair产率补足缺口的预测概率约`76.5%`，90%/95%把握需158/172个已批准候选。
- 在服务器内生成权限`0600`的42条最低机械风险语义审核包；42/42通过SQLite反向无序扫描稳定性探针，但自动语义批准仍为0，必须逐题裁决instruction—gold—SQL对应关系。敏感prompt、SQL、答案、task ID和证据不进入Git。
- 本地全项目回归为`335 passed`；新增Python/JSON、远端shell语法、敏感文件权限、空checkpoint目录与NPU释放门均通过。

### v1.8.0 — 2026-08-13

- 修正v1.6.0容量规划中“复用现有22对”的隐含假设：这22条现已冻结为eval22，训练可用数必须记为0；当前review138即使全部批准，独立达到48对的后验预测概率也只有约50%，且该值尚未扣除语义审核淘汰。
- 服务器侧CPU复用既有原生full25逐题结果：27个已观测首错中16个与eval22重叠并排除，余下11个与chosen-only calibration16的额外重叠为0；其中10个为可执行但证据错误/不足、1个为schema/语法/执行错误。这些状态尚未通过chosen构建和token门，因此训练继续关闭。
- 将下一优先动作改为：先把原生11条构造成严格pair并完成CPU门，再用Step120做纯前向margin；随后以42条最低风险review任务估计真实语义批准率。若11条全部通过，剩余37对在review138全部批准的理想条件下达到概率约87.6%；90%/95%把握分别需要143/156个已批准候选，仍需额外容量安全余量。

### v1.7.0 — 2026-08-13

- 把 full25 已得到、但未达到训练数量门的 22 个真实 Step 120 首错状态转为独立 evaluation-only 契约：44 行 chosen/rejected 邻接、真实首错与工具结果、机械正确 chosen、相同前缀、token mask、集合隔离和内容哈希全部 fail closed；数据显式禁止训练和模型晋级。
- 完成 Step 120 纯前向基线：semantic delta 正确候选仅 `3/22` 占优，平均 margin `-1.1454`；17 条 aggregation 状态只有 `2/17` 占优。与旧 frozen16 合并后，共 38 个互不重叠失败状态只有 `3/38` 正确候选占优，系统性误排得到独立复现。
- 完成现有 chosen-only 一步 checkpoint 的同 22 条对照：`18/22` margin 改善且零更早分叉回退，但正确占优仍为 `3/22`、full-SQL 仍为 `1/22`，未达到预注册 `17/22`。按合同拒绝候选，不跑 full64、不追加训练、不晋级；下一步冻结 eval22，并从不重叠任务另建至少 48 条真实首错训练 pair。

### v1.6.0 — 2026-08-13

- 综合原生/Step 120 full25、chosen-only、pairwise、schema-oracle 和真实首错预算阶梯重新排序下一步：当前不训练，不降低 48 个独立 pair 门槛，也不优先在 oracle schema 上造合成 operator pair。
- 以 full25 的 `22/64 = 34.4%` 实测真实 pair 产率倒推，补足 26 对预计需要约 76 个新任务；规划先从 138 条 review-required current-definition 队列中按 aggregation/operator 分层，新增至少 80 个经语义裁决且无冻结集重叠的独立任务。该 80 是 CPU 容量规划值，真实开门仍要求累计 `≥48` 个实际 Step 120 首错 pair。
- 固化后续顺序为：CPU 语义裁决扩池 → 32 题一批 full25 真实状态采集 → Step 120 forward-only margin → 条件式一步 pairwise → 同 64 题 Pareto 复评；任何一层失败均不初始化下一层训练或晋级 checkpoint。

### v1.5.0 — 2026-08-13

- 完成获准的一步 train48 chosen-only 全参 SFT：退出码 0，loss `1.347718`、grad norm `60.4354`、峰值 HBM `26.53 GiB`、墙钟 `358s`；最终 model/extra checkpoint 含有效 `.metadata`，optimizer 文件为 0。
- 完成相同 calibration16 的训练后纯前向：SQL NLL 相对改善 `12.75%` 且 `16/16` 逐题改善，但 greedy token 仅增加 5 个，未达到预注册 `+12`；7 项门禁中 6 项通过、1 项失败，因此不做自由回放、不续训、不晋级。
- 增加首个 non-greedy 边界安全聚合：仅 2/16 从 query-start 移到更后位置，14/16 保持同一首分叉；原有 aggregation 障碍 `9/9` 全部未清除。下一阶段先做不重叠、经语义复核的 aggregation/operator 对比数据 CPU 门禁，训练继续关闭。

### v1.4.1 — 2026-08-13

- 修复 chosen-only 一步启动前 CPU 门的模块解析顺序：项目根目录现在位于容器自带 `/verl` 之前，避免其同名 `scripts` 包遮蔽本仓库门禁模块；失败发生在训练启动前，未初始化 optimizer、未更新参数、未生成 checkpoint。

### v1.4.0 — 2026-08-13

- 新增 chosen-only 首动作加权 dataset：loss 仅由互斥的 tool structure `0.25×` 与解码 SQL `8×` 组成；不含工具结果或最终答案分量。
- 新增获准的一步 train48 全参 SFT 入口：Step 120 model-only 初始化、新 CPU-offload Adam、batch48 恰好一次、`1e-6`、TP4/PP2/CP2、只保存 final `model,extra`；启动器再次核对 train48 哈希、calibration16 隔离、CPU gate 和 canary decision。
- 新增 post-forward calibration16 决策器：严格复用 v1.3.0 阈值，并增加逐题 first-nongreedy SQL offset 不得提前的边界退化检查；全门通过也只开放一次 calibration16 单动作自由回放，不开放追加训练或晋级。

### v1.3.0 — 2026-08-13

- chosen-only 64 条 CPU 门实跑通过：序列 `1,637–2,035` tokens、首动作 `61–112` tokens、SQL `6–57` tokens，全部无截断，loss 仅覆盖唯一 assistant tool action，system/user loss 为 0。
- calibration16 的 Step 120 teacher-forced 纯前向完成：SQL NLL `1.2913`，381 个 SQL tokens 中 greedy `277`、top-5 `344`、平均 rank `18.78`；16/16 均有非 greedy SQL token，完整 SQL 全 greedy `0/16`。运行 `exit 0`、无 optimizer/checkpoint。
- 预注册且仅开放一步 train48 chosen-only 金丝雀：tool-structure/SQL 权重 `0.25/8`、新 CPU-offload Adam、model-only checkpoint。只有 calibration16 SQL NLL 至少相对改善 5%、至少 12/16 逐题改善、greedy tokens 至少 `+12`、top-5 不低于 344、mean rank 改善且结构 NLL 退化不超 5%，才允许后续自由回放；promotion 仍关闭。

### v1.2.0 — 2026-08-13

- teacher-forced component runner 将 final-answer mask 改为可选组件，同时保持 assistant、tool-turn、tool-structure 和 SQL mask 为必需组件；既有两回合修复诊断继续输出 final-answer，本轮单首动作数据无需伪造最终回答。
- 新增 calibration16 专用 Step 120 teacher-forced 启动入口：TP4/PP2/CP2、精确 vocab-parallel SQL rank、forward-only、空 load/save contents、无 optimizer/checkpoint；启动前重跑 chosen-only CPU tokenization/loss-mask 门。
- train48 不进入基线；本阶段只测 Step 120 对正确首动作与 SQL token 的初始概率，得到结果前 training 与 promotion 继续关闭。

### v1.1.0 — 2026-08-13

- 新增 chosen-only Qwen3.6 首动作 tokenization/loss-mask 门：完整渲染 `system,user,assistant(tool_call)` 与老板四工具 schema，`truncation=error`，逐条要求 loss 恰好等于唯一 assistant 动作且非 assistant 上下文 loss 为 0。
- 首动作 component mask 将监督目标无重叠、无遗漏地拆成工具结构与解码 SQL 内容，SQL shell 外层引用不冒充语义 token；任一空 mask、越界、重复 command、模板 token 漂移或合同哈希不一致都 fail-closed。
- CPU 门禁通过后只开放 calibration16 的 Step 120 teacher-forced 纯前向基线；training 与 promotion 仍关闭。

### v1.0.0 — 2026-08-13

- 新增 chosen-only schema-conditioned first-action 数据构建器：对完整 64 题重新执行只读 gold SQL、核对非空结果与 expected value 支持，只生成 `system,user,assistant(tool_call)`，不加入工具结果或最终答案。
- gold SQL 只存在于唯一监督 assistant 标签；用户 prompt 仅包含 gold 选中表的 SQLite metadata 与固定 `/workspace/logistics.sqlite` 动作要求，并显式标记 `oracle_relevant_table_selection=true`、`deployment_ready=false`，禁止把诊断输入当成可部署方案。
- 按固定 seed 和 answer type 分层拆为 train48/calibration16；合同初始保持 CPU tokenization、teacher-forced、training 和 promotion 全部关闭，下一步只允许 Qwen3.6 chat-template 与首动作 assistant-only loss mask 门禁。

### v0.99.0 — 2026-08-13

- Step 120 task-specific schema-oracle 有效 64 题运行完成：强制 checkpoint→rollout 同步、greedy `2/1`、`exit 0`、64/64、纯前向且无 checkpoint；首查询为正确/等价 `4`、带真实结果错误 `35`、无只读查询 `25`。
- 正确门 `4/64 < 32/64`、错误 pair 门 `35/64 < 48/64` 均失败；schema runtime、pair 构建、训练与晋级全部关闭。错误查询以可执行但证据错误/不足 `23`、空结果 `10`、schema/语法/执行错误 `2` 为主。
- 固化安全汇总、完整报告和结果锁定测试；下一步只构建 chosen-only schema-conditioned 正确首动作，先过 CPU 防泄漏、SQL 等价、tokenization、loss mask 与 48/16 分割，再决定是否做 teacher-forced NPU 基线。

### v0.98.4 — 2026-08-13

- 修正共享首查询 outcome adapter 的合同分派：query-initiation 与 structured-SQLite 继续严格要求 `3/3`，task-specific schema-oracle 则严格要求 `2` 个助手回合与 `1` 次工具反馈；不再因旧协议硬编码拒绝有效轨迹。
- 新增两类合同的正反回归覆盖；分析器仍核对数据行数、Parquet 哈希、training=false 与只读数据库执行，错误合同继续 fail-closed。

### v0.98.3 — 2026-08-13

- 补齐老板原版轨迹转换器与在线 agent-loop 的边界一致性：在线 runtime 可能接受一个 token 截断的 Qwen tool block、返回真实 parser error 后继续；转换器现仅在“恰有一个开放 tool call 且紧随真实 tool response”时保留该调用和错误响应，并单独审计 `truncated_nonterminal_tool_calls`。
- 仍对无前序调用响应、完整/截断调用混组、多开放调用、无参数或无真实响应保持 fail-closed；不会补造工具结果，也不会把未执行的查询记为观察样本。

### v0.98.2 — 2026-08-13

- 同步修正 schema-oracle 数据合同的 `next_action` 机器字段：明确记录“首个 greedy action + 一次真实工具结果”协议，移除遗留的 `one_turn` 描述，并新增合同回归测试，防止启动配置与密封数据元数据再次漂移。

### v0.98.1 — 2026-08-13

- 修正 schema-oracle action 诊断的 agent-loop 轮次合同：首次 64 题运行虽全部生成工具调用，但 `max_assistant_turns=1` 会在第一条工具结果执行前终止，导致 `64/64` 响应缺失，不能用于门禁判定。
- 有效协议改为最多 2 个助手回合、仅 1 次工具反馈：第一回合必须发出唯一 bash/SQLite 查询，真实工具结果返回后第二回合禁止再调用工具并只作简短确认；仍为 Step 120 强制同步、greedy 纯前向，optimizer/checkpoint 关闭。

### v0.98.0 — 2026-08-13

- 新增完整不重叠 64 题的 task-specific schema-oracle action 门禁：schema 只从只读 SQLite `table_info/foreign_key_list` 提取，表集合由已机械验证 gold SQL 的 FROM/JOIN 决定；prompt 不含数据库行、工具结果、expected value、gold SQL 或答案。
- 首版曾将每题限制为一个助手回合和一次工具反馈；实跑后由 v0.98.1 更正 agent-loop 轮次语义。该门禁是由 gold SQL 选相关表的诊断上界，不是可部署输入；若正确/等价首查询 `≥32/64`，下一步必须改用不依赖 gold 表选择的全库 schema 在冻结 val20 上验证，否则只有不同任务的带结果错误查询 `≥48/64`，才允许构造 correct-vs-actual-wrong pairs。
- 通过 wrong-pair 数量门也只开放 pair 构建与 CPU 审计，不开放 optimizer；两门都失败则转为 chosen-only schema-conditioned action supervision 设计，训练与晋级仍保持关闭。

### v0.97.0 — 2026-08-13

- 结构化 SQLite realization 实跑完成：Step 120 强制同步、同一 41 题、greedy 3/3、`exit 0`，但可识别只读查询为 `0/41 < 31/41`；没有未观测 SELECT/WITH，可排除工具结果缺失导致的假阴性。
- 相比通用提示，总工具调用 `172→132`、重复 Bash `112→107`、未观测调用 `63→48`，但路径/CLI-only 仍有 `81`、schema catalog/definition `22`，`41/41` 均未从探索转成查询。更强通用运行时指令路线停止。
- 下一无训练门改为完整不重叠 64 题的单回合 task-specific schema oracle：若正确/等价首查询 `≥32/64`，优先 runtime schema injection；否则只有不同任务的带结果错误查询 `≥48`，才允许构造同状态 correct-vs-actual-wrong pairs，训练仍默认关闭。

### v0.96.0 — 2026-08-13

- 新增结构化 SQLite realization 门禁：严格复用 v0.95 的同一 41 个 Step 120 full25 无查询任务和 hidden verifier，只替换通用干预为“路径定位最多一次、schema 检查最多一次、第三回合必须非交互 SELECT/WITH”，禁止交互 shell、重复命令和猜答案。
- 数据构建器必须验证上游 41 行合同与 Parquet 哈希，并机械移除旧通用干预，防止两条指令叠加；新合同冻结 `3 assistant / 3 tool-result`、`31/41` 恢复线、task-specific 信息零披露和训练/晋级关闭。
- 结果适配器扩展为显式支持两种诊断合同，结构化判定器同时核对首查询与命令族行数。过线只允许先验证 full64/val20 运行时约束；不过线才进入至少 48 条机械验证 schema-grounded action pairs 数据门，门禁本身不初始化 optimizer。

### v0.95.1 — 2026-08-13

- 移除 v0.95.0 新报告与测试文件末尾的多余空白行，恢复 `git diff --check` 零警告；实验数据、`0/41` 判定、下一门槛和训练关闭状态均未改变。

### v0.95.0 — 2026-08-13

- 固化 Step 120 查询启动干预证据包：有效运行 `41/41`、带真实工具结果的只读查询 `0/41 < 31/41`，训练与晋级继续关闭；报告同时保留 `2` 条未观测查询和 `39` 条无查询，避免把截断混入成功。
- SQLite 细分显示干预后 `41/41` 已进入 SQLite、`18/41` 做过 schema discovery，但 `91` 次停在路径/CLI、`28` 次停在 catalog/schema、重复 Bash `112`。瓶颈收敛为非交互调用、去循环和 schema→SELECT 实现，而不是完全忽略工具指令。
- 下一无训练门预注册为同 41 题的结构化三回合 `path→schema→SELECT/WITH` 工作流，仍以 `31/41` 带结果查询为通过线；失败后才允许准备 task-specific schema grounding/action supervision，且机械验证训练对仍须至少 48 条。

### v0.94.3 — 2026-08-13

- 查询启动干预门禁最终为 `0/41` 条带真实工具结果的只读查询，另有 `2/41` 发出可识别查询但结果未观测、`39/41` 无可识别只读查询，远低于预注册 `31/41`；因此否定“只加一句运行时查询指令即可修复”，训练与晋级继续关闭。
- 41 条轨迹共含 `172` 次工具调用，其中 Bash `170`、read `2`；初版聚合发现 `119` 次 SQLite 相关但非已识别只读查询、`45` 次 `ls`、重复 Bash `112`。这表明模型并非完全忽略 SQLite 指令，需进一步区分 schema meta、路径/CLI、未解析只读和不安全命令。
- 命令族审计器新增 SQLite 安全聚合细分，并增加“任意 SQLite”和“schema discovery”逐题覆盖计数；仍不输出命令、SQL、prompt 或工具结果。下一训练目标暂锁定为 schema discovery / tool realization，细分审计完成前不初始化 optimizer。

### v0.94.2 — 2026-08-13

- Step 120 查询启动诊断已真实完成：强制 actor→vLLM 权重同步标记存在，`41/41` 输入全部落盘、运行 `exit 0`；全程为 greedy n1、forward-only、val-only，未初始化 optimizer、未保存 checkpoint。
- 新增查询启动结果专用适配器。它拒绝把原 64 题候选合同冒充干预合同，且必须同时满足 Parquet 哈希、合同/Parquet/轨迹行数一致、`3 assistant / 3 tool-result` 和 `training_allowed=false`，才复用统一的首只读查询机械分类。
- 此版本只补齐结果审计链，尚不提前写入恢复数或门禁结论；最终数字必须来自适配器与预注册 `31/41` 判定器的实际输出。

### v0.94.1 — 2026-08-13

- 查询启动门禁的工具反馈预算由 `2` 修正为 `3`，与最多 `3` 个助手回合严格对齐；因此模型在第 3 个助手回合发出的只读查询也能获得真实工具结果，不会因协议截断被误记为未观测失败。
- 新增合同回归测试锁定 `3 assistant / 3 tool-result`，修复仅影响尚未启动的诊断运行，不改变 41 题选择、无泄漏干预、`31/41` 通过线或训练关闭状态。

### v0.94.0 — 2026-08-13

- 冻结下一项最高信息价值的无训练门禁：对 Step 120 完整 25 回合仍为 `no_readonly_query` 的预计 `41` 题，追加任务无关、无答案泄漏的查询启动约束，仅运行最多 3 个助手回合；原问题与 hidden verifier 保持不变。
- 预注册通过线为至少 `31/41` 题产生带真实工具结果的可识别只读查询。过线说明查询能力主要受策略路由约束，先验证运行时指令；中等恢复转向 native-anchored 启动/完成对比采集，低恢复才转向 schema discovery / tool realization 修复。
- 数据构建器对完整 64 题候选合同、原 Step 120 轨迹、严格 `no-query` 数量和干预泄漏标志全部 fail closed；结果分析器无论分支均保持 `training_allowed=false`、`promotion_allowed=false`，不会初始化 optimizer 或保存 checkpoint。

### v0.93.0 — 2026-08-13

- 完成原生 Qwen3.6-27B 与真实 Step 120 的同一 64 题完整 26/25 回合公平对照：两侧均 `exit 0`、`64/64` 落盘、prompt 完全一致，原生明确跳过同步，Step 120 明确完成 dist checkpoint 强制同步；老板原版评分两侧均 `64/64` 成功。
- 类奖励黑客的 `result_wrong_process_ok` 在原生/Step 120 为 `13/8`，工具调用 `1078/991`、重复 Bash `141/120`，因此缺陷在训练前已存在且未被训练放大。Step 120 numeric correct `5→7`，但只读 SQL 覆盖 `30→23`、完整回答 `40→35`、平均总奖励 `0.2858→0.2612`；逐题差异均未达双侧 `p<0.05`，作为方向性覆盖回退而非确定因果。
- 新增首 SQL 结果审计、逐题匿名转移比较、完整归因汇总和 native-anchored Pareto 晋级门禁。未来候选必须同时达到 SQL 覆盖 `≥30`、完整回答 `≥40`、numeric correct `≥7`、平均总奖励 `≥0.285840625`、高过程分错答 `≤8`；任一失败都不晋级，门禁本身不授权追加训练。

### v0.92.0 — 2026-08-13

- 强制同步后的 Step 120 已在同一 64 条不重叠任务上完成 3/12/完整 26 助手回合的预算阶梯：合格首错 pair 分别为 `1/23/22`，全部低于 `48`，训练继续 fail closed；完整预算运行 `exit 0`、`64/64` 行落盘，未初始化 optimizer、未产生 checkpoint。
- 完整 25 次工具反馈将终止回答从 12 回合运行的 `27` 提高到 `35`、缺失工具响应从 `93` 降到 `51`，但仍有 `41/64` 题没有可识别只读 SQL；延长同一 64 题已无足够数据收益，禁止降低 pair 门槛或把无 SQL/未观测调用伪造成 rejected。
- 原生回放入口的行数合同从写死 16 改为显式 `EXPECTED_EVAL_ROWS`，默认行为不变，并同步用于实验合同和跨节点结果回收。下一步补齐同一 64 题原生模型完整 25 回合对照，再决定是否对 `138` 条 review-required current-definition 任务做额外语义复核扩池。

### v0.91.0 — 2026-08-12

- 强制同步后的 Step 120 在 64 条不重叠任务、最多 3 个助手回合上完成有效采集，但只得到 `1` 个实际首错 pair；其余为 `60` 条无只读 SQL、`3` 条首 SQL 无真实工具响应。`1 < 48` 数量门禁 fail closed，未启动 optimizer、未生成 checkpoint。
- 新增无敏感载荷的 rollout 命令族审计器。同一 64 题取证对照显示原生/真实 Step 120 分别只有 `1/64`、`4/64` 进入可识别只读 SQLite，两边均有 `83` 次重复 Bash；主要预算都用于 `ls/find/grep/read/head`。因此短预算探索模式在原生模型中已存在，Step 120 未新制造或放大，3 回合协议本身也不足以生成训练 pairs。
- 撤回旧报告中通过未同步 vLLM 得到的 Step 120 自由回放归因，只保留不经过该路径的相同状态 forward-only margin 证据；新增安全报告与 JSON 汇总。下一步在相同 64 题上延长 Step 120 交互预算，仍须达到至少 48 个真实观测首错 pair 才允许 margin/训练门禁。

### v0.90.2 — 2026-08-12

- val-only 强制 checkpoint 同步补丁新增 Ray task 内的明确审计标记：只有真实进入 `_fit_update_weights()` 前才打印 `force actor-to-rollout weight sync`，不再仅靠“没有看到 skip”作间接判断。
- repair replay 无人值守启动器在退出码 0 后仍会检查 driver 日志：出现 skip 标记则改判退出码 9，缺少 force 标记则改判退出码 10，并写入 `CHECKPOINT_SYNC_INVALID`；两种情况都不会继续结果回收或下游 pair 门禁。

### v0.90.1 — 2026-08-12

- 64 条短回放日志暴露出权重归因漏洞：actor 虽加载 Step 120 dist checkpoint，但 Ray `OneStepTaskRunner` 未继承提交 shell 的 `LLIN_VAL_ONLY_FORCE_DIST_SYNC` 环境变量，明确打印了跳过 actor→vLLM 初始同步；`-01/-02` 采集因此判为原生权重无效运行，已保留现场并终止，未进入 pair 数据或训练。
- 强制同步开关改为序列化 trainer config `val_only_force_dist_sync=True`，由 Hydra 命令行显式传入 Ray task；补丁可将服务器上的旧环境变量条件原位迁移为 config 条件，并保持幂等。
- 此发现不影响原生/Step 120 teacher-forced checkpoint 概率对照，但此前通过 repair replay 入口得到的“Step 120/SFT 自由回放”不再作为 checkpoint 归因证据；修复后必须重跑并看到实际 actor→rollout 同步标记，才能恢复该证据链。

### v0.90.0 — 2026-08-12

- pairwise trainer 从固定 32 行扩展为“一个正偶数、恰好覆盖全数据集的 global batch”，仍保持 DP=1、顺序 sampler、每个 microbatch 一对相邻 `chosen → rejected` 和一次 optimizer step；原冻结 16 对默认批量继续兼容。
- 新增不重叠 48–64 对一步金丝雀入口，启动前交叉验证数据合同、token gate 与 Step 120 margin 决策的 pair/row 数和所有 fail-closed 标志；任何旧合同、数量漂移或未授权 margin 都不会初始化 Adam。
- 新 checkpoint 只保存 model+extra、不保存 optimizer，且不得先跑自由回放；唯一下一门是原冻结 16 题的 `chosen ≥12/16`、margin 改善 `≥12/16`、更早分叉回退为 0。

### v0.89.0 — 2026-08-12

- 新增 48–64 对可变批量的 CPU token 门禁与 Step 120 forward-only margin 入口；不再复用冻结 16 题脚本中写死的 32 行、16 对和 critical-token 身份。
- token 门禁要求实际 pair 数达到 48、每对严格 `chosen → rejected` 相邻、semantic-delta mask 非空且完全位于 SQL、候选 sign/pair index 与行序一致；任何一项失败均在加载 NPU 模型前停止。
- margin 分析按实际 pair 数使用向上取整的 75% 正确偏好阈值，并复用既有 aggregation/query-start/clause/identifier/operator 家族口径；只有正确 delta 未达到 75% 偏好时才允许一次新数据 pairwise 金丝雀，之后仍必须回到原冻结 16 题做 `12/16` 外部概率门禁。

### v0.88.1 — 2026-08-12

- 首次 64 条过滤在遇到“第二助手回合生成 SQL、但因一次工具反馈上限而未执行”的题时 fail closed；没有输出半成品 pair，也没有启动训练。
- 首错 pair 构建器现在先要求首条只读 SQL 的 call ID 存在真实 tool response；未观测 SQL 结果单独计入排除项，不执行合成结果、不借用其他工具返回，也不阻断其余候选的聚合门禁。

### v0.88.0 — 2026-08-12

- 新增不重叠首错 pair 构建器：逐题只读执行 Step 120 第一条 SQL，并同时检查当前 gold 支持与教师结果等价；只保留确实错误/不足的实际查询，禁止把正确等价查询误标为 rejected。
- 每对数据冻结模型实际 assistant tool call 与实际工具输出作为相同前缀，chosen 为当前权威 verification SQL、rejected 为同一次回放的实际首错 SQL；两条候选后的工具/答案尾部固定为不计分 stub。
- pair 数量门禁固定为至少 48；通过数量门禁后仍只授权 Step 120 forward-only token-family/semantic-delta 审计，不直接授权 optimizer 或 checkpoint 晋级。

### v0.87.1 — 2026-08-12

- 将 Step 120 repair replay 的无人值守合同从写死 16 行改为显式 `EXPECTED_EVAL_ROWS`，并把实验用途与 split 设为参数；默认值仍保持原冻结 16 题行为不变。
- 结果跨节点回收现在使用同一个期望行数，避免 64 条不重叠首错采集成功后被旧 `16` 行检查误判为无效；采集仍固定 greedy n=1、老板四工具和传入的最大助手/工具回合数。

### v0.87.0 — 2026-08-12

- current-definition 正式池审计已在 5 号机 train236 全量实跑：236/236 条 verification SQL 均可只读执行、非空且支持当前 gold；排除冻结 16 题、val20、test20 的 task/instruction/SQL 身份及语义高风险题后，得到 `64` 条 strict-available 新任务，达到 `≥48` 数据门禁。
- 新增 48–64 条不重叠首错采集输入构建器：历史 source prompt 仅提供老板 system/tool 运行契约，user instruction、verification SQL、expected value、required fields/tables 均从当前权威 manifest 重建，并逐项核对审计哈希。
- 构建产物只授权 Step 120 greedy 首查询采集，不是训练集；必须先按只读执行结果过滤出机械错误首查询并完成 pair 数据门禁，才允许启动 pairwise optimizer。

### v0.86.0 — 2026-08-12

- 新增不重叠 pair 数据池的训练前审计器：不再把 193 条 source-instruction 漂移任务直接判死或静默放行，而是以当前权威任务定义重建候选身份，再逐条执行只读 verification SQL 并核对 expected value。
- 门禁同时隔离 val20、test20 和冻结 16 题的 task ID、user instruction hash、verification SQL hash；输出只包含 task ID、哈希、风险标签和聚合计数，不包含原始问题、SQL、答案或工具结果。
- 当前已确认旧严格候选共 25 条、冻结后仅余 9 条，因此在新审计达到至少 48 个 strict-available 新任务前禁止启动 NPU pairwise 训练；两机 NPU 当前均无运行进程。

### v0.85.0 — 2026-08-12

- 完成原生 Qwen3.6-27B 与 Step 120 的严格同状态概率归因：两者正确 semantic delta 均为 `0/16` 占优，平均 margin `-1.2057/-1.1877`；Step 120 有 `12/16` 逐题朝正确方向、`4/16` 朝错误方向，平均变化仅 `+0.0180`，核心 misranking 明确为训练前已存在且未被 Step 120 放大。
- 完成同 16 题、同 prompt、greedy n=1、老板四工具、48K/25 工具回合的原生自由回放和老板原版评分：原生/Step 120 的 `result_wrong_process_ok` 为 `13/16`、`12/16`，数值正确 `3/16`、`2/16`，重复命令均值 `12.19/10.25`，老板总奖励却为 `0.7813/0.7000`。因此高过程分错答在原生模型中已经大量存在，本次训练没有创造或放大该代理错配。
- 新增安全归因分析器、报告和 JSON 汇总；术语边界冻结为“proxy-aligned failure pattern”，不把现有证据夸大成模型意图。两项评估均无 optimizer/checkpoint，原生回放 exit code 0、墙钟 `49分15秒`，结束后两机 NPU 已释放。

### v0.84.0 — 2026-08-12

- 新增原生 Qwen3.6-27B 与 Step 120 的奖励代理行为归因入口：原生模型可在完全相同的 16 个 Step 120 首错状态和 correct-vs-actual-wrong pairs 上执行 forward-only semantic-delta margin，不初始化 optimizer、不保存 checkpoint，也不把归因结果误接到训练授权。
- 新增原生权重的同 16 题、greedy n=1、老板四工具、48K/25 工具回合自由回放入口；与既有 Step 120 回放保持 prompt、任务、解码和运行时合同一致，供老板原版 `result_wrong_process_ok`、过程分、数值正确、重复命令和完成率配对比较。
- 归因设计分离“相同错误状态下的模型条件偏好”和“各模型自然生成的端到端行为”：前者判断训练是否改变正确/错误 SQL 的相对概率，后者判断类似代理奖励利用是否在原生模型中已经自然出现；两项都只作诊断，不允许模型晋级。

### v0.83.0 — 2026-08-12

- 完成 Step 120 的一次 reference-free semantic-delta pairwise 全参金丝雀：16 对固定 `chosen → rejected` 数据、TP4/PP2/CP2、fresh CPU-offload Adam、`1e-6`、beta `1.0`，唯一训练步 loss `1.4948`、grad norm `98.01`，保存约 `51G` 的 model+extra checkpoint 且无 optimizer 文件。
- 同数据前后门禁显示 `16/16` margin 改善、均值 `-1.1877 → -0.7646`，但正确候选只从 `0/16` 升到 `3/16 < 12/16`；0 个更早分叉回退、0 个冻结 target 非法。流水线 exit code 0、总墙钟 `10分02秒`，质量门禁 fail closed，停止回放和追加训练，checkpoint 不晋级。
- 新增不含原始问题、SQL、答案和服务器路径的完整报告与 JSON 汇总；后续优先冻结当前 16 对为评价集并获取不重叠、机械验证、按首分叉家族分层的训练 pairs，在数据就绪前不占用 NPU。

### v0.82.3 — 2026-08-12

- 修复首次 pairwise 训练在首个 batch 前退出的问题：固定顺序 sampler 现在提供 veRL `SFTTrainer.fit()` 每个 epoch 必调的 `set_epoch()` 接口，同时仍严格保持 16 对数据的 `chosen → rejected` 邻接顺序且不 shuffle。
- `-01` 失败运行未执行 optimizer step、未保存 checkpoint，保留作审计证据；修复经单元测试和服务器语法检查后使用新运行编号重试，避免覆盖失败现场。

### v0.82.2 — 2026-08-12

- 修正自包含 pairwise 流水线的 token gate 交接：训练阶段显式读取刚由最终 32 行 baseline-forward 生成的 `token_gate.json`，不再回落到数据目录中增加 pair-order/sign 校验之前的旧门禁文件。
- 该交接仍发生在 optimizer 初始化前；若 baseline 的 delta mask、chosen critical target、相邻 pair 顺序、candidate sign 或 pair index 任一不合格，训练直接 fail closed。

### v0.82.1 — 2026-08-12

- pairwise 一步流水线不再复用增加 `pair_index` 列之前的旧 baseline diagnostic；训练前自动用 Step 120 在最终 32 行 Parquet 上重跑同一 forward-only margin，随后训练一步并做 post-forward，确保 baseline/post 的数据文件哈希、token mask 与 pair 顺序完全一致。
- baseline、训练和 post-forward 各用隔离运行目录，最终比较器只接受 result v2 与相同 16 个 task；额外一次约数分钟的 Step 120 纯前向换取自包含、可复现的严格前后证据链。

### v0.82.0 — 2026-08-12

- 新增一次 reference-free pairwise plan-to-SQL 金丝雀：数据固定为 16 个相邻 `chosen → rejected` pair，关闭 shuffle，global batch 为 32、microbatch 为 2；loss 只比较两侧 semantic-delta token 的平均 log probability，并以 logistic ranking 直接纠正 Step 120 在 `0/16` 对中偏好实际首错 SQL的问题。
- 训练复用已实跑的 Step 120 TP4/PP2/CP2 全参 SFT 资源合同、`1e-6` 学习率、fresh CPU-offload Adam、全量重计算和 model+extra-only checkpoint；总 optimizer step 硬锁为 1，不保存 optimizer，不运行中间验证，不把 reference-free 目标称为 DPO。
- 流水线在训练前重建 baseline result v2，训练后自动用 Step 1 checkpoint 做同数据 forward-only margin，只有正确 delta `≥12/16` 占优、`≥12/16` margin 改善、0 个更早 non-greedy 回退且冻结 offset 处 target 合法时才允许短的一回合 semantic-plan replay；否则停止且不追加 pairwise 步数。

### v0.81.0 — 2026-08-12

- 完成 Step 120 的 16 题三臂 semantic-plan sufficiency gate：Control/operator/full plan 的 verified 或机械等价恢复分别为 `1/16`、`1/16`、`2/16`；operator aggregation-critical 为 `0/9 < 4/9`，full plan 为 `2/16 < 8/16`，自动决策锁定 plan-to-SQL realization/recovery。
- 完成相同首错状态的 16 对 semantic-delta forward-only margin：正确候选占优 `0/16`，平均/中位 margin 为 `-1.1877/-1.1270`；aggregation `0/9`、query-start `0/3`、identifier/literal `0/3`、clause `0/1` 全部偏好实际错误 SQL。冻结首个 non-greedy token `16/16` 精确重建，运行退出码 0，无 optimizer、无 checkpoint。
- 冻结一次 pairwise chosen-vs-rejected 金丝雀门槛：正确 delta `≥12/16` 占优、`≥12/16` margin 改善、0 个更早分叉回退；概率门禁通过前不跑完整自由回放。新增不含原始问题、SQL、答案和服务器路径的安全汇总与报告。

### v0.80.2 — 2026-08-12

- 修正 rejected 候选之后重复长工具结果导致个别配对序列达到 `13,589` tokens 的无关开销：chosen/rejected 都只在待评分 SQL 调用之后追加相同的短工具/助手占位，候选前的真实“首错 SQL + 实际工具结果”状态保持逐 token 不变。
- 合同明确该尾部不参与 semantic-delta 结论；因自回归因果遮罩，未来占位不会影响更早候选 SQL 的 logits。这样继续使用已验证的 8K forward-only 配置，不为无关的候选后内容扩大上下文、显存与运行时间。

### v0.80.1 — 2026-08-12

- 修正真实 Step 120 首错 SQL 与 teacher-forced mask 的 shell 包装合同差异：共享前缀仍保留模型实际调用及实际工具结果，rejected 候选仅把等价 SQL payload 规范化为冻结的 `sqlite3 -json` 包装，SQL 文本、语义与查询哈希不变。
- 数据合同显式记录该规范化；回归测试同时校验候选包装满足 mask 契约、规范化前后解析出的 SQL 完全相同，避免把工具 wrapper 差异错误计入 semantic-delta margin。

### v0.80.0 — 2026-08-12

- 新增 Step 120 correct-vs-actual-wrong semantic-delta margin gate：16 题各构造 chosen/rejected 两行，二者共享完全相同的“首错 SQL + 实际工具结果”前缀；chosen 保留机械验证的纠正 SQL，rejected 重放模型真实首错 SQL，不把计划、结果或答案作为新增提示。
- token 级门禁用确定性序列对齐只标记两条 SQL 的 edit span，并为纯插入/删除在另一侧设置相邻锚点，保证两侧长度归一化 NLL 可比；输出正 margin 定义为正确 candidate 的平均 token log-probability 更高，并按既有 critical-token family 分层。
- 训练前硬校验 32/32 行无截断、delta mask 非空且属于 SQL mask、chosen 的冻结首个 non-greedy offset/target ID 16/16 精确重建；只有正确 semantic delta 未在至少 `12/16` 对中占优时，才允许进入一次 chosen-vs-rejected 训练金丝雀，否则转向受限 SQL planner 与 bash-only 工具策略。全流程 forward-only，不初始化 optimizer、不保存 checkpoint。

### v0.79.1 — 2026-08-12

- 修正 semantic-plan 一回合输出适配器对并行工具调用的处理：仍硬拒绝任何新工具结果，但不再因同一助手回合含多条未执行调用而丢弃整批生成；完整保留调用顺序，评分只取首条只读 SQL，并将多调用视为独立协议违规指标。
- 输出合同升级为 v2，逐行记录工具调用数、bash 调用数、名称顺序与“恰好一个 bash”合规性，汇总记录调用直方图和最大并行数；分析结果同步公开协议合规数，避免把格式服从性与首条 SQL 语义恢复能力混为一项。

### v0.79.0 — 2026-08-12

- 新增无训练的 semantic-plan sufficiency gate：同一 16 条首错状态严格复制为 `control / operator_oracle / full_plan_oracle` 三臂，唯一变量为结构化计划；operator 臂只含聚合、分组、过滤、时间、排序算子与计数，full 臂再加入机械核验的表、join edge 和分角色列，不含原 SQL、结果、最终答案或字面量。
- 冻结为一次 Step 120 模型加载、48 行贪心 `n=1`、`max_assistant_turns=1`：模型生成下一条 bash SQL 工具调用后即停止，代理不执行该调用；分析阶段才在 immutable/read-only SQLite 上执行并复用既有 gold 支持与教师结果等价口径，且不初始化 optimizer、不保存 checkpoint。
- 固化自动决策阈值：operator oracle 至少恢复 `4/9` 个 aggregation-critical task 且不回退 Control 已成功题，则转向 plan selection/contrast；否则 full plan 达到 `8/16` 才转向 schema grounding/compositional plan；仍未达到则锁定 plan-to-SQL realization/recovery。新增 5 个单元测试覆盖计划隔离、Parquet 合同、零工具结果、阈值优先级与无训练启动器。

### v0.78.2 — 2026-08-12

- 清理 semantic critical-token 最终报告日期行的 Markdown 尾随空格；实验数据、证据哈希、停止决策和下一目标均不变，并重新执行差异检查。

### v0.78.1 — 2026-08-12

- 完成 Step 120 的 16 题 semantic critical-token 一步金丝雀：训练退出码 0，loss `1.823778`、grad norm `171.6451`、峰值 HBM `26.59 GiB`；最终 checkpoint 含 32 个 model 与 8 个 extra distcp 分片、无 optimizer 文件。
- 严格同数据 forward-only 对比显示 SQL NLL `1.292942 → 1.143501`（`16/16` 改善），greedy `221 → 225`、top-5 `266 → 270`、mean rank `21.54 → 16.76`；原 critical token 中 query-start `3/3` 转为 greedy，但 aggregation `9/9`、identifier/literal `3/3`、clause keyword `1/1` 均仍为首个 non-greedy。
- 完整 SQL 概率 `>0.5` 保持 `2/16 → 2/16`，未达到 `12/16` 主门禁；按合同停止，不跑短/完整 replay、不做 held-out 或老板完整评分、不继续增加步数/单 token 权重。下一步先做无 NPU 的错误/纠正 SQL 语义差异 span 提取与 mask 门禁；实验结束后两机各 8 张 NPU 空闲。

### v0.78.0 — 2026-08-12

- 新增 critical-token canary 归因审计：逐题校验冻结 offset/target ID 与 Step 120 首分叉一致，再区分原临界 token 已转为 greedy、仍是首个非 greedy、或出现更早新分叉；输出仅含任务 ID、类别、offset、rank、概率和聚合状态，不复制原始问题、SQL 或答案。
- 审计器硬校验诊断 v3、相同任务/数据哈希、双端 forward-only 且均未初始化 optimizer；即使临界 token rank 改善，也只有完整纠正 SQL 概率 `>0.5` 达到 `12/16` 才允许进入短 replay。

### v0.77.0 — 2026-08-12

- 真实 16 题联合审计显示：`13/16` 涉及聚合/grouping 差异，主错误 `11/16` 为聚合；semantic critical token 中聚合函数 `9/16`、query start `3/16`，两者合计 `12/16`，另有 identifier/literal `3/16`、clause keyword `1/16`。
- 冻结下一严格单变量金丝雀：继续使用 Step 120、同一 16 条首错零-loss 状态、相同 `0.25/8/1` 基础权重和一步更新，只把每题 semantic-mask v3 的首个非 greedy SQL token 从 `8×` 提到 `32×`；不加入人工提示、不改变目标 SQL、不增加训练步。
- 新增 critical-token 数据附标、token ID/offset 复核、恰好单 token `32×` mask 门禁和隔离启动器；CPU 门禁通过前不初始化模型，训练后仍要求整条纠正 SQL 概率门槛而不是只报告被加权 token。

### v0.76.0 — 2026-08-12

- 新增首错 SQL 结构差异与 semantic critical-token 联合审计：使用有界词法 clause signature 区分表 grounding、join、时间口径、聚合/grouping、过滤、select 和 ordering/limit，并与 query-start、聚合函数、clause keyword、identifier/literal 等首分叉家族交叉。
- 审计器只读取服务器侧状态条件 Parquet 和 v3 纯前向结果，仓库/输出均不包含原始问题、SQL、答案或服务器路径；该 CPU 证据完成前不冻结下一训练配方。

### v0.75.1 — 2026-08-12

- 复核状态条件 Step 120 的 16 条首个非 greedy token，发现 `16/16` 都是 shell 外层单引号、offset 均为 0，不能解释表/字段/过滤等 SQL 语义失败；在该偏差修复前不启动新的训练。
- SQL 分项 mask 升级为“解码后 SQL 内容”契约：排除 `shlex.quote` 添加的首尾包装引号，保留 SQL 自身字符串字面量及其 shell 转义；外层包装 token 回归 tool structure。诊断契约升级到 v3，比较器仍兼容既有 v2 结果但拒绝跨版本比较。

### v0.75.0 — 2026-08-12

- 完成 Step 120、通用 SFT Step 5、SQL-weighted Step 1 的全查询只读语义基线：首条均为 `0/16`；前三条内分别为 `3/16、2/16、3/16`，任意位置为 `4/16、4/16、5/16`。后续恢复确实存在，但 SQL-weighted 的额外任意位置命中伴随更多查询，不能替代有界恢复门禁。
- 构造并 CPU 核验 16 条 Step 120 首错状态样本：错误 assistant 回合/实际工具结果 loss 为 0，纠正 SQL 和最终答案受监督；真实序列 `1608–7604` tokens，以 8K、`truncation=error` 完整保留。一步训练退出码 0，loss `1.637145`、grad norm `73.0647`，32 个模型分片完整，无 optimizer state。
- 状态条件纯前向对比显示纠正 SQL NLL `1.6815 → 1.4118`（`-16.04%`）且 `16/16` 改善，greedy `221 → 225`、top-5 `284 → 298`；但逐题概率超过 0.5 仅 `1/16 → 1/16`，未达到回放门槛 `12/16`。候选 fail closed：不做原始 prompt 诊断、短/完整回放、held-out 或老板完整评分，不追加训练；两机各 8 张物理 NPU 已释放。
- 下一阶段先做首错 SQL 的表/字段/过滤/时间/聚合/join 语义分类，并结合首个非 greedy critical token 构造机械验证的语义对比恢复样本；不直接增加训练步或 SQL 权重。

### v0.74.4 — 2026-08-12

- 将 `MAX_LENGTH=8192` 的静态断言从旧 SQL-only 金丝雀用例移到状态条件化金丝雀用例；该修正只影响测试归属，不改变已冻结的 8K 训练配置。
- v0.74.3 首次提交时定向/全量测试均因上述误放断言失败，但顺序命令仍错误地完成提交推送；本版本在任何服务器同步或训练启动前修正，并重新执行完整回归。

### v0.74.3 — 2026-08-12

- 16 条状态条件化样本已完成机械构造；首错均不是正确或等价证据，首错工具结果均来自实际 Step 120 回放，纠正查询全部机械验证。首次 4K tokenizer 门禁因一条 `7604`-token 真实错误状态 fail closed，未加载模型。
- 16K 纯 CPU 探测确认其余 15 条为 `1608–2441` tokens，最大值仅该条 `7604`；金丝雀因此冻结为最小无截断上限 `8192`，继续使用 `truncation=error`，不裁剪真实错误结果。

### v0.74.2 — 2026-08-12

- 真实 OpenAI 回放中的 `function.arguments` 既可能是 JSON 字符串也可能是映射；状态条件化构造器现统一通过既有严格解析器转换为工具参数映射，并冻结 `type=function`、`name=bash`，消除与教师工具调用 struct schema 的混用，同时保持命令原文不变。
- 第二次真实构造仍在 Parquet 写入前 fail closed，确认没有部分训练数据被消费、没有模型或 optimizer 初始化；新增字符串 arguments 回归用例后重跑。

### v0.74.1 — 2026-08-12

- 状态条件化数据构造器将真实回放中可能为 `null`、结构体或列表的 assistant content 统一规范化为确定性 JSON 文本，避免 PyArrow 在消息列中混合 struct/non-struct 类型；不改变工具调用、实际工具结果或监督回合。
- 首次真实数据构造在 Parquet 写入前因上述 schema 不一致 fail closed，未生成训练样本、未加载模型、未占用 NPU；新增结构化 content 回归用例后再执行 CPU 门禁。

### v0.74.0 — 2026-08-12

- 新增全查询语义基线，支持对任意数量的同题回放逐条只读执行 SQL，并聚合前 1/2/3 条与任意后续查询首次获得正确或等价证据的任务数；只输出查询哈希和分类，不复制原始问题、SQL 或答案。
- Qwen3.6 assistant-only 数据集支持逐 assistant 回合选择性监督；状态条件化样本把 Step 120 首个错误查询和已观察工具结果保留为上下文且 loss 严格为 0，只训练机械验证的纠正 SQL 与最终答案。分项诊断和 `0.25/8/1` 加权 mask 已同步支持三 assistant 回合形态。
- 新增状态条件化数据构造、CPU fail-closed mask 门禁和一步启动器；训练仍固定从 Step 120 起跑、16 题单 batch、一步更新、只保存 `model,extra`。完整回归 `222 passed`，两台服务器检查时均无 NPU 进程。

### v0.73.0 — 2026-08-12

- 完成从无训练门禁到一步 SQL 加权训练、final model+extra checkpoint、训练后 exact rank、16 题 greedy 48K 自由回放和 immutable 首条 SQL 语义门禁的完整流程；训练退出码 0，模型 32 个 distcp 主分片完整，optimizer state 文件为 0。
- SQL NLL `2.4484 → 2.0612`（`-15.81%`）且 `16/16` 改善，greedy token `166 → 173`、top-5 `254 → 261`、平均 rank `56.59 → 41.35`；但逐题教师 SQL 概率超过 0.5 仍为 `0/16`（门槛 `12/16`），整段全 greedy 仍为 `0/16`。
- 首条查询仍为 `0/16` 支持 gold、`0/16` 与教师结果等价；回放耗时约 `55m23s`，工具调用 328 次、终止回答 `13/16`、缺失工具响应 9 个。候选 fail closed：不续训、不跑 held-out、不作完整老板评分或晋升声明。
- 下一训练目标切换为模型首个错误查询与实际工具结果条件下的 SQL 恢复监督；两机实验结束后各 8 张物理 NPU 均已释放。

### v0.72.1 — 2026-08-12

- teacher-forced 比较器支持不绑定自由回放的纯 checkpoint NLL/rank 对比，避免把其他模型的 rollout 指标误配给当前候选；既有前后回放流水线继续传入并严格校验 16 个同 prompt 的 rollout comparison。
- 分项比较新增逐题几何平均目标概率超过 `0.5/0.8` 的计数，便于直接执行 SQL 金丝雀的冻结晋级阈值；无 rollout 时诊断状态显式标记为 pending，不推断自由运行质量。

### v0.72.0 — 2026-08-12

- 完成 Step 120 与通用 SFT Step 5 的双 checkpoint NPU exact token-rank 门禁：329 个教师 SQL token 的 greedy 命中 `166 → 231`（`50.46% → 70.21%`），top-5 命中 `254 → 302`（`77.20% → 91.79%`），平均 rank `56.59 → 17.53`，但整段全 greedy 仍均为 `0/16`。
- 两份运行 task id、数据哈希一致，均为 forward-only、optimizer 未初始化、checkpoint 未保存；核心前向 `85.38s`，端到端约 `253.3s`。结合首条 SQL 语义门禁仍为双方 `0/16`，训练目标继续锁定 SQL grounding/semantics，一步 `0.25/8/1` 金丝雀前置门禁已全部清空。
- 更新安全聚合、执行说明与回归断言；三次兼容失败均保留为不可晋升运行，成功运行不覆盖失败证据。

### v0.71.3 — 2026-08-12

- 适配当前 veRL `get_non_tensor_data` 的必需 `default` 参数：exact rank 读取 `model_vocab_size` 时显式传入空默认，并对缺失元数据给出独立硬错误；同时扫描项目内同类调用，确认无其他遗漏。
- `-03` 已证明 BSHD 强制长度对齐生效，随后在词表元数据 API 参数处失败；仍未进入 Step 5、未训练且未写 checkpoint。完整回归后继续使用新运行名重试。

### v0.71.2 — 2026-08-12

- 修复 exact token-rank 在 veRL BSHD 微批次统一填充下的标签对齐：rank 标签现在复用模型前向的 `forced_max_seqlen`，保证 CP 分片后的 logits 与 labels 具有完全相同的序列长度，不做截断或静默补齐。
- `-02` 已加载 Step 120 并进入真实模型前向，但在自定义 rank 处理器中以 `840 != 768` 主动失败；未进入 Step 5、未训练且未写 checkpoint。补充静态前向契约测试后使用新运行名重试。

### v0.71.1 — 2026-08-12

- 修复 teacher-forced exact token-rank 门禁对 Qwen3.6 嵌套 Hugging Face 配置的兼容读取：优先支持顶层 `vocab_size`，并兼容 `text_config` / `language_config`，缺失或非法值继续 fail closed。
- 补充平铺、嵌套对象、嵌套字典与非法配置单元测试；失败的 `-01` 运行未进入模型前向、未初始化 optimizer、未训练且未写 checkpoint，后续使用全新运行名重试。

### v0.71.0 — 2026-08-12

- 完成重复 SQL 因果诊断：Step 120 与通用 SFT Step 5 在同一 16 题上的首条查询均为 `0/16` 支持 gold、`0/16` 与教师结果机械等价，确认准确率故障先于重复循环发生。
- 同题自由回放显示通用 SFT 后平均 SQL `6.63 → 10.44`、重复命令 `10.25 → 14.81`，正确数保持 `2/16`、完整收尾 `15/16 → 13/16`；据此将重复查询定位为耗时/完成率放大器，而非当前准确率的充分根因。
- 结合 48K 强制收尾 `3/4` 救回但 sentinel6 数值正确仍为 `0/6`，以及正确证据 oracle 将 12 题正确数 `1 → 8`，冻结下一步顺序为“先修首条 SQL，再做 duplicate-cache 配对 A/B”；不优先扩到 64K/96K。
- 新增自包含技术报告及 canonical artifact；报告 schema、来源、载荷一致性与语义回退结构验证通过。增强 reader 在本机落入 fallback，因此不声明交互式浏览器验收通过。

### v0.70.0 — 2026-08-12

- 完成不占用 NPU 的首条 SQL 语义门禁：Step 120 与通用 SFT Step 5 均为 `0/16` 首条查询支持 gold、`0/16` 与教师结果机械等价；前者为 13 条错误/不足证据和 3 条空结果，后者为 13 条错误/不足、2 条空结果和 1 条执行错误。
- 新增 bounded、immutable、只读 SQLite 分类器，拒绝非 `SELECT/WITH`，限制执行时间与返回行数；原始 SQL 不进入安全聚合或 Git。
- teacher-forced forward-only 入口新增 TP 跨分片 exact token rank、greedy/top-5 命中与首个非 greedy SQL token 定位；该门禁已完成代码和 CPU 单元验证，但因 NPU 正由他人使用尚未实跑。
- 将下一轮冻结为 Step 120 的一步 SQL-only 单变量金丝雀：工具结构/SQL/最终答案权重为 `0.25/8/1`，不加入模型状态纠正样本；16 行 CPU mask 门禁逐行通过，SQL 占加权 loss mass 均值 `78.05%`。已准备 final-only checkpoint 启动器，训练未启动。
- 项目完整测试结果为 `213 passed`；Python 编译与新增 shell 入口语法检查通过。

### v0.69.0 — 2026-08-11

- 新增无 optimizer、无保存的 veRL/Megatron teacher-forced 分项纯前向评估；同一 16 条数据的工具回合、工具结构、SQL payload 和最终答案 mask 均通过非空、互斥及 assistant loss-mask 重建门禁。Step 120/SFT Step 5 端到端流水线 `4m31s`，核心前向合计 `87.5s`，单卡峰值 allocated HBM `12.07 GiB`。
- 官方 assistant loss 从 `1.873814` 降至 `0.414605`（`-77.87%`），且 Step 120 值与原 SFT 第一步更新前日志精确一致；工具结构、SQL、最终答案的目标概率分别提高到 `0.855/0.339/0.668`，三个分项均为 `16 改善 / 0 恶化`。
- 老板自由回放的两份模型均为 `16/16` 第一条 SQL 偏离教师目标、目标 SQL 后续命中 `0/16`。因此 checkpoint、mask 和 loss 计算不是主因；通用 loss 中 `832` 个易学结构 token 相对 `329` 个 SQL token 过重，并且教师轨迹不覆盖模型首个错误查询后的状态，是当前最符合证据的解释。
- 下一轮固定为 1–2 步 SQL-focused 金丝雀：SQL payload 权重提高 `4–8×`，工具模板降权，并加入模型首错状态的短纠正轨迹；只有 SQL NLL 为 `16/16` 改善、至少 `12/16` 的 SQL 目标概率超过 0.5 且自由生成出现非零首条正确/机械等价 SQL，才重跑完整老板回放。
- 新增纯前向数据集、组件 mask、无人值守前后比较、自由回放首处分叉分析、安全聚合 JSON 与自包含报告；项目完整测试结果为 `203 passed`。报告 schema、来源和自包含构建通过；Windows 浏览器 QA 首轮发现的约一条滚动条宽度横向溢出已加入本地 CSS 修复，最终交互复验未声明通过。

### v0.68.0 — 2026-08-11

- 完成 Step 120 与 SFT Step 5 在完全相同 16 个 task、相同 prompt、老板四工具、greedy、48K/25 回合配置下的老板原始 `reward_judge.py` 配对复评；两个回放均以退出码 `0` 完成，分别耗时 `29m42s` 和 `37m07s`。
- 同题准确率没有提升：exact result success 均为 `2/16`；平均奖励从 `0.7000` 降至 `0.60625`，完整收尾从 `15/16` 降至 `13/16`，配对结果为 `2 胜 / 4 负 / 10 平`，正式门禁失败。
- 训练后模型的平均工具回合 `12.19 → 14.69`、SQL 次数 `6.63 → 10.44`、重复命令 `10.25 → 14.81`，而过程分均值保持 `0.93125`；说明 teacher-forcing loss 的明显下降没有转化为自由运行时的单次查询和及时收尾行为。
- 停止将该 16 条配方扩展到 48–64 条；保留 checkpoint 只用于诊断。下一步应先做 teacher-forced token/结构命中与自由 rollout 的差异定位，再决定是提高 SFT 有效监督、修改轨迹格式，还是加入显式反重复/收尾约束。

### v0.67.0 — 2026-08-11

- 从正式 train236 仅选择 16 条已审核、SQL 可执行且与 expected 自洽的真实纠错任务；与 val20/test20 的 task-id 重叠均为 0。Qwen3.6 完整模板 tokenization/mask 门禁为 `16/16`，并修正历史 `function.arguments` 字符串与当前模板要求 mapping 的格式差异。
- 5 号机使用 Step 120 模型、TP4/PP2/CP2、16 NPU 完成 5 个全参数 SFT 更新：loss `1.8738 → 0.5764`（下降约 `69.2%`），墙钟 `11m04s`，单卡峰值 `26.34 GiB`，CPU Adam 峰值 `1072.06 GiB`。
- 最终仅保存 model + extra：32 个非空 dist-checkpoint 分片、总计 `54,720,369,973` 字节，完整性校验通过；未保存 optimizer，避免再次产生约 438 GiB 的 Adam 状态。
- 新增任意 Megatron dist checkpoint 的 val-only 强制权重同步补丁，以及 Step 120/SFT Step 5 同题、greedy、老板四工具、48K/25 回合的无人值守原始评分器前后回放。该版本提交时基线回放已启动；最终结果见 v0.68.0，且不产生 held-out 准确率提升声明。
- 新增数据准备、训练、回放、自动流水线及契约测试；完整测试结果为 `196 passed`。

### v0.66.0 — 2026-08-11

- 确认并实跑 veRL 官方 `verl.trainer.sft_trainer`：从 Step 120 的 Megatron distributed model checkpoint 只加载模型参数，禁用 GRPO optimizer/dataloader 恢复，重建全新的 SFT Adam。
- 新增 Qwen3.6 完整对话自定义数据集：解决官方逐消息 `MultiTurnSFTDataset` 与 Qwen3.6 system/tools/user 联合模板不兼容的问题；合成工具样本 418 tokens 中仅 65 个 assistant tokens 进入 loss，353 个 system/user/tool tokens 全部遮罩。
- 5 号机 16 NPU 的 TP4/PP2/CP2 单步全参 SFT 以退出码 `0` 完成：loss `0.9603356`、grad norm `141.0989`、峰值 `26.27 GiB/卡`、CPU 内存 `821.63 GiB`；总墙钟 `5m00s`，运行目录仅 `282 MiB`，未复制大模型或保存 Adam。
- 四个隔离运行依次修复 Hydra 覆盖语法、Megatron-Bridge 版本固定和 Megatron `no_padding` 要求；失败与成功目录均不可晋升。测试完成后两台项目容器均已停止并释放 NPU。
- 新增数据生成、token 掩码门禁、SFT 启动器、Qwen3.6 custom dataset、实验报告和契约测试；项目测试为 `187 passed`。

### v0.65.0 — 2026-08-11

- 复核远端无人值守流水线时间戳：三条件 oracle 实测 `3h42m30s`、Step 125 五次 `2×8` 金丝雀运行实测 `2h36m57s`；结合 Step 120 val20 `4054.6s` 和 checkpoint 保存 `89s` 建立下一轮墙钟基线。
- 固化效率优先的两机调度：5 号机训练、6 号机准备/核验/推理，复用 oracle 和 3,200 条奖励回放；不为小规模 SFT 建立 32 卡跨机拓扑，也不在开发门禁失败时运行密封 test20。
- 预计启动后 `4–6h` 得到 16 条纠错 SFT 的首个 go/no-go，`7–10h` 完成 48–64 条纠错 SFT 与 dev20；全部门禁通过时，含两次 mixed-only GRPO 更新和一次 test20 的完整路径为 `12–16h`。首次 SFT checkpoint/data-loader 不兼容时下行情形为 `18–24h`。
- 新增自包含时间预算报告和安全聚合 JSON；报告 schema、来源、载荷与 HTML 结构验证通过，本机增强 reader 间歇性 fallback，因此最终浏览器交互验收未声明通过。

### v0.64.0 — 2026-08-11

- 新增 Step 125 五步金丝雀的只读组内信号分析器：在服务器端只输出聚合结果，不传输原始敏感轨迹；确认 10 个 prompt 中 4 个 mixed-correct、6 个 all-wrong，mixed groups 的正确/错误奖励排序为 `4/4` 严格一致。
- 完成 Step 100/120/125/200 同一 val20、12 题 oracle 梯度和五步训练信号的联合诊断：当前奖励分层能选择已有正确候选，但 evidence/SQL 获取能力不足使多数 prompt 无正确候选，继续同配方更新缺少收益证据。
- 下一轮固定为低成本三段门禁：16 条机械验证纠错 SFT 过拟合冒烟、48–64 条纠错 SFT 金丝雀、最多 2 个 mixed-only GRPO optimizer updates；保留 Step 120，并把反复使用的 val20 降级为开发集，最终只使用一次 untouched test20。
- 新增自包含复盘报告、可复查聚合 JSON 和分析器测试；项目测试为 `183 passed`。报告 schema、来源、载荷与 HTML 结构验证通过；本机增强 reader 未就绪，因此浏览器交互验收未声明通过。

### v0.63.0 — 2026-08-11

- 修正 `2 groups × 8 responses` 续训的累计 rollout 上限：预热只控制初始队列深度，不再额外增加可被训练消费的 groups，避免目标 Step 125 因 `+4` 预热组继续更新到 Step 127。
- 无人值守流水线新增 `stage4_post_train` 恢复入口：可用已保存的 Step 125 老板验证结果做五步门禁，只统计对应的 `rollouts/122–126`，并从完整的 Step 127 检查点续跑 18 步到 Step 145，使 Step 120→145 的参数更新总数仍为 25。
- 启动器新增通过 Ray object store 从 rollout 节点回收末次 validation 的机制，并校验 20 行 JSONL 后原子落盘，解决两机 `/data3` 为节点本地目录时验证文件只出现在 6 号机、5 号机评分阶段不可见的问题。

- 官方昇腾 veRL 镜像已通过中国大陆镜像站拉取，并重新标记为 `llin-verl-a3:20260730`。
- 两台机器均完成官方镜像的软件栈和 Qwen3.6-27B 模型识别检查。
- Ray 两节点集群已连通，可见 32 张 NPU；角色测试确认训练任务落在 5 号机、rollout 任务落在 6 号机。
- 4 条真实验证任务已转换为 Parquet；两台机器上的只读数据库查询和奖励闭环均为 `4/4` 满分。
- 本地覆盖 Megatron 拓扑、Continuous Token、TP8/DP2 权重同步、48K、fully-async、Fastest-K、完整 PI 工具、奖励、boss-aligned source join/人工审核门禁、冻结基线、checkpoint 完整性、vLLM public abort、老板 KB/DWH 影子回放、老板原版前后配对评测、Step 100→200 退化归因、连续正确性离线门禁、Step 120 配对诊断和无人值守准确率流水线，项目测试为 `182 passed`。
- 老板评测影子回放使用 1,500/1,500 唯一 task_id 的同源 Qwen3.6 v15 文件；KB/DWH 共 1,000 条完整评估，DWH `277/280` 结构化 verifier 自洽、严格正确 6 条，KB 500 条全部保持非在线可用。
- 影子回放定位并修复 `/workspace/` 被字符串删除后误判为宿主 `/` 的安全规则缺陷；真实根目录和宿主路径扫描仍被阻止。
- 完整 PI Agent 已通过 6 号机真实 veRL 容器门禁：`bash/read/write/edit` 全部加载，同一轨迹共享可写沙箱，sqlite3 只读代理可查询 v15 数据，失败状态正确记录，轨迹释放后工作区不存在；门禁结束后容器已停止。
- 历史 V2 数据 `/data3/llin/qwen3.6-27b-verl-grpo/data/formal_pi_v2_20260803` 曾完成 `160/20/20` 工程审计，但后续发现其混用 Qwen3.7-Max manifest、Qwen3.6 conversation 和项目 fallback，现仅保留复现用途，正式入口已拒绝。
- 正式 prompt 连同四工具 schema 的 token 范围为 train `773–809`、val `775–799`、test `775–806`，均远低于 4,096-token 初始 prompt 预算；三个 Parquet 的 SHA256 已写入服务器侧审计报告。
- 经明确授权，两个新建的 `llin` 容器已重建为特权容器；两侧 NPU 探针均通过，未改动其他人的镜像、容器或目录。
- 两机 2-rank HCCL all-reduce 和 1→16 rollout fan-out 均通过；256 MiB stateless PyHCCL 广播、普通 broadcast 与 all-reduce 均验证成功。正式配置使用 3 GiB 权重广播 bucket，并将 vLLM HBM 利用率限制为 60%。
- `llin-pi-grpo-one-step-20260730-08` 已完成 1 个真实 GRPO 更新并以退出码 `0` 结束：16 条轨迹均完成 8 轮交互，平均奖励 `0.096875`（最小 `0.05`、最大 `0.20`），actor loss `0.013279`，梯度范数 `0.855277`。
- 单步实测生成耗时 `198.47s`、跨机权重同步 `7.35s`、actor 更新 `35.47s`；已保存完整的 `global_step_1` FSDP2 actor、优化器、额外状态、模型配置和 tokenizer 检查点。
- `pi-grpo-megatron-tp4-pp2-cp2-20260730-07` 已在 5 号机 16 张 NPU 上完成 TP4/PP2/CP2 全参更新，并以退出码 `0` 结束；6 号机继续使用 8 张 NPU 的 vLLM TP8 rollout。
- Megatron 单步共处理 `30,479` tokens，平均奖励 `0.146875`（最小 `0.05`、最大 `1.0`），actor loss `-0.0178175`，梯度范数 `1.16987`；actor 峰值 NPU 显存约 `29.63 GiB`，5 号机进程统计的主机内存约 `824.35 GiB`。
- 本次生成耗时 `206.70s`、跨机权重同步 `6.50s`、actor 更新 `260.59s`，不含保存的训练步骤耗时 `473.81s`。首次验证保存的全量 `global_step_1` 检查点约 `456 GiB`，保存另耗时 `129.05s`；后续 smoke 默认不保存，设置 `SAVE_FREQ=1` 时才保存。
- `pi-grpo-megatron-tp4-pp2-cp2-tp8-dp2-20step-20260730-11` 已完成 20/20 个 One-Step-Off-Policy 全参更新并以退出码 `0` 结束；20 份 rollout 均已落盘，Continuous Token 全程无 processor fallback。
- 18 个稳态步平均整步 `230.52s`：生成 `173.02s`、actor 更新 `49.83s`、1→16 权重同步 `7.37s`。生成占整步均值 `73.19%`，最慢/平均轨迹比值均值 `1.96`、p95 `3.33`；第 12 步达到生成 `399.96s`、整步 `553.99s`。
- 稳态 NPU 采样显示 trainer AICore 非零记录占比仅 `6.67%`，rollout 为 `80.39%`，确认训练机存在明显等待。两路 vLLM prefix cache 累计命中 `469,248 / 1,447,944` tokens，命中率 `32.41%`。
- 第 20 步仅保存 `model,extra`，最终 `global_step_20` checkpoint 约 `48 GiB`；HF safetensors 索引引用的 13 个实际分片全部存在，未保存优化器状态。
- `pi-grpo-fully-async-bounded-3step-20260730-02` 已完成 3/3 个 bounded fully-async 更新并以退出码 `0` 结束。每步消费 4 个完整 GRPO group，最大 6 个在途/排队 group，40k queued-token 背压生效，未出现 queue drop 或 Continuous Token fallback。
- fully-async 三批队列等待为 `159.41s / 111.96s / 177.26s`；第二批利用训练期间生成的 backlog，将等待降低约 `29.8%`。后续参数同步为 `7.79s / 7.75s / 7.28s`，cache 命中率 `32.41%`。前两个完整日志步的 trainer 资源利用率为 `46.40% / 47.40%`，但仍有约一半 trainer 时间在等待，需用更长稳态运行继续评估吞吐。
- 48K 容量估算以每卡实际可用 `61.27 GiB` 为上限：训练侧从实测 `29.63 GiB` 峰值出发，直接可计算的增量为 `4.28 GiB`，另留 `10 GiB` workspace/碎片预算后规划峰值约 `43.91 GiB`，余量约 `17.36 GiB`。rollout 侧每个 48K 活跃序列约需 `0.89 GiB` KV+GDN cache，16 并发约 `14.25 GiB`；加 TP8 权重分片与 `12 GiB` runtime 预算后约 `32.54 GiB`，低于 60% HBM 预算 `36.76 GiB`，规划余量约 `4.23 GiB`。
- 48K 阶梯实测已完成：8K、16K 真实环境均完成 16 条轨迹和全参数更新；32K 容量探针处理 `122,021` tokens，actor 峰值 allocated/reserved 为 `27.86/31.35 GiB`；48K 容量探针实际处理 `43,848`-token prompt 和 `190,900` 个总 tokens，峰值为 `32.42/37.78 GiB`，退出码均为 `0`。
- 48K 相对每卡实际可用 `61.27 GiB` 仍有约 `23.49 GiB` reserved 余量，证明当前 TP4/PP2/CP2、micro-batch 1、激活重计算及 optimizer/gradient CPU offload 能完成长上下文全参数前反向。该结果不等于 16 条轨迹同时接近 48K 的 rollout 吞吐验证。
- 源轨迹 system/user prompt 已精确保留；`25 assistant turns / 24 工具反馈批次 / 单轮 4 个并行调用` 覆盖抽查源轨迹的 `6–11 / 9–20 / 2–4` 范围。16K 实跑每条产生 `4–19` 次工具调用，单轮并行峰值为 3。
- 当前 veRL runtime 仍只提供 `query_sqlite`，尚未等价复现老板的 PI `bash/read/edit/write` 工具环境，因此“prompt、上下文和轮次已对齐”不能表述为“完整真实 Agent 环境已对齐”。
- Fastest-K 严格单步 A/B 已跑通：`4→4` baseline 与 `6→最快4` 均完成 16 条轨迹和一次全参数更新，退出码均为 `0`。过量采样将 trainer 收集等待从 `383.81s` 降至 `283.85s`（`-26.04%`），完整训练步从 `464.60s` 降至 `364.69s`（`-21.50%`）。
- 同一 A/B 中 Fastest-K 平均 reward 从 `0.500000` 降至 `0.290625`，完全答对从 `6/16` 降至 `2/16`，平均输出字符数下降 `12.62%`。单步结果不足以证明因果，但已确认最快选择存在质量偏差风险，正式默认前必须完成多步同 prompt 调度 A/B。
- 历史 v0.10 单步 A/B 的四个 group 均形成 `6 candidates → 4 selected + 2 discarded`，但当时 `physical_aborts=0`；后续 V2–V4 专项门禁已定位并修复该假取消，不能再把历史 0 次取消视为当前实现状态。
- 单步配置矩阵已扩展为 TP8×DP2 的 `4→4 / 5→4 / 6→4` 和 TP4×DP4 的 `4→4 / 5→4`；完整 step 分别为 `464.60 / 412.34 / 364.69 / 482.61 / 388.97s`，当前最优仍是 TP8×DP2 `6→最快4`。
- `llin-tp8-dp2-fastest-k6of4-prewarm8-8k-20step-20260731-01` 已完成 20/20 个全参数更新并以退出码 `0` 结束；8-group 预热为 `375.76s / 104,761 tokens`，20 份 rollout 共 320 条轨迹，最终 `global_step_20` checkpoint 约 `47.57 GiB`，索引引用的 13 个 safetensors 分片全部存在。
- step 2–20 的平均完整 step 为 `182.67s`：队列等待 `152.93s`、actor 更新 `19.54s`，累计等待占比 `83.72%`。预热只让前两个 batch 基本无等待，随后队列再次耗尽，证明瓶颈是长期 rollout 生产率而不是队列深度。
- 77 个可审计 Fastest-K quorum 的均值/p95 为 `291.06/394.01s`，共丢弃 154 个候选、stale drop 为 0、物理 vLLM abort 仍为 0；不能宣称未选候选的底层生成已经被实际中止。
- 本轮旧奖励平均 `0.39625`，轨迹任意位置含目标值为 `83/320`，严格最终答案正确仅 `3/320`。按新语义离线 replay 后平均 reward 为 `0.19625`、满分 3 条；项目代码已改为只有最终可见答案正确才能满分，历史运行值保持不回写。
- `llin-strict-reward-gate-tp8-dp2-4of4-8k-5step-20260801-01` 已以退出码 0 完成 5 个在线更新；80 条 rollout 的在线 score 与严格离线 replay 完全一致，55 条严格满分、平均 reward `0.740625`。四个 prompt 的满分数为 `0/20、17/20、19/20、19/20`，因此该值只证明新 reward 已上线，不能外推为总体准确率。
- Fastest-K V2 门禁记录到 8 个活跃物理请求、8 个 RPC acknowledgement，但 8 个服务端结果均为 request not found；V3 增加最多 20 次、总计 1 秒注册重试后仍有 6 个 retry exhausted，排除了单纯注册窗口不足。
- V3 首次启动暴露补丁升级链的前向幂等 bug：旧 V2 补丁无法识别 V3 marker，按旧 anchor 重复替换并在模型加载前退出。现已让旧补丁识别后续 marker，并加入 V1→V2→V3→V4 连续执行的幂等测试。
- 源码审计确认 vLLM 0.18 的 `request_states` 以 internal ID 为键，而 veRL 旧取消服务错误地用 external ID 直接查询。V4 改用 `external_req_ids` 验证注册状态并调用公开 `AsyncLLM.abort(external_id)`，保留 `reset_prefix_cache=False`。
- `llin-abort-gate-tp8-dp2-6of4-8k-1step-20260801-04` 已以退出码 0 完成：4 个 group 共 8 个落后候选全部物理取消，`active_requests=8`、`abort_acks=8`、`physical_aborts=8`、`retry_exhausted=0`、`failures=0`；训练 step 的 queue wait/actor update/完整耗时为 `0.051/57.045/65.101s`。
- V4 结束后仅重启本项目两个 `llin` 容器；两机均无 Ray、vLLM、Megatron 或 NPU 运行进程，16 张 NPU 每卡仅保留约 `2.88–3.13 GiB` 驱动基线 HBM，训练和 rollout 显存已释放。
- `llin-pi-formal-frozen-baseline-20260803-04` 已在完整 PI 四工具、48K 上下文和正式 200-task 合集上以退出码 `0` 完成冻结模型基线；总时长 `2h 29m 38s`，结果为 `200/200`，verifier 异常为 0。
- 冻结基线平均 reward 为 `0.07175`；严格最终答案正确 `4/200`、SQL 证据正确 `6/200`、使用必需表 `150/200`、产生最终答案 `67/200`，工具协议和 bash 成功率均为 `200/200`。该结果作为正式 GRPO 前的未训练能力基线，不表示训练收敛。
- 冻结基线四次启动依次修复了 val-only 误建完整 Adam、Fastest-K 对标准配置强依赖 `async_training`、以及同源基础模型评测仍重复首次全量权重广播的问题；最终运行同时暴露 200 条单批 barrier 和 TP8×DP2 后段负载不均的评测长尾。
- `llin-pi-formal-grpo-4of4-50step-20260803-03` 已完成 50/50 全参数更新并以退出码 0 结束；总时长 `12h 21m 44s`，50 个 fully-async step 的平均队列等待/actor 更新/完整 step 为 `486.55/159.41/655.08s`，等待占比 `74.27%`。
- 本轮 800 条训练 rollout 的平均 reward 为 `0.068`，严格正确 `4/800`；最后十步 reward 和安全/收尾率有弱改善，但 step 10/20/30/40/50 的 20-task 贪心 validation 在最终答案、SQL 证据和 strict acc 上始终为 0，因此 checkpoint 只记作工程成功，不记作质量收敛。
- 质量审计确认正式 V2 的主要阻塞是 instruction/gold 语义错位、source system prompt 缺失和沙箱根目录枚举，而不是显存或 HCCL；在 V3 数据人工复核完成前暂停 V2 续训和 Fastest-K 正式化。
- `llin-v15-dwh-bossreward-5step-20260804-03` 已在同源 v15 DWH、老板 system/四工具和 48K 上完成 5/5 次全参数更新并以训练退出码 0 结束；80 条训练 rollout 的平均 score/boss/evidence 为 `0.352856/0.413812/0.215000`，reward 公式不匹配和 verifier 异常均为 0。
- 固定 val20 的混合分数/老板奖励/strict evidence 从冻结基线 `0.243075/0.452250/0.132500` 变为 `0.391000/0.490000/0.160000`；SQL 证据从 `0/20` 到 `1/20`，但老板答案正确仍为 `2/20`、strict acc 仍为 `0/20`，只记作短程正向信号。
- 5 步平均队列等待/actor 更新/整步为 `726.52/208.63/943.05s`，等待占比 `77.04%`；8-group 预热耗时 `1960.22s`、累计 `846,859 tokens`，证明完整 PI 48K 的长期瓶颈仍是 rollout 生产率。
- 本轮 HF checkpoint 虽返回成功，但独立核验只包含基础模型 `905/1199` tensors，缺失第 32–63 层等 294 个 tensor，已标记 `CHECKPOINT_INVALID`、不可续训或部署。正式配置现改存 Megatron distributed model checkpoint，并新增成功退出前的 fail-closed 完整性门禁。
- 实验完成后两个 `llin` 容器均已停止；两机 NPU 无运行进程，16 张卡 HBM 均回到约 `2.88–3.13 GiB` 驱动基线。

## 参考实现

- [veRL 官方仓库](https://github.com/verl-project/verl)
- [veRL 昇腾安装说明](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/get_start/install_guidance.rst)
- [veRL One-Step-Off-Policy 说明](https://github.com/verl-project/verl/blob/main/docs/advance/one_step_off.md)
- [veRL 昇腾模型与算法支持](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/model_support/model_and_algorithm_support.md)

## 版本记录

### v0.65.0 — 2026-08-16

- 修复开放DWH长轨迹补跑的分片尾部空转：原调度必须等待整批中最慢轨迹达到30分钟上限才启动下一批，导致两机过去6小时平均NPU利用率仅约23%–27%。新增按单轨迹完成即时补位的滚动准入调度，同时把在途数量硬限制在每机真实vLLM容量内。
- 滚动调度仍按原任务分片原子落盘，异常恢复时只重做尚未形成完整原子分片的轨迹；采样参数、Step120模型、96K上下文、30分钟单轨迹超时及每题8条group身份均不变。开放DWH无人值守接力器默认启用该模式。

### v0.64.0 — 2026-08-15

- 新增开放DWH容量修复无人值守接力器：每台机器独立等待v15超时槽位回填完成，立即衔接v20超时槽位回填，再按本机真实序列容量全量重跑v21，避免每小时巡检之间出现NPU空转。
- 接力器逐阶段检查退出码并fail closed，安全状态仅记录阶段、臂标签和聚合计数；v20仍只替换原超时槽位，v21使用新目录完整重跑，不混入已停止的过载分片。

### v0.63.0 — 2026-08-15

- 修复开放DWH独立rollout的队列型假超时：`AgentLoopWorker`会在vLLM准入前同时启动整批轨迹，因此新增硬门禁，要求`task_batch_size × samples_per_task`不得超过`DP × max_num_seqs`的真实序列容量。
- 本轮双机容量合同固定为5号机`6题×8=48`、6号机`8题×8=64`；超出容量会在模型加载与轨迹生成前直接失败，不再让排队时间消耗单轨迹1800秒预算。
- v15/v20既有非超时槽位保持不变，只对原超时槽位做一对一回填；不完整的v21保留停止证据并按修复后的容量合同全量重跑，所有新结果继续`training_allowed=false`。

### v0.62.0 — 2026-08-11

- 修复 banded 断点续跑的 rollout 累计上限：fully-async 会按 checkpoint 的累计策略步恢复投喂索引，五步/二十步入口现在使用 `最终策略步 × 每步 groups + 预热 groups`，不再把新增 groups 数误当作绝对上限。
- Step 120→125 将从累计索引 241 投喂至 254，Step 125→145 将从 251 投喂至 294；两段分别保持原定的 `10+4` 与 `40+4` 个 groups，同时延续 checkpoint 数据游标而不重置采样顺序。
- 运行契约新增 rollout 起始索引和累计上限，便于在模型加载完成后、首个更新前直接审计续跑数量。

### v0.61.0 — 2026-08-11

- 无人值守准确率流水线支持从已通过的 oracle/replay 结果直接进入 Stage 4，避免恢复失败后重复执行约 3 小时 42 分钟的三组诊断。
- Step 120 的结构完整 Adam checkpoint 在 Megatron/MindSpeed HybridDeviceOptimizer 恢复时触发参数映射 `KeyError`；banded 入口新增显式 optimizer 加载开关，本轮从模型与 extra 恢复并重新初始化 Adam，最终 checkpoint 仍保存完整 `model,optimizer,extra`。
- Stage 4 与 Stage 5 均在运行契约中记录 optimizer 重置原因；断点续跑会复用并重新验证原离线奖励门槛，未通过时保持 fail closed。

### v0.60.0 — 2026-08-10

- 新增可脱离SSH运行的准确率无人值守流水线：自动完成oracle三条件冻结评测、老板原版评分、分层奖励离线回放、5步金丝雀、条件式20步续训、最终评测、checkpoint完整性检查和两机Ray资源清理。
- 将fully-async group形状参数化，正式金丝雀固定为 `2 groups/update × 8 responses/group`，总轨迹量仍为16；`fastest_k=oversample_candidates=8`，保证8条响应全部训练，避免速度选择偏差。
- 新增 `banded_v1` 正确性优先奖励：错误但过程完整的轨迹最高0.50，正确最终答案最低0.65，正确答案与正确SQL最低0.80；安全、协议和gold SQL有效性继续硬归零。
- 在真实前后100步共3,200条轨迹上预检通过：800/800完整group，133个mixed-correct group的正确轨迹排序率100%，全部必需字段完整，错误奖励上限与正确奖励下限门禁均通过。
- 本地完整回归测试为 `176 passed`；服务器端三套oracle Parquet schema与离线奖励门禁均已使用真实资产预检通过。

### v0.59.0 — 2026-08-10

- 复核 Step 100/120/200 老板原版同题结果，确认总奖励与准确率脱钩：Step 120 完成与过程改善但数值正确由 `3/20` 降为 `2/20`，Step 200 继续降至 `1/20`。
- 对前后两个100步共800个完整GRPO group重新归因：全错组分别占 `78.25%/78.5%`，mixed-correct仅 `18.75%/18%`，证明多数更新缺少二值正确性方向，单纯增加正确性奖励权重不足。
- 固化新的准确率提升顺序：先做同运行 oracle 梯度诊断，再用人工确认的train子集做纠错SFT，随后以 `2 groups/update × 8 responses/group` 跑5步GRPO金丝雀；正确率门禁通过前继续冻结64K/96K和长程训练。

### v0.58.0 — 2026-08-10

- 实现可配置的 PI 强制收尾策略：按助手回合或剩余 response token 触发，禁用后续工具调用，将最终回答限制为 4K tokens，并对违规工具调用最多纠正重试一次；所有轨迹补齐稳定审计字段，避免异构批次拼接失败。
- 新增 Step 120 小型冻结门禁的数据准备、启动、配置补丁和配对分析入口；增强老板原版评分适配器，保留终端工具响应与强制收尾纠正消息，不为未收尾轨迹伪造最终答案。
- 完成 6 题和 `task_000196` 单题实跑：强制收尾改善完成率和老板部分分，但最终数值正确仍未提升；据此暂停直接扩到 64K/96K或续训100步，下一步优先预算感知拦截、纠正监督及同运行配对门禁。

### v0.57.0 — 2026-08-10

- 逐题解析 Step 120 固定 val20：4 道未收尾题全部走到 26 回合、以 1 个未返回工具调用结束，平均 24.5 条 SQL、30.75 条重复命令且 4/4 出现冗余振荡；已收尾题平均为 15 回合、8.56 条 SQL 和 19.13 条重复命令。因此单独把上下文从48K升到96K并不对症。
- 核对服务器模型配置，Qwen3.6-27B 原生 `max_position_embeddings=262144`；96K无需RoPE外推。训练侧规划峰值约 `48.6 GiB/卡`、仍低于 `61.27 GiB`，但48K→96K每个跑满序列增加约 `0.75 GiB`缓存，24序列/副本最坏增加18 GiB，超过当前同步后约11 GiB余量。
- 给出自适应快速实验顺序：先做48K强制收尾sentinel6，失败时再测64K+32轮；96K仅对剩余失败题做定向推理并从8序列/副本容量探针开始。同时做零GPU奖励/反循环回放，最终候选必须通过5步、2 groups/update的可学习性金丝雀。
- 按 Step 120 实测耗时，100步更新加一次完整val20和保存约 `18.14h`，5步约 `2.00h`；新增可复现分析脚本、notebook、聚合JSON与技术报告，避免用100步盲跑验证单一假设。

### v0.56.0 — 2026-08-10

- 完成 Step 100→120 dense30 隔离试验的老板原版复评：总奖励 `0.443750→0.563745`（`+0.119995`，7胜/3负/10平），完整收尾 `13/20→16/20`、必需表命中 `15/20→18/20`；但数值正确 `3/20→2/20`，说明提升来自完成与过程而非最终正确性。
- 对 Step 100/120/200 的同一 val20 统一复算 dense30；Step 100→120 仅 `0.324059→0.324087`（`+0.000028`），配对 bootstrap 区间跨0。三版 task、prompt、ground truth 均为20/20一致，verifier error为0，奖励公式复算无偏差。
- 新增可复现分析脚本、已执行 notebook、聚合 JSON 与自包含技术报告；20题置信区间明确纳入结论，当前建议保留 Step 120、先扩到80–100道密封题并提高 mixed-correct group 比例，再做单因素短程 A/B。
- 纠正结束阶段耗时归因：最终验证耗时 `4054.6s`（约67.6分钟），checkpoint 保存仅约 `89s`；此前观察到的约69分钟不是保存模型耗时。

### v0.55.1 — 2026-08-10

- 为20步候选试验增加固定30%权重的 `compute_score_dense30` 奖励入口，并由启动脚本显式选择，避免复用中的Ray预启动worker不继承driver环境变量而静默回落到旧奖励。

### v0.55.0 — 2026-08-10

- 新增最终可见答案专用的连续正确性：数字误差按相对距离给部分分，表格标签提供次级信号，日期/时间不参与数值命中，过量输出数字会被降权；危险命令、无效协议和不可验证 gold 仍保持硬归零。
- 对前后200步共 `3,200` 条、`800` 个完整 group 完成离线回放：`75.75%` 的 group 产生至少 `0.05` 连续分差，原本全错的627组中 `70.65%` 获得可学习排序；严格正确排序一致率 `97.74%`、老板宽松数字口径一致率 `93.88%`，全部799条无最终答案轨迹保持0分，离线门禁通过。
- 将候选奖励接为默认关闭的环境权重，并新增Step 100→120短程入口：候选权重 `30%`，保持4 groups/update、4 responses/group、12个在途groups、48K、0.80 HBM和16K batched tokens不变，只在Step 120验证和保存完整 `model,optimizer,extra`。

### v0.54.0 — 2026-08-07

- 使用同一分析器复算前100步与后100步各 `400` 个完整 GRPO group：数值正确性有对有错比例从 `75/400`（`18.75%`）降至 `72/400`（`18.00%`），仅下降 `0.75pp`、少3个group。
- 两个阶段均为 `1,600` 条轨迹、无缺组；四条全错仅从313组增至314组、四条全对从12组增至14组。因此正确性信号并非后100步突然恶化，而是前后两个阶段都长期维持在约18%的低水平。
- 将前后100步的同口径组内信号表追加到现有 Step 100/200 canonical HTML 报告，保留原有全部章节、图表、来源与 caveat。

### v0.53.0 — 2026-08-07

- 完成 Step 100→200 老板评分下降的逐题可加总诊断：20 题总奖励净少 `0.8813`，其中数值正确性贡献 `-0.5000`（`56.7%`）、过程与字段质量贡献 `-0.2876`（`32.6%`）、完成状态切换净贡献 `-0.0937`（`10.6%`）。
- 复核 6 道退化题，确认失败集中在遗漏整体汇总、最新一期统计周期错位、遗漏期望表/必需字段，以及两道歧义温度任务达到 26 回合后仍未给最终答案。
- 新增 1,600 条续训 rollout 的首末四分位与组内信号分析：在线老板奖励和过程分改善，但数值正确率仅提高 `0.25pp`；400 个 GRPO group 中只有 `72` 个（`18%`）含正确/错误混合 response，说明当前相对正确性信号过稀。
- 更新现有 Step 100/200 canonical HTML 报告，加入精确归因图、训练信号对比、六题失败表、optimizer/data cursor 重置与 fully-async staleness 的证据边界；新增可复现诊断脚本和测试。

### v0.52.0 — 2026-08-07

- 完成 Step 200 老板原版评分器复评：20/20 task_id 匹配，Step 100/200 的 system+user 输入逐题完全一致；原版总奖励从 `0.443750` 降至 `0.399685`（`-9.93%`），逐题为 3 胜、6 负、11 平。
- 退化主要来自数值正确从 `3/20` 降到 `1/20`、过程分从 `0.765000` 降到 `0.723750`、必需字段命中均值下降 `0.117647`；完整收尾保持 `13/20`，因此不能归因于完成率变化。
- 原版评测转换器现在能把 token 边界截断的最终工具调用保留为“未响应调用”，不会把调用前的推理文字伪造成最终答案；新增可复现的老板评测配对汇总器和 canonical HTML 报告。

### v0.51.0 — 2026-08-07

- 修正长期训练 checkpoint 契约：正式 100-step 及 step100→step200 入口均保存 `model,optimizer,extra`，不再遗漏 Adam 一阶/二阶矩、master parameters 和学习率调度器状态。
- 新增在线 worker checkpoint 配置工具，可对当前 16 个 Megatron rank 先盘点、再更新并逐 rank 回读；用于在不中断当前续训的情况下确保 step-200 最终保存 optimizer。
- checkpoint 完整性门禁现在会读取 manifest；只要声明保存 optimizer，就强制要求 optimizer distributed metadata 和非空分片存在，否则最终作业返回失败。
- 当前续训已在线验证 `16/16` 个训练 rank 全部回读为 `model,optimizer,extra`，且训练持续推进；完整回归测试为 `135 passed`。

### v0.50.0 — 2026-08-06

- 新增 step-100 → step-200 专用续训入口：累计训练目标 200、rollout 目标 800，对应从恢复计数新增 100 次参数更新和 400 个完整 groups；其余 48K、`4 groups/update × 4 responses`、12 个在途 groups、0.80 vLLM cache、16K batched tokens、24 seq/副本、12 workers、学习率 `1e-7` 配置保持不变。
- 新增角色隔离的 resume view：训练节点恢复 step-100 的完整 Megatron model/RNG，rollout 节点不加载与旧 train237 绑定的 `data.pt`，改在修正后的 train236 上重置数据游标；原 checkpoint 没有 Adam 状态，因此续训使用同配置但重新初始化的 optimizer，并在运行目录写入明确契约。
- 最终验证和保存均仅在累计 step-200 触发；启动器要求唯一最终 checkpoint 为 `global_step_200`，并继续执行完整性门禁。

### v0.49.0 — 2026-08-06

- 追溯唯一重复 prompt 的老板 v15 task manifest 与两条原始 Qwen3.6-27B 事件轨迹；保留相对更贴近题意的 `task_000147`，从 train 剔除 `task_000033`，val/test 保持原 20/20 不变。
- 新增相同 instruction 绑定不同 gold 的 fail-closed 构建门禁、可审计质量剔除器和全量标签重放器；未来正式入口切换为 `boss_v15_dwh_full276_20260806` 的 `236/20/20`。
- 两台服务器的数据契约与三份 Parquet SHA256 独立一致；修正后 `276/276` 条 hidden SQL 可执行、非空且 expected value 匹配，冲突数为 0。语义审计仍有 `271/276` 条预警，明确不把机械自洽冒充人工语义正确。
- 新增来源与正确性技术审计报告，确认归档轨迹为 `my-local Qwen3.6-27B` 且只供 SFT/reference；GRPO 输入不包含其答案，hidden label 来自老板 task manifest。
- 全项目回归测试为 `129 passed`；报告 canonical payload、来源结构和语义 fallback 通过，因本机 Chromium 与增强 reader 不兼容采用 `structural_only` 验收。

### v0.48.0 — 2026-08-06

- 新增面向领导的技术实验汇报方法论，提炼背景、结果、原因分析和下一阶段计划的固定四段结构。
- 固化最多三位小数、从起点到终点、百分点与相对变化区分、整体与条件性子集分离、reward hacking 谨慎结论及外部信息边界。

### v0.47.0 — 2026-08-06

- 原样保留 step-100 的 32-shard Megatron distributed checkpoint；复检 `54,720,369,973 bytes`、格式和元数据均有效，可继续训练。
- 新增 Qwen3.6 专用离线导出器：在 CPU/Gloo TP1/PP1/CP1 上恢复完整 64 层，再由 Megatron Bridge 流式写入 HF safetensors；针对上游未实现的 MTP 映射，仅从基础模型继承训练中未启用的 15 个 MTP tensor。
- 独立 HF 目录通过 `1199/1199` tensor、15/15 shard、0–63 层、GDN 权重族和零 shape mismatch 门禁；全新 TP8 vLLM 成功加载并生成 `HF export works`，验收后 8 张 NPU 全部释放。
- 新增可复现的 HF 导出、严格校验和 vLLM 最小生成脚本及路径隔离/MTP fallback 测试；清理本次失败尝试的约 49 GiB 临时目录，正式模型、日志和恢复 checkpoint 保留；全项目回归测试为 `125 passed`。

### v0.46.0 — 2026-08-06

- 新增 Step 100 原版评分器评测的外部汇报版；原始报告保持不变。
- 将样本计数改为聚合比例，移除评测集规模、逐题明细、内部服务器路径、脚本/数据文件名和 SHA256，同时保留总奖励、过程质量、完成率与下一轮建议。

### v0.45.0 — 2026-08-06

- 使用老板原始 `judge_trajectory.py`、`judge_trajectory_openai.py`、`reward_judge.py`、v15 task manifest 和原始 `logistics.sqlite`，在同一固定 val20 上严格配对重算冻结模型与 step-100；两轮均 20/20 task_id 匹配，manifest SHA256 完全一致。
- 老板原版总奖励均值从 `0.479065` 降至 `0.443750`（`-7.37%`）；数值正确从 `2/20` 增至 `3/20`、verdict correct 从 `1/20` 增至 `2/20`、过程分从 `0.670625` 增至 `0.765000`，但完整收尾从 `15/20` 降至 `13/20`，新增硬门控归零抵消了能力收益。
- 新增 fail-closed 的 Qwen 文本轨迹到 OpenAI messages 适配器，支持单轮并行工具调用并忠实记录缺失 tool response；逐题配对为 6 胜、5 负、9 平，两轮都完成的 10 题均值从 `0.61188` 升至 `0.70938`，确认下一轮应优先修复最终回答预算和重复探索，而不是原样增加训练步数。
- 全项目回归测试为 `119 passed`。

### v0.44.0 — 2026-08-05

- 修正后的正式实验 `llin-v15-dwh-bossreward-12groups-100step-20260805-03` 已通过真实运行门禁：两路 TP8 各加载 15/15 个模型分片，`2560 MiB` bucket 的初始权重同步耗时 `13.06s`，并连续实测 `active_tasks_size=12`、`staleness_samples=12`，确认 12 个完整 groups 实际满载。
- `gpu_memory_utilization=0.80` 下，12-group rollout 带载 HBM 约 `53.8–56.1 GiB/卡`；首个 actor 更新后的新权重同步成功，耗时 `8.69s`，同步后 HBM 约 `54.4–54.7 GiB/卡`，仍余约 `10.8–11.1 GiB/卡`，未出现 OOM 或最大张量断言。
- 首次预热完成 8 groups、`743,730` tokens，耗时 `1709.87s`；第 1 步 actor 更新耗时 `266.72s`、整步 `275.73s`，参数版本推进到 1。第 2 步已从库存直接取满 4 groups，队列等待约 `0.09s`；因此保留 0.80、16K batched tokens、24 seqs/副本、12 workers 和 12-group 并发，无需同步下调其他参数。

### v0.43.0 — 2026-08-05

- `llin-v15-dwh-bossreward-12groups-100step-20260805-02` 已证明 `gpu_memory_utilization=0.80` 可以完成两路 TP8 vLLM 模型加载，但首次权重同步在发送前 fail closed：Qwen3.6-27B 的 `model.language_model.embed_tokens.weight` 为 `[248320, 5120]` BF16，单个不可拆分张量约 `2425 MiB`，不能装入 v0.42.0 误设的 `512 MiB` bucket；本轮尚未 rollout 或更新参数，退出码 `1`。
- 正式入口将同步 bucket 修正为 `2560 MiB`，这是能容纳该最大张量的最小实用对齐档位；按 HCCL send/receive 缓冲与昇腾 PyHCCL 广播输出估算，同步瞬态约 `7.5 GiB/卡`，仍比原 `3072 MiB` bucket 的约 `9 GiB/卡` 低 `16.7%`。
- 保持 `gpu_memory_utilization=0.80`、`max_num_batched_tokens=16,384`、`max_num_seqs=24/副本`、12 Agent workers、12 个在途 groups、48K 上下文、100 步及仅末步验证/保存不变；能否保留这些并发参数以后续首次更新后的真实同步 HBM 峰值为最终门禁。

### v0.42.0 — 2026-08-05

- `llin-v15-dwh-bossreward-12groups-100step-20260805-01` 在 12 个在途 groups 中完成 11 个、训练端消费首批 4 groups 并执行第 1 次 actor 更新后，于新权重同步阶段 OOM；退出码 `1`，未完成 step 指标落盘、未验证、未保存 checkpoint，因此该内存更新不可恢复。
- 根因是 `gpu_memory_utilization=0.85` 的 vLLM 常驻预算与 `3072 MiB` 权重 bucket 叠加：HCCL send/receive 双缓冲加昇腾 PyHCCL 同尺寸广播输出，使同步瞬态接近三个 bucket；日志对应表现为已经分配 6 GiB 后再次申请 3 GiB，而每卡只余 `65–473 MiB`。
- 修正版将 vLLM 预算降至 `0.80`，并曾把正式入口的同步 bucket 降至 `512 MiB`；后续 `-02` 启动证明该档位小于约 `2425 MiB` 的最大 embedding 张量，已由 v0.43.0 修正为 `2560 MiB`。16K batched tokens、24 seqs/副本、12 Agent workers、12 个在途 groups、48K 上下文和其他训练/奖励参数保持不变。

### v0.41.0 — 2026-08-05

- 已启动正式实验 `llin-v15-dwh-bossreward-12groups-100step-20260805-01`：boss-aligned 237/20/20 契约门禁通过，两机 train/val Parquet 大小与 SHA256 完全一致，Ray 角色确认训练固定在 5 号机、rollout 固定在 6 号机。
- 5 号机 16 个 Megatron worker 与 6 号机两路 TP8 vLLM 均成功加载；两路 vLLM 各完成 15/15 个 safetensors 分片，在 `gpu_memory_utilization=0.85` 下初始化 HBM 为约 `55.4–55.7 GiB/卡`，12 groups 实际带载后约 `58.9–59.3 GiB/卡`，仍余约 `6.2–6.6 GiB/卡`，未出现 OOM。
- 首次跨机权重同步耗时 `13.99s`；fully-async 运行时监控实测 `active_tasks_size=12`、`max_concurrent_samples=12`、`staleness_samples=12`，证明当前确实同时生成 12 个完整 groups（48 条轨迹），不是只修改静态上限。第 1–99 步不验证、不保存，只有第 100 步执行最终验证并保存一个模型。

### v0.40.0 — 2026-08-05

- 新增独立的 100-step/12-group 正式训练入口：每次参数更新仍消费 `4 groups × 4 responses`，100 步共消费 400 个完整 groups、1,600 条轨迹；`staleness=2.0`、12 个 Agent workers 与两路 TP8 各 24 个 sequence slots 共同形成 12-group 在途上限。
- 按最终口径关闭所有中途验证与保存：第 1–99 步不验证、不写 checkpoint，仅第 100 步执行一次 val20 并保存 `global_step_100`；启动器要求最终迭代严格等于 100，随后执行 checkpoint 完整性门禁，最多保留一个模型。
- 正式推理容量原固定为 `gpu_memory_utilization=0.85`、`max_num_batched_tokens=16,384`、`max_num_seqs=24/副本`；`-01` 的首次更新后权重同步证明 0.85 与 3072 MiB bucket 组合会 OOM，后续由 v0.42.0 修正为 0.80 与 512 MiB bucket。

### v0.39.0 — 2026-08-05

- 完成 48K GRPO 推理容量与并发提效评估：结合最近 11 步正式运行、当前 TP8×DP2/8-group 调度和 vLLM/vLLM Ascend 官方参数语义，明确 `gpu_memory_utilization=0.85` 主要扩大 cache 容量，不等同于直接提速。
- 给出逐级 A/B：先比较 `0.60/8K/16` 与 `0.85/8K/16`，再单独加入 `16K max_num_batched_tokens`；更高并发从 10 groups/20 seqs/副本开始，并把 staleness=1.5 的质量代价单独验证，不直接将 12 groups/staleness=2 写成正式默认。
- 新增 HBM 瞬态、KV cache/preemption、prefill/decode 吞吐、工具后端、新鲜度和训练质量门禁，以及端到端 Amdahl 上限模型，避免把局部吞吐收益当作等比例 step 提速。

### v0.38.0 — 2026-08-05

- 按明确指令停止 `llin-v15-dwh-bossreward-4groups-50step-20260805-02`：最终完成 `11/50` 次更新并保留 11 份 rollout，启动器记录退出码 `1`；由于只在 step 50 保存，本轮没有 checkpoint。
- 先停止 6 号机 rollout 容器、再停止 5 号机 trainer 容器；两容器最终均为 `Exited (137)`，两机 32 张 NPU 均无运行进程，AICore 回到 0、HBM 回到约 `2.9–3.1 GiB/卡` 驱动基线。

### v0.37.0 — 2026-08-05

- 首次从完全停止的容器启动 `llin-v15-dwh-bossreward-4groups-50step-20260805-01` 时，正式入口在数据契约门禁阶段因缺少项目 `PYTHONPATH` 立即退出；尚未加载模型、生成 rollout 或占用 NPU，失败目录原样保留用于审计。
- 正式 50-step 入口现在自行导出项目与 runtime Python 路径，不再依赖容器重启前的交互式会话环境；新增契约测试覆盖冷启动所需的导入路径。
- 修复后启动 `llin-v15-dwh-bossreward-4groups-50step-20260805-02`：boss-aligned 契约与两机 Parquet 哈希门禁通过，16 个 Megatron 训练 worker 和两个 TP8 vLLM 副本均已创建，两个推理副本各完成 `15/15` 个 safetensors 分片加载；作业继续运行并等待 8-group 预热完成。

### v0.36.0 — 2026-08-05

- 按明确指令停止 `llin-v15-dwh-bossreward-2groups-50step-20260804-03`：已完成 `36/50` 次更新并落盘 36 份 rollout；由于只在 step 50 保存，本轮没有最终 checkpoint。两个项目容器均已停止，32 张 NPU 的 AICore 回到 0、HBM 回到约 `2.9–3.1 GiB/卡` 驱动基线，已有日志和 rollout 保留。
- 从并发容量重新校准正式入口：默认恢复为 `4 groups/update × 4 responses = 16 trajectories/update`，`PREWARM_GROUPS` 与 `MAX_QUEUE_GROUPS` 继续按两个 update batch 计算为 8；在 `staleness=1` 下最多暴露 8 个完整 group、32 条轨迹，与现有 `TP8×DP2`、每副本 16 sequences 的容量对齐。
- 50 步、48K、完整 PI 工具、学习率、每 10 步固定验证、无 Fastest-K 过量采样和仅 step 50 保存最终分布式 checkpoint 的其余语义不变。

### v0.35.0 — 2026-08-04

- `llin-v15-dwh-bossreward-2groups-50step-20260804-02` 成功越过 fresh HF 初始化并创建 fully-async 组件，但在 step 0 被预热门禁阻止：2 groups/update 使 staleness=1.0 对应的物理队列容量为 4 groups，旧固定 `PREWARM_GROUPS=8` 超过容量；没有 rollout 文件、参数更新或 checkpoint。
- 正式入口将预热量和队列 group 预算改为 `2 × GROUPS_PER_STEP`：当前即 4 groups，仍保持两个 update batch 的预热深度与 staleness=1.0，不通过硬扩 8-group 队列引入额外 policy-version 陈旧度。

### v0.34.0 — 2026-08-04

- 将 fresh HF 初始化兼容修复部署到两台节点并完成清洁容器/Ray 重启；训练/rollout 角色、双机 train/val 哈希和脚本语法门禁再次通过。
- 已后台启动 `llin-v15-dwh-bossreward-2groups-50step-20260804-02`；实际参数保持 `2 groups/update`、50 步、每 10 步验证和仅 step 50 保存，运行已越过 `-01` 的 `stat(None)` 失败时间点且主进程继续存活。

### v0.33.0 — 2026-08-04

- `llin-v15-dwh-bossreward-2groups-50step-20260804-01` 在 step 0 模型初始化时退出：`use_dist_checkpointing=True` 且 fresh run 没有 `dist_checkpointing_path`，veRL 将 `None` 传给 Megatron loader 并触发 `TypeError: stat(None)`；没有 rollout、参数更新或 checkpoint，两机 NPU 已释放。
- 新增幂等的 Megatron 初始化兼容补丁：只有真实 dist checkpoint 路径存在时才从分布式权重加载；fresh run 继续从基础 HF 模型初始化，同时保留最终 `model` 槽使用 Megatron distributed checkpoint 的保存语义。
- 完整项目回归为 `112 passed`；补丁已在容器真实 veRL 源码临时副本上通过首次应用、重复应用和编译门禁。

### v0.32.0 — 2026-08-04

- 已将提交 `1d82af3` 的正式启动脚本同步到两台项目节点，双端 SHA256 一致；训练/rollout Ray 角色、train/val 文件存在性与哈希、脚本语法门禁全部通过。
- 已后台启动 `llin-v15-dwh-bossreward-2groups-50step-20260804-01`；实际主进程参数确认为 `ppo_mini_batch_size=2`、`rollout.n=4`、`total_training_steps=50`、`test_freq=10`、`save_freq=50`，当前处于模型初始化阶段。

### v0.31.0 — 2026-08-04

- 正式长文本 GRPO 默认改为每次参数更新消费 `2 groups × 4 responses = 8 trajectories`，训练总步数保持 `50`，其余 rollout、48K 上下文、25 轮工具反馈、学习率、验证和拓扑配置不变。
- checkpoint 频率绑定到总训练步数，50-step 正式运行只在 step 50 保存一次最终 `model,extra` Megatron distributed checkpoint，不再写入中间模型；最终完整性验证和 fail-closed 门禁保持启用。

### v0.30.0 — 2026-08-04

- 完成 v15 DWH 48K 五步训练的单步耗时复核：预热库存耗尽后的 step 3–5 平均 `23.79min`，其中等待 4 个完整 group 平均 `20.18min`、Actor 更新 `3.48min`，确认主要瓶颈是 rollout 长期供给而非训练计算。
- 核清当前 batch 为 `4 groups × 4 responses = 16 trajectories/update`；给出 `2 groups/update` 约 `12–14min/step` 的容量估算，并明确其不会自动提高等样本总吞吐，且会增加更新次数和梯度噪声。
- 复核纯 `6→最快4` 旧 A/B 的 `-21.50%` 整步收益与 reward/正确率下降风险；建议改测“4 个主候选 + 2 个延迟备用”，并以等 40 groups、selected/shadow-discarded 质量差异作为上线门禁。

### v0.29.0 — 2026-08-04

- 完成老板 v15 DWH 主奖励 5-step 真实训练与固定 val20：5/5 全参数更新、80 条训练 rollout、最终贪心验证均完成；记录逐步队列等待、actor 更新、长轨迹 token/轮次、安全原因和 numeric/table 分项。
- 确认短程混合分数与老板奖励改善但 strict acc 未改善，明确禁止把 5-step 工程成功写成质量收敛。
- 发现 PP=2 mbridge 在线 HF 导出静默缺失后半 pipeline 的 294 个 tensor；本轮 checkpoint 标记无效，正式训练改用 Megatron distributed model checkpoint，并新增成功退出前的 checkpoint fail-closed 验证器及测试。
- 记录 `mstx.range_end`、NPU→CPU 算子回退、最终聚合行缺失等非致命观测问题；修复短跑分析器硬编码 50 步的缺步误报；实验后停止两个 `llin` 容器并验证显存释放。

### v0.28.0 — 2026-08-04

- 复用已完整落盘的固定 val20 轨迹建立冻结基线：混合分数/老板奖励/严格证据均值为 `0.243075 / 0.452250 / 0.132500`，老板数字答案正确 `2/20`、严格最终答案正确 `1/20`、SQL 证据与 strict acc 均为 `0/20`；reward 公式不匹配和 verifier 异常均为 0。
- 修复 3 条无 `must_use_fields` 任务把 `boss_fields_used=None` 交给 veRL reducer 后触发的 NumPy 求平均错误；内部 process 权重不变，对外指标统一为可聚合数值。
- 新增在线安全原因数值指标和离线验证汇总器；网络、破坏性、宿主路径、Python 网络和根目录扫描分别计数，避免 `safe=0` 只有结果、没有原因。
- `llin-v15-dwh-bossreward-5step-20260804-02` 在 `global_step=0` 主动终止以补安全观测，没有更新权重；清洁重启后的最终门禁运行 `-03` 已通过老板 full277 契约并开始初始化。
- 项目完整回归门禁为 `107 passed`；奖励类型、安全原因、验证离线重放和历史训练分析均保持兼容。

### v0.27.0 — 2026-08-04

- 以老板 v15 原始 instruction、gold、同源 sandbox 和 evaluator 为权威契约，277 条可执行 DWH 全部纳入正式数据资产；确定性分割为 `237 train / 20 val / 20 test`，唯一重复 prompt 组留在同一 split，跨 split prompt 泄漏为 0。
- 正式在线奖励改为 `70% boss_reward + 30% strict_evidence_reward`；危险工具、无效协议、不可执行或空 gold SQL 直接归零，并分别落盘老板奖励、严格证据奖励和 strict acc。
- 新增数据选择、真实 Parquet 契约检查、老板奖励重放字段、分析器和 277 条审计报告；272 条语义预警继续保留，明确“评测器对齐”不等于“自然语言语义已经人工证明无误”。
- 新增冻结 val20 成功后自动衔接 5-step GRPO 的监督启动器；基线非零退出时硬阻断训练，并分别记录基线与目标运行状态。
- 项目完整回归门禁为 `104 passed`；两台服务器上的 train/val/test/contract SHA256 一致，真实 Parquet 的奖励契约与必需字段检查通过。

### v0.26.0 — 2026-08-04

- 完成 DWH 与 KB 奖励判分差异案例审计：DWH `task_000001` 的老板/本项目分数为 `0.92/0.15`，KB `KT-LOG-0301` 为 `0.69/0.05`。
- 报告写明老板 `judge_trajectory + reward_judge` 与本项目 `boss_reward_shadow` 的实际奖励定义，并保留原始 instruction、gold、SQL、文档访问和最终回答摘录。
- 区分两类根因：DWH 为严格全结果相等误杀语义正确的附加投影；KB 为 `unanswerable + 空 source_documents` 与真实可读冷链文档冲突，需先修数据血缘而非直接调高规则奖励。

### v0.25.0 — 2026-08-04

- 新增老板 KB/DWH 影子奖励与 OpenAI 消息适配器：DWH 仅允许 gold SQL 自洽的 numeric/table，答案数字碰撞但无 SQL 的分数封顶为 `0.15`；KB 只记录真实文档访问、数字/文本锚点与拒答信号，未校准前永不进入在线奖励。
- 对 task-id 唯一的 Qwen3.6 v15 原轨迹完成 1,000 条 KB/DWH 回放：DWH 500 条中 277 条在线候选、3 条 gold 不一致、220 条需要语义 judge，严格正确 6 条；KB 400 answerable/100 unanswerable 全部保持 shadow-only。
- 发现旧 converted 文件重复 task_id 并由门禁拒绝；修复合法 `ls/find /workspace/` 被误判为宿主根目录扫描的问题，保留网络、真实 `/`、宿主路径和破坏性命令硬隔离。
- 新增对抗与回放测试、完整验证报告；正式训练入口和当前 reward V2 均未切换，必须先完成人工差异复核和 KB 语义 judge 校准。
- 全项目回归门禁为 `92 passed`；远端回放只使用 CPU/SQLite，没有启动 Ray、模型或占用 NPU。

### v0.24.0 — 2026-08-04

- 废止固定 `160/20/20` 的 formal V2 默认链路；旧构建器改为必须显式 `--allow-legacy-v2`，正式 50-step 入口只接受 full、已审核、带完整哈希的 `boss-pi-aligned-grpo-v1`，并在模型加载前拒绝 V2、pilot、fallback 和未审核数据。
- 从老板 `pi_to_openai.py` 原样冻结 `DEFAULT_SYSTEM` 与 `bash/read/edit/write` schema，记录源脚本和内容 SHA256；runtime 修正为 1-based read offset、2,000 行/50 KiB 输出和 900 秒工具边界，同时明确保留网络/宿主机/破坏性命令安全隔离这一差异。
- 追溯确认旧 V2 manifest 来自 Qwen3.7-Max，而讨论/对比对象为 Qwen3.6 conversation；改用 v15 原始 Qwen3.6 事件文件名 `task_id` 连接同一 sandbox task，不再按 row order、模糊文本或跨模型 manifest 拼接。
- v15 1,500 条源轨迹已全量进入 SFT/reference；1,000 条 KB/Hybrid、220 条无严格 verifier 样本不进入当前 GRPO，3 条 gold 执行不一致被拒绝，277 条可执行 DWH numeric/table 进入显式 alignment review queue。当前批准 GRPO 为 0，未启动新训练。
- 新增 boss-aligned 构建器、task manifest 无损导出、正式契约门禁、逐项复盘和回归测试；项目门禁为 `81 passed`。

### v0.23.0 — 2026-08-04

- 正式 `llin-pi-formal-grpo-4of4-50step-20260803-03` 已完成 `50/50`、退出码 0；800 条 rollout 完整、奖励重放 0 差异、verifier 0 异常，最终 `global_step_50` checkpoint 约 48 GiB，两机 NPU 已释放。
- 完成训练质量诊断：严格正确仅 `4/800`，最终答对 `24/800`、SQL 证据正确 `19/800`；200 个 GRPO group 中 43 个零奖励方差，同一 prompt 平均仅 1.25 次 group exposure。五次 validation strict/final/SQL 均为 0，reward 的小幅上升来自安全和最终回答率。
- 审计 200 条正式 V2 数据，191 条命中至少一个语义复核触发器：161 条“最新/最近”没有时间 SQL、71 条 `LIMIT` 无 `ORDER BY`、99 条广泛分析问题用唯一 hidden target 打分；暂停 V2 续训，下一步重建人工确认的 V3。
- 修复 fully-async validation 轨迹覆盖：trainer 将真实 policy step 传给 rollouter，后者验证期间临时使用并恢复数据计数；两台 Ray 启动前幂等应用，后续应保留 `10/20/30/40/50.jsonl`。
- 禁止 `find/ls/du/tree /` 枚举容器根目录；fallback prompt 明确唯一 `/workspace/logistics.sqlite`。正式 builder 现保留 source system 并记录血缘，不再把 fallback 冒充为老板原始 prompt。
- 新增可复现的 50-step 分析器、instruction/gold 对齐审计及回归测试；项目门禁为 `73 passed`。当前没有直接改奖励权重，避免在错位数据上把目标变化与权重变化混在一起。

### v0.22.0 — 2026-08-03

- 新增正式 PI 故障复盘与复现文档，按七次关键运行逐项记录环境、数据、配置、决定性错误、根因、修复、验证证据和安全复现条件。
- 明确区分冻结 `val_only` 误建优化器与正式训练 FP32 master shard 二次锁页卸载这两类 207001 OOM，并补充双机 Parquet 可见性、可选 Fastest-K 配置、冗余首次同步及全量 barrier 长尾的诊断方法。
- 将 Apex/ModelOpt/Triton/SciPy/KV connector、`mstx.range_end` 等非致命信息与真正致命异常分表记录；正式 `-03` 截至证据快照已完成 `4/50`，文档明确保留“运行中”边界，不提前宣称整轮成功。

### v0.21.0 — 2026-08-03

- 正式 `-02` 已完成双机数据门禁、首批完整 PI rollout 和 actor 前反向，但在更新退出阶段把 FP32 optimizer master shard 以 `non_blocking=True` 搬回 CPU 时触发 CANN host-pinned allocator OOM；主机当时仍有约 `1.9 TiB available`，不是普通主机内存耗尽，也不是 NPU 前反向显存不足。
- 正式配置保留 MindSpeed `optimizer_cpu_offload=True`，Adam 状态继续位于 5 号机 CPU；仅关闭 veRL engine 的二次阶段 `megatron.optimizer_offload`，使 FP32 master shard 常驻 NPU，避免每步巨量锁页 D2H/H2D 复制。
- 预计每卡增加约 `6–8 GiB` 常驻 HBM；48K 容量门禁实测 reserved 峰值 `37.78 GiB`、相对每卡可用 HBM 仍有约 `23.49 GiB`，因此先用真实第 1 步验证后再继续 50-step。

### v0.20.0 — 2026-08-03

- 正式 50-step 首次启动在 step 0 暴露双机数据可见性问题：fully-async rollouter 位于 6 号机并会直接读取 train/val Parquet，而正式数据当时只部署在 5 号机；本次没有发生 rollout 或参数更新。
- 通过 SSH 加密流将已审计 train/val 文件直接同步到 6 号机，不在本地落盘；两端 SHA256 分别一致为 `0f22b2...ac25` 与 `f06b15...85b8`。
- 新增双角色 Ray 数据门禁，在模型初始化前验证 train/val 在 5、6 号机均存在、大小及 SHA256 完全一致，避免再次将数据部署错误延迟暴露到远端 actor 初始化阶段。

### v0.19.0 — 2026-08-03

- 新增正式 50-step 全参数 GRPO 入口：完整 PI 四工具、奖励 V2、48K 上下文、TP4/PP2/CP2 训练和 TP8/DP2 rollout，固定 `4→4` 而不创建过量候选。
- 数据严格使用 train 160 条训练、val 20 条每 10 step 贪心 `n=1` 验证，test 20 条不进入运行；学习率默认 `1e-7`，每 10 step 保存并仅保留一份 `model,extra` checkpoint。
- bounded fully-async 预热 8 个完整 group、队列最多 8 group/`1,572,864` tokens、staleness 上限 1 个 policy version；新增启动参数和数据隔离契约测试。

### v0.18.0 — 2026-08-03

- 完成完整 PI Agent、48K 上下文、贪心 `n=1` 的 200-task 冻结模型基线；退出码 0、结果 200/200、总时长 `2h 29m 38s`，两机训练和推理显存均已释放。
- 记录平均 reward `0.07175`、严格最终答案正确 `4/200`、SQL 证据正确 `6/200`、工具协议有效 `200/200` 等正式训练前基线指标，明确区分严格正确率与 verifier 满分 `acc`。
- 新增完整复盘文档，串联 `-01` 到 `-04`：修复 val-only 优化器误初始化、可选 Fastest-K 配置兼容和冗余首次权重同步，并记录全量评测的 barrier 长尾与 DP 后段失衡。

### v0.17.0 — 2026-08-03

- 冻结基线第三次启动已保持进程存活并完成 actor、两套 TP8 vLLM、944/944 权重转换，但在 0/200 处停在冗余的首次 1→16 actor-to-rollout 广播超过 60 分钟；失败证据保留在 `llin-pi-formal-frozen-baseline-20260803-03`。
- 为 `val_only + resume_mode=disable` 增加受限优化：actor 与 rollout 均从同一只读 `MODEL_PATH` 加载时跳过首次权重广播；训练、恢复 checkpoint 或非 val-only 运行仍保留原同步行为，避免把该修复扩散到正式训练语义。

### v0.16.0 — 2026-08-03

- 冻结基线第二次启动已成功越过 forward-only、模型装载、TP8×DP2 vLLM 与完整四工具初始化，但在 0/200 处发现 Fastest-K 补丁假定所有运行都存在 `async_training` 配置；失败证据保留在 `llin-pi-formal-frozen-baseline-20260803-02`。
- 将 Fastest-K 设为真正可选能力：标准 One-Step/val-only 没有 `async_training` 时自动视为关闭；同时支持对容器内已带旧 marker 的补丁做前向幂等升级，避免只能重建镜像才能修复。

### v0.15.0 — 2026-08-03

- 冻结基线首次启动在 step 0 暴露 veRL `val_only` 仍构建完整 Adam 并申请锁页主机内存的问题，尚未生成任何评测轨迹；失败证据保留在 `llin-pi-formal-frozen-baseline-20260803-01`。
- 将冻结 actor 显式切换为 Megatron `forward_only`，同时关闭 optimizer/gradient offload，使评估只初始化模型与跨机 rollout 权重，不再创建、卸载或保存优化器和梯度状态。

### v0.14.0 — 2026-08-03

- 新增 200 条正式任务的统一冻结模型评测文件 `pi_formal_all.parquet`，保留每条任务原始 train/val/test 标记，便于在不混淆数据血缘的前提下建立全量能力基线。
- 新增冻结模型基线运行与后台启动入口：完整 PI `bash/read/write/edit` 环境、48K 上下文、贪心 `n=1` 评估、禁止参数更新和 checkpoint 保存，并将全部轨迹落盘用于后续奖励/正确率/工具行为审计。
- 修正工具轮次的 off-by-one 边界：正式配置使用 `26 assistant turns / 25 tool-result turns`，允许完成第 25 次工具调用后再生成最终答案；历史 `25/24` 只能容纳 24 次工具反馈。

### v0.13.0 — 2026-08-03

- 将正式 rollout 环境从简化 `query_sqlite` 升级为老板 PI 的 `bash/read/write/edit` 四工具契约；实现轨迹级共享沙箱、工具事件审计、统一清理和安全边界。
- 定位并修复官方 veRL 昇腾镜像缺少 sqlite3 CLI 的环境差异，加入只读兼容代理；真实容器四工具闭环、状态共享、SQL 查询和清理门禁通过。
- 上线 evidence-grounded 奖励 V2：最终答案占主导，agent SQL 必须在隐藏环境中执行并与 gold SQL 结果一致； unsafe/非法协议硬置零。
- 从 v15/v20/v21 三个独立环境构建 200 条 numeric/table DWH 正式任务，按 `160/20/20` 隔离 train/val/test；执行所有 gold SQL、剔除 7 条标签不一致候选并完成独立复审。
- 修复 mixed numeric/table gold 无法写入 Arrow 的序列化 bug，以及 tokenizer 审计误报 2 tokens 的统计 bug；项目回归测试更新为 `54 passed`。

### v0.12.0 — 2026-08-01

- 完成严格奖励 5-step 在线门禁；80 条 score 与严格 replay 完全一致，并按 prompt 分组揭示 55/80 高均值来自三易一难的极小 smoke 集，避免将其误报为总体正确率。
- 通过 V2 可观测性、V3 注册重试和 V4 源码修复，定位 Fastest-K `physical_aborts=0` 的根因是 vLLM 0.18 external/internal request ID 混用；最终 4 个 group 的 8 个落后候选全部完成真实物理取消。
- 修复补丁升级链的前向幂等问题；新增取消状态分析、public abort 补丁和回归测试，项目测试更新为 `43 passed`。
- 新增完整技术复盘，记录每次运行、失败、假设、排除证据和最终配置；本轮结束后停止两台项目 Ray 环境并释放全部训练/推理显存。
- 自包含 HTML 报告通过 canonical artifact、exact-payload 与 semantic fallback 的 `structural_only` 验证；本机 Chromium reader 出现横向溢出/启动超时，未将增强交互验收冒充为已通过。

### v0.11.0 — 2026-07-31

- 增加 fully-async 精确阶段计时与 8-group 预热观测，完成 TP8×DP2、TP4×DP4 的五组单步配置矩阵，并选定 TP8×DP2 `6→最快4` 进行 20-step 稳态验证。
- 20/20 更新以退出码 0 完成；稳态平均 step `182.67s`，其中队列等待 `152.93s`、actor 更新 `19.54s`，确认预热和加深队列不能修复长期 rollout 供给不足。
- 将旧奖励的“整条轨迹含目标值”与“最终可见答案正确”拆分；320 条 replay 显示旧命中 `83` 条、严格正确 `3` 条，后续满分条件已改为严格最终答案正确。
- 新增可复现分析器、canonical HTML 技术报告和回归测试，项目测试为 `41 passed`；报告通过桌面与窄屏稳定态 QA，官方 500ms browser receipt 因 renderer 自身 16px 瞬态布局误报未通过，证据边界已保留。

### v0.10.0 — 2026-07-31

- 在 bounded fully-async 中实现可配置 Fastest-K 过量采样：保持 `rollout.n=4` 的 GRPO group 原子性，默认并发生成 6 条候选并选择最快 4 条，未选任务通过逻辑到物理 request 映射执行可确认的 vLLM 取消且不重置 prefix cache。
- 完成 `4→4` 与 `6→最快4` 的真实单步全参数 A/B；trainer 等待下降 `26.04%`，完整 step 缩短 `21.50%`，验证了绕开长尾候选的吞吐收益。
- 记录平均 reward、答对数和输出长度同步下降的选择偏差风险，并明确物理 abort 本轮未被实际命中的证据边界；新增验证报告、幂等补丁和回归测试，项目测试为 `36 passed`。

### v0.9.0 — 2026-07-31

- 完成 8K、16K、32K、48K 长上下文阶梯实跑；48K 探针以 `43,848`-token prompt 完成 TP8×DP2 rollout、TP4×PP2×CP2 全参数前反向和 CPU Adam 更新，退出码为 `0`。
- 48K actor 峰值 allocated/reserved 为 `32.42/37.78 GiB`，相对每卡实际可用 `61.27 GiB` 仍有约 `23.49 GiB` reserved 余量；记录 32K/48K 的 tokens、耗时和适用边界。
- 将 One-Step 与 bounded fully-async 的单轮并行工具调用上限显式设为 4；结合 25 assistant turns 和 24 工具反馈批次，覆盖抽查老板源轨迹的轮数与并行度。
- 新增长上下文与工具轮次验证报告，明确 system/user prompt 已按数据血缘对齐，但 `query_sqlite` runtime 尚未等价复现 PI `bash/read/edit/write` 环境。

### v0.8.0 — 2026-07-31

- 将 One-Step 与 bounded fully-async 的默认多轮预算提高为 25 个 assistant turns、24 个工具反馈轮，最大上下文提高为 48K（4K initial prompt + 44K multi-turn response），并显式开启 8K chunked prefill。
- 将 fully-async queued-token 下限同步提高到一个最坏情况完整训练 batch（`4 groups × 4 responses × 49,152 = 786,432` tokens），避免 48K oversized group 在 trainer 收齐首批 4 groups 前触发背压死锁。
- 数据转换改为优先保留老板源轨迹的 system prompt，不再无条件替换为项目短提示词；无 source system 时仍保留可复现 fallback。
- 新增 48K 显存容量估算器与回归测试：训练规划峰值约 `43.91 GiB/卡`，rollout 规划占用约 `32.54 GiB/卡`，均低于当前硬件预算，但明确要求 8K/16K/32K/48K 阶梯实测。
- 明确剩余环境差异：当前 `query_sqlite` 工具并不等价于老板的 PI `bash/read/edit/write` runtime，48K/25 轮和 system prompt 对齐不能单独证明真实 Agent 环境已经复现。

### v0.7.0 — 2026-07-31

- 完成老板轨迹与本次 320 条 rollout 的数据血缘审计；确认指定 v15 归档不包含本次 4 个 prompt，并按训练元数据回溯 v24/v26 同源轨迹，实现 `4/4` exact-prompt 配对。
- 新增轨迹长度与长尾调度调查报告和可重复分析脚本，量化同 prompt response 差异、20-step 生成长尾、300 秒硬超时阈值与丢弃下界。
- 给出“8 个完整 group 预热 + 约 70k token 队列 + 每步消费 4 group”的 bounded fully-async 建议，并明确 GRPO group 原子性、vLLM 真取消、prefix cache 保留和 off-policy staleness 风险。

### v0.6.0 — 2026-07-31

- 新增完整训练复盘报告，记录初始两机环境、原始轨迹数据审计、prompt-only 数据转换、工具与奖励定义。
- 按时间线整理容器权限、Ray 角色、HCCL fan-out、FSDP2、Megatron、TP8/DP2、Continuous Token、checkpoint 和 fully-async 队列的十余次关键尝试、失败根因及修复。
- 固化最终 One-Step-Off-Policy 与 bounded fully-async 的环境、配置、启动步骤、实测指标、证据边界、局限和后续建议。

### v0.5.0 — 2026-07-30

- 完成 1→16 HCCL/PyHCCL fan-out 验证，将 rollout 扩展为 vLLM TP8×DP2，并修正 Ascend DP2 权重同步 IPC rank 映射。
- 完成 20-step One-Step-Off-Policy 全参训练，降低 checkpoint 频率和内容规模，加入两机 NPU、Continuous Token、prefix-cache 与长尾稳态统计。
- 修复 One-Step-Off-Policy rollout dump、20-step 数据迭代和 text-only Continuous Token worker 路径，记录 20 步耗时、利用率、cache 和 48 GiB checkpoint 结果。
- 长尾判据触发后切换并实测 bounded fully-async：完整 GRPO group 原子入队、40k queued-token 背压、staleness=0.5、partial rollout，3 步训练以退出码 0 完成。
- 将 pytest 收集范围固定为本项目 `tests/`，避免误收集已忽略的上游 `reference/` 测试树。

### v0.4.0 — 2026-07-30

- 将 5 号机训练后端切换为 Megatron TP=4、PP=2、CP=2，关闭 LoRA，并保留 6 号机 vLLM TP=8 异步多轮工具 rollout。
- 配置 actor 参数常驻 NPU、CPU Adam/梯度卸载、全量激活重计算和 CP KV all-gather，加入拓扑约束检查。
- 引入官方昇腾验证版 Megatron-Bridge 源码及最小 veRL 兼容层，完成 16-NPU 全参单步 GRPO，记录奖励、loss、显存、耗时和检查点规模。
- smoke 默认关闭约 `456 GiB` 的全量检查点保存，仍可通过 `SAVE_FREQ=1` 显式开启。

### v0.3.0 — 2026-07-30

- 按授权将 5、6 号机上两个新建的 `llin` 容器重建为特权 Ascend 容器，并验证 NPU 可用性。
- 固定 HCCL 内网接口和通信端口范围，新增 2-rank all-reduce 与 1→8 权重广播拓扑检查。
- 将 Qwen3.6 工具调用切换到 `qwen3_coder` XML 解析，配置 16-NPU FSDP2 训练和 8-NPU TP rollout。
- 完成真实数据上的单步轨迹 GRPO 更新，记录奖励、loss、性能数据并保存 `global_step_1` 检查点。

### v0.2.0 — 2026-07-30

- 完成 5 号机训练、6 号机推理的 veRL One-Step-Off-Policy 架构和 Ray 角色固定。
- 使用官方昇腾 A3 镜像建立 `llin` 独立环境，并验证跨机 Ray 通信。
- 新增老板 PI 轨迹数据的 prompt-only Parquet 转换、只读 SQLite 工具和可审计数值奖励。
- 新增 Qwen3.6-27B 单步轨迹 GRPO 启动配置、真实数据集成检查和单元测试。
- 记录当前 NPU 容器权限边界，避免在未明确授权时使用全特权容器。

### v0.1.0 — 2026-07-29

- 初始化项目仓库。
- 建立版本记录规范：每次更新都在本节记录变更，并提交、推送到远程仓库。

## 版本维护约定

- 每次代码、配置、文档或功能更新，都必须同步更新本文件的“版本记录”。
- 版本记录应包含版本号、日期以及简明的变更说明。
- 完成并验证更新后，将 README 与相关改动一并提交并推送到 `origin/main`。
