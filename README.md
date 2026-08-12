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
- [`docs/repeated_sql_causal_diagnosis_20260812.html`](docs/repeated_sql_causal_diagnosis_20260812.html)：把首条 SQL 语义门禁、同题自由回放、48K 强制收尾和正确证据 oracle 串成因果链，区分重复查询对准确率、完成率和墙钟的不同作用。
- [`docs/repeated_sql_causal_diagnosis_20260812.artifact.json`](docs/repeated_sql_causal_diagnosis_20260812.artifact.json)：上述重复 SQL 因果诊断的 canonical report artifact、聚合数据、图表和来源定义。
- [`docs/leadership_experiment_update_methodology_20260806.md`](docs/leadership_experiment_update_methodology_20260806.md)：从多轮实际修订中提炼的领导汇报方法论，固化四段结构、数字精度、口径边界、抗奖励投机表述、行动项口吻和自检清单。
- `llin_verl/pi_sqlite_tool.py`：只读 SQLite 轨迹工具。
- `llin_verl/pi_workspace_tools.py`、`llin_verl/pi_agent_loop.py`：完整 PI 四工具、轨迹级共享沙箱、事件审计和统一清理。
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
- `scripts/patch_verl_vllm_abort_api.py`：修复 vLLM 0.18 external/internal request ID 混用，使用公开 `AsyncLLM.abort(external_id)` 真正终止物理请求。
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
- `scripts/teacher_forced_token_ranks.py`：在 TP 词表分片上计算教师 SQL token 的精确 rank，并定位首个非 greedy 关键 token，不收集完整 logits。
- `scripts/qwen36_sql_weighted_sft_dataset.py`、`scripts/check_sql_weighted_sft_dataset.py`、`scripts/run_repair_sft_sql_weighted_canary.sh`：构造和 CPU 核验 SQL 加权 loss mask，并从 Step 120 启动仅一步、单变量、只保存最终模型的金丝雀。
- `scripts/prepare_state_conditioned_repair_sft.py`、`scripts/check_state_conditioned_sft_dataset.py`、`scripts/run_repair_sft_state_conditioned_canary.sh`：从 Step 120 首错 SQL 和真实工具结果构造零-loss 上下文，机械核验纠正查询并执行一步状态条件化金丝雀。

## 已验证状态

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
