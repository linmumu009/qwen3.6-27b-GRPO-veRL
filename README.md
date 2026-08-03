# qwen3.6-27b-GRPO-veRL

Qwen3.6 27B 的 GRPO / veRL 训练项目。

## 当前方案

- 5 号机负责 Megatron 全参训练，拓扑为 TP=4、PP=2、CP=2，Ray 自定义资源名为 `llin_trainer`。
- 6 号机负责 vLLM 异步轨迹推理，拓扑为 TP=8、DP=2，Ray 自定义资源名为 `llin_rollout`。
- 两台机器通过内网 Ray 集群通信；训练权重使用 veRL 的 `nccl` 检查点后端，在昇腾环境中实际注册为 HCCL 广播。
- HCCL 固定使用 `eno0` 和 `192.168.202.0/24` 内网，并为 host/NPU socket 分配互不重叠的端口范围，避免自动选中不可达的管理网卡。
- actor 模型参数常驻 NPU，不做参数卸载；Adam 优化器与梯度卸载到 5 号机内存，并开启全量激活重计算。LoRA 已关闭（`lora_rank=0`）。
- rollout 开启 Continuous Token、prefix caching 和两路 vLLM cache 计数；checkpoint 每 20 步保存一次，仅保存 `model,extra` 并只保留 1 份。
- 长上下文配置将最大上下文设为 `49,152` tokens（默认初始 prompt `4,096` + 多轮 response `45,056`）；正式 PI 配置将允许最多 25 次工具反馈和随后 1 次最终回答，单轮最多 4 个并行工具调用，单次工具返回放宽到 `32,768` 字符；vLLM 保持 `8,192`-token chunked prefill 和每副本最多 16 个活跃序列。
- 数据转换优先保留源轨迹的 system prompt；只有源记录没有 system message 时才使用项目的物流分析 fallback prompt。
- 20-step One-Step-Off-Policy 的长尾切换判据已触发；后续推荐使用 bounded fully-async：一个 prompt 的 4 条 GRPO 轨迹作为不可拆分 group。48K 配置的 queued-token 上限至少容纳一个最坏情况训练 batch（默认 `786,432` tokens），group 数上限继续控制 staleness，满载时背压而不是丢弃旧样本。
- bounded fully-async 已支持 Fastest-K 过量采样：默认物理生成 6 条候选、最先完成的 4 条组成完整 GRPO group，剩余候选取消；可用 `OVERSAMPLE_CANDIDATES=4` 恢复无过量采样的 baseline。该能力已验证吞吐收益，但仍需多步质量 A/B 后才能作为正式训练默认策略。
- Fastest-K 的逐请求取消已改为 vLLM 0.18 的公开 external-request API；V4 门禁实测 8/8 个落后候选完成物理取消，且不清空 prefix cache。当前实验已停止，两台 `llin` 容器保持空闲。
- 所有新增镜像、容器、工作目录和实验名均以 `llin` 开头，不复用或修改其他人的环境。

当前服务器部署：

| 项目 | 5 号机 | 6 号机 |
| --- | --- | --- |
| 角色 | 训练 | rollout 推理 |
| 工作目录 | `/data3/llin/qwen3.6-27b-verl-grpo` | `/data3/llin/qwen3.6-27b-verl-grpo` |
| 容器 | `llin-verl-trainer-m05-20260730` | `llin-verl-rollout-m06-20260730` |
| 镜像 | `llin-verl-a3:20260730` | `llin-verl-a3:20260730` |
| 容器权限 | 特权模式（仅重建上述 `llin` 容器） | 特权模式（仅重建上述 `llin` 容器） |
| 当前实验 NPU | 0（实验已停止；容器空闲） | 0（实验已停止；容器空闲） |

## 数据结论

老板原有的 `trajectories_v15_27B_table.tar.gz` 是 PI agent 事件轨迹，共包含 1,500 个 JSONL 文件。它包含提示、模型消息、工具调用和工具输出，但没有可直接供 GRPO 使用的显式 reward，因此不能原样传给 veRL。

早期 smoke 采用只读 `query_sqlite`；正式数据阶段已切换到以下完整 PI 契约：

1. 完整保留老板 PI 的 system prompt，并提供同名、同 schema 的 `bash/read/write/edit` 四工具。
2. 每条轨迹从对应 `sft/<version>` 环境复制独立可写工作区；四个工具在整条轨迹中共享状态，结束后统一清理。
3. 昇腾 veRL 镜像缺少 `sqlite3` CLI，项目提供只读兼容代理，保持模型仍按原 system prompt 在 Bash 中调用 `sqlite3`。
4. 奖励 V2 以最终答案正确性为主，同时重新执行 agent SQL，与隐藏的 gold `verification_sql` 结果进行独立比对；只提到表名或在工具输出里出现目标值不能获得满分。
5. 首批正式数据为 200 条可执行验证的 DWH numeric/table 任务：v15 train 160、v20 val 20、v21 test 20；task ID、instruction hash 和 environment 均跨 split 隔离。

原始轨迹、验证清单、Parquet、模型、checkpoint 和运行日志均不会提交到 Git。

## 目录

- [`docs/training_experiment_report_20260731.md`](docs/training_experiment_report_20260731.md)：从初始环境、数据改造、十余次关键尝试到最终 One-Step 与 bounded fully-async 跑通的完整复盘报告。
- [`docs/trajectory_rollout_investigation_20260731.html`](docs/trajectory_rollout_investigation_20260731.html)：同 prompt 轨迹长度对比、长尾 rollout 超时、完整 GRPO group 队列与 vLLM 真取消方案的可交互调查报告。
- [`docs/context_48k_tool_turn_validation_20260731.md`](docs/context_48k_tool_turn_validation_20260731.md)：8K/16K/32K/48K 阶梯实跑、显存峰值、system prompt 血缘和工具调用轮次对齐报告。
- [`docs/fastest_k_oversampling_validation_20260731.md`](docs/fastest_k_oversampling_validation_20260731.md)：`4→4` 与 `6→最快4` 的严格单步 A/B、吞吐收益、质量选择偏差和物理 vLLM 取消证据边界。
- [`docs/fastest_k_efficiency_20step_20260731.html`](docs/fastest_k_efficiency_20step_20260731.html)：五组拓扑/过量采样矩阵、8-group 预热的 20-step fully-async 时序、奖励泄漏复核和下一步效率实验的自包含技术报告。
- [`docs/fastest_k_abort_debug_20260801.html`](docs/fastest_k_abort_debug_20260801.html)：严格奖励在线门禁、Fastest-K V2–V4 假取消故障链、external/internal request ID 根因、最终 8/8 物理取消和显存释放的完整技术复盘。
- `llin_verl/pi_sqlite_tool.py`：只读 SQLite 轨迹工具。
- `llin_verl/pi_workspace_tools.py`、`llin_verl/pi_agent_loop.py`：完整 PI 四工具、轨迹级共享沙箱、事件审计和统一清理。
- `llin_verl/pi_sqlite_cli.py`：为官方昇腾镜像补齐的受限只读 sqlite3 CLI 兼容层。
- `llin_verl/pi_reward.py`：最终答案、可执行 SQL 证据、必需表和安全协议联合奖励 V2。
- `runtime/sitecustomize.py`：将训练池固定到 5 号机、rollout 池固定到 6 号机。
- `scripts/prepare_pi_dataset.py`：验证轨迹到 veRL Parquet 的转换程序。
- `scripts/prepare_pi_formal_dataset.py`、`scripts/audit_pi_formal_dataset.py`：200 条正式 PI 数据构建、gold SQL 执行、split 隔离和独立质量门禁。
- `scripts/start_ray_m05.sh`、`scripts/start_ray_m06.sh`：两机 Ray 启动程序。
- `scripts/check_ray_roles.py`：跨机角色落点验证。
- `scripts/check_hccl.py`：两机基础 HCCL all-reduce 验证。
- `scripts/check_hccl_fanout.py`：1 个训练 rank 到 16 个 rollout rank 的权重广播拓扑验证。
- `scripts/run_pi_grpo_smoke.sh`：Qwen3.6-27B 单步轨迹 GRPO 冒烟实验。
- `scripts/launch_pi_grpo_smoke.sh`：带退出码、起止时间和完整日志的后台实验启动器。
- `scripts/run_pi_grpo_megatron_tp4_pp2_cp2.sh`：16-NPU Megatron TP4/PP2/CP2 全参轨迹 GRPO 配置。
- `scripts/launch_pi_grpo_megatron_smoke.sh`：Megatron 单步实验的日志、时间和退出码启动器。
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

## 已验证状态

- 官方昇腾 veRL 镜像已通过中国大陆镜像站拉取，并重新标记为 `llin-verl-a3:20260730`。
- 两台机器均完成官方镜像的软件栈和 Qwen3.6-27B 模型识别检查。
- Ray 两节点集群已连通，可见 32 张 NPU；角色测试确认训练任务落在 5 号机、rollout 任务落在 6 号机。
- 4 条真实验证任务已转换为 Parquet；两台机器上的只读数据库查询和奖励闭环均为 `4/4` 满分。
- 本地覆盖 Megatron 拓扑、Continuous Token、TP8/DP2 权重同步、监控分析、48K 容量估算、bounded fully-async 队列、Fastest-K、完整 PI 工具、奖励 V2、正式数据门禁和 vLLM public abort，项目测试为 `54 passed`。
- 完整 PI Agent 已通过 6 号机真实 veRL 容器门禁：`bash/read/write/edit` 全部加载，同一轨迹共享可写沙箱，sqlite3 只读代理可查询 v15 数据，失败状态正确记录，轨迹释放后工作区不存在；门禁结束后容器已停止。
- 正式数据 `/data3/llin/qwen3.6-27b-verl-grpo/data/formal_pi_v2_20260803` 已生成并通过独立审计：train/val/test 为 `160/20/20`，共 200 个唯一 task 和 instruction hash，跨 split 重叠为 0；每条 gold SQL 均重新执行并与标签一致。
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

## 参考实现

- [veRL 官方仓库](https://github.com/verl-project/verl)
- [veRL 昇腾安装说明](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/get_start/install_guidance.rst)
- [veRL One-Step-Off-Policy 说明](https://github.com/verl-project/verl/blob/main/docs/advance/one_step_off.md)
- [veRL 昇腾模型与算法支持](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/model_support/model_and_algorithm_support.md)

## 版本记录

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
