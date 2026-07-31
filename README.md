# qwen3.6-27b-GRPO-veRL

Qwen3.6 27B 的 GRPO / veRL 训练项目。

## 当前方案

- 5 号机负责 Megatron 全参训练，拓扑为 TP=4、PP=2、CP=2，Ray 自定义资源名为 `llin_trainer`。
- 6 号机负责 vLLM 异步轨迹推理，拓扑为 TP=8、DP=2，Ray 自定义资源名为 `llin_rollout`。
- 两台机器通过内网 Ray 集群通信；训练权重使用 veRL 的 `nccl` 检查点后端，在昇腾环境中实际注册为 HCCL 广播。
- HCCL 固定使用 `eno0` 和 `192.168.202.0/24` 内网，并为 host/NPU socket 分配互不重叠的端口范围，避免自动选中不可达的管理网卡。
- actor 模型参数常驻 NPU，不做参数卸载；Adam 优化器与梯度卸载到 5 号机内存，并开启全量激活重计算。LoRA 已关闭（`lora_rank=0`）。
- rollout 开启 Continuous Token、prefix caching 和两路 vLLM cache 计数；checkpoint 每 20 步保存一次，仅保存 `model,extra` 并只保留 1 份。
- 下一阶段真实环境配置将最大上下文设为 `49,152` tokens（初始 prompt `4,096` + 多轮 response `45,056`），assistant/tool 交互上限设为 `25/24` 轮，单次工具返回放宽到 `32,768` 字符；vLLM 保持 `8,192`-token chunked prefill 和每副本最多 16 个活跃序列。
- 数据转换优先保留源轨迹的 system prompt；只有源记录没有 system message 时才使用项目的物流分析 fallback prompt。
- 20-step One-Step-Off-Policy 的长尾切换判据已触发；后续推荐使用 bounded fully-async：一个 prompt 的 4 条 GRPO 轨迹作为不可拆分 group。48K 配置的 queued-token 上限至少容纳一个最坏情况训练 batch（默认 `786,432` tokens），group 数上限继续控制 staleness，满载时背压而不是丢弃旧样本。
- 所有新增镜像、容器、工作目录和实验名均以 `llin` 开头，不复用或修改其他人的环境。

当前服务器部署：

| 项目 | 5 号机 | 6 号机 |
| --- | --- | --- |
| 角色 | 训练 | rollout 推理 |
| 工作目录 | `/data3/llin/qwen3.6-27b-verl-grpo` | `/data3/llin/qwen3.6-27b-verl-grpo` |
| 容器 | `llin-verl-trainer-m05-20260730` | `llin-verl-rollout-m06-20260730` |
| 镜像 | `llin-verl-a3:20260730` | `llin-verl-a3:20260730` |
| 容器权限 | 特权模式（仅重建上述 `llin` 容器） | 特权模式（仅重建上述 `llin` 容器） |
| 当前实验 NPU | 16（Megatron TP=4、PP=2、CP=2，全参） | 16（vLLM TP=8、DP=2） |

## 数据结论

老板原有的 `trajectories_v15_27B_table.tar.gz` 是 PI agent 事件轨迹，共包含 1,500 个 JSONL 文件。它包含提示、模型消息、工具调用和工具输出，但没有可直接供 GRPO 使用的显式 reward，因此不能原样传给 veRL。

本项目采用以下转换：

1. 从已验证轨迹中提取 prompt、环境 ID 和 verifier。
2. 转换成 veRL 的 prompt-only Parquet，并指定 `tool_agent`。
3. rollout 期间通过只读 `query_sqlite` 工具查询每个任务的 `logistics.sqlite`。
4. 只有实际查询了必需表并给出验证数值时才获得满分；查询到正确表但最终答案错误只给进度分。

原始轨迹、验证清单、Parquet、模型、checkpoint 和运行日志均不会提交到 Git。

## 目录

- [`docs/training_experiment_report_20260731.md`](docs/training_experiment_report_20260731.md)：从初始环境、数据改造、十余次关键尝试到最终 One-Step 与 bounded fully-async 跑通的完整复盘报告。
- [`docs/trajectory_rollout_investigation_20260731.html`](docs/trajectory_rollout_investigation_20260731.html)：同 prompt 轨迹长度对比、长尾 rollout 超时、完整 GRPO group 队列与 vLLM 真取消方案的可交互调查报告。
- `llin_verl/pi_sqlite_tool.py`：只读 SQLite 轨迹工具。
- `llin_verl/pi_reward.py`：数值结果、工具证据和必需表联合奖励。
- `runtime/sitecustomize.py`：将训练池固定到 5 号机、rollout 池固定到 6 号机。
- `scripts/prepare_pi_dataset.py`：验证轨迹到 veRL Parquet 的转换程序。
- `scripts/start_ray_m05.sh`、`scripts/start_ray_m06.sh`：两机 Ray 启动程序。
- `scripts/check_ray_roles.py`：跨机角色落点验证。
- `scripts/check_hccl.py`：两机基础 HCCL all-reduce 验证。
- `scripts/check_hccl_fanout.py`：1 个训练 rank 到 16 个 rollout rank 的权重广播拓扑验证。
- `scripts/run_pi_grpo_smoke.sh`：Qwen3.6-27B 单步轨迹 GRPO 冒烟实验。
- `scripts/launch_pi_grpo_smoke.sh`：带退出码、起止时间和完整日志的后台实验启动器。
- `scripts/run_pi_grpo_megatron_tp4_pp2_cp2.sh`：16-NPU Megatron TP4/PP2/CP2 全参轨迹 GRPO 配置。
- `scripts/launch_pi_grpo_megatron_smoke.sh`：Megatron 单步实验的日志、时间和退出码启动器。
- `scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh`：TP4/PP2/CP2 训练、TP8/DP2 rollout 的 bounded fully-async 配置，按完整 GRPO group 入队并以 queued tokens 做背压。
- `scripts/monitor_npu_utilization.py`、`scripts/monitor_vllm_cache_metrics.py`：两机 NPU 稳态利用率与两路 vLLM prefix-cache 计数采样。
- `scripts/analyze_grpo_steady_state.py`：汇总 20-step 稳态耗时、长尾、NPU 利用率和 cache 命中率，并输出 fully-async 切换判据。
- `scripts/analyze_trajectory_comparison.py`：只读扫描老板轨迹、同源 converted 轨迹、本次 320 条 rollout 与 20-step 日志，输出可复查的长度和超时统计。
- `scripts/estimate_48k_capacity.py`：依据已验证的 6K 实测峰值、Qwen3.6 64 层混合 GDN/全注意力结构及 TP/PP/CP 切分，估算 48K 训练激活和 rollout KV/GDN cache 容量。
- `llin_verl/megatron_bridge_compat.py`、`scripts/patch_verl_megatron_bridge_compat.py`：为昇腾验证版 Megatron-Bridge 补齐当前 veRL 所需的最小兼容接口。

## 已验证状态

- 官方昇腾 veRL 镜像已通过中国大陆镜像站拉取，并重新标记为 `llin-verl-a3:20260730`。
- 两台机器均完成官方镜像的软件栈和 Qwen3.6-27B 模型识别检查。
- Ray 两节点集群已连通，可见 32 张 NPU；角色测试确认训练任务落在 5 号机、rollout 任务落在 6 号机。
- 4 条真实验证任务已转换为 Parquet；两台机器上的只读数据库查询和奖励闭环均为 `4/4` 满分。
- 本地覆盖 Megatron 拓扑、Continuous Token、TP8/DP2 权重同步、监控分析、48K 容量估算及 bounded fully-async 队列测试，项目测试为 `35 passed`。
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
- 上述 48K 结论是公式与既有实测结合的容量门禁，尚不是 48K 实跑证据；正式训练前必须按 `8K → 16K → 32K → 48K` 做 NPU 分配与一条完整前反向阶梯探针。当前 veRL runtime 仍只提供 `query_sqlite`，尚未等价复现老板的 PI `bash/read/edit/write` 工具环境，因此不得把本次参数调整称为“完整真实环境已对齐”。

## 参考实现

- [veRL 官方仓库](https://github.com/verl-project/verl)
- [veRL 昇腾安装说明](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/get_start/install_guidance.rst)
- [veRL One-Step-Off-Policy 说明](https://github.com/verl-project/verl/blob/main/docs/advance/one_step_off.md)
- [veRL 昇腾模型与算法支持](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/model_support/model_and_algorithm_support.md)

## 版本记录

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
