# qwen3.6-27b-GRPO-veRL

Qwen3.6 27B 的 GRPO / veRL 训练项目。

## 当前方案

- 5 号机负责 Megatron 全参训练，拓扑为 TP=4、PP=2、CP=2，Ray 自定义资源名为 `llin_trainer`。
- 6 号机负责 vLLM 异步轨迹推理，Ray 自定义资源名为 `llin_rollout`。
- 两台机器通过内网 Ray 集群通信；训练权重使用 veRL 的 `nccl` 检查点后端，在昇腾环境中实际注册为 HCCL 广播。
- HCCL 固定使用 `eno0` 和 `192.168.202.0/24` 内网，并为 host/NPU socket 分配互不重叠的端口范围，避免自动选中不可达的管理网卡。
- actor 模型参数常驻 NPU，不做参数卸载；Adam 优化器与梯度卸载到 5 号机内存，并开启全量激活重计算。LoRA 已关闭（`lora_rank=0`）。
- 所有新增镜像、容器、工作目录和实验名均以 `llin` 开头，不复用或修改其他人的环境。

当前服务器部署：

| 项目 | 5 号机 | 6 号机 |
| --- | --- | --- |
| 角色 | 训练 | rollout 推理 |
| 工作目录 | `/data3/llin/qwen3.6-27b-verl-grpo` | `/data3/llin/qwen3.6-27b-verl-grpo` |
| 容器 | `llin-verl-trainer-m05-20260730` | `llin-verl-rollout-m06-20260730` |
| 镜像 | `llin-verl-a3:20260730` | `llin-verl-a3:20260730` |
| 容器权限 | 特权模式（仅重建上述 `llin` 容器） | 特权模式（仅重建上述 `llin` 容器） |
| 单步实验 NPU | 16（Megatron TP=4、PP=2、CP=2，全参） | 8（vLLM TP=8、DP=1） |

## 数据结论

老板原有的 `trajectories_v15_27B_table.tar.gz` 是 PI agent 事件轨迹，共包含 1,500 个 JSONL 文件。它包含提示、模型消息、工具调用和工具输出，但没有可直接供 GRPO 使用的显式 reward，因此不能原样传给 veRL。

本项目采用以下转换：

1. 从已验证轨迹中提取 prompt、环境 ID 和 verifier。
2. 转换成 veRL 的 prompt-only Parquet，并指定 `tool_agent`。
3. rollout 期间通过只读 `query_sqlite` 工具查询每个任务的 `logistics.sqlite`。
4. 只有实际查询了必需表并给出验证数值时才获得满分；查询到正确表但最终答案错误只给进度分。

原始轨迹、验证清单、Parquet、模型、checkpoint 和运行日志均不会提交到 Git。

## 目录

- `llin_verl/pi_sqlite_tool.py`：只读 SQLite 轨迹工具。
- `llin_verl/pi_reward.py`：数值结果、工具证据和必需表联合奖励。
- `runtime/sitecustomize.py`：将训练池固定到 5 号机、rollout 池固定到 6 号机。
- `scripts/prepare_pi_dataset.py`：验证轨迹到 veRL Parquet 的转换程序。
- `scripts/start_ray_m05.sh`、`scripts/start_ray_m06.sh`：两机 Ray 启动程序。
- `scripts/check_ray_roles.py`：跨机角色落点验证。
- `scripts/check_hccl.py`：两机基础 HCCL all-reduce 验证。
- `scripts/check_hccl_fanout.py`：1 个训练 rank 到 8 个 rollout rank 的权重广播拓扑验证。
- `scripts/run_pi_grpo_smoke.sh`：Qwen3.6-27B 单步轨迹 GRPO 冒烟实验。
- `scripts/launch_pi_grpo_smoke.sh`：带退出码、起止时间和完整日志的后台实验启动器。
- `scripts/run_pi_grpo_megatron_tp4_pp2_cp2.sh`：16-NPU Megatron TP4/PP2/CP2 全参轨迹 GRPO 配置。
- `scripts/launch_pi_grpo_megatron_smoke.sh`：Megatron 单步实验的日志、时间和退出码启动器。
- `llin_verl/megatron_bridge_compat.py`、`scripts/patch_verl_megatron_bridge_compat.py`：为昇腾验证版 Megatron-Bridge 补齐当前 veRL 所需的最小兼容接口。

## 已验证状态

- 官方昇腾 veRL 镜像已通过中国大陆镜像站拉取，并重新标记为 `llin-verl-a3:20260730`。
- 两台机器均完成官方镜像的软件栈和 Qwen3.6-27B 模型识别检查。
- Ray 两节点集群已连通，可见 32 张 NPU；角色测试确认训练任务落在 5 号机、rollout 任务落在 6 号机。
- 4 条真实验证任务已转换为 Parquet；两台机器上的只读数据库查询和奖励闭环均为 `4/4` 满分。
- 本地新增 Megatron 拓扑和兼容补丁测试，完整测试为 `10 passed`。
- 经明确授权，两个新建的 `llin` 容器已重建为特权容器；两侧 NPU 探针均通过，未改动其他人的镜像、容器或目录。
- 两机 2-rank HCCL all-reduce 和 1→8 rollout fan-out 均通过；单步配置使用 5 GiB 权重广播 bucket，并将 vLLM HBM 利用率限制为 60%。
- `llin-pi-grpo-one-step-20260730-08` 已完成 1 个真实 GRPO 更新并以退出码 `0` 结束：16 条轨迹均完成 8 轮交互，平均奖励 `0.096875`（最小 `0.05`、最大 `0.20`），actor loss `0.013279`，梯度范数 `0.855277`。
- 单步实测生成耗时 `198.47s`、跨机权重同步 `7.35s`、actor 更新 `35.47s`；已保存完整的 `global_step_1` FSDP2 actor、优化器、额外状态、模型配置和 tokenizer 检查点。
- `pi-grpo-megatron-tp4-pp2-cp2-20260730-07` 已在 5 号机 16 张 NPU 上完成 TP4/PP2/CP2 全参更新，并以退出码 `0` 结束；6 号机继续使用 8 张 NPU 的 vLLM TP8 rollout。
- Megatron 单步共处理 `30,479` tokens，平均奖励 `0.146875`（最小 `0.05`、最大 `1.0`），actor loss `-0.0178175`，梯度范数 `1.16987`；actor 峰值 NPU 显存约 `29.63 GiB`，5 号机进程统计的主机内存约 `824.35 GiB`。
- 本次生成耗时 `206.70s`、跨机权重同步 `6.50s`、actor 更新 `260.59s`，不含保存的训练步骤耗时 `473.81s`。首次验证保存的全量 `global_step_1` 检查点约 `456 GiB`，保存另耗时 `129.05s`；后续 smoke 默认不保存，设置 `SAVE_FREQ=1` 时才保存。

## 参考实现

- [veRL 官方仓库](https://github.com/verl-project/verl)
- [veRL 昇腾安装说明](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/get_start/install_guidance.rst)
- [veRL One-Step-Off-Policy 说明](https://github.com/verl-project/verl/blob/main/docs/advance/one_step_off.md)
- [veRL 昇腾模型与算法支持](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/model_support/model_and_algorithm_support.md)

## 版本记录

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
