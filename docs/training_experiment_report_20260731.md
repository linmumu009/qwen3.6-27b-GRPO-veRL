# Qwen3.6-27B veRL 两机轨迹 GRPO 训练复盘报告

> 报告日期：2026-07-31
> 实验日期：2026-07-29 至 2026-07-30
> 报告范围：数据审计、两机环境、通信验证、FSDP2 单步、Megatron 全参单步、20-step One-Step-Off-Policy、bounded fully-async 短跑
> 最终状态：One-Step-Off-Policy 20/20 步成功；bounded fully-async 3/3 步成功
> 证据边界：本报告只把有仓库代码、提交记录、运行摘要或日志支持的结果写成“已验证”

## 技术摘要

本次实验最终跑通了 Qwen3.6-27B 在两台昇腾 A3 服务器上的多轮工具交互轨迹 GRPO：

- 5 号机使用 16 张 NPU 做 Megatron 全参训练，拓扑为 `TP=4、PP=2、CP=2`。
- 6 号机使用 16 张 NPU 做 vLLM rollout，拓扑为 `TP=8、DP=2`。
- 两台机器通过 Ray 分配角色，通过 HCCL 完成训练权重从 1 个发送端到 16 个 rollout rank 的广播。
- 训练关闭 LoRA，actor 参数常驻 NPU；优化器和梯度卸载到 5 号机主机内存，并开启全量激活重计算。
- 数据不是直接复用老板的历史轨迹，而是把其中可验证任务转成 prompt-only Parquet；模型在 rollout 时重新调用只读 SQLite 工具，奖励函数同时检查工具证据、必需表和最终数值。
- 最终 One-Step-Off-Policy 实验完成 20/20 步，退出码为 0。18 个稳态步平均耗时 `230.52s`，其中生成平均 `173.02s`，占整步 `73.19%`；trainer 明显在等待 rollout。
- 两路 vLLM prefix cache 的累计命中率为 `32.41%`，Continuous Token 全程未回退。
- 长尾判据被触发后，我们实现了“完整 GRPO group + queued-token budget”的 bounded fully-async 队列，并完成 3/3 步短跑。第二批队列等待从 `159.41s` 降到 `111.96s`，下降约 `29.8%`。

这次实验已经证明“环境、数据、工具调用、奖励、跨机权重同步、全参更新、checkpoint 和异步队列”能够闭环，但还没有证明训练质量已经收敛，也没有证明 3 步 fully-async 在长稳态下必然比 One-Step-Off-Policy 吞吐更高。当前最重要的后续工作是扩充验证数据、运行更长的 fully-async 稳态实验，并补做可恢复的优化器 checkpoint。

## 最终结论一览

| 检查项 | 最终结果 | 证据或口径 |
| --- | --- | --- |
| 两机资源 | 32 张 NPU 可见，训练和 rollout 角色落点正确 | Ray 角色检查 |
| 数据闭环 | 4 条真实验证任务，两机 SQLite 查询与奖励检查均为 `4/4` | 集成检查 |
| 基础通信 | 2-rank HCCL all-reduce 成功 | 通信预检 |
| 权重 fan-out | 1→16 all-reduce、broadcast、256 MiB stateless PyHCCL broadcast 成功 | fan-out 预检 |
| FSDP2 单步 | 1 个真实 GRPO 更新成功，退出码 0 | `llin-pi-grpo-one-step-20260730-08` |
| Megatron 单步 | TP4/PP2/CP2 全参更新成功，退出码 0 | `pi-grpo-megatron-tp4-pp2-cp2-20260730-07` |
| One-Step 长跑 | 20/20 步成功，20 份 rollout 落盘 | `pi-grpo-megatron-tp4-pp2-cp2-tp8-dp2-20step-20260730-11` |
| Continuous Token | 正式 20 步无 processor fallback | driver log |
| Prefix cache | `469,248 / 1,447,944` tokens，命中率 `32.41%` | 两路 vLLM `/metrics` |
| 最终 checkpoint | `global_step_20` 约 48 GiB，13 个 HF 分片齐全 | checkpoint 完整性检查 |
| bounded fully-async | 3/3 步成功，无 queue drop，无 Continuous Token fallback | `pi-grpo-fully-async-bounded-3step-20260730-02` |
| 本地回归测试 | `29 passed` | pytest |

## 1. 初始目标、约束和环境

### 1.1 目标

目标不是普通单轮问答 GRPO，而是训练一个能够在物流数据环境中多轮调用工具的 Qwen3.6-27B agent。一次 rollout 需要经历：

1. 读取用户问题；
2. 判断需要查询哪些表和字段；
3. 调用 `query_sqlite`；
4. 读取工具返回；
5. 必要时继续查询；
6. 给出带明确数值的中文答案；
7. 由规则奖励核对工具证据和答案。

同时希望采用“一台机器训练、一台机器 rollout”的拆分，避免训练和推理抢同一组 NPU。

### 1.2 基础设施约束

| 项目 | 初始情况 | 对方案的影响 |
| --- | --- | --- |
| 服务器 | 5 号机和 6 号机，各 16 张昇腾 NPU | 可形成 16-NPU 训练池和 16-NPU rollout 池 |
| 网络 | 两机内网连通 | Ray 和 HCCL 均固定走内网接口 |
| 外网 | 只能稳定访问中国大陆站点 | 优先使用官方昇腾 veRL 镜像的大陆镜像来源 |
| 隔离要求 | 不能动其他人的镜像、容器和目录 | 所有资源统一以 `llin` 开头 |
| 初始容器权限 | 新建的非特权容器无法初始化 NPU | 经明确授权后只重建两个 `llin` 容器为特权容器 |
| 工作目录 | 5 号机已有 `/data3/llin`；6 号机需保持同构 | 两机统一使用 `/data3/llin/qwen3.6-27b-verl-grpo` |

### 1.3 最初部署

| 项目 | 5 号机 | 6 号机 |
| --- | --- | --- |
| 角色 | trainer | rollout |
| 主机工作目录 | `/data3/llin/qwen3.6-27b-verl-grpo` | `/data3/llin/qwen3.6-27b-verl-grpo` |
| 容器工作目录 | `/workspace/llin-verl-grpo` | `/workspace/llin-verl-grpo` |
| 容器 | `llin-verl-trainer-m05-20260730` | `llin-verl-rollout-m06-20260730` |
| 镜像 | `llin-verl-a3:20260730` | `llin-verl-a3:20260730` |
| 模型 | `/models/Qwen3.6-27B` | `/models/Qwen3.6-27B` |
| Ray 自定义资源 | `llin_trainer` | `llin_rollout` |

镜像来自官方昇腾 veRL 配置，通过大陆可访问的镜像源取得，再重命名为 `llin-verl-a3:20260730`。没有改动或复用其他人的镜像和容器。

## 2. 原始数据不能直接训练，必须重建可验证 rollout

### 2.1 原始数据

原始归档位于：

```text
/data/renjunxiang/coding/huawei_train/archives/trajectories_v15_27B_table.tar.gz
```

审计结果：

- 共 1,500 个 JSONL 轨迹文件；
- 记录了 prompt、模型消息、工具调用和工具输出；
- 适合作为任务来源、轨迹参考和 verifier 来源；
- 没有可直接交给 veRL GRPO 的显式 reward；
- 历史工具输出不能作为新策略的在线行为，因此不能把旧轨迹原样当作 on-policy GRPO batch。

结论是：数据“能用”，但用途是构造可验证任务，而不是直接把 DPO 轨迹喂给 GRPO。

### 2.2 转换后的训练样本

首轮实验选择 4 条已经核验的真实任务，生成：

```text
data/pi_verified_smoke.parquet
```

每条样本包含：

- `agent_name=tool_agent`；
- system prompt 和用户问题；
- `environment_id`，用于绑定对应的 `logistics.sqlite`；
- `verifier_id`；
- 期望数值；
- 必须查询的表；
- 工具参数和 `query_sqlite` 选择信息。

转换程序是 `scripts/prepare_pi_dataset.py`。它只接受同时存在 prompt 与 verifier 的记录，并把 verifier 信息放入 `reward_model.ground_truth`。

### 2.3 在线工具环境

`llin_verl/pi_sqlite_tool.py` 提供只读 `query_sqlite`。安全边界是：

- 只允许 `SELECT` 或 `WITH`；
- 每个任务只能访问其 `environment_id` 对应的 SQLite；
- 查询日志写回样本的 `extra_info.llin_sql_queries`；
- reward 不相信模型声称“我查过了”，只检查真实工具日志。

### 2.4 奖励定义

奖励函数位于 `llin_verl/pi_reward.py`，分数定义如下：

| 条件 | reward |
| --- | ---: |
| 成功调用工具、覆盖全部必需表、最终数值正确 | `1.0` |
| 成功调用工具、覆盖全部必需表、最终数值错误 | `0.2` |
| 成功调用工具，但未覆盖全部必需表 | `0.05` |
| 没有成功工具调用 | `0.0` |

数值核对支持逗号和小误差容忍。这个设计使 reward 同时约束：

1. 是否真的使用工具；
2. 是否查询了正确的数据来源；
3. 是否给出了正确的最终结果。

### 2.5 数据边界

当前 4 条任务足以验证工程闭环，但不足以评价模型泛化或训练收益。20 步训练会重复使用很小的数据集合，因此本报告中的 reward 只能作为“训练管道能产生非零且可变化的信号”的证据，不能解释为模型质量已经提升。

## 3. 方案如何从 FSDP2 演进到 Megatron 全参

### 3.1 第一版：FSDP2 单步闭环

第一版真实训练采用：

| 配置项 | 值 |
| --- | --- |
| trainer | 5 号机 16 NPU，FSDP2 |
| rollout | 6 号机 8 NPU，vLLM `TP=8、DP=1` |
| actor 参数卸载 | 开启 |
| optimizer 卸载 | 开启 |
| gradient checkpointing | 开启 |
| prompt 最大长度 | 2,048 tokens |
| response 最大长度 | 4,096 tokens |
| vLLM 最大模型长度 | 6,144 tokens |
| train batch | 4 prompts |
| 每个 prompt 的 rollout | `n=4` |
| 每步轨迹数 | 16 |
| 多轮上限 | assistant 4 轮、user 3 轮 |
| 单次工具回复上限 | 1,024 tokens |
| 权重同步 bucket | 5,120 MiB |
| checkpoint | 单步保存完整 actor、optimizer 和 extra |

这版的目的不是追求吞吐，而是先证明“真实数据 → 多轮工具 rollout → reward → GRPO update → checkpoint”全部连通。

在完成数据、Ray、HCCL 和角色预检后，初始单步通过以下入口启动：

```bash
RUN_NAME=llin-pi-grpo-one-step-20260730-08 \
bash scripts/launch_pi_grpo_smoke.sh
```

实际配置由当时版本的 `scripts/run_pi_grpo_smoke.sh` 注入，launcher 将完整日志、起止时间和退出码写入独立运行目录。

`llin-pi-grpo-one-step-20260730-08` 最终成功：

- 16 条轨迹均完成 8 轮交互；
- reward 平均 `0.096875`，最小 `0.05`，最大 `0.20`；
- actor loss `0.013279`；
- gradient norm `0.855277`；
- generation `198.47s`；
- 跨机权重同步 `7.35s`；
- actor update `35.47s`；
- 退出码 `0`；
- 保存了完整 `global_step_1`。

### 3.2 为什么没有继续使用 FSDP2

FSDP2 可以跑通，但不适合我们计划中的长上下文全参训练：

- 参数 all-gather 会在层边界产生显存峰值；
- 27B 模型加长 prompt/response 后，激活和临时 buffer 继续增长；
- 为降低峰值而大量卸载参数，会增加主机与 NPU 之间的数据搬运；
- 目标不是 LoRA，而是利用 5 号机 16 张 NPU 做全参训练。

因此训练后端改为 Megatron，并采用用户最终确认的 `TP4 × PP2 × CP2`。

### 3.3 最终训练拓扑

`TP=4、PP=2、CP=2` 的乘积是 16，刚好占满 5 号机：

- TP 把单层权重和计算切到 4 张 NPU；
- PP 把模型层分成 2 个 pipeline stage；
- CP 把长序列上下文切到 2 路；
- 训练侧数据并行度为 1；
- rollout 侧采用两套 `TP=8` engine，即 `TP8 × DP2`，占满 6 号机 16 张 NPU。

该拓扑避免了 FSDP2 的逐层全参数 all-gather，同时保留 CP 对长序列的分摊能力。

### 3.4 Megatron 全参单步里程碑

`pi-grpo-megatron-tp4-pp2-cp2-20260730-07` 首次证明该拓扑能够做真实全参更新：

| 指标 | 实测值 |
| --- | ---: |
| 训练 tokens | 30,479 |
| 平均 reward | 0.146875 |
| reward 范围 | 0.05–1.0 |
| actor loss | -0.0178175 |
| gradient norm | 1.16987 |
| actor 峰值 NPU 显存 | 约 29.63 GiB |
| 5 号机进程统计主机内存 | 约 824.35 GiB |
| generation | 206.70s |
| 权重同步 | 6.50s |
| actor update | 260.59s |
| 不含保存的整步 | 473.81s |
| checkpoint 保存 | 129.05s |
| 完整 checkpoint | 约 456 GiB |

该结果回答了“能否去掉 LoRA 做全参”的问题：可以。代价是优化器状态和 checkpoint 很大，因此后续保留全参训练，但把 optimizer/gradient 卸载到主机，并降低 checkpoint 频率和保存内容。

## 4. 十余次关键尝试及问题解决

下表把环境预检、通信实验和训练实验放在同一条时间线上，但明确标注类型。早期部分尝试没有保留逐次独立 run 目录，因此按可审计的问题阶段归并；不会为缺失日志编造编号。

| 序号 | 类型 / 实验 | 当时配置或目标 | 现象 | 根因 | 解决方法 | 结果 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 数据审计 | 直接评估 1,500 条历史 DPO/PI 轨迹能否用于 GRPO | 有轨迹但没有显式 reward | 历史轨迹不是新策略的在线 rollout，也缺少 veRL 所需奖励闭环 | 提取 prompt、环境和 verifier，重新在线 rollout | 形成数据改造方案 |
| 2 | 镜像与软件栈 | 在大陆网络部署 veRL Ascend 环境 | GitHub/海外镜像不可作为稳定依赖 | 服务器网络访问限制 | 使用官方昇腾 veRL 镜像的大陆可达来源，并重标记为 `llin-*` | 两机模型和软件栈识别通过 |
| 3 | 非特权容器 NPU 探针 | 两个新建 `llin` 容器先按非特权运行 | 容器内无法初始化 NPU | 昇腾设备和运行时权限不足 | 获得明确授权后，只重建两个新建 `llin` 容器为特权容器 | 两侧 NPU 探针通过 |
| 4 | Ray 两机角色 | 5 号机训练、6 号机 rollout | 需要避免 worker 漂移到错误节点 | 默认 Ray 资源不能表达本项目的角色边界 | 加入 `llin_trainer`、`llin_rollout` 自定义资源，并用 `sitecustomize.py` 固定 placement | 32 NPU 可见，角色检查通过 |
| 5 | 数据与 reward 集成 | 4 条真实任务在两机执行 | 需要确认 DB 路径、工具日志和 verifier 一致 | 跨容器路径及 metadata 容易错位 | 使用 `environment_id` 绑定 SQLite，执行只读查询和奖励闭环检查 | 两机均为 `4/4` |
| 6 | 2-rank HCCL | 先验证基础跨机 all-reduce | 自动网卡可能选中不可达管理网 | 通信接口和端口未固定 | 固定内网 `eno0`，host/NPU socket 使用不重叠端口段 | all-reduce 成功 |
| 7 | 1→8 HCCL fan-out | FSDP2 trainer 向 8 个 TP rollout rank 发权重 | 非对称 communicator 比 2-rank 更复杂 | 默认 HCCL 路径不一定支持该 rank 分布 | 新增 `check_hccl_fanout.py`，先单独验证广播 | 1→8 通过 |
| 8 | FSDP2 单步 | 16-NPU FSDP2 + 8-NPU TP8 rollout | 需要验证完整轨迹 GRPO 更新 | 数据、工具格式、HCCL、checkpoint 任一处都可能失败 | 切换 `qwen3_coder` 工具格式，逐层完成预检 | `llin-pi-grpo-one-step-20260730-08` 成功 |
| 9 | Megatron-Bridge 接入 | 改为 TP4/PP2/CP2 全参 | veRL 所需 Bridge 接口与镜像内版本不完全一致 | 上游版本组合存在最小接口缺口 | 引入指定 Megatron-Bridge 源码并添加幂等兼容补丁 | 模型和 optimizer 构建成功 |
| 10 | Megatron 单步 | 16-NPU TP4/PP2/CP2 + 8-NPU rollout | 第一次完整 checkpoint 体积和保存时间很大 | 全量模型、优化器和 extra 都保存 | 先保留一次完整证据，后续降低保存频率并缩减内容 | `...tp4-pp2-cp2-20260730-07` 成功 |
| 11 | checkpoint 压力 | 单步 checkpoint 约 456 GiB，保存另耗时 129.05s | 频繁保存会破坏稳态测量并快速占满磁盘 | 优化器状态占据主要空间 | 正式 20 步只在第 20 步保存 `model,extra`，只保留 1 份 | 最终约 48 GiB |
| 12 | 1→16 HCCL 扩展 | rollout 从 TP8/DP1 扩为 TP8/DP2 | 默认/部分算法在非均匀 1+16 rank 拓扑失败或不支持 | 非对称跨机 fan-out 对算法有要求 | 使用官方 HCCL 语法 `broadcast=level0:NA;level1:NHR`，并以 256 MiB stateless PyHCCL 做独立验证 | 1→16 all-reduce、broadcast 均通过 |
| 13 | TP8/DP2 权重同步 | 6 号机启用两套 TP8 engine | 两个 DP replica 都使用 IPC rank `0..7`，第二组应为 `8..15` | vLLM multiprocessing DP 子进程重置了 DP 配置 | 从 `ASCEND_RT_VISIBLE_DEVICES` 恢复 DP replica 的本地偏移 | 16 张 rollout NPU 全部工作 |
| 14 | vLLM worker 导入 | Ray 在 6 号机启动 rollout/AgentLoop | worker 侧无法稳定找到 vLLM 模块或补丁 | 只在 driver 环境设置路径不足 | 两机启动脚本显式加入 `/vllm` 到 `PYTHONPATH`，worker 侧补丁在 `ray start` 前应用 | worker 初始化稳定 |
| 15 | 正式 20 步 `-07` | TP4/PP2/CP2 + TP8/DP2 | 完成首批 rollout 后，落盘调用缺少 `_dump_executor` | One-Step trainer 绕过了基类构造，但继承的 dump 方法仍依赖 executor | 增加 `_init_dump_executor()` 幂等补丁 | 进入下一轮尝试 |
| 16 | 正式 20 步 `-08` | 加入 rollout dump 修复 | 完成 1 个真实更新后报 `TypeError: cannot unpack None` | 只有 4 条训练样本，dataloader 迭代器在计划的 20 步前耗尽 | 将 `trainer.total_epochs` 同步设置为总训练步数 | 数据可循环供给 20 步 |
| 17 | 正式 20 步 `-09/-10` | 开启 Continuous Token | 日志仍出现 processor fallback；重启后结果未改变 | 补丁只应用在 5 号机 driver，真正的 AgentLoop worker 在 6 号机重新加载 processor | 在 6 号机 `ray start` 前也修补 AgentLoop；仅在明确 text-only 时清空 processor | 正式运行无 fallback |
| 18 | 正式 20 步 `-11` | 完整最终 One-Step 配置 | 无阻塞错误 | 前述问题均修复 | 开启 NPU/cache 监控并执行 20 步 | 20/20 步、退出码 0 |
| 19 | cache 统计补充 | 需要回答 prefix cache 是否命中 | driver 日志没有可靠的汇总计数 | vLLM 指标在动态 `LLMServerManager` endpoint | 监控程序从日志发现 endpoint，轮询 `/metrics` 并汇总两个 engine | 命中率 `32.41%` |
| 20 | fully-async `-01` | 完整 group + token budget 队列 | 训练开始前触发 `lr_decay_steps > 0` assertion | fully-async Megatron 路径要求显式正数 scheduler 步数 | 设置 constant scheduler，`lr_decay_steps=TOTAL_TRAINING_STEPS` | 配置阶段问题消除 |
| 21 | fully-async `-02` | 6-group、40k token、staleness 0.5 | 需要确认队列不会拆 group 或静默丢样本 | 上游 sample-wise `deque(maxlen=...)` 可能淘汰旧样本 | 队列改存完整序列化 group 和精确 token 数，满载背压 | 3/3 步、无 drop、退出码 0 |

### 4.1 Megatron 兼容问题的处理原则

没有直接修改仓库外的上游源码并把不可追溯状态留在容器里，而是把每个最小改动写成幂等 patch：

- `patch_verl_megatron_bridge_compat.py`
- `patch_verl_vllm_dp_weight_sync.py`
- `patch_verl_one_step_dump_executor.py`
- `patch_verl_one_step_continuous_token.py`
- `patch_verl_agent_loop_continuous_token.py`
- `patch_verl_fully_async_continuous_token.py`
- `patch_verl_fully_async_group_token_queue.py`

启动脚本每次先应用补丁，再启动训练。补丁已经应用时返回 `already-patched`，从而保证容器重启后的行为可重复。

### 4.2 权重广播 bucket 为什么改为 3 GiB

最终 bucket 设为 `3,072 MiB`，原因是 Qwen3.6-27B 最大 embedding 参数约 `2.37 GiB`。bucket 不能小于最大的不可拆参数，否则权重同步无法正常封包；同时没有必要继续保留早期的 5 GiB bucket。3 GiB 在可容纳最大参数和控制 vLLM 侧瞬时内存之间取得了更合适的平衡。

## 5. 最终成功的 One-Step-Off-Policy 配置

### 5.1 环境

| 维度 | 最终值 |
| --- | --- |
| trainer 主机 | 5 号机 |
| rollout 主机 | 6 号机 |
| 每机 NPU | 16 |
| 容器 | `llin-verl-trainer-m05-20260730` / `llin-verl-rollout-m06-20260730` |
| 镜像 | `llin-verl-a3:20260730` |
| veRL 模式 | `experimental.one_step_off_policy` |
| 模型 | Qwen3.6-27B |
| 数据 | `data/pi_verified_smoke.parquet` |
| 通信 | Ray + HCCL，固定内网接口 |
| HCCL broadcast | `level0:NA;level1:NHR` |

### 5.2 训练侧配置

| 配置项 | 最终值 |
| --- | --- |
| strategy | Megatron |
| TP / PP / CP | `4 / 2 / 2` |
| trainer NPU | 16 |
| LoRA | 关闭，`lora_rank=0` |
| dtype | bfloat16 |
| actor 参数卸载 | 关闭 |
| optimizer CPU offload | 开启，比例 1.0 |
| gradient offload | 开启 |
| distributed optimizer | 开启 |
| activation recompute | `uniform / full / 1 layer` |
| sequence parallel | 开启 |
| context parallel algorithm | `kvallgather_cp_algo` |
| attention | FlashAttention，backend auto |
| actor learning rate | `1e-6` |
| PPO mini-batch | 4 groups |
| micro-batch / NPU（配置键仍为 `per_gpu`） | 1 |
| dynamic batch | 关闭 |
| KL loss | 关闭 |
| entropy coefficient | 0 |

这里的“全参”指 actor 的全部模型参数都参加训练，不使用 LoRA。为了容纳 27B 模型和 6,144-token 上限，参数常驻 NPU，优化器与梯度卸载到主机内存，激活通过 TP/PP/CP 和 recompute 控制。

### 5.3 rollout 与多轮工具配置

| 配置项 | 最终值 |
| --- | --- |
| vLLM TP / DP | `8 / 2` |
| rollout NPU | 16 |
| 每个 prompt 的响应数 | `n=4` |
| prompt 上限 | 2,048 tokens |
| response 上限 | 4,096 tokens |
| max model len | 6,144 tokens |
| max batched tokens | 8,192 |
| max sequences | 16 |
| HBM utilization | 0.60 |
| mode | async，enforce eager |
| prefix caching | 开启 |
| Continuous Token | 开启，`qwen35`，text-only |
| agent workers | 8 |
| 工具格式 | `qwen3_coder` |
| assistant / user 轮数上限 | `4 / 3` |
| 单次工具回复上限 | 1,024 tokens |
| checkpoint sync bucket | 3,072 MiB |

### 5.4 batch 的准确含义

One-Step 每步：

- 从数据集中取 4 个 prompt；
- 每个 prompt 生成 4 条 rollout；
- 因此每次 GRPO update 使用 4 个比较 group，共 16 条轨迹；
- group 内 4 条轨迹共享同一个 prompt，GRPO 的相对 advantage 只在 group 内有意义；
- 不能为了凑 token budget 把 group 拆散到不同 update。

### 5.5 checkpoint 策略

正式 20 步实验采用：

- `save_freq=20`；
- 只保存 `model,extra`；
- `max_actor_ckpt_to_keep=1`；
- 第 20 步保存；
- 最终目录约 48 GiB；
- safetensors index 引用的 13 个实际分片全部存在。

这能避免早期每步保存约 456 GiB 全量状态、额外消耗约 129 秒的问题。但因为没有保存 optimizer，当前 checkpoint 更适合模型留档和后续加载，不是严格意义上的无损训练续跑点。

## 6. 最终 One-Step 是怎么启动的

以下命令应在各自 `llin` 容器中执行，不包含 SSH、密钥或外部访问方式。

### 6.1 准备数据

```bash
python3 scripts/prepare_pi_dataset.py \
  --prompts /path/to/verified_prompts.jsonl \
  --verifier-manifest /path/to/verifier_manifest.jsonl \
  --output data/pi_verified_smoke.parquet \
  --limit 4
```

然后运行两机集成检查，确认数据库、工具和奖励均为 `4/4`。

### 6.2 启动 Ray

先在 5 号机：

```bash
bash scripts/start_ray_m05.sh
```

再在 6 号机：

```bash
bash scripts/start_ray_m06.sh
```

启动脚本会：

- 固定 HCCL 内网接口和端口；
- 注册 `llin_trainer` / `llin_rollout`；
- 应用 worker 侧必要补丁；
- 将 5 号机作为 Ray head，6 号机作为 rollout node。

### 6.3 训练前预检

```bash
python3 scripts/check_ray_roles.py
python3 scripts/check_hccl.py
ROLLOUT_RANKS=16 python3 scripts/check_hccl_fanout.py
```

必须先确认：

- Ray 可见 32 张 NPU；
- trainer actor 只落在 5 号机；
- rollout actor 只落在 6 号机；
- 2-rank all-reduce 通过；
- 1→16 fan-out 通过；
- 两路 TP8 engine 的可见设备分别为 `0..7` 和 `8..15`。

### 6.4 启动正式 20 步

```bash
RUN_NAME=pi-grpo-megatron-tp4-pp2-cp2-tp8-dp2-20step-20260730-11 \
TOTAL_TRAINING_STEPS=20 \
SAVE_FREQ=20 \
ROLLOUT_NPUS=16 \
WEIGHT_BUCKET_MB=3072 \
bash scripts/launch_pi_grpo_megatron_smoke.sh
```

核心配置全部保存在 `scripts/run_pi_grpo_megatron_tp4_pp2_cp2.sh`。launcher 记录：

- driver PID；
- 开始时间；
- 完整 `driver.log`；
- 结束时间；
- 退出码。

训练期间使用：

- `scripts/monitor_npu_utilization.py` 采集两机 NPU 利用率；
- `scripts/monitor_vllm_cache_metrics.py` 发现 vLLM endpoint 并采集 cache counter；
- `scripts/analyze_grpo_steady_state.py` 汇总稳态、长尾和 fully-async 切换判据。

## 7. 20 步实测结果证明管道稳定，但训练机空转明显

正式实验：

```text
pi-grpo-megatron-tp4-pp2-cp2-tp8-dp2-20step-20260730-11
```

完成 `20/20` 步，退出码 `0`，产生 20 份 rollout JSONL，Continuous Token 全程没有 processor fallback。

### 7.1 稳态耗时

前 2 步作为预热，下面统计余下 18 步：

| 指标 | 均值 | 中位数 | p95 / 最大值 |
| --- | ---: | ---: | ---: |
| 整步耗时 | 230.52s | 208.35s | 553.99s |
| rollout 生成 | 173.02s | 177.75s | 399.96s |
| actor update | 49.83s | 40.49s | 146.38s |
| 权重同步 | 7.37s | 7.33s | 7.84s |
| 平均 reward | 0.2413 | 0.2422 | 0.3250 |

生成平均占整步的 `73.19%`，中位数占比更高，为 `80.40%`。这说明主要瓶颈不是 1→16 权重同步，而是多轮长轨迹生成。

### 7.2 长尾

| 长尾指标 | 均值 | 中位数 | p95 / 最大值 |
| --- | ---: | ---: | ---: |
| 最慢轨迹 / 平均轨迹生成时间 | 1.96× | 1.86× | 3.33× |
| 生成耗时 / 整步耗时 | 73.19% | 80.40% | 88.31% |

第 12 步是最明显的长尾：

- generation `399.96s`；
- actor update `146.38s`；
- 整步 `553.99s`；
- 最慢轨迹约为平均轨迹的 `3.33×`。

分析器的切换规则为：

```text
p90 tail ratio >= 1.75
且
mean generation share >= 0.55
```

本次两项均满足，因此 `recommend_bounded_fully_async=true`。

### 7.3 NPU 利用率

| 主机角色 | AICore 均值 | NPU utilization 均值 | AICore 非零记录占比 |
| --- | ---: | ---: | ---: |
| 5 号机 trainer | 3.47% | 5.15% | 6.67% |
| 6 号机 rollout | 5.78% | 16.00% | 80.39% |

这些是固定时间间隔采样的设备级描述性指标，不等同于 kernel 理论利用率。即便如此，trainer 的 AICore 非零记录只占 `6.67%`，足以支持“5 号机大部分时间在等 rollout”的诊断。

### 7.4 Prefix cache

两路 vLLM engine 的最终累计计数：

```text
hits    = 469,248 tokens
queries = 1,447,944 tokens
rate    = 32.4079%
```

因此本次确实开启并命中了 prefix cache。命中率约 32.41%，并不能消除多轮工具轨迹的 response 生成成本，但证明 cache 路径有效。

## 8. 为什么要改成 bounded fully-async

One-Step-Off-Policy 的同步边界是：

1. rollout 机生成当前 batch；
2. trainer 等待完整 batch；
3. trainer 更新；
4. 同步权重；
5. 下一批 rollout。

在轨迹长度差异很大时，整个 batch 被最慢轨迹拖住。trainer 在生成阶段没有可训练的上一批数据，因此空转。

fully-async 的目标是让：

- 6 号机持续生产完整 GRPO group；
- 5 号机只要队列中已有足够 group 就立即训练；
- trainer 更新时，rollout 可以继续为下一步生产；
- 权重在允许的 staleness 范围内同步。

## 9. 最终 bounded fully-async 设计

### 9.1 为什么不能只用普通流式 sample 队列

GRPO 需要对同一 prompt 的多条 response 做组内相对比较。若 sample-wise 队列：

- group 可能被拆散；
- `deque(maxlen=...)` 可能静默淘汰较旧样本；
- token 很长时，按“样本个数”限制不能反映实际内存；
- 不同策略版本的样本可能被错误组合。

因此队列元素不是单条 response，而是一个完整 GRPO group。

### 9.2 队列约束

最终短跑配置：

| 配置项 | 值 |
| --- | ---: |
| 每个 prompt 的轨迹数 | 4 |
| 每个训练步消费 group | 4 |
| 每步轨迹总数 | 16 |
| 最大在途/排队 group | 6 |
| queued-token 上限 | 40,000 |
| 总短跑 group | 12 |
| 并发样本 / replica | 16 |
| staleness threshold | 0.5 |
| parameter sync trigger | 1 step |
| partial rollout | 开启 |
| 满载行为 | producer 等待，绝不淘汰旧 group |

token 数使用生成后 `attention_mask.sum()` 的精确值。若一个 group 自身超过 40k：

- 仅当队列为空时允许它进入，保证系统不会永久死锁；
- 它仍然不会被拆分；
- 后续 producer 等待，直到该 group 被消费。

### 9.3 调度和 off-policy 设置

fully-async 额外配置：

- `actor.use_rollout_log_probs=True`；
- `rollout.calculate_log_probs=True`；
- `algorithm.rollout_correction.bypass_mode=True`；
- `staleness_threshold=0.5`；
- `trigger_parameter_sync_step=1`；
- `lr_decay_style=constant`；
- `lr_decay_steps=TOTAL_TRAINING_STEPS`。

本次是 One-Step-Off-Policy / bounded fully-async 工程验证，不应解释为已经完成了严格的 off-policy 校正比较。正式长跑前应再确认 staleness 定义和 correction 策略是否符合最终算法目标。

## 10. fully-async 3 步结果：管道成立，吞吐收益仍需长跑

实验：

```text
pi-grpo-fully-async-bounded-3step-20260730-02
```

结果：

- `3/3` 步完成；
- 退出码 `0`；
- 无 queue drop warning；
- 无 Continuous Token fallback；
- 两路 cache 命中率 `32.41%`；
- 每步消费 4 个完整 group；
- 40k queued-token 背压生效。

### 10.1 队列等待和参数同步

| 批次 | 等待 group 的时间 | 解释 |
| ---: | ---: | --- |
| 1 | 159.41s | 冷启动，没有 backlog |
| 2 | 111.96s | 利用上一轮训练期间积累的 rollout，较第一批下降 29.8% |
| 3 | 177.26s | 3 步有限生产在结束边界出现尾部等待 |

参数同步：

```text
14.0762s / 7.7875s / 7.7522s / 7.2762s
```

第一次包含初始化成本，后续稳定在约 7.3–7.8 秒，与 One-Step 的约 7.37 秒一致。

### 10.2 前两个完整日志步

| 指标 | 第一个日志步 | 第二个日志步 |
| --- | ---: | ---: |
| step | 312.05s | 227.66s |
| generation / queue wait | 159.44s | 111.98s |
| actor update | 144.80s | 107.91s |
| trainer resource utilization | 46.40% | 47.40% |
| actor loss | — | -0.04190 |
| gradient norm | 0 | 0.79298 |

第一步 gradient norm 为 0，是因为该批 group 内 reward 相同，GRPO 组内 advantage 为 0；第二步出现非零 loss 和 gradient norm，证明参数实际发生更新。

### 10.3 结论边界

短跑证明：

- producer/consumer 能并发；
- 完整 group 不会被拆；
- token budget 能背压；
- 没有静默丢样本；
- 第二批确实使用了训练期间产生的 backlog；
- 非零梯度更新能够发生。

短跑没有证明：

- 长时间稳态下 trainer 一定持续满载；
- 40k / 6-group 是最优队列大小；
- staleness 0.5 对最终训练质量没有影响；
- fully-async 的总 tokens/s 已稳定超过 One-Step。

当前两个完整日志步的 trainer 利用率仍只有约 46%–47%，说明队列设计解决了“完全串行”的结构问题，但 rollout 生产速度仍可能低于 trainer 消费速度。

## 11. 验证口径和稳健性检查

### 11.1 已完成的检查

- 拓扑约束：`TP × PP × CP = trainer NPU`；
- rollout 约束：`ROLLOUT_NPUS % TP = 0`；
- Ray 角色落点；
- 两机 2-rank HCCL；
- 1→16 非对称 fan-out；
- 256 MiB stateless PyHCCL broadcast；
- 4 条真实任务两机工具/reward 闭环；
- 补丁幂等性；
- Continuous Token 无 fallback；
- 两个 vLLM engine cache counter 汇总；
- NPU 采样窗口与训练稳态对齐；
- 20 个 rollout 文件存在；
- 第 20 步 checkpoint 分片完整；
- fully-async 无 queue drop；
- 本地 `29 passed`。

### 11.2 指标定义

- **整步耗时**：日志中的 `timing_s/step`。
- **生成耗时**：日志中的 `timing_s/gen`；fully-async 下还包含等待队列满足训练 group 的时间语义。
- **actor update**：`timing_s/update_actor`。
- **权重同步**：训练权重经 checkpoint engine 广播到 rollout worker 的耗时。
- **稳态步**：20 步中排除最前面 2 个预热步后的 18 步。
- **cache hit rate**：两个 engine 的累计 prefix cache hits 除以 queries。
- **AICore active record ratio**：采样记录中 AICore 利用率大于 0 的比例。
- **长尾比**：一个 batch 内最慢轨迹生成时间除以平均轨迹生成时间。

### 11.3 没有画趋势图的原因

本报告的核心证据是离散的环境变更、失败根因和两次最终运行。One-Step 有 18 个稳态样本，fully-async 只有 3 步，把两者画成连续趋势容易放大短跑波动并暗示不存在的可比性。因此采用逐次实验表和精确汇总表，而不做误导性的趋势图。

## 12. 局限、不确定性和风险

### 12.1 数据量太小

只使用 4 条验证任务，能够证明流程，不能证明泛化。重复采样还可能使 prefix cache 命中率高于大规模多样化数据时的实际水平。

### 12.2 fully-async 样本太少

3 步只能做功能验证。第三批等待升高可能来自有限生产任务的收尾，不应据此否定或确认长期吞吐。

### 12.3 NPU 指标是采样值

设备采样可说明“长时间没有 kernel 活动”，但不能替代算子级 profiling。需要结合 tokens/s、queue depth、group token 分布和算子 trace 才能做精细容量规划。

### 12.4 checkpoint 不能无损续训

20 步最终 checkpoint 未保存 optimizer。模型权重完整，但重新开始训练时优化器动量和调度器状态不能严格恢复。

### 12.5 off-policy 校正仍需算法确认

短跑使用 `rollout_correction.bypass_mode=True`。如果未来增加允许 staleness 或一次积累多个旧策略 batch，需要明确是否采用重要性校正、裁剪方式和最大策略版本差。

### 12.6 第一批出现零梯度是数据多样性信号

当同一 group 四条轨迹得到相同 reward 时，GRPO 组内 advantage 为 0。小数据集上这种情况更常见，意味着即使工程流程正常，也可能产生没有学习信号的 batch。

### 12.7 早期逐次日志不完整

仓库保存了最终脚本、补丁、提交历史和最终运行摘要，但没有把每次早期容器调试都下载为独立 run 目录。因此本报告按“可验证的问题阶段”归并早期尝试；正式 `-07` 至 `-11` 和 fully-async `-01/-02` 则按运行编号记录。

## 13. 下一步建议

按优先级建议：

1. **扩大数据后再做质量结论。** 从 1,500 条轨迹中建立更大的 verified prompt 集，统计缺失 verifier、DB 不可用、答案不可数值核对和超长 prompt 的比例。
2. **运行至少 20–50 步 fully-async 稳态。** 排除预热和收尾，比较 tokens/s、trainer idle、rollout idle、queue depth、queued tokens、策略版本差和 reward。
3. **同时记录 group token 分布。** 用 p50/p90/p95/p99 group tokens 决定 40k budget 和 6-group 上限，而不是只凭短跑。
4. **调节生产并发。** 如果 trainer 仍然饥饿，逐步增加 rollout 并发；如果 queue 经常打满，再判断是 token budget 太小还是 trainer 变慢。
5. **加入低频完整 checkpoint。** 例如模型 checkpoint 高频、包含 optimizer 的恢复 checkpoint 低频，兼顾磁盘和灾难恢复。
6. **做真实恢复演练。** 从 checkpoint 拉起至少 1 步，核对模型分片、额外状态、optimizer 策略和权重同步。
7. **确定最终 off-policy 规则。** 在增加 staleness 之前明确 correction、裁剪和可接受版本差，并把它写成训练配置测试。
8. **补充算子级 profiling。** 对长尾步分别分析 vLLM decode、工具等待、数据库查询和 Megatron update，避免只看设备平均利用率。

## 14. 仍需回答的问题

- 1,500 条历史轨迹中有多少任务能够自动生成可靠 verifier？
- 真实大数据下 prefix cache 命中率是否仍能保持约 32%？
- 长轨迹的耗时主要来自 decode token 数、工具轮次、SQLite 查询，还是少数异常任务？
- 6 号机两路 TP8 engine 的最优并发是多少？
- 40k queued tokens 和 6 groups 是否会在更长 response 下频繁背压？
- 允许多大的策略 staleness 才不会显著破坏 GRPO 组内比较？
- 保存 optimizer 后 checkpoint 体积、保存时延和恢复时延是多少？
- 若未来 prompt/response 上限继续增加，CP2 是否足够，还是需要调整 TP/PP/CP？

## 15. 证据和可复现文件索引

| 用途 | 文件 |
| --- | --- |
| 最终 One-Step 配置 | `scripts/run_pi_grpo_megatron_tp4_pp2_cp2.sh` |
| 最终 fully-async 配置 | `scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh` |
| 两机 Ray 启动 | `scripts/start_ray_m05.sh`、`scripts/start_ray_m06.sh` |
| Ray 角色检查 | `scripts/check_ray_roles.py` |
| 2-rank HCCL | `scripts/check_hccl.py` |
| 1→16 HCCL | `scripts/check_hccl_fanout.py` |
| 数据转换 | `scripts/prepare_pi_dataset.py` |
| SQLite 工具 | `llin_verl/pi_sqlite_tool.py` |
| 奖励函数 | `llin_verl/pi_reward.py` |
| NPU 采样 | `scripts/monitor_npu_utilization.py` |
| cache 采样 | `scripts/monitor_vllm_cache_metrics.py` |
| 稳态分析 | `scripts/analyze_grpo_steady_state.py` |
| One-Step 本地结果摘要 | `runs/pi-grpo-megatron-tp4-pp2-cp2-tp8-dp2-20step-20260730-11/steady_state_summary.json` |
| fully-async 本地结果摘要 | `runs/pi-grpo-fully-async-bounded-3step-20260730-02/validation_summary.json` |

`runs/`、原始数据、模型和 checkpoint 均按仓库策略保持忽略，不提交到 Git；报告中的数字来自这些本地证据与已提交的版本记录。
