# 正式 PI 基线与 50-step GRPO 故障复盘（2026-08-03）

## 技术摘要

本文记录从“完整 PI Agent + 正式 200 条数据”冻结模型基线，到首轮 `4→4`、50-step 全参数 GRPO 的七次关键运行。它回答四个问题：当时使用了什么配置、发生了什么、根因是什么、如何修复并验证。

截至本文证据快照：

- 冻结模型基线经历 `-01` 至 `-04` 四次启动，最终 `-04` 完成 `200/200`，退出码为 `0`。
- 正式训练 `-01` 因 6 号机缺少 Parquet 在 step 0 退出；`-02` 在首批 rollout 和 actor 前反向完成后，因 CANN 锁页主机内存申请失败退出。
- 正式训练 `-03` 应用双机数据门禁和优化器驻留修复后，已连续完成 `4/50` 个真实全参数更新；截至 `2026-08-03 13:22 UTC` 未再次出现 pinned-host OOM。它仍在运行，因此本文只把它记为“修复验证已越过原失败点”，不提前宣称 50-step 已全部成功。
- 日志中的 Apex、ModelOpt、SciPy、Triton、KV connector 和 `mstx.range_end` 信息并不都代表作业失败。本文单独给出判别表，避免以后看到 `ERROR` 字样就误判根因。

这批问题不是“旧工程流程突然失效”，而是工作负载发生了实质变化：从 4 个短 prompt、简化 SQLite 工具和小规模更新，切换为完整四工具沙箱、48K 上下文、200 条正式任务、冻结全量评测以及持续 50-step fully-async 训练。新路径首次触发了 val-only 优化器生命周期、可选 Fastest-K 配置、双机文件可见性和每步大规模 optimizer master shard 搬运等旧 smoke 没覆盖的分支。

## 1. 文档范围与证据口径

### 1.1 纳入的运行

| 阶段 | 运行名 | 结果 | 首个决定性问题 |
| --- | --- | --- | --- |
| 冻结基线 | `llin-pi-formal-frozen-baseline-20260803-01` | 退出码 `1` | val-only 仍创建并卸载完整 Adam |
| 冻结基线 | `llin-pi-formal-frozen-baseline-20260803-02` | 退出码 `1` | Fastest-K 补丁无条件访问 `async_training` |
| 冻结基线 | `llin-pi-formal-frozen-baseline-20260803-03` | 人工停止，无 `exit_code` | 同源基础权重仍做冗余 1→16 首次广播 |
| 冻结基线 | `llin-pi-formal-frozen-baseline-20260803-04` | 退出码 `0` | 成功，但有全量 barrier 长尾和 DP 后段失衡 |
| 正式训练 | `llin-pi-formal-grpo-4of4-50step-20260803-01` | 退出码 `1` | rollout 节点缺少正式 Parquet |
| 正式训练 | `llin-pi-formal-grpo-4of4-50step-20260803-02` | 退出码 `1` | FP32 optimizer master shard 二次卸载触发 pinned-host OOM |
| 正式训练 | `llin-pi-formal-grpo-4of4-50step-20260803-03` | 运行中 | 修复后已越过原失败点，快照为 `4/50` |

### 1.2 证据来源

- 服务器运行目录：`/workspace/llin-verl-grpo/runs/<RUN_NAME>/driver.log`、`exit_code`、`started_at`、`finished_at`。
- 冻结基线入口：`scripts/run_pi_frozen_baseline.sh`。
- 正式训练入口：`scripts/run_pi_formal_50step.sh`。
- 双机数据门禁：`scripts/check_formal_data_on_ray.py`。
- 公共训练拓扑：`scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh`。
- 版本修复提交：`647de60`、`b915ee9`、`c349155`、`ecb8030`、`d039110`、`ff20e3e`、`b254136`。

日志、Parquet、原始轨迹、模型和 checkpoint 不进入 Git；本文只记录足以复现和理解问题的配置、错误摘要与证据定位。

## 2. 共同环境、数据与拓扑

### 2.1 硬件与角色

| 组件 | 配置 |
| --- | --- |
| 5 号机 | 16 × Ascend 910，64 GiB/卡；负责全参数训练 |
| 6 号机 | 16 × Ascend 910，64 GiB/卡；负责 rollout |
| 训练并行 | Megatron `TP=4 × PP=2 × CP=2 × DP=1` |
| rollout 并行 | vLLM Ascend `TP=8 × DP=2` |
| 调度 | 两节点 Ray；资源标签 `llin_trainer` / `llin_rollout` 固定角色 |
| 权重同步 | veRL checkpoint engine，配置名为 `nccl`，Ascend 实际使用 HCCL |
| 通信 | `eno0` / `192.168.202.0/24`；1→16 fan-out 已单独验证 |
| 容器 | 两个仅属于本项目的 `llin-verl-*` 特权容器 |

### 2.2 模型与 Agent

- 模型：Qwen3.6-27B，LoRA 关闭，正式训练为全参数更新。
- 上下文：`49,152` tokens，其中初始 prompt 上限 `4,096`，多轮 response 上限 `45,056`。
- Agent：保留老板源 system prompt，使用 `bash/read/write/edit` 四工具。
- 多轮上限：`26` 个 assistant turns、`25` 个工具反馈批次；第 25 次工具调用后仍允许生成最终答案。
- 单轮并行工具调用上限：`4`；单次工具返回上限：`32,768` 字符。
- Continuous Token、prefix cache、8K chunked prefill 均开启。

### 2.3 正式数据和奖励

- 数据版本：`formal_pi_v2_20260803`。
- train / val / test：`160 / 20 / 20`；test 在本轮保持封存。
- reward：证据约束的 V2 奖励；最终答案正确性为主，agent SQL 会在隐藏环境中重新执行并与 gold SQL 结果比较，非法协议或不安全行为硬置零。
- 训练采样：精确 `4→4`，不启用 Fastest-K 过量候选。
- 每步：消费 4 个 prompt group，每个 group 4 条轨迹，共 16 条训练轨迹。
- 队列：预热 8 groups，最大 8 groups / `1,572,864` queued tokens，staleness 上限为 1 个 policy version。

### 2.4 正式 50-step 的关键训练参数

| 参数 | 值 |
| --- | --- |
| `total_training_steps` | `50` |
| 学习率 | `1e-7`，constant |
| actor micro-batch | 每卡 `1` |
| 参数驻留 | `param_offload=False`，参数常驻 NPU |
| 激活 | full recompute / gradient checkpointing |
| MindSpeed optimizer | `optimizer_cpu_offload=True` |
| 正式 `-01/-02` 的 veRL engine optimizer | `megatron.optimizer_offload=True` |
| 正式 `-03` 的 veRL engine optimizer | `megatron.optimizer_offload=False` |
| validation | 每 10 step，val 20 条，贪心 `n=1` |
| checkpoint | 每 10 step，仅 `model,extra`，只保留 1 份 |

这里存在两个名字相似但作用不同的卸载开关：

1. `override_optimizer_config.optimizer_cpu_offload=True`：MindSpeed CPU Adam，决定 Adam 状态和更新逻辑位于 CPU；正式训练继续保留。
2. `actor.megatron.optimizer_offload`：veRL engine 在一次训练上下文退出时，是否再次把 optimizer 相关张量搬回 CPU；正式 `-03` 将其关闭，避免 FP32 master shard 每步经过 CANN pinned-host allocator 往返搬运。

## 3. 故障一：冻结基线 `-01` 在 val-only 中误建完整优化器

### 3.1 出现时的配置

- 目标是冻结模型评测：`trainer.val_only=True`、`val_before_train=True`、`n=1` 贪心解码。
- 数据为 200 条统一评测集，完整 PI 四工具，48K 上下文。
- 训练拓扑仍沿用 Megatron TP4/PP2/CP2 的普通 actor 初始化路径。
- 当时只设置了 val-only，没有显式 `actor.megatron.forward_only=True`；公共配置仍包含完整 Adam、optimizer/gradient offload。

### 3.2 症状与错误

作业在 step 0、尚未生成任何评测轨迹时退出。决定性日志为：

```text
ray::WorkerDict.actor_init_model()
torch.OutOfMemoryError: allocate_host_memory
CachingHostAllocator.cpp:230
aclrtMallocHostWithCfg, error code is 207001
rtsMallocHost execution failed, reason=driver error:out of memory
```

### 3.3 根因

veRL 的 `val_only` 控制训练流程，但没有自动把 Megatron actor 变成纯前向模块。actor 初始化仍创建完整 Adam 状态，并尝试通过 Ascend 锁页主机内存完成优化器卸载。冻结评测根本不需要梯度或优化器，因此这是生命周期配置错误，不是“评测 batch 太大”。

### 3.4 修复

冻结基线入口增加：

```text
actor_rollout_ref.actor.megatron.forward_only=True
actor_rollout_ref.actor.megatron.optimizer_offload=False
actor_rollout_ref.actor.megatron.grad_offload=False
trainer.save_freq=-1
```

这样评测只初始化模型和 rollout，不创建、卸载或保存 Adam/梯度状态。正式训练入口不使用 `forward_only`，因此不受影响。

### 3.5 验证与复现提示

- 验证：`-02` 已越过 actor 初始化、完成两套 TP8 vLLM 和四工具加载，说明本问题被解除。
- 安全复现：只需对比冻结入口是否缺少 `forward_only=True`，不建议为了复现再次运行完整 200-task 并耗尽 pinned memory。
- 防回归：`tests/test_frozen_baseline_contract.py` 固定检查 forward-only、offload 关闭和禁止 checkpoint。

## 4. 故障二：冻结基线 `-02` 的 Fastest-K 补丁错误假定存在 fully-async 配置

### 4.1 出现时的配置

- 已应用 `forward_only=True`，optimizer/gradient offload 关闭。
- 使用标准 One-Step / val-only 路径，不是 bounded fully-async trainer。
- Fastest-K 是历史上给 fully-async rollouter 增加的可选能力，本次冻结评测并未启用它。

### 4.2 症状与错误

模型、TP8×DP2 vLLM 和完整四工具已经初始化，但在 `0/200` 开始生成时退出：

```text
ray::AgentLoopWorker.generate_sequences()
omegaconf.errors.ConfigAttributeError:
Key 'async_training' is not in struct
```

### 4.3 根因

历史 Fastest-K 补丁直接访问 `config.async_training`。标准 One-Step/val-only 配置根本没有该节点；“功能未启用”和“字段不存在”没有被区分。与此同时，容器可能已应用旧 marker，旧补丁不能自动升级到兼容实现。

### 4.4 修复

- 只有配置存在 `async_training` 且显式启用 Fastest-K 时才读取其参数。
- 标准 One-Step 和 val-only 缺少该字段时自动视为 Fastest-K 关闭。
- 增加前向幂等：旧 marker 能识别后续版本，不需要重建镜像即可升级补丁。

### 4.5 验证与复现提示

- 验证：`-03` 已通过 AgentLoop 创建和模型加载，不再出现 `ConfigAttributeError`。
- 最小复现条件：运行不含 `async_training` 节点的标准 One-Step 配置，同时应用旧版 Fastest-K AgentLoop 补丁。
- 防回归：补丁测试覆盖“字段缺失时关闭”和旧 marker 连续升级。

## 5. 故障三：冻结基线 `-03` 在同源模型上进行冗余首次 1→16 权重广播

### 5.1 出现时的配置

- `val_only=True`、`forward_only=True`、`resume_mode=disable`。
- actor 和两套 rollout 引擎都直接从相同的只读 `MODEL_PATH` 加载同一份 Qwen3.6-27B 基础权重。
- 仍沿用训练模式的“actor 初始化后，把全量权重同步到 rollout”逻辑。

### 5.2 症状

- 16 个训练 rank、两套 TP8 vLLM 和 `944/944` 权重转换均已完成。
- 评测一直停在 `0/200`。
- 首次 1→16 actor-to-rollout 广播超过 60 分钟仍未完成。
- 该运行被人工停止，因此没有 `exit_code`；不能写成框架异常退出。

### 5.3 根因

冻结基线不存在训练后的新权重：actor 与 rollout 已从同一路径加载完全相同的基础模型。再次执行 Megatron→HF 转换和跨机全量广播没有语义价值，却进入了最昂贵的 checkpoint-engine 路径。

这与正式训练的权重同步不同。正式训练每步 actor 权重变化，必须同步；冻结评测首次同步则是可证明的冗余操作。

### 5.4 修复

增加严格受限的跳过条件：

```text
val_only=True
resume_mode=disable
actor model path == rollout model path
```

只有三项同时满足时才跳过首次 actor-to-rollout 权重同步。训练、checkpoint 恢复、模型路径不同或非 val-only 运行继续保留原同步。

### 5.5 验证与复现提示

- 验证：`-04` 成功进入 rollout，并完成 200/200。
- 最小复现条件：同源模型的 val-only 运行不启用上述跳过逻辑，观察首次同步阶段而非生成阶段。
- 风险边界：绝不能把该优化无条件应用到训练，否则 rollout 会使用旧 policy。

## 6. 现象四：冻结基线 `-04` 成功但存在全量 barrier 长尾和 DP 失衡

### 6.1 配置与结果

- 200 条任务一次性贪心评测，`n=1`。
- vLLM `TP8×DP2`，两套推理副本。
- 结果：`200/200`，退出码 `0`，总时长 `2h 29m 38s`，verifier 异常 `0`。

### 6.2 观察到的低效现象

运行后段，一个 TP8 副本已经排空，另一个副本仍持有较多长轨迹。框架需要等待整个 200-task validation batch 完成后才统一写出结果，因此：

- 空闲副本不能接管另一副本已分配的剩余长任务。
- 已完成的部分结果在 barrier 前不可见。
- 最后少量长轨迹决定整批完成时间。

### 6.3 根因与处理

这是静态分派加批次 barrier 的尾部效应，不是模型、reward 或工具正确性错误。冻结基线的目标是建立一次性能力基准，`-04` 结果仍然有效；但正式训练不应复制这种“200 条全部结束才继续”的调度方式。

正式训练改用 bounded fully-async：以完整 GRPO group 为单位入队、按 queued tokens 背压、训练端只需取够 4 个完整 group 即可更新。后续仍需逐轨迹 deadline 和动态补充机制进一步削弱长尾。

## 7. 故障五：正式训练 `-01` 只有训练节点拥有 Parquet

### 7.1 出现时的配置

- 正式 50-step、完整 PI、48K、精确 `4→4`。
- 5 号机训练 TP4/PP2/CP2，6 号机 rollout TP8/DP2。
- train/val 路径在两个容器内都是 `/workspace/llin-verl-grpo/data/formal_pi_v2_20260803/...`。
- 正式 Parquet 当时只部署在 5 号机。

### 7.2 症状与错误

本地启动前检查只验证了发起命令的容器，因此通过；远端 `FullyAsyncRollouter` 在 6 号机创建时直接读取 train 数据并失败：

```text
ray::FullyAsyncRollouter.__init__()
FileNotFoundError:
Unable to find '/workspace/llin-verl-grpo/data/formal_pi_v2_20260803/pi_formal_train.parquet'
```

运行在 step 0 退出，没有 rollout、奖励计算或参数更新。

### 7.3 根因

相同的容器内路径不等于共享文件系统。两台服务器分别挂载自己的 `/data3/llin`；fully-async rollouter 不是只接收训练机传来的 token batch，它会在 rollout 节点读取 prompt 数据。因此单节点 `test -f` 无法证明跨节点可见性。

### 7.4 修复

1. 将已经审计的 train/val Parquet 直接同步到 6 号机对应的 `llin` 目录。
2. 在模型初始化前执行 `scripts/check_formal_data_on_ray.py`。
3. 该门禁分别调度到 `llin_trainer` 和 `llin_rollout` 节点，核对：
   - 文件存在；
   - 字节数一致；
   - SHA256 一致。

当前固定摘要：

```text
train: 0f22b2c6d3385f3aff201eefd402e18da2c8a3be7d7b1a331b86a307cf6bac25
val:   f06b159548326f78567eae73cc860e1109db779946bc242aa41c7bcf800485b8
```

### 7.5 验证与复现提示

- 验证：正式 `-02` 和 `-03` 的双角色数据门禁均通过，rollouter 成功读取 prompt 并生成轨迹。
- 最小复现条件：删除或改名 rollout 节点上的 Parquet，门禁现在应在模型加载前直接失败，而不是等 Ray actor 创建后才失败。
- 防回归：`tests/test_formal_training_contract.py` 检查正式入口必须调用双机门禁。

## 8. 故障六：正式训练 `-02` 在更新退出阶段触发 CANN pinned-host OOM

### 8.1 出现时的配置

- 正式配置完整生效：TP4/PP2/CP2 训练、TP8/DP2 rollout、48K、4→4、CPU Adam。
- `param_offload=False`。
- MindSpeed：`optimizer_cpu_offload=True`、`optimizer_offload_fraction=1`。
- veRL engine：`actor.megatron.optimizer_offload=True`、`grad_offload=True`。
- 主机当时仍约有 `1.9 TiB available`；48K 容量门禁的 actor reserved 峰值为 `37.78 GiB/卡`，每卡可用约 `61.27 GiB`。

### 8.2 已经完成到哪个阶段

这次不是初始化失败。它已经完成：

1. 双机数据门禁；
2. 完整 PI rollout；
3. 奖励计算；
4. actor 前向和反向；
5. 优化器更新的主体。

错误发生在 `train_mini_batch` 上下文退出、engine 从 NPU 切回 CPU 的阶段。没有形成可确认的 durable `global_step_1`/checkpoint，因此不能把它计为已完成训练步。

### 8.3 决定性调用栈

```text
train_mini_batch
  -> transformer_impl.__exit__
  -> _context_switch("cpu")
  -> engine.to(... optimizer=...)
  -> offload_megatron_optimizer
  -> offload_megatron_copy_params
  -> shard_fp32_from_float16_groups
  -> tensor.data.to("cpu", non_blocking=True)
  -> CachingHostAllocator.cpp:230
  -> aclrtMallocHostWithCfg, error code 207001
```

### 8.4 为什么不是普通“内存不够”

- 报错来自 `torch_npu` 的 `CachingHostAllocator` 和 `aclrtMallocHostWithCfg`，申请的是 CANN 锁页 host memory，不是普通 pageable RAM。
- 5 号机仍有约 1.9 TiB 普通内存可用，排除整机 RAM 耗尽。
- actor 前反向已经完成，排除训练激活直接导致的 NPU OOM。
- 栈明确指向 FP32 master shard 的 `non_blocking=True` D2H 搬运。

### 8.5 根因

两层卸载叠加：MindSpeed CPU Adam 已经把优化器状态放在 CPU；veRL engine 的 `megatron.optimizer_offload=True` 又在每次训练上下文退出时，把 `shard_fp32_from_float16_groups` 等 master shard 以 non-blocking 方式搬回 CPU。这会申请大量 pinned memory，并在 Ascend CANN allocator 上触发 207001。

### 8.6 修复

正式入口增加覆盖：

```text
actor_rollout_ref.actor.megatron.optimizer_offload=False
```

同时保留：

```text
optimizer_cpu_offload=True
optimizer_offload_fraction=1
```

结果是 Adam 状态和 CPU 优化器逻辑仍在主机内存，但 FP32 master shard 不再由 veRL engine 每步做第二次 pinned D2H/H2D 往返，改为常驻 NPU。

预计每卡多占约 `6–8 GiB` 常驻 HBM。已有 48K 容量门禁还剩约 `23.49 GiB` reserved 余量，因此该交换在当前 64 GiB NPU 上可接受。

### 8.7 验证与复现提示

- 正式 `-03` 已连续完成 4 个全参数更新，越过 `-02` 的首步退出点，未再次出现 207001。
- 最小复现条件：在同一正式负载下恢复 `actor.megatron.optimizer_offload=True`，同时保持 CPU Adam；预计在训练上下文退出时复现。该操作代价高，不建议在正式机器上主动复现。
- 防回归：`tests/test_formal_training_contract.py` 固定检查正式入口必须覆盖为 `False`。
- 尚待验证：50-step 全程、每 10-step validation 和 checkpoint 保存是否全部稳定，必须等 `-03` 自然结束后再更新最终结论。

## 9. 正式训练 `-03`：当前修复验证状态

### 9.1 相比 `-02` 的唯一关键配置差异

```text
# 保留
optimizer_cpu_offload=True

# 修改
actor_rollout_ref.actor.megatron.optimizer_offload=False
```

数据、奖励、Agent、并行拓扑、48K、多轮上限、4→4、学习率、validation 和 checkpoint 频率保持不变，因此它能够作为 optimizer engine offload 修复的直接验证。

### 9.2 证据快照

截至 `2026-08-03 13:22 UTC`：

```text
global_steps: 4
Training Progress: 4/50
queue_wait_s=114.230161
update_actor_s=136.040174
step_s=258.401232
```

- 4 次实际 actor update 已完成。
- 两个 TP8 rollout EngineCore 均存活。
- 16 张 rollout NPU 均有活动。
- 未出现 `allocate_host_memory`、207001 或 NPU OOM。
- 首次 validation 在 step 10，当前尚未发生；因此 validation/checkpoint 路径仍是未验证项。

### 9.3 当前可下的结论

可以确认 `megatron.optimizer_offload=False` 已越过 `-02` 的决定性故障点；不能确认整个 50-step、五次 validation 或末次 checkpoint 已成功。后续文档应在运行结束后补充退出码、最终步数、validation 指标和 checkpoint 完整性。

## 10. 非致命告警与真正故障的判别

| 日志内容 | 当前判定 | 判定依据 | 处理建议 |
| --- | --- | --- | --- |
| `MixedFusedLayerNorm ... AttributeError` | 非致命可选导入失败 | 成功的 baseline `-04` 和正式 `-03` 同样出现 | 保留观察；不要当作退出根因 |
| ModelOpt `already registered` | 非致命可选插件警告 | 日志明确写 `You may ignore`，主路径继续 | 不升级依赖，除非后续需要对应插件 |
| Triton 3.2 低于建议 3.3 | 非致命版本警告 | 现有 NPU路径已完成多次训练 | 记录技术债，不在正式运行中热升级 |
| SciPy 要求 NumPy `<2.3`，实际 `2.4.6` | 当前非致命 | vLLM/评测继续运行 | 若使用 SciPy/sklearn 数值功能再固定版本 |
| `reset_connector called but no KV connector` | 非致命 | 未配置 KV connector 的提示，推理继续 | 不需处理，除非启用 KV connector |
| `mstx.range_end() missing range_id`，级别为 `[ERROR]` | 当前是 profiling/instrumentation 缺陷 | `-03` 在持续出现时仍从 step 3 进到 step 4 | 与训练异常分开统计；后续可降噪或修 profiler wrapper |
| `tp_group is None` deprecation | 非致命弃用告警 | 当前版本仍回退到默认 TP group | 升级 Megatron 前修接口 |
| `FileNotFoundError ... pi_formal_train.parquet` | 致命 | Ray rollouter 创建失败，退出码 1 | 双节点存在性/大小/SHA 门禁 |
| `ConfigAttributeError: async_training` | 致命 | AgentLoopWorker 生成前退出 | 可选配置安全读取 |
| `CachingHostAllocator ... 207001` | 致命 | actor 初始化或更新退出，退出码 1 | 先根据调用栈区分“不该建 optimizer”与“二次 pinned offload” |

## 11. 为什么真实数据/奖励切换会暴露这些问题

### 11.1 smoke 已跑通的是工程骨架，不是所有运行分支

早期 smoke 证明了 Ray、HCCL、FSDP/Megatron、vLLM、工具循环和一次 GRPO update 可以工作，但没有覆盖：

- val-only actor 生命周期；
- 200-task 全量评测 barrier；
- 标准 One-Step 在 Fastest-K 补丁存在时的字段缺失；
- 两台独立磁盘上的正式 Parquet 部署；
- 完整 PI 长轨迹下每步 optimizer master shard 的反复 pinned 搬运；
- 每 10 step validation/checkpoint 的长期组合。

### 11.2 数据和奖励本身没有直接造成已知框架异常

截至当前证据：

- 正式数据通过 160/20/20 隔离、gold SQL 重执行和双机 SHA 门禁。
- baseline 200/200 的 verifier 异常为 0。
- 正式 `-02` 已完成 rollout、奖励和前反向，失败栈位于 optimizer context switch。
- 正式 `-03` 已完成 4 次更新。

因此已发生的致命问题主要属于生命周期、配置兼容、部署和内存传输路径，而不是“奖励函数把训练算坏了”或“数据文件内容无法解析”。

## 12. 以后复现和排障的固定顺序

### 12.1 启动前

1. 确认运行类型：冻结评测、One-Step 训练还是 fully-async 训练。
2. 冻结评测必须同时满足 `val_only + forward_only + no optimizer/grad offload + no checkpoint`。
3. 正式训练必须执行双角色 Ray 数据门禁，不只在 driver 容器中 `test -f`。
4. 记录最终 Hydra overrides，注意同一个 key 可能在公共脚本和正式入口各出现一次，最后一个值才生效。
5. 记录两层 optimizer offload 的最终值，不能只写“优化器卸载已开/已关”。

### 12.2 初始化阶段

按以下门槛逐层判断：

1. 16 个 Megatron worker 创建；
2. Qwen3.6 `944/944` 权重加载；
3. 两个 `EngineCore_DP0/DP1` 创建；
4. vLLM 两套 TP8 分片加载；
5. 首次权重同步；
6. 首个完整 group 入队。

停在哪一层，就只检查该层之前最近的决定性异常，不要被更早的可选导入 warning 干扰。

### 12.3 训练阶段

一个“完成的 step”至少需要同时看到：

- `global_steps` 增长；
- `[LLIN_TRAIN_STAGE]` 中 `update_actor_s` 和 `step_s`；
- 训练进度条增加；
- 无随后抛出的同一步异常。

只完成 rollout、reward、backward 或 optimizer 主体但在 context exit 失败，不能记为 durable completed step。

### 12.4 看到 OOM 时

先按 allocator 和调用栈分类：

- NPU HBM：通常包含 device allocation 和卡上显存信息；检查激活、KV cache、bucket、并发。
- 普通 CPU RAM：检查系统 available、RSS、OOM killer。
- CANN pinned host：包含 `CachingHostAllocator`、`aclrtMallocHostWithCfg`、207001；检查 non-blocking D2H/H2D 和重复 offload。

不要因为错误文本都写 `out of memory` 就使用同一个修复。

### 12.5 运行结束后

1. 检查 `exit_code`；人工停止的运行必须明确标成“无 exit code”。
2. 检查最终 `global_steps`。
3. 核对 validation 文件数与每次 20 条完整性。
4. 检查 checkpoint 索引引用的所有分片实际存在。
5. 记录 trainer/rollout NPU 是否释放。
6. 只有以上均通过，才能把“运行中”升级为“50-step 成功”。

## 13. 已固化的防回归措施

| 风险 | 固化措施 |
| --- | --- |
| val-only 创建 Adam | 冻结入口显式 `forward_only`；契约测试 |
| Fastest-K 污染标准配置 | 字段缺失时关闭；前向幂等升级测试 |
| 同源冻结模型冗余同步 | 三条件受限 skip；训练路径不跳过 |
| 双机数据不一致 | trainer/rollout Ray 角色分别计算大小和 SHA256 |
| 二次 pinned optimizer offload | 正式入口覆盖 engine `optimizer_offload=False` |
| test 泄漏 | 正式入口只传 train/val；test 保持封存 |
| 巨型 checkpoint 占盘 | 只保存 `model,extra`，最多保留一份 |
| 把 warning 当 fatal | 本文的告警分类表和退出码/阶段判据 |

## 14. 限制与待补证据

- 正式 `-03` 在本文写作时仍运行；最终退出码、step 10/20/30/40/50 validation、checkpoint 完整性和最终显存释放尚未获得。
- `mstx.range_end` 虽已证明不会阻止前 4 步，但它仍属于应修复的观测噪声；不能永久忽略所有相同 `[ERROR]`。
- `-03` 的前 4 步证明 optimizer 修复有效，不等于证明奖励改善或模型收敛；质量判断必须比较冻结基线和后续 validation。
- 主机 `1.9 TiB available` 与每卡 HBM 余量来自故障现场和既有 48K 门禁，不应外推到更大模型、更高并发或取消激活重计算的配置。

## 15. 下一步

1. 让正式 `-03` 自然运行，不因本文写作重启 Ray 或容器。
2. 到 step 10 后首先审计 20 条 val：严格最终答案正确、SQL 证据、reward、工具协议、安全率和 verifier 异常。
3. 同时验证第一个滚动 checkpoint 的索引与分片完整性。
4. 50-step 结束后在本文追加最终运行结论；若中途再失败，沿用同一模板增加“配置—症状—根因—修复—验证”条目，而不是覆盖历史证据。
5. 完成质量判断前不恢复 `6→最快4`，避免吞吐选择偏差干扰正式奖励曲线。
