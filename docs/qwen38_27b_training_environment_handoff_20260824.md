# Qwen3.8-27B Ascend veRL/GRPO 训练环境交接手册

**版本：** 1.0

**更新日期：** 2026-08-24

**适用范围：** Qwen3.8-27B、Ascend 910、veRL fully-async、Megatron 全参训练、vLLM 多轮工具轨迹

**目标读者：** 接管本项目训练、换数据、换卡数或迁移到同类机器的工程人员

## 1. 先读结论

这套工程已经证明：Qwen3.8-27B 能在 5 号机 16 卡训练、6 号机 16 卡推理的组合下，完成模型加载、训练到推理权重同步、多轮工具轨迹生成和 1 个 optimizer step 的端到端闭环。

但截至 2026-08-24，当前 `approved43 + tiered-query-cost-v1 + strict-mixed` 新训练路线还没有通过“5 个实际 optimizer step”金丝雀，**不能把正式 43 题全量训练写成已经跑通**。最近一次完成轨迹的金丝雀为 `PASS=0 / FAIL=14 / UNKNOWN=82`，随后已修复数据库、请求身份、工具 token 计数与表格解析接线；再后的运行分别暴露了 NPU 占用和 pinned host memory 初始化问题。当前规定仍是：先从原始 Qwen3.8 基座重跑金丝雀，审核通过后才允许全量。

接手人应把以下两句话同时记住：

1. **底层 Qwen3.8 双机训练工程可运行。**
2. **当前安全奖励配方尚未取得五步有效训练信号，正式全量仍锁定。**

本手册不使用图表：路径、参数、状态与故障是一组精确查表问题，表格和命令比架构图更不容易隐藏边界条件。

## 2. 当前推荐主线与废弃路线

### 2.1 当前唯一推荐主线

- actor/ref：都从原始 `/models/Qwen3.8-27B` 初始化。
- 训练集：冻结 `approved43`，不能扫描原始 100 题自动扩充。
- 奖励：`llin_verl/pi_reward.py::compute_score_tiered_query_cost_v1`。
- 判定：`PASS / FAIL / UNKNOWN` 三态；UNKNOWN 整组 mask。
- 组门：每题 8 条轨迹，只有同时存在明确 PASS 与明确 FAIL 的完整组才更新。
- uniform 全对、uniform 全错、全 UNKNOWN、不完整或陈旧组：清空 advantage、return、response mask，并跳过 optimizer。
- actor 约束：固定原始 ref，`KL loss=0.001`、`low_var_kl`；KL 不混入 reward。
- 异步陈旧度：`staleness=0`。
- 放行顺序：CPU/资产门禁 → 最多 5 个实际 optimizer step 金丝雀 → 独立自动审计与主任务审核聚合结果 → 正式训练。

### 2.2 禁止作为新实验默认入口的历史路线

以下脚本保留作审计和复现，不应交给新使用者直接启动：

- `scripts/launch_qwen38_train70_host.sh`
- `scripts/run_pi_qwen38_train70_2x_banded_v2.sh`
- `scripts/launch_qwen38_step70_mixed27_host.sh`
- `scripts/run_pi_qwen38_step70_mixed27_4x_banded_v2.sh`
- `scripts/run_qwen38_mixed27_sealed6_eval_host.py`

原因不是这些脚本“跑不起来”，而是旧 `mixed27 × 4 + banded-v2 + 无固定 ref KL` 配方已经实测退化：108 组中 78 组八条全错却仍有奖励方差，近一半更新完全没有正确轨迹，sealed6 从训练前 `6/6` 至少一次正确降到训练后 `1/6`。详见 `docs/qwen38_mixed27_training_collapse_diagnosis_20260821.md`。

## 3. 机器、网络与角色

| 名称 | SSH 别名 | 内网地址 | 已验证角色 | NPU | 当前模型宿主路径 |
|---|---|---|---|---:|---|
| 5 号机 | `huawei-05` | `192.168.202.5` | Ray head、Megatron actor/ref 训练 | 16 | `/data3/llin/base_model/Qwen3.8-27B` |
| 6 号机 | `huawei-06` | `192.168.202.4` | Ray worker、vLLM rollout | 16 | `/data3/llin/base_model/Qwen3.8-27B` |
| 0 号机 | `huawei-00` | `10.10.2.2` | 可选 rollout 扩容/基准，不在当前正式金丝雀中 | 最多 16；使用前实查 | `/data3/models/Qwen3.8-27b` |

注意：0 号机宿主目录最后一个字符是小写 `b`；容器内统一挂为 `/models/Qwen3.8-27B`。不要按旧口述写成宿主 `/data3/models/Qwen3.8-27B`。

网络合同：

- 5/6 号机使用 `eno0` 和 `192.168.202.0/24`。
- 0 号机使用 `enp196s0f0` 和 `10.10.2.2`。
- 当前双机专用 Ray 地址为 `192.168.202.5:36379`。
- Docker 必须使用 host network；Ray、HCCL 和 vLLM 不能自动选到管理网卡。
- 当前金丝雀只使用 5+6。将 0 号机加入正式训练不是简单把 `ROLLOUT_NNODES=2`，还要增加 0 号机 Ray 启动、资源标签、HCCL 网卡/IP 和数据/runtime 同步门；没有完成这些改动前不得假定三机入口可直接用。

## 4. 已验证软件与容器

5/6 号机金丝雀容器基于同一镜像：

| 项 | 值 |
|---|---|
| 镜像名 | `llin-verl-a3:20260730` |
| 镜像 ID | `sha256:cdc98ccb6b739429aa81f3dc6e4bca9543d1e99202cd7e8686055fb5e9fae7b7` |
| Python | 3.11.15 |
| PyTorch | 2.9.0 |
| torch-npu | 2.9.0.post2 |
| veRL | 0.9.0.dev0 |
| Ray | 2.56.1 |
| PyArrow | 24.0.0 |
| vLLM 包 | 0.18.0+empty（昇腾镜像定制包，不要用普通 CUDA wheel 覆盖） |
| MindSpeed | 0.16.0 |
| Transformers | 5.10.4 |
| Docker | `--privileged --network host --shm-size 64g` |

当前容器名：

- 5 号机：`llin-verl-qwen38-canary-m05-20260822`
- 6 号机：`llin-verl-qwen38-canary-m06-20260822`
- 0 号机基准容器：`llin-verl-qwen38-bench-m00-20260817`

5/6 号机容器必须至少有这些挂载：

| 宿主 | 容器 | 权限/用途 |
|---|---|---|
| `/data3/llin/qwen3.6-27b-verl-grpo` | `/workspace/llin-verl-grpo` | 项目和 run 输出，可写 |
| `/data3/llin/base_model/Qwen3.8-27B` | `/models/Qwen3.8-27B` | 原始模型，只读 |
| `/data/renjunxiang/pi/sandbox` | `/pi_sandbox` | 原始任务环境，只读 |
| `/usr/local/Ascend/driver` | 同路径 | 只读 |
| `/usr/local/Ascend/add-ons` | 同路径 | 只读 |
| `/etc/hccn.conf` | 同路径 | 只读 |
| `/etc/ascend_install.info` | 同路径 | 只读 |
| `/usr/local/dcmi`、`/usr/local/sbin` | 同路径 | 只读 |

不要在旧容器里手工 `pip install -U`。本项目会对 `/verl` 做幂等运行时补丁；镜像、代码 commit、模型和数据必须一起冻结，才能复现。

## 5. 模型位置与完整性门

容器统一模型路径：

```text
/models/Qwen3.8-27B
```

当前冻结值：

| 资产 | 预期值 |
|---|---|
| `config.json` SHA256 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| safetensors 分片数 | 18 |
| 18 分片复合 SHA256 | `e2c3b44e4e198e94fcd74903983fc8997f8e504a21575e397f9d59db1cc2fc8f` |
| 模型总文件清单摘要（初始复制审计） | `83fdde44333639d15e1ba3e8fe363d90629382b38c71eec54fa2a42b9c7d988f` |
| 初始总字节数 | `55,586,115,141` |

在容器内检查：

```bash
sha256sum /models/Qwen3.8-27B/config.json
find /models/Qwen3.8-27B -maxdepth 1 -name 'model-*-of-00018.safetensors' | wc -l
LC_ALL=C sha256sum /models/Qwen3.8-27B/model-*-of-00018.safetensors \
  | sha256sum
```

复合哈希依赖绝对路径文本；不同宿主路径直接计算会得到不同值。请在各容器统一的 `/models/Qwen3.8-27B` 下计算。

**绝对禁止**用 Qwen3.6、Step120、Step70、旧 Qwen3.8 GRPO checkpoint 作为本路线的恢复点。actor 和 ref 都从同一原始 Qwen3.8 权重开始；金丝雀失败后也必须新建 run，从原始基座重来。

## 6. 项目、数据与输出位置

### 6.1 项目根目录

| 位置 | 路径 |
|---|---|
| 5/6 号机宿主 | `/data3/llin/qwen3.6-27b-verl-grpo` |
| 容器 | `/workspace/llin-verl-grpo` |
| Windows 开发副本 | `D:\Files\ShunFengWork\qwen3.6-27b-GRPO-veRL` |
| 所有运行输出 | `<项目根>/runs/<RUN_NAME>` |

### 6.2 当前 approved43 私有训练资产（5 号机）

公共父 run：

```text
/data3/llin/qwen3.6-27b-verl-grpo/runs/
  llin-v15-codex-model2-100-step120-8x-20260821-01
```

| 资产 | 精确路径 | 权限/计数 | SHA256 |
|---|---|---|---|
| 43 题批准 Parquet | `grpo_readiness_audit_20260822-05/private/grpo_approved43.sensitive.parquet` | `0600`，43 行 | `d86b53d906806b150d43a508dce9b0dd6d05105c07e03961e8e7bf9439ccd944` |
| 43 题 manifest | `grpo_readiness_audit_20260822-05/private/grpo_approved43_manifest.sensitive.jsonl` | `0600`，43 行 | `1426bc09a3dbaf4709fd89227790603afb7a2bf11beeba80946057d490e0f424` |
| 冻结 100 题任务索引 | `data/tasks.jsonl` | `0600/0644 依部署而定`，100 行 | 启动器按成员和索引核验 |
| 原始 100 题 Parquet | `data/rollout_100.sensitive.parquet` | 私有，100 行 | `c0befda32166340bf68e6b948a1e8fcc6f8f0887d7a5f38a4e6b1051b8f9f7af` |

完整批准路径为：

```text
/data3/llin/qwen3.6-27b-verl-grpo/runs/
llin-v15-codex-model2-100-step120-8x-20260821-01/
grpo_readiness_audit_20260822-05/private/
grpo_approved43.sensitive.parquet
```

私有文件不得提交 Git，不得复制到公共文档目录。新 run 的 `private/`、`private_recovery/` 和 `audit/` 应分别为 `0700`，敏感文件为 `0600`。

### 6.3 当前金丝雀还依赖的历史只读证据

`scripts/launch_qwen38_tiered_canary5_host.sh` 当前把冻结 96 条观测重放源写为：

```text
/workspace/llin-verl-grpo/runs/
llin-qwen38-approved43-tiered-v1-canary5-20260823-03
```

新机器若没有这个 run，在线观测 CPU 门会失败。迁移时必须复制其 `private/rollouts` 与 `private/canary20.sensitive.parquet`，保持私有权限；或者把启动器中的 `FROZEN96_RUN_CONTAINER` 改成另一个经过同等审计的冻结重放集。不得为了启动方便删除这道门。

## 7. 核心脚本地图

### 7.1 当前金丝雀主线

| 作用 | 脚本 |
|---|---|
| 宿主总控、资产门、双机数据同步、Ray、清理、状态 | `scripts/launch_qwen38_tiered_canary5_host.sh` |
| 容器内 5 实际更新步合同 | `scripts/run_pi_qwen38_approved43_tiered_canary_v1.sh` |
| 20 题交替 numeric/table 金丝雀集 | `scripts/prepare_qwen38_tiered_canary_data.py` |
| 8 题密封评测集 | `scripts/prepare_qwen38_tiered_canary_sealed8.py` |
| 按任务身份复制只需的 SQLite 环境 | `scripts/stage_bound_pi_sandbox.py` |
| 冻结 96 条观测重放 | `scripts/replay_qwen38_tiered_observability.py` |
| 当前在线数据观测门 | `scripts/validate_qwen38_tiered_online_observability.py` |
| 5 号机 Ray head | `scripts/start_ray_qwen38_smoke_m05.sh` |
| 6 号机 Ray worker | `scripts/start_ray_qwen38_smoke_m06.sh` |
| Ray 角色与卡数检查 | `scripts/check_qwen38_smoke_ray_cluster.py` |
| 通用 fully-async Megatron/vLLM 入口 | `scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh` |

### 7.2 奖励和 veRL 运行时补丁

| 作用 | 文件 |
|---|---|
| tiered-v1 奖励入口 | `llin_verl/pi_reward.py` |
| 奖励主体与三态判断 | `llin_verl/tiered_query_cost_reward.py` |
| EvidencePlan/身份绑定合同 | `llin_verl/outcome_gated_contract.py` |
| Fastest-K 过量采样 | `scripts/patch_verl_fastest_k_oversampling.py` |
| UNKNOWN/硬门重采 | `scripts/patch_verl_hard_gate_resampling.py` |
| strict mixed 组门和 skip optimizer | `scripts/patch_verl_grpo_strict_variance_gate.py` |
| 金丝雀 rollout/参数审计 | `scripts/patch_verl_canary_rollout_audit.py` |

### 7.3 正式训练入口

| 作用 | 脚本 |
|---|---|
| 宿主正式总控 | `scripts/launch_qwen38_tiered_formal_host.sh` |
| 43×4 正式容器入口 | `scripts/run_pi_qwen38_approved43_tiered_formal_v1.sh` |
| 43×4 调度生成 | `scripts/prepare_qwen38_approved43_outcome_training.py` |

重要：截至本手册日期，正式宿主入口仍落后于金丝雀的 v1.12.28 数据库/identity/token 观测接线。即使旧文件中出现 `formal_training_allowed=true`，也不能绕过“新金丝雀通过并把同一接线移植到正式入口”的要求。

## 8. 当前冻结训练合同

| 参数 | 金丝雀 | 计划中的正式训练 |
|---|---:|---:|
| 训练题 | approved43 中 20 个唯一题，numeric/table 各10 | 43 个唯一题 |
| 每题曝光 | 1 | 4 |
| 名义组 | 最多20 | 172 |
| 每个名义 batch 的组 | 2 | 2 |
| 每组接受轨迹 | 8 | 8 |
| 每组候选 | 16，Fastest-8 | 16，Fastest-8 |
| 最大名义 batch | 10 | 86 |
| 目标实际 optimizer step | 5，达到即停 | 由 eligible mixed 组决定，最多86个名义 batch |
| actor/ref 初始模型 | 原始 Qwen3.8 | 原始 Qwen3.8 |
| 训练拓扑 | TP4×PP2×CP2，16卡 | 同左 |
| rollout 拓扑 | TP4×DP4，16卡 | 同左 |
| prompt/response/context | 4,096 / 90,112 / 94,208 | 同左 |
| 轨迹超时 | 1,800秒 | 同左 |
| 采样 | `temperature=1, top_p=.95, top_k=20` | 同左 |
| reasoning effort | medium，通过 chat template/system 指令实现 | 同左 |
| rollout 最大序列 | 16/TP4副本 | 同左 |
| vLLM cache 比例 | 0.80 | 同左 |
| batched tokens | 16,384 | 同左 |
| Agent worker | 12 | 同左 |
| 每副本并发 | 6 | 同左 |
| LR | `5e-8 constant` | 同左 |
| entropy | 0 | 同左 |
| 固定 ref KL | `0.001 low_var_kl` | 同左 |
| KL 进 reward | 否 | 否 |
| staleness | 0 | 0 |
| optimizer offload | CPU=false，engine=false | 同左，除非重新做内存金丝雀 |
| checkpoint | 实际 step5 临时 `model,optimizer,extra` | 最终仅 `model,extra` 一份 |

这里的 30 分钟必须从轨迹请求进入系统时开始，覆盖队列等待、vLLM 准入、生成和工具执行；不能在真正开始生成时才计时。

## 9. 第一次接管时的标准启动流程

下面默认在 5 号机操作，并通过内网 SSH 控制 6 号机。

### 9.1 同步一个明确 Git 版本

先确认本地和 5/6 号机项目都指向同一个已审核提交。不要边训练边 `git pull`，也不要直接从会变化的工作区运行。

```bash
cd /data3/llin/qwen3.6-27b-verl-grpo
git status --short
git rev-parse HEAD
```

若工作区有未提交改动，先查清归属；不要用 `git reset --hard` 清理别人的工作。

### 9.2 为本次 run 冻结 runtime

为每个新 run 使用唯一名称：

```bash
export PROJECT_ROOT=/data3/llin/qwen3.6-27b-verl-grpo
export RUN_NAME=llin-qwen38-approved43-tiered-v1-canary5-$(date +%Y%m%d-%H%M%S)
export RUN_HOST="$PROJECT_ROOT/runs/$RUN_NAME"
mkdir -p "$RUN_HOST/runtime" "$RUN_HOST/audit" "$RUN_HOST/private"
chmod 700 "$RUN_HOST" "$RUN_HOST/runtime" "$RUN_HOST/audit" "$RUN_HOST/private"

git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_HOST/audit/runtime_git_commit.safe.txt"
git -C "$PROJECT_ROOT" archive HEAD | tar -x -C "$RUN_HOST/runtime"
tar -C "$RUN_HOST/runtime" -cf "$RUN_HOST/private/runtime.tar" .
sha256sum "$RUN_HOST/private/runtime.tar" > "$RUN_HOST/audit/runtime_tar.sha256"
chmod 600 "$RUN_HOST/private/runtime.tar" "$RUN_HOST/audit/runtime_tar.sha256"
```

同步同一个 runtime 到 6 号机相同宿主路径：

```bash
ssh root@192.168.202.4 \
  "mkdir -p '$RUN_HOST/runtime' '$RUN_HOST/audit' '$RUN_HOST/private' && chmod 700 '$RUN_HOST' '$RUN_HOST/runtime' '$RUN_HOST/audit' '$RUN_HOST/private'"
scp -p "$RUN_HOST/private/runtime.tar" root@192.168.202.4:"$RUN_HOST/private/runtime.tar"
ssh root@192.168.202.4 \
  "tar -C '$RUN_HOST/runtime' -xf '$RUN_HOST/private/runtime.tar' && sha256sum '$RUN_HOST/private/runtime.tar'"
sha256sum "$RUN_HOST/private/runtime.tar"
```

两端 tar SHA256 必须相同。`runtime/` 不需要 `.git`；commit 已单独记入审计文件。

### 9.3 不占卡预检

```bash
ssh root@192.168.202.4 true
docker inspect llin-verl-qwen38-canary-m05-20260822 >/dev/null
ssh root@192.168.202.4 \
  'docker inspect llin-verl-qwen38-canary-m06-20260822 >/dev/null'

npu-smi info
ssh root@192.168.202.4 npu-smi info
ss -lntp | grep ':36379' || true
ssh root@192.168.202.4 "ss -lntp | grep ':36379' || true"
```

要求：两机无用户 NPU 进程、36379 无旧 Ray、容器存在、SSH 无交互提示。

随后核对模型和数据哈希。宿主 Parquet/manifest 用 `sha256sum`；模型分片必须在两个容器的统一路径内计算。任何不一致都停止，不允许“先跑起来再说”。

### 9.4 启动容器

当前金丝雀容器可能处于 `Exited (137)`。确认没有别人的任务且资源空闲后，在各自主机启动现有容器：

```bash
docker start llin-verl-qwen38-canary-m05-20260822
ssh root@192.168.202.4 \
  'docker start llin-verl-qwen38-canary-m06-20260822'
```

若需重建容器，必须复用第 4 节的镜像、host network、privileged、64 GiB shm 和全部挂载；不要只写一条不完整的 `docker run` 命令交接。

### 9.5 后台启动金丝雀

从冻结 runtime 的宿主脚本启动，而不是从可变项目目录启动：

```bash
nohup env \
  RUN_NAME="$RUN_NAME" \
  HOST_PROJECT_ROOT="$PROJECT_ROOT" \
  CONTAINER_PROJECT_ROOT=/workspace/llin-verl-grpo \
  TRAINER_CONTAINER=llin-verl-qwen38-canary-m05-20260822 \
  ROLLOUT_CONTAINER=llin-verl-qwen38-canary-m06-20260822 \
  ROLLOUT_HOST=192.168.202.4 \
  RAY_ADDRESS=192.168.202.5:36379 \
  bash "$RUN_HOST/runtime/scripts/launch_qwen38_tiered_canary5_host.sh" \
  > "$RUN_HOST/supervisor.log" 2>&1 &

echo $! > "$RUN_HOST/launcher.pid"
```

宿主总控会依次执行：

1. 双机空闲、容器、模型和私有数据哈希门。
2. 容器内相关 CPU 回归。
3. 备份 `/verl` 即将被补丁修改的文件。
4. 生成 canary20 和 sealed8。
5. 以 `0600` 把两份数据同步到 6 号机并复核 SHA256。
6. 只复制批准任务与 sealed 任务绑定的 SQLite 环境到 run 私有 sandbox。
7. 执行冻结96条重放和两机在线观测门。
8. 启动隔离 Ray head/worker。
9. 启动最多 5 个实际更新步的训练。
10. 退出时清理 Ray，写入状态、退出码和审计文件。

### 9.6 监控

```bash
watch -n 10 "cat '$RUN_HOST/state' 2>/dev/null; tail -n 30 '$RUN_HOST/training.log' 2>/dev/null"
```

常用文件：

| 文件 | 含义 |
|---|---|
| `state` | 当前阶段或最终状态 |
| `supervisor.log` | 宿主总控输出 |
| `training.log` | veRL/Megatron/vLLM 主日志 |
| `ray_m05.log`、`ray_m06.log` | Ray 两节点启动日志 |
| `audit/container_cpu_gate.log` | 容器 CPU 门禁 |
| `audit/*observability*` | identity/database/token 可观测性 |
| `audit/ray_status_before.safe.txt` | 物理资源和角色 |
| `private/rollouts/` | 敏感轨迹，只留服务器 |
| `private/parameter_audit/` | 参数/更新审计 |
| `private_recovery/checkpoints/` | 金丝雀临时恢复点 |
| `exit_code`、`finished_at` | 真实退出状态与结束时间 |

不要只看 `nohup` 进程还在不在，也不要把 shell 返回 0 当训练成功。至少同时核对：`state`、`exit_code`、实际 optimizer step、checkpoint 结构、actor 参数变化、Ray 清理和 NPU 释放。

## 10. 金丝雀放行标准

只有以下条件全部满足，才允许准备正式训练：

- 达到恰好 5 个**实际** optimizer step，而不是 5 个名义 batch。
- 至少出现可信的 PASS/FAIL strict-mixed 组；不能靠把 UNKNOWN 改成 FAIL 凑方差。
- UNKNOWN、uniform 与陈旧组的 advantage/return/response mask 均为 0，optimizer 和 policy version 不变。
- `actor loss`、`grad norm`、固定 ref KL、LR 都是有限数，无 NaN/Inf。
- 参数审计证明每个实际 step 才发生参数变化；skip batch 不改变 Adam state。
- 原始模型目录的 18 分片哈希前后不变。
- step5 临时 checkpoint 完整，包含 `model,optimizer,extra`。
- sealed8 前后评测完整，且没有越过预注册退化线。
- 没有 reward boundary 违规、guess-correct 获利、unsafe/预算超限获正奖励或表格漏行漏列获 PASS。
- 两机退出后 Ray、vLLM 孤儿、36379 监听和 NPU 用户进程都为 0。

满足这些只是“允许审核正式训练”，不是自动启动正式训练。当前正式入口还必须同步金丝雀的 run-local sandbox 与 observability 接线后重新做静态/CPU 合同测试。

## 11. 如何换数据

当前代码不是任意 Parquet 即插即用；数据、manifest、tasks、SQLite 环境和硬编码哈希构成一个绑定资产包。换数据要改以下位置并重新测试：

| 要改内容 | 文件/变量 |
|---|---|
| 批准 Parquet、manifest、tasks、raw/heldout 路径 | 两个 `launch_qwen38_tiered_*_host.sh` 的 `PACKAGE_*`、`TASKS_CONTAINER`、`RAW100_CONTAINER` |
| 数据 SHA256 | 两个宿主 launcher 的 `EXPECTED_APPROVED_SHA256`、`EXPECTED_MANIFEST_SHA256`、`EXPECTED_RAW100_SHA256` |
| 数据生成器冻结哈希与成员数 | `prepare_qwen38_approved43_outcome_training.py`、`prepare_qwen38_tiered_canary_data.py` |
| 题目曝光次数 | 正式准备器的 `EXPOSURES` 与正式 runner 的总组数 |
| canary numeric/table 配比 | `prepare_qwen38_tiered_canary_data.py` |
| sealed 成员与配比 | `prepare_qwen38_tiered_canary_sealed8.py` |
| run-local 数据库 | 数据中 `environment_id`，并由 `stage_bound_pi_sandbox.py` 精确复制 |
| 奖励所需字段/身份 | `reward_model.ground_truth`、`extra_info`、批准 manifest |

新批准数据至少应满足：

- 每个题目身份唯一，使用稳定的 `instruction_sha256`。
- `extra_info.global_index` 能唯一映射冻结 `tasks.jsonl`。
- `extra_info.training_allowed` 在原始数据中仍为 `false`；本路线通过独立批准包授权，不篡改原始标志。
- `reward_model.ground_truth` 包含 answer type、完整 gold/verifier、environment identity。
- `tasks.jsonl` 提供 EvidencePlan、expected tables、verification criteria/must-use fields。
- numeric/table verifier 必须先做离线重放；表格使用全行、全列、顺序、数值容差和 TopN/占比等完整校验。
- 每个环境必须有非空 `logistics.sqlite`，且不得包含符号链接。
- 训练、sealed 和其他评测身份零交集。
- 所有敏感产物保持 `0600`，Git 只提交代码和去标识摘要。

换数据后的正确顺序是：数据质量审计 → 生成独立批准包 → 冻结哈希 → CPU shadow/reward 对抗测试 → 小金丝雀。不要跳过审计直接增加采样次数。

## 12. 如何换卡数和拓扑

### 12.1 必须满足的公式

训练：

```text
TRAIN_TP × TRAIN_PP × TRAIN_CP = TRAIN_NPUS
```

推理：

```text
ROLLOUT_DP = ROLLOUT_NNODES × ROLLOUT_NPUS_PER_NODE ÷ ROLLOUT_TP
```

并且总 rollout 卡数必须能被 `ROLLOUT_TP` 整除。

### 12.2 当前 16+16 推荐值

```text
TRAIN_TP=4 TRAIN_PP=2 TRAIN_CP=2 TRAIN_NPUS=16
ROLLOUT_TP=4 ROLLOUT_NPUS=16 ROLLOUT_NNODES=1  # DP=4
ROLLOUT_MAX_SEQS=16
```

Qwen3.8 的 24 个 attention heads、4 个 KV heads 与线性注意力头能被 TP4 整除。94K 长轨迹下，TP4 每副本并发从 16 起步；不要未经基准直接升到 24/32。

### 12.3 常见调整

| 资源 | 可用配置 | 说明 |
|---|---|---|
| rollout 16卡 | TP4×DP4，`max_num_seqs=16/副本` | 当前主线，长轨迹效率较好 |
| rollout 8卡 | TP4×DP2 | 可运行但吞吐下降；相同总体并发不能照搬 |
| rollout 12卡 | TP4×DP3 | 适合0号机只空12卡时的单独推理任务 |
| rollout 16卡保守基线 | TP8×DP2 | 显存更宽松、通信更多、独立副本更少 |
| rollout TP2 | 不建议94K | 单卡权重与KV压力过大 |
| 训练少于16卡 | 当前未验证 | 需要重新设计 TP/PP/CP、显存和吞吐，不是改一个数字 |

改卡数时同时检查：Ray 每节点 `--num-gpus`/NPU 自定义资源、placement group、HCCL 网卡、`ROLLOUT_NNODES`、`ROLLOUT_NPUS`、`ROLLOUT_TP`、`ROLLOUT_MAX_SEQS`、Agent worker 与每副本并发。只改 runner 里的卡数会造成 Ray 资源永远等不到或 placement group 跨机错误。

### 12.4 效率基准方法

用同一小批任务比较拓扑，并记录：

- 完整轨迹/小时。
- 有效生成 token/小时。
- timeout、runtime error、取消候选比例。
- 每副本峰值 HBM、KV cache 使用率。
- Ray 排队时间和整步墙钟。

历史同批小基准中 TP4 的平均吞吐比 TP8 高约 4.14%，但差距不大；不能把它外推为所有长度和所有数据都更快。94K 真实负载仍以 TP4 并发16作为保守起点，显存有稳定余量后再试20。

## 13. 如何换模型

换成另一个 Qwen3.x 27B 或不同分片版本时，至少要重新完成：

1. 容器内 config、tokenizer、分片数和复合哈希。
2. Megatron Bridge tensor key、层数、形状与 HF→Megatron 转换门。
3. 训练到 vLLM 权重同步。
4. tokenizer 对工具 response token 的真实计数。
5. chat template 中 reasoning effort 和 preserved thinking 行为。
6. 1-step 零风险工程 smoke。
7. 5-step 奖励/质量金丝雀。

不能只改 `MODEL_PATH` 后复用旧 `EXPECTED_*`，也不能复用其他模型的 optimizer/checkpoint。

## 14. 已遇到的错误、根因和解决办法

### 14.1 宿主路径与容器路径混用

- 现象：宿主 wrapper 找不到 `/workspace/...`，在启动 NPU/Ray 前退出。
- 根因：宿主脚本使用了容器路径。
- 修复：严格区分 `HOST_PROJECT_ROOT=/data3/...` 与 `CONTAINER_PROJECT_ROOT=/workspace/...`。
- 预防：所有 scp/chmod/docker 外操作用宿主路径；`docker exec` 内才用容器路径。

### 14.2 双机本地盘不共享

- 现象：5 号机能读训练/评测 Parquet，6 号 rollout actor 报文件不存在。
- 根因：相同路径名不代表共享文件系统。
- 修复：Ray 启动前确定性生成数据，`scp -p` 到相同路径，设 `0600` 并核对双端 SHA256。
- 预防：runtime、训练 Parquet、sealed Parquet、run-local SQLite 环境都做跨机哈希门。

### 14.3 在宿主错误 Python/PyArrow 中生成 Parquet

- 现象：嵌套字段 Parquet 在训练容器里读取损坏或 schema 不一致。
- 根因：宿主 Python 3.13/PyArrow 25 与容器 Python 3.11/PyArrow 24 的嵌套类型兼容问题。
- 修复：所有训练 Parquet 都在正式容器内生成和回读。
- 预防：宿主只做文件搬运和哈希，不直接重写训练 Parquet。

### 14.4 `/pi_sandbox` 只读，且数据库不可观测

- 现象：`database_unavailable` 大量 UNKNOWN；尝试直接写只读挂载失败。
- 根因：原始 sandbox 是只读挂载，且旧入口没有把任务绑定数据库复制到 run-local 目录。
- 修复：`stage_bound_pi_sandbox.py` 只按训练/密封数据中的 environment identity 复制所需环境到 `<run>/private/pi_sandbox`，双机同步并哈希；运行时用 `PI_AGENT_SANDBOX_LOWER` 指向它。
- 预防：启动前要求数据库数量大于0、两机数量和复合哈希一致。

### 14.5 request/environment identity 缺失

- 现象：轨迹无法与题目、workspace、工具事件可靠连接，fail-closed 为 UNKNOWN。
- 根因：旧 runtime 没有贯通同一 request/environment identity。
- 修复：v1.12.28 将 identity 写入请求、workspace、工具事件并在奖励端核验。
- 预防：canary20/sealed8 在两机都执行 online observability CPU gate；任何 identity missing 都停止。

### 14.6 工具响应 token 不可观测

- 现象：`tool_response_cost_unobservable`，所有工具 token 被旧日志记成0。
- 根因：没有在运行时用实际 tokenizer 计数；0 既可能是真0也可能是缺失。
- 修复：用冻结 Qwen3.8 tokenizer 真实计数并持久化；缺失继续 UNKNOWN，禁止伪填0。
- 预防：设置 `PI_AGENT_TOKENIZER_PATH=/models/Qwen3.8-27B`，online observability 门必须覆盖 token 字段。

### 14.7 表格最终答案被系统性误判

- 现象：旧 47 道全错中 27 道其实是奖励器假阴性；22 道修复后为 mixed，5 道变全对。
- 根因：旧评分只接受特定 `category/value` 对象，不能正确比较完整二维表。
- 修复：`corrected_full_table_ordered_v1` 做全行、全列、顺序与数值容差校验；在线 parser 只接受能明确恢复完整表格的结构。
- 预防：漏行、漏列、错序、多个 final、附加叙述均 fail closed；表格 verifier 必须先离线重放。

### 14.8 reward 附加字段与 veRL 内建列冲突

- 现象：step0 sealed 汇总把32个样本扩成64个 reward 值或发生列冲突。
- 根因：自定义奖励返回名为 `reward` 的附件，与框架内建 reward 列重名。
- 修复：审计字段改名 `tiered_reward`。
- 预防：测试所有奖励返回键不得覆盖框架保留列。

### 14.9 非数值奖励附件被 NumPy 求均值

- 现象：验证指标归约对 dict/list/null 报类型错误。
- 根因：veRL 会对 extra reward fields 做数值汇总。
- 修复：dict/list/null 规范化为稳定 JSON 字符串或字符串 `null`；真正训练标量保持数值。
- 预防：以32条 sealed 样本做列长度和类型回归。

### 14.10 三态样本字段集合或顺序不一致

- 现象：UNKNOWN-first 批次对后续正常样本索引 `infrastructure_error_type` 触发 KeyError；或 `DataProto.concat` 报 metadata 不一致。
- 根因：不同状态返回的键集合和插入顺序不同。
- 修复：所有状态总是返回 `infrastructure_error_type`，正常值为空字符串；`score` 固定第一，其余键按名字排序。
- 预防：跨 PASS/FAIL/UNKNOWN 回归同时检查键集合、顺序和类型。

### 14.11 ref log-prob 阶段缺 Flash Attention 配置

- 现象：actor 已加载，但首批 reference log-prob 因 `use_flash_attn`/micro batch 配置缺失退出。
- 根因：只给 actor 设置了 MindSpeed Flash Attention，没有给冻结 ref 同步配置。
- 修复：显式添加 `+actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True`。
- 预防：合同测试同时检查 actor/ref attention 配置。

### 14.12 三态组门仍断言旧 `acc`

- 现象：tiered reward 已返回 `success/train_mask`，严格组门却因没有 legacy `acc` 退出。
- 根因：补丁残留旧奖励字段假设。
- 修复：三态主路径使用 `success/train_mask`；只有旧奖励才 fallback 到 `acc`。
- 预防：容器门禁使用一个明确不带 `acc` 的 mixed 组。

### 14.13 TensorDict 不可哈希

- 现象：`unhashable type: TensorDict`，发生在任何 optimizer update 前。
- 根因：把 TensorDict 当普通 key 迭代器传给 `set.issubset`。
- 修复：必需字段逐键 membership 检查。
- 预防：用模拟真实 TensorDict 迭代行为的测试对象回归。

### 14.14 uniform/UNKNOWN skip 后永久空等

- 现象：首批约29分钟完成并跳过 optimizer，此后数小时 AICore 0%，进程只在 wait/poll。
- 根因：skip 分支同时跳过 `rollouter.reset_staleness()`；在 `staleness=0` 下两组许可耗尽后不再开放同 policy 采样窗口。
- 修复：skip 时仍调用 `reset_staleness()`，但不更新 optimizer、参数、Adam、policy version或广播权重。
- 预防：测试连续两个 skip 后第三批仍能生成；生产运行已验证此活性修复。

### 14.15 超时轨迹缺 rollout log-prob

- 现象：旧 Step70 在第4步被单条超时轨迹中止；外层队列曾错误返回成功，但没有完整 checkpoint。
- 根因：取消/超时轨迹没有完整 log-prob，且外层只看队列结束码。
- 修复：成功条件改为完整目标 checkpoint + manifest +分片门，不能只看进程返回0；取消轨迹元数据和 log-prob 路径做失败关闭。
- 预防：30分钟覆盖排队、准入、生成和工具执行；最终必须核对 checkpoint，不接受部分 step。

### 14.16 等待时间没有计入30分钟

- 现象：旧系统名义30分钟，实际轨迹可因排队等待远超30分钟。
- 根因：计时点放在开始生成之后。
- 修复：deadline 从请求入队开始，覆盖 queue wait、vLLM admission、生成、工具执行和收尾。
- 预防：日志分别记录 queue wait 与整条墙钟；整条超过1800秒必须 timeout。

### 14.17 失败后遗留 vLLM/Ray worker

- 现象：新任务启动前 NPU 仍有上一轮 `VLLM::` worker，占卡或触发 OOM。
- 根因：Ray stop 后孤儿 worker 未跟随退出，外层误判完成。
- 修复：清理顺序为停止队列/接力器 → Ray stop → 二次检查并清理专用容器的孤儿 worker → 核对 NPU/端口。
- 预防：`assert_idle` 是启动硬门；不要仅看 `docker ps`。

### 14.18 placement group 瞬时资源竞态

- 现象：多个 TP4 副本并发创建时，一个脚本看到“空闲卡够”，实际 PG 同时争抢后失败。
- 根因：每副本用非原子的即时空闲检查。
- 修复：移除每副本瞬时检查，交给 Ray 原子 placement group 调度。
- 预防：启动后检查每个 PG 都完整落在单机，资源标签和物理卡数一致。

### 14.19 PP2 在线 HF 导出不完整

- 现象：训练中直接导出的 HF 模型缺 tensor，不能用于后续评测。
- 根因：PP2 分布式权重未完整汇聚。
- 修复：训练只保存 Megatron distributed checkpoint；结束后单独导出 HF，并核对18分片、1,199 tensor key、层覆盖、形状和跨机哈希。
- 预防：任何缺 manifest/分片/tensor 的导出都禁止发布或评测。

### 14.20 权重同步 bucket 太小

- 现象：embedding tensor 大于通信 bucket，权重同步失败。
- 根因：默认 bucket 小于最大单 tensor。
- 修复：`WEIGHT_BUCKET_MB=2560`。
- 预防：换模型时先计算最大 tensor 大小，bucket 必须不小于它。

### 14.21 NPU OOM 与 pinned host memory OOM 混淆

- 现象A：NPU 上只剩约5 MiB，加载 ref MLP 再申请172 MiB失败。
- 现象B：清空 NPU 后 actor/ref 可装入、约30 GiB/卡，但16个 worker 初始化 `HybridDeviceOptimizer` 时容器退出137或报 host pinned allocation。
- 根因：A 是启动前已有占用或错误内存放置；B 是 CPU optimizer offload 并行克隆 Adam 状态，耗尽可锁页主机内存，不是 HBM 不足。
- 当前修复：先保证 NPU 真空闲；当前 approved43 金丝雀固定 `OPTIMIZER_CPU_OFFLOAD=false`、`ENGINE_OPTIMIZER_OFFLOAD=false`。
- 预防：用 `npu-smi` 区分 HBM，使用宿主内存/locked memory 与日志区分 pinned host OOM。不要看到 OOM 就机械切换 CPU offload。

说明：2026-08-17 的 Qwen3.8 工程 smoke 也通过“关闭 optimizer CPU/engine offload”解决 Ascend `207001`；v1.12.29 曾短暂尝试恢复 CPU offload，v1.12.30 根据清卡后证据撤销。以 v1.12.30 当前配置为准。

### 14.22 奖励黑客与全错组代理学习

- 现象：旧 banded-v2 中错误轨迹奖励低于正确轨迹，但 78/78 个全错组仍有奖励方差并产生梯度；最终 sealed 严重退化。
- 根因：GRPO 只看组内相对奖励，全错组仍会优化报告长度、过程完整性、表命中等代理；无固定 ref KL 又允许小漂移累积。
- 修复：strict-mixed-only；全错/全对/UNKNOWN整组 mask；固定原始 ref KL；staleness0；短金丝雀与sealed止损。
- 预防：训练信号统计必须报告 strict correct、mixed/allwrong/allcorrect/UNKNOWN 组、实际更新数，而不是只看平均 reward。

## 15. 故障恢复与安全停机

### 15.1 先判断是否真的需要停止

若 `training.log` 仍在产生新 batch、NPU AICore 有活动、Ray 队列在变化，不要因为单步长达20–30分钟就强杀。94K 多轮工具轨迹的长尾是真实存在的。

若已确认死锁、目标不可达或硬门失败，优先终止宿主 supervisor，使其 `trap` 执行清理；不要直接随机杀 Ray worker。

```bash
kill "$(cat "$RUN_HOST/supervisor.pid")"
```

等待后核对：

```bash
cat "$RUN_HOST/state"
cat "$RUN_HOST/exit_code"
docker exec llin-verl-qwen38-canary-m05-20260822 ray stop --force || true
ssh root@192.168.202.4 \
  'docker exec llin-verl-qwen38-canary-m06-20260822 ray stop --force || true'
npu-smi info
ssh root@192.168.202.4 npu-smi info
ss -lntp | grep ':36379' || true
```

只清理本次专用容器和明确识别的本次 PID；不要用宽泛 `pkill -9 python`。

### 15.2 是否可以续跑

- 金丝雀硬失败：不续跑，从原始 Qwen3.8 新建 run。
- 数据/模型/runtime 哈希失败：修复资产后新建 run。
- optimizer/参数已发生但 checkpoint 不完整：不续跑。
- 只有完整、经过门禁的 Megatron checkpoint 才能作为恢复点；当前金丝雀合同仍要求从原始基座复跑，而不是从失败金丝雀续。

## 16. 正式训练流程（当前为受控待放行）

当第10节全部通过，并且正式入口已经补齐与金丝雀相同的 sandbox/identity/token/observability 接线后：

1. 新建唯一 `RUN_NAME`，冻结新的 runtime。
2. 从原始 Qwen3.8 actor/ref 开始，不使用金丝雀权重。
3. 容器内生成 `approved43x4`：43题×4=172组。
4. 生成与训练零交集的 sealed8。
5. 两机同步训练/密封数据、runtime 和 run-local SQLite，逐项复核 SHA256。
6. 运行完整 CPU 合同与 observability 门。
7. 启动隔离 Ray。
8. 用 `scripts/run_pi_qwen38_approved43_tiered_formal_v1.sh` 执行最多86个名义 batch。
9. uniform/UNKNOWN 仍可使实际 optimizer step 少于86；报告必须写实际步数。
10. 最终只保留一个完整 `model,extra` Megatron checkpoint，生成 `final_model_manifest.safe.json`。
11. 单独导出 HF，做完整 tensor/分片/哈希门，再执行未训练数据的 `2+2+2` 密封评测。

不要直接运行现有 `launch_qwen38_tiered_formal_host.sh` 来绕过当前锁；它是“待完成金丝雀后使用的入口”，不是当前 green path。

## 17. 交给新使用者的最小改动清单

如果机器环境、镜像和奖励都不变，新使用者原则上只改这些：

1. `RUN_NAME`：每次唯一。
2. `TRAINER_CONTAINER`、`ROLLOUT_CONTAINER`、`ROLLOUT_HOST`、`RAY_ADDRESS`。
3. 宿主/容器项目根和模型挂载；容器内模型统一为 `/models/Qwen3.8-27B`。
4. 数据包四个路径及所有 SHA256。
5. 训练题数、曝光次数、名义组/批次数、sealed 配比。
6. `TRAIN_TP/PP/CP/NPUS` 与 `ROLLOUT_TP/NPUS/NNODES`。
7. 根据 rollout 副本数调整 `ROLLOUT_MAX_SEQS`、Agent worker、每副本并发。
8. 若最大长度变化，同时满足 `prompt + response = context`，再评估 KV cache、超时和 queue token。
9. 每次先运行容器 CPU 门和 5 实际更新步金丝雀。

不应随手改的安全参数：

- strict-mixed-only、UNKNOWN mask、staleness0。
- 固定 ref KL 和 `0.001 low_var_kl`。
- `temperature=1/top_p=.95/top_k=20` 的同批比较口径。
- 30分钟从入队开始的完整轨迹 timeout。
- private 文件权限和跨机哈希。
- 当前 `OPTIMIZER_CPU_OFFLOAD=false / ENGINE_OPTIMIZER_OFFLOAD=false`。
- 最终 checkpoint 完整性门。

## 18. 最终交付检查表

训练开始前：

- [ ] 两机/三机 SSH、内网 IP、HCCL 网卡正确。
- [ ] 容器镜像 ID、挂载、host network、privileged、shm 正确。
- [ ] 模型 config、18分片和复合哈希一致。
- [ ] 数据、manifest、tasks、raw/heldout、SQLite 完整且权限正确。
- [ ] runtime 来自明确 Git commit，所有参与节点 tar SHA 一致。
- [ ] NPU 空闲、无旧 Ray、无36379监听、无 vLLM 孤儿。
- [ ] 容器 CPU 回归与 observability 门全通过。

训练过程中：

- [ ] 名义 batch、实际 optimizer step 分开统计。
- [ ] PASS/FAIL/UNKNOWN 与 mixed/uniform 分布可见。
- [ ] queue wait 计入1800秒。
- [ ] skip batch 不改参数/Adam/policy version，但会恢复采样许可。
- [ ] KL、loss、grad norm、reward boundary、HBM、主机 pinned memory 可见。

训练结束后：

- [ ] `state` 与 `exit_code` 一致。
- [ ] checkpoint 目录、metadata、分片和内容符合合同。
- [ ] 原始模型目录哈希未变。
- [ ] Ray/vLLM/NPU/端口完全释放。
- [ ] sealed/heldout 与训练身份零交集并完成评测。
- [ ] Git 只提交代码、测试、README版本和安全摘要；敏感数据、轨迹、SQL、gold、runtime tar 不提交。

## 19. 参考报告

- `docs/qwen38_27b_engineering_smoke_20260817.md`：Qwen3.8 双机 1-step 工程闭环。
- `docs/qwen38_grpo_candidate_audit_20260818.md`：v15/v20/v21 70道候选审核。
- `docs/qwen38_banded_v2_strict_reward_replay_20260818.md`：旧 banded-v2 修复与回放。
- `docs/qwen38_mixed27_training_collapse_diagnosis_20260821.md`：旧 mixed27 退化根因。
- `docs/qwen38_grounded_tristate_calibration_20260822.md`：三态奖励校准边界。
- `docs/qwen38_tiered_canary5_unreachable_20260823.md`：活性修复通过但训练信号不可用。
- `README.md` 的 v1.12.16–v1.12.30：当前金丝雀每次生产故障与修复的时间线。

## 20. 当前现场状态快照

本节只描述 2026-08-24 核验时状态，使用时必须重新检查：

- 5 号机项目和模型存在；`llin-verl-qwen38-canary-m05-20260822` 为 `Exited (137)`。
- 6 号机项目和模型存在；金丝雀容器也曾以 `Exited (137)` 结束，旧 smoke 容器仍在运行。
- 0 号机模型实际位于 `/data3/models/Qwen3.8-27b`，基准容器挂载后可见 `/models/Qwen3.8-27B`。
- 最近 `canary ...-09` 的 `state=failed_full_training_locked`、`exit_code=137`。
- 当前代码基线在本手册编写前为 `68e971f`（v1.12.30）；接手时应以 `origin/main` 上包含本手册的新提交为准，并记录实际 commit。
- 正式全量训练仍未放行。
