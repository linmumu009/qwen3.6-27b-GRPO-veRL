---
报告日期: 2026-09-14
更新类型: 多机多卡 GRPO 训练情况报告（27B，事实核验版）
更新人: llin（Codex 协助核验与整理）
状态: 已核验历史双机训练闭环；不代表当前正在运行或训练效果达标
---

# 多机多卡 GRPO 训练情况报告（Qwen3.6-27B）

## 一、背景

本报告描述“一台机器训练、一台机器 rollout”的 27B GRPO 架构：5 号机提供 16 个计算 chip 运行 Megatron Trainer，6 号机提供 16 个计算 chip 运行 vLLM Rollout，通过 Ray 编排和 HCCL 权重同步连接起来。

模型选用已有完整训练记录的 **Qwen3.6-27B**。仓库后续另有 Qwen3.8-27B 实验，但模型、奖励和验收状态不同，不能将其混写为一个 27B 实验。本报告沿用原 sources 报告的 Markdown 格式，使用 2026-09-14 在 5 号机实际读取的配置、日志与检查点记录核实历史事实。

结论：**双机分离式 GRPO 已有实际训练、rollout、权重同步及 Step120 检查点证据。** 这证明工程链路运行过，不单独证明任务能力提升，也不代表 9 月 14 日仍在运行这项 GRPO 作业。本次仅核验，没有发起新训练。

## 二、已完成的工作

### 2.1 基础架构搭建与角色

| 机器 | 角色 | 计算 chip | 并行配置 | 证据 |
|------|------|------:|------|------|
| 5 号机 | Ray head、Megatron Trainer | 16 | TP4×PP2×CP2，训练 DP1 | 现场容器/脚本；历史 Trainer 日志 |
| 6 号机 | Ray worker、vLLM Rollout | 16 | TP8×DP2×PP1 | 5 号机保存的展开配置与来自 `192.168.202.4` 的 Rollouter 日志 |

5 号机现场设备枚举为 8 个 NPU ID、每个 2 chip；表中“16”采用计算 chip 口径。6 号机在本次没有另行登录，表中其配置与参与运行情况来自保存在 5 号机的历史双机日志，不能作为 6 号机当前空闲和健康状态证明。

服务器上确认存在的训练容器为 `llin-verl-trainer-m05-20260730`，镜像标签 `llin-verl-a3:20260730`，模型挂载为 `/models/Qwen3.6-27B`，项目挂载为 `/workspace/llin-verl-grpo`。

### 2.2 训练脚本核心配置

入口为 `scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh`，实际显式指定 `actor.strategy=megatron`，不是原报告所述“通过 megatron 参数推断”。核心参数摘录如下，非独立完整启动命令：

```bash
python3 -m verl.experimental.fully_async_policy.fully_async_main \
  --config-path=config \
  --config-name=fully_async_ppo_megatron_trainer.yaml \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.strategy=megatron \
  actor_rollout_ref.model.path=/models/Qwen3.6-27B \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.actor.megatron.use_mbridge=True \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
  actor_rollout_ref.actor.megatron.context_parallel_size=2 \
  actor_rollout_ref.actor.megatron.dtype=bfloat16 \
  actor_rollout_ref.actor.megatron.param_offload=False \
  actor_rollout_ref.actor.megatron.optimizer_offload=True \
  actor_rollout_ref.actor.megatron.grad_offload=True \
  actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
  actor_rollout_ref.rollout.data_parallel_size=2 \
  trainer.nnodes=1 trainer.n_gpus_per_node=16 \
  rollout.nnodes=1 rollout.n_gpus_per_node=16
```

`trainer.nnodes=1` 和 `rollout.nnodes=1` 分别计算角色池中的机器数，总部署仍是两台机器，不应将两项都改成 2。训练 DP 为 16/(4×2×2)=1；rollout 的 DP2 是推理侧并行维度，不能称作 Trainer DP2。

脚本还配置 Bridge 兼容补丁、分布式优化器及 CPU optimizer offload、activation recompute、Flash Attention、sequence parallel 和 `kvallgather_cp_algo`。这些是已用工程配置，不意味着换模型后全部参数都可原样迁移。

### 2.3 历史训练完成证据

主要核验运行：`runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/`。

| 项目 | 现场读取结果 | 可支持的结论 |
|------|------|------|
| 模型路径 | 日志为 `/models/Qwen3.6-27B` | 本次证据属于 Qwen3.6-27B |
| 训练步指标 | 检出 `training/global_step` 102–120 | 至少这些步有训练日志；不能将续训写成从零新训 120 步 |
| 最终步 | `training/global_step:120.0` | 日志到达 Step120 |
| 权重同步 | `_fit_update_weights`，参数版本到 120；最后一次耗时 7.7294 秒 | Trainer→Rollout 同步链路执行过；单次耗时不是平均性能基准 |
| rollout 所在机器 | `FullyAsyncRollouter ... ip=192.168.202.4` | 6 号机参与 rollout |
| 退出状态 | `exit_code=0` | 包装进程正常结束，需结合其余证据理解 |
| 完成时间 | `2026-08-10T08:16:14+00:00`，北京时间 16:16:14 | 历史完成时间，不是本次新训练 |
| 检查点指针 | `latest_checkpointed_iteration.txt=120`，Step120 目录存在 | 保存目标一致 |
| 既有完整性记录 | `checkpoint_integrity.json` 中模型和 optimizer 均 `valid=true`、各 32 个分片 | 历史完整性检查有记录；本次未重读数百 GB 张量验证全部内容 |

交叉核验早期 `pi-grpo-fully-async-bounded-3step-20260730-02`：退出码 0、权重版本推进到 3，存在 6 号机 rollout 日志。其输出的 `step:3` 与同一行 `training/global_step:2.0` 口径不同，因此不单靠 run 名称或进度编号证明完成了几个 optimizer step。

## 三、遇到的问题与事实边界

### 3.1 原报告中需要修正的表述

| 原表述或容易产生的理解 | 核验后表述 |
|------|------|
| llin 的 Megatron strategy 是推断的 | 脚本显式配置 `strategy=megatron`，日志也展开为 megatron |
| 16 trainer + 16 rollout 是单机配置 | 本报告是两台机器各 16 chip，共 32 chip |
| 两份报告只是模型大小不同 | 还存在 FSDP2/Megatron、one-step/fully-async、数学单轮/工具多轮差异 |
| 27B 跑通过，所以 9B 换成 Megatron 就能跑通 | 9B 仍需模型映射、资源隔离和完整训练闭环验收 |
| `rollout.mode=async` 就等于完全相同的异步训练实现 | 调度方式需看入口与配置，本项目双机使用 fully-async-policy |
| 镜像相同即可完整复现 | 还需冻结容器补丁、源码、依赖、模型、数据和实际 overrides |

### 3.2 507001 的对照

本次读取的 Step100→120 `driver.log` 未检出字符串 507001。这只说明该选定日志中未发现此码，不说明所有双机历史运行从未遇到 NPU 错误。

9B 日志的已定位异常发生在 FSDP 的参数搬运/聚合及权重发送路径；尚不能从该对照断言 Megatron 天然消除 507001，或 FSDP2 在 NPU 上不可用。veRL 官方确有 NPU FSDP2 路线，参见[官方快速上手](https://verl.readthedocs.io/en/latest/ascend_tutorial/zh/get_start/quick_start.html)。

### 3.3 训练完成与训练有效性

检查点存在、loss 出现、进程退出码为 0，各自都不是能力提升证据。GRPO 还需核对同题分组、实际 reward/advantage、是否跳过 optimizer、陈旧轨迹处理、训练前后独立评测。历史 27B 工程跑通记录不能替代新数据/新奖励的效果验收，也不能借用后续 CPT/SFT 结果证明 GRPO 收益。

## 四、关键发现：我们实际的 GRPO 流程

### 4.1 训练与 rollout 的闭环

1. 6 号机使用当前参数版本执行 rollout；本项目包含多轮工具轨迹，不只是输出一次数学答案。
2. 自定义奖励对轨迹评分，同一道题的多条回答构成 GRPO 组，供训练侧计算组内相对优势。
3. fully-async 管理轨迹队列与参数版本，并按配置处理陈旧样本；5 号机 Megatron 读取可用训练批次执行更新。
4. 新参数通过 checkpoint engine 同步到 rollout 侧，继续生成后续轨迹；训练侧按保存策略写入检查点。

角色分离不保证任何时候都能满负载重叠；同步、等待完整组、工具调用和队列背压都会造成等待。吞吐和加速比需要专门采样，不能从“双机”直接推出 2 倍速度。

### 4.2 默认参数与真实运行参数需要分开

| 参数 | 当前通用脚本默认值 | Step100→120 已查日志 |
|------|------|------|
| 训练并行 | TP4 / PP2 / CP2 | TP4 / PP2 / CP2 |
| rollout 并行 | TP8 / DP2 | TP8 / DP2 |
| 陈旧度阈值 | `STALENESS_THRESHOLD=0.5` | `staleness_threshold=2.0` |
| 训练总步 | 默认 20，可由环境覆盖 | 本次是 Step100→120 续训，以最终日志与检查点为准 |

当前脚本还提供最快 K 条选择、候选超采样、队列 token 上限、partial rollout 等扩展。其默认 `FASTEST_K=4`、候选数 6、每组保留响应 4，并非所有历史实验都固定相同。复现应读取目标运行的完整配置，不能把今天的脚本默认值倒填到旧实验。

## 五、关键问题解答

### 问题 1：我们的 GRPO 是否确实一机训练、一机 rollout？

是，就本报告核验的 Qwen3.6-27B 运行而言，5 号机日志确认 Trainer 16 chip、6 号机 Rollouter 16 chip，同时有更新与保存证据。它是训练/推理角色分离的双机架构，不是把同一个 Trainer 的 TP/PP 跨两台机器展开。

### 问题 2：为什么训练侧使用 Megatron？

因为本项目已有 Qwen3.6-27B 的 Bridge 和 TP4/PP2/CP2 适配、补丁及成功运行证据，适合作为本报告的可核验方案。不能由此推导出 NPU 只支持 Megatron。

### 问题 3：offload 与并行选项是否属于必要条件？

它们属于已用配置。optimizer/grad offload 与重计算会改变显存占用和耗时；能否去掉应由实际峰值显存与稳定性测试决定。`use_mbridge` 涉及本项目模型转换链路，不能把它当作与模型无关的通用开关。

### 问题 4：能否直接用于新的 27B 实验？

拓扑可以作为基线，训练配方仍需重新固定数据、reward、KL、组大小、上下文、陈旧度和保存策略。尤其当前通用脚本默认 checkpoint 内容为 `[model,extra]`，而所核验历史 Step120 完整性记录包含 optimizer；需要可续训 checkpoint 时必须显式包含 optimizer，并验收实际保存结果。

## 六、推荐配置与交付结论

### 6.1 双机基线

采用 Qwen3.6-27B，5 号机 16 chip Megatron TP4×PP2×CP2，6 号机 16 chip vLLM TP8×DP2，fully-async-policy 入口。网络、Ray 标签、模型挂载和项目运行时需在两机保持一致；当前通用脚本默认 Ray 地址为 `192.168.202.5:26379`，不能与 Qwen3.8 金丝雀使用的另一端口混用。

第二章为配置摘录，完整工程入口见附录。脚本会应用运行时补丁，不能在事实核验过程中直接执行。新运行应先检查两机资源与通信，固定完整配置，然后以少量实际 optimizer step 验证分组奖励、权重版本推进和可恢复检查点，再扩大规模。

### 6.2 对外报告可采用的结论

“Qwen3.6-27B 已在 5 号机 16 chip 训练、6 号机 16 chip rollout 的分离式架构下完成 GRPO 工程闭环。训练侧使用 Megatron TP4/PP2/CP2，推理侧使用 vLLM TP8/DP2；历史续训运行具有到达 Step120 的训练日志、参数同步记录和检查点。上述结果证明工程可运行，模型能力收益需以独立评测另行判定。”

单机 9B 当前应写为“8+8 方案已尝试，权重同步及数据/奖励接口仍有阻塞”，详见[单机多卡 9B 报告](single_machine_9b_grpo_report_20260914.md)。

## 七、附录

### A. 文件位置

5 号机宿主根目录：`/data3/llin/qwen3.6-27b-verl-grpo`；容器根目录：`/workspace/llin-verl-grpo`。

| 文件 | 用途 |
|------|------|
| `scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh` | 完整通用工程入口，含兼容补丁与配置 |
| `runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/driver.log` | 实际配置、训练、远端 rollout 与同步日志 |
| 同运行目录 `exit_code`、`finished_at` | 退出与完成时间 |
| 同运行目录 `checkpoint_integrity.json` | 既有检查点完整性记录 |
| 同运行目录 `checkpoints/latest_checkpointed_iteration.txt` | 保存步指针 |
| 同运行目录 `checkpoints/global_step_120/` | Step120 检查点 |

### B. 证据身份与环境边界

| 文件 | SHA256 |
|------|------|
| 服务器通用脚本 | `65bfbe1c90d963844ab02e8847aa92f1b027706c7856c05f5c48cf52aa3b28af` |
| Step100→120 `driver.log` | `d29189e71341741f32de88cc564e1b33892089d128fc3ef533f1ac6ed8d6c5b1` |

9 月 14 日训练容器版本：veRL 0.9.0.dev0、PyTorch 2.9.0、torch-npu 2.9.0.post2、Ray 2.56.1、vLLM 0.18.0+empty、Transformers 5.10.4。原容器长期使用并应用补丁，这组现场版本不能当作 8 月 10 日完整的不可变环境清单。

### C. 原材料及相关报告

- [原单机分析报告](sources/single_machine_grpo_analysis_20260913.md)：本次对照材料，保留原文。
- [Qwen3.8-27B 环境交接手册](../docs/qwen38_27b_training_environment_handoff_20260824.md)：仅用于辨别后续模型路线，不作为本报告 Qwen3.6 的混合证据。

---

**报告完成时间**：2026-09-14

**核验方式**：5 号机只读核验历史脚本、配置、日志及检查点记录；未登录 6 号机复查当前状态，未启动训练。
