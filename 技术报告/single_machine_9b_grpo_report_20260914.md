---
报告日期: 2026-09-14
更新类型: 单机多卡 GRPO 训练情况报告（9B，事实核验版）
更新人: llin（Codex 协助核验与整理）
状态: 已核验脚本与历史日志；8+8 方案尚未完成训练闭环
---

# 单机多卡 GRPO 训练情况报告（Qwen3.5-9B）

## 一、背景

目标是在 5 号机单台服务器上，以 Qwen3.5-9B 完成 GRPO 冒烟训练：16 个 Ascend 计算 chip 中，8 个用于 Trainer，8 个用于 vLLM Rollout。训练侧使用 FSDP2，推理侧使用 TP4、DP2。

本报告沿用原报告的章节、参数块、对照表与附录格式。2026-09-14 实际通过 `huawei-05` 登录服务器，核对容器、脚本、12 份数学冒烟日志、模型配置、数据结构和奖励调用接口。**现有证据支持“已经搭建并尝试运行”，不支持“单机 9B 已跑通 GRPO”。** 本次没有启动训练，也没有修改服务器环境。

设备口径：5 号机 `npu-smi info -l` 显示 8 个 NPU ID，每个包含 2 个 chip，共 16 个计算 chip。以下“8 卡”“16 卡”均沿用项目习惯，指计算 chip 数，不能写成 16 块物理板卡。

## 二、已完成的工作

### 2.1 基础架构与资产核验

| 核验项 | 服务器事实 | 结论 |
|------|------|------|
| 容器 | `llin-verl-trainer-m05-20260730`，镜像标签 `llin-verl-a3:20260730` | 已确认；同一镜像标签不保证容器后续修改一致 |
| 模型 | 项目内 `models/Qwen3.5-9B/config.json`，架构 `Qwen3_5ForConditionalGeneration` | 已确认配置身份；未逐张量重新校验完整权重 |
| 数据 | `data/math_smoke.parquet`，100 行；`data_source` 标记为 gsm8k 50、math 50 | 已确认条数与来源标签，未独立审核题目出处和答案正确性 |
| 数据字段 | `data_source`、`prompt`、`ground_truth` | 与当前 naive reward 所需字段不匹配，见第三章 |
| 启动脚本 | `scripts/run_math_grpo_smoke.sh` | 已确认存在，入口是 `one_step_off_policy.main_ppo` |
| 实际配置 | v10/v11 日志展开值均为 Trainer 1×8、Rollout 1×8、FSDP size 8、Rollout TP4/DP2 | 已确认配置；再次核验补得 v11 训练组与权重接收组物理映射，见 3.3，尚缺全部 vLLM 进程快照 |
| 完成状态 | 12 份日志均未检出 `training/global_step` 指标；`runs/math*/checkpoints/global_step_*` 未发现检查点 | 未获得实际训练步完成证据 |

原报告“Ray 正确识别 16 NPU”“所有 NPU 健康 OK”“vLLM 服务成功启动”包含历史现场判断。本次确认设备枚举和 vLLM 初始化日志，但没有将当前设备枚举等同于 9 月 13 日的 Ray 可用资源，也没有将权重加载进度条等同于推理请求与权重更新均成功。

### 2.2 训练脚本核心配置

以下是服务器现有脚本的关键配置摘录，表示已尝试方案，**不是已经验证通过的启动配方**：

```bash
python3 -m verl.experimental.one_step_off_policy.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.data_parallel_size=2 \
  trainer.nnodes=1 trainer.n_gpus_per_node=8 \
  rollout.nnodes=1 rollout.n_gpus_per_node=8
```

其余现有参数：训练 batch 4、每题采样 4 条、学习率 1e-6、prompt 上限 1024、response 上限 2048、rollout 模型长度 3072、计划训练 3 步；关闭 reward KL 和 actor KL loss。`Total training steps: 3` 是计划预算，不是完成计数。GRPO 在当前配置下禁用 critic，设置 `critic.strategy=fsdp2` 不代表还运行一个独立价值网络。

`TRAIN_TP=4` 虽在 shell 中导出，但没有作为 FSDP2 训练 TP 参数传入；不能把训练端写成 TP4。FSDP 的分片组与 Megatron TP/PP/CP 是不同概念。

## 三、遇到的问题

### 3.1 历次运行的实际失败位置

| 日志文件（均在项目 `runs/` 下） | 观察到的失败 | 解释边界 |
|------|------|------|
| `math_grpo_smoke.log`、`math_grpo_smoke_v2.log` | 模型绝对路径被当成 Hugging Face repo ID，随后 OSError | 先核对容器可见路径与挂载，不能归因于并行策略 |
| `math_grpo_smoke_v3.log` | processor 缺少 chat template，处理样本失败，最终 `num_samples=0` | 数据预处理失败 |
| `math_grpo_smoke_v4.log`、v5、v8 | `Total available GPUs 0 is less than total desired GPUs 8` | 资源池/设备类型/占用需要核对；报错里的 GPU 是框架用语 |
| `math_grpo_smoke_v6.log`、v7 | `cannot import name 'LLM' from 'vllm'` | Python 导入路径或包布局失败 |
| `math_grpo_smoke_v9.log` | `TRAIN_NPUS: unbound variable` | shell 变量失败 |
| `math_grpo_smoke_v10.log`、v11 | `WorkerDict.actor_update_weights()` 抛 RuntimeError，伴随 507001 / EE9999 | 已定位到权重同步阶段，底层根因尚未确定 |
| `math_grpo_smoke_debug.log` | 同样出现 507001 与设备事件查询错误 | 调试输出未给出可唯一归因的结论 |

这些是不同尝试的故障，不能统一概括为“FSDP2 初始化卡住”。当前脚本与原报告附录也不是逐字一致，例如当前脚本没有原附录中的 `PYTORCH_NPU_ALLOC_CONF` 导出；该变量是否由某次启动的外部环境设置，需该次环境快照证明。

### 3.2 507001 的定位结果

v11 的实际堆栈为：

```text
WorkerDict.actor_update_weights()
  → engine_workers.py::update_weights
  → hccl_checkpoint_engine.py::send_weights
  → fsdp/transformer_impl.py 中权重迭代器
  → param.to(device, non_blocking=True).full_tensor()
  → NPU RuntimeError；随后出现 507001 / EE9999
```

因此优先检查 FSDP/DTensor 的 CPU→NPU 参数恢复、完整参数聚合、权重同步通信组和设备映射。异步设备错误可能延后显现，这条堆栈仍不能单独证明内存、通信、固件或框架哪一项是根因。原报告对五类原因的“概率排序”没有证据支撑，应改为待排查清单。下述再次核验从设备日志补充了更明确的通信超时证据。

脚本虽配置 `checkpoint_engine.backend=nccl`，本次实际堆栈进入的是 `hccl_checkpoint_engine.py`，不能据字符串就判定“错误地使用 CUDA NCCL”。关闭 offload、减小同步 bucket 或调整 sleep/wake 都只能作为后续单变量诊断实验，不能写成已修复方案。

### 3.3 2026-09-14 再次现场核验：AllGather 超时及真实设备映射

本次约 10:54 起再次只读登录五号机，重查 12 份日志、容器源码及保留的 CANN plog；没有重启服务、重置 NPU 或启动训练。启动脚本及 v10/v11 日志 SHA256 与附录一致。

**直接失败原因已缩小为权重同步期间的 HCCL 集合通信超时。** v10 第 72307 行、v11 第 72459 行、debug 第 159962 行均明确记录 `Communication_Error_Timeout(EI0002)`，操作是 `AllGather_group_name_0ringAllGatherSmallCount_device`，等待对端 `remote rank:[0]`；随后的设备记录为 `timeout=1836s`，再出现 `rtMemcpyAsync` / 事件查询 `507001`。因此不是只有“未知 NPU 错误”可用，也不能把日志中 16 字节拷贝失败解释成模型显存不足。

v11 的时间线（服务器北京时间）：

| 时间 | 证据 |
|------|------|
| 14:08:16 | rank 0/1 plog 建立训练 AllGather 资源 |
| 14:11:16 起 | rank 1 每三分钟记录同一 stream 46、task 2957 未完成 |
| 14:26:33 | rank 0 首先被心跳标记 STUCK |
| 14:28:52 / 14:28:54 | chip 15 / 8 随后被标记 STUCK |
| 14:39:07 | 主日志报 EI0002，等待 rank 0，设备超时 1836 秒 |
| 14:39:08 | 多个训练 worker 连带报 507001 |

**脚本中的超时值没有在所查 worker 中生效。** `plog-3160854_20260913140515100.log:126–127` 和 rank 1 对应日志明确写出 `HCCL_CONNECT_TIMEOUT ... [7200]s`、`HCCL_EXEC_TIMEOUT set by default to [1836]s`。不能以 shell 中导出 60000 就宣称实际通信等待阈值为 60000 秒；具体是在 Ray 环境传递还是初始化覆盖时丢失，尚未定位。延长超时也不能修复停住的 rank。

**历史设备分配已有部分直接证据。** rank 0 plog 第 373 行记录 FSDP 训练组 `0/0 ... 7/7`；`plog-3162498_20260913140608982.log:333` 记录权重同步组 `0/0; 1/8; 2/9; ...; 8/15`。这证明 v11 训练组在 chip 0–7，权重接收组在 chip 8–15；同步组包含 trainer rank 0 是实现设计，并非抢卡证据。它补充了前次核验缺失的历史映射，但仍不替代全部 vLLM 进程的设备快照。

源码表明所有训练 rank 都要消费 DTensor 参数生成器以参与 AllGather，只有 rank 0 负责向 rollout 广播：`hccl_checkpoint_engine.py:242–245`、`:254–270`，`transformer_impl.py:971–978`。本次报错来自非零 rank 消费生成器的分支。优先排查 rank 0 在训练聚合与 rollout 通信之间的阻塞、执行顺序和同步；**现有 plog 没有给出 rank 0 当时的 Python 栈或逐算子完整轨迹，尚不能宣称已经证明某一行代码死锁、offload 缺陷或硬件故障。** 所查 rank 0 plog 最后保留的是 AllGather 建链、心跳 STUCK 和退出后的 socket 警告。

华为对 [EI0002 的说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/hccl/hcclug/hcclug_000036.html)也要求追查未进入同步的 rank、任务序列和通信链路；错误本身不唯一指向硬件。

现场 `npu-smi info` 显示 16 chip 均为 OK、AICore 利用率为 0、无运行中的 NPU 进程，HBM 约 2.8–3.1 GiB；宿主机 uptime 79 天。**确认训练失败，未发现机器当前持续故障或必须重启的证据；当前健康快照不能反证周末没有瞬时设备异常。**

### 3.4 核验新增发现：数据与奖励接口不匹配

**数据结构问题：** 当前数据仅有顶层 `ground_truth`，而容器的 `naive.py:121` 读取 `data_item.non_tensor_batch["reward_model"]["ground_truth"]`。所查 `RLHFDataset` 没有对应字段转换。需补齐框架要求的结构，例如保留顶层字段时额外生成：

```python
{"reward_model": {"style": "rule", "ground_truth": {"golden_answer": "2"}}}
```

**函数接口问题：** 现有函数是 `compute_score(response, ground_truth, **kwargs)`，框架调用传入 `data_source`、`solution_str`、`ground_truth`、`extra_info`。在原容器中仅用合成答案执行 CPU 调用，已复现：

```text
TypeError: compute_score() missing 1 required positional argument: 'response'
```

后续应把接口适配为 `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`，内部使用 `solution_str`。这两项属于进入奖励阶段前需要修复的独立问题，不是此次 507001 的已证根因。

此外，现有数学评分是数值近似或字符串匹配，不是完整符号等价判定；原附录解析 `-\frac{1}{2}` 时正则没有把外层负号用于结果，混合 `boxed/fbox` 的提取也不是按全文位置取最后一个答案。修复接口后仍须验证评分样例，不能直接据此给出 MATH 准确率。

## 四、关键发现：并行策略与资源分配

### 4.1 单机 8+8 的可行性

服务器的 `experimental/separation/utils.py` 按 `trainer.nnodes × trainer.n_gpus_per_node` 创建训练资源池；rollout replica 按 TP×DP×PP 计算设备数并创建资源池。本方案 Trainer 8，Rollout 4×2×1=8，在 16 chip 上设备数预算成立。

但 Ray 的 `llin_trainer` / `llin_rollout` 是节点资源标签，不是指定 chip 0–7 或 8–15 的掩码。同一节点同时拥有两种标签，仍需要 Ray 设备分配、容器可见设备和进程实际设备映射共同保证不重叠。8+8 只是资源数量配置，不能省略映射验证。再次核验已从 v11 plog 确认训练组与权重接收组的上述分配，详见 3.3。

### 4.2 与已验证双机方案的差异

| 维度 | 单机 9B 现有尝试 | 双机 27B 已有工程记录 |
|------|------|------|
| 模型 | Qwen3.5-9B | Qwen3.6-27B |
| 入口 | one-step-off-policy | fully-async-policy |
| 训练后端 | FSDP2 | Megatron |
| 训练并行 | FSDP size 8，未设置训练 TP | TP4×PP2×CP2，训练 DP1 |
| 设备分配 | 同机 8+8，v11 训练组/权重接收组映射已从 plog 补证 | 5 号机 16 训练、6 号机 16 rollout |
| rollout | TP4、DP2 | TP8、DP2 |
| 证据状态 | 权重同步阶段失败，无完成步证据 | 有 Step120 日志、同步记录及检查点 |

`rollout.mode=async` 只说明 rollout 接口模式，不能据此把 one-step-off-policy 写成与 fully-async-policy 完全相同的调度机制。

## 五、关键问题解答

### 问题 1：NPU 上是否只支持 Megatron？

不是。veRL 官方快速上手列出了 vLLM + FSDP2 路线，官方 CI 也包含 Qwen3.5-2B FSDP2 NPU 脚本。但这些材料不等于本容器的 Qwen3.5-9B、one-step 分离式训练已经通过验证。[官方快速上手](https://verl.readthedocs.io/en/latest/ascend_tutorial/zh/get_start/quick_start.html)、[官方 Ascend CI](https://github.com/verl-project/verl/blob/main/.github/workflows/nightly_ascend.yml)。

### 问题 2：是否可以单机一半 rollout、一半训练？

可以作为本报告的目标方案，当前代码具备分离资源池机制，卡数预算也匹配。当前证据尚不能把“可配置”升级成“已跑通”，需要补齐数据/奖励接口、设备映射、权重同步及真实训练步验收。

### 问题 3：507001 是否由 FSDP2 导致？

本次异常出现在 FSDP 权重导出/恢复路径；尚无对照实验证明后端本身是根因。所核对的 27B Step100→120 日志未检出 507001，不能外推整个项目历史都没出现过此错误。

### 问题 4：是否改为 Megatron TP2×PP2×CP2？

该组合占 8 chip，算术成立，但本次没有验证 9B 的 Bridge 映射、注意力/线性注意力分片、PP 切分、CP 算子及显存预算。可列为备选，不能直接复制 27B 配置并宣布可用。优先将已存在的 FSDP2 8+8 方案的阻塞点逐项缩小；如转 Megatron，需作为独立适配实验。

## 六、推荐配置与验收安排

### 6.1 单机基线

保留 Qwen3.5-9B、Trainer 8 chip / FSDP2、Rollout 8 chip / TP4×DP2、短上下文 3072、每题 4 条、计划 3 步。第二章给出拓扑摘录；完整参数以附录所定位服务器脚本为核验对象。该脚本当前仍有阻塞，不作为“拿来即跑”的成品交付。

### 6.2 下一次运行前后需要的证据

1. 数据适配后确认 100 行可被实际 dataset 读取，聊天模板不导致样本全被过滤；用合成答案通过真实 reward manager 的正确、错误、空回答、分数负号用例。
2. 在没有资源冲突的运行窗口，保存 Ray 节点/可用资源、各 worker 的可见设备与实际 chip ID；证明训练和 rollout 各 8 且不重叠。
3. 单独验证初始化权重同步及一次真实 rollout；如继续出现 DTensor 错误，围绕参数 offload、聚合和 HCCL 路径做单变量诊断。
4. 完成 3 个实际 optimizer step，记录奖励、组内优势、loss、梯度与参数版本，确认不是全零优势或跳过更新；保存可读检查点并验证新权重可用于 rollout。

训练与验证使用同一 100 题只适合冒烟，不构成泛化评测；在上述验收完成前，本报告状态保持“单机训练闭环待验证”。

## 七、附录

### A. 核验位置与环境

原始材料：[单机原报告](sources/single_machine_grpo_analysis_20260913.md)。保留原文作为来源，不覆盖原作者的历史记录。

5 号机宿主项目根目录：`/data3/llin/qwen3.6-27b-verl-grpo`；容器内根目录：`/workspace/llin-verl-grpo`。核验时版本：veRL 0.9.0.dev0、PyTorch 2.9.0、torch-npu 2.9.0.post2、Ray 2.56.1、vLLM 0.18.0+empty、Transformers 5.10.4。这是 9 月 14 日现场版本，不是所有历史尝试的冻结版本证明。

### B. 证据身份

| 文件（相对项目根目录） | SHA256 |
|------|------|
| `scripts/run_math_grpo_smoke.sh` | `dd6c480b7d2fa08e258f8fc96ba70d1a9d9eb71a56e8c247f9d220660327ee66` |
| `runs/math_grpo_smoke_v10.log` | `ec80f632d837208ba044a05ebb213284b542d85da312844c1974e2a35688a9b2` |
| `runs/math_grpo_smoke_v11.log` | `c63725e1681b0df8a443c385168e7fc42642b9dffac45c5f604ba93176097458` |

### C. 与另一份报告的关系

双机配置、真实完成证据和迁移边界见[多机多卡 27B 报告](multi_machine_27b_grpo_report_20260914.md)。两份报告分别描述不同模型与工程状态，不把双机成功记录移植为单机成功结论。

---

**报告完成时间**：2026-09-14

**核验方式**：5 号机只读现场核验、CPU 合成奖励接口调用、官方资料交叉核对。
