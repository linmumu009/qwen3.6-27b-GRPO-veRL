# 五号机单机 9B GRPO 实际复现结果

日期：2026-09-15。依据 [9 月 14 日核验报告](single_machine_9b_grpo_report_20260914.md)和 [9 月 13 日原报告](sources/single_machine_grpo_analysis_20260913.md)实际运行。

## 结论

已复现初始化权重同步阻塞，并通过同一次运行的单变量干预定位到 **IPC 地址的物理编号与逻辑编号错配**。只补齐地址映射，阻塞即解除，程序继续到第二个确定故障：**3880 MiB 的 FP32 embedding 张量放不进 2560 MiB 的权重同步桶**。因此不能把本次问题归结为 NPU 硬件损坏或 FSDP2 不支持。

本次未等待 1836 秒再次触发设备超时，而是在抓取实时堆栈后解除阻塞；复现的是历史报告中的同步停滞及其故障链，不宣称此次重新产生了 507001。无完成 optimizer step 证据，无检查点，未跑通训练闭环。

## 运行条件与控制变量

- 五号机原容器 `llin-verl-trainer-m05-20260730`，原 Ray head 单节点 16 NPU 全部可用；启动前 NPU 全部健康、无计算进程，主机可用内存约 1.9 TiB，磁盘余量约 665 GB。
- 原脚本 `scripts/run_math_grpo_smoke.sh` SHA256 仍为 `dd6c480b7d2fa08e258f8fc96ba70d1a9d9eb71a56e8c247f9d220660327ee66`，复制进独立运行目录后执行。
- 保持 Qwen3.5-9B、FSDP2 8 chip、rollout TP4/DP2 8 chip、原始 100 题、原始奖励函数、offload 和 2560 MiB 同步桶不变。
- 仅更改运行名称、输出目录、驱动日志缓冲，增加外层 1200 秒执行上限；原始失败会先于该上限终止作业。没有为此重启 Ray、重置设备或修改共享源码。
- 容器内证据目录：`/workspace/llin-verl-grpo/runs/math-repro-20260915-1005`；宿主目录：`/data3/llin/qwen3.6-27b-verl-grpo/runs/math-repro-20260915-1005`。目录后缀是标识，不作为精确启动时间。

## 1. 第一处故障：权重两端 IPC 地址不一致

实时堆栈将等待链明确拆成四段：

| 进程 | 捕获到的等待位置 | 含义 |
|---|---|---|
| Trainer rank 0，PID 3448958 | `hcclBroadcast → send_weights:263` | 正在向 rollout 广播同步桶 |
| Trainer rank 1，PID 3448959 | DTensor 参数迭代，`transformer_impl.py:975` | 后续聚合/参数恢复等待 rank 0 继续 |
| CheckpointEngineWorker，PID 3450598 / 3450609 | `BucketedWeightSender._init_buffer:180 → socket.send_pyobj` | 尚未与本地 vLLM 完成握手，因此还未消费 HCCL 接收生成器 |
| vLLM Worker TP0，PID 3451689 | `BucketedWeightReceiver._init_buffer:306 → recv_pyobj` | 等待本地权重发送端，但地址错误 |

本次 Ray job 为 `08000000`。发送端实际监听 `/tmp/rl-colocate-zmq-08000000-replica-0-rank-0.sock` 至 `rank-7.sock`，已从 `/proc/net/unix` 核对。

vLLM 两个 DP 组的 `ASCEND_RT_VISIBLE_DEVICES` 实际分别为 `8,9,10,11` 和 `12,13,14,15`。容器 `utils.py:45–62` 的 `_resolve_vllm_weight_sync_local_rank()` 直接返回可见设备列表中的物理编号；`_get_zmq_handle():383–398` 再把这个编号用于 socket 路径，因而接收端寻找 `rank-8.sock` 至 `rank-15.sock`。job 与 replica 标识一致，差异在 rank 编号。

对应逻辑来自仓库 [patch_verl_vllm_dp_weight_sync.py](../scripts/patch_verl_vllm_dp_weight_sync.py) 的 `NEW` 补丁。它在 rollout 独占整机、物理编号与逻辑编号一致时可用；在本次 rollout 只占后半组 chip 的情况下不成立。不能只去掉 DP 偏移，否则两个 DP 组又会重复使用 0–3。

### 同一次运行的因果验证

北京时间 10:09:31（记录为 `2026-09-15T02:09:31.958751+00:00`），仅给本 job 的八个 socket 创建临时别名：`rank-(8+i).sock → rank-i.sock`，i=0…7。创建前逐一确认目标是真实 socket、别名路径不存在；不改任何模型/通信算法/桶大小参数，也未重启进程。

创建别名后程序随即继续，下一次检查已得到 embedding 大于桶的 AssertionError；原来的 HCCL 广播、IPC 发送及接收等待不再停留。这构成地址错配导致本次阻塞的直接干预证据。别名只用于诊断，退出后已全部移除，不作为长期修复。

## 2. 第二处故障：完整 FP32 参数大于同步桶

`driver.log:72428`：

```text
AssertionError: Weight model.language_model.embed_tokens.weight(torch.Size([248320, 4096]), torch.float32) is too large to fit in the bucket.
```

计算为 `248320 × 4096 × 4 = 4,068,474,880 bytes = 3880 MiB = 3.7890625 GiB`，而脚本设置桶为 `2560 MiB = 2.5 GiB`。这不是“16 字节参数拷贝导致显存不足”，而是单个完整张量超出同步器的容量合同。

`hccl_checkpoint_engine.py:254–279` 在发现当前张量装不下时，先广播现有桶，再断言单张量能否放入。此处在张量过大、桶尚为空时也可能先发空桶，导致原本应该快速报出的容量错误被第一处 IPC 阻塞掩盖。此次恢复通信后实际抛出上述断言。

后续修正方向：启动前检查最大传输张量；至少使桶能容纳该 3880 MiB 张量，或实现正确的分块传输/经验证的传输 dtype 转换。4096 MiB 是容纳已知 embedding 的候选值，不是已验证完整模型都可用的配方；还需检查其余张量和显存，因为 HCCL 两个缓冲区本身就占用 `2 × bucket_size`，另有 IPC 缓冲及模型/KV 开销。

## 3. 其他问题与后续验收

本次 worker 的限定环境检查再次确认：`HCCL_EXEC_TIMEOUT` 缺失、`HCCL_CONNECT_TIMEOUT=7200`。`get_ppo_ray_runtime_env()` 显式传递了 PYTHONPATH 等字段，却未传递脚本中的 HCCL 超时变量；独立启动的 Ray worker 不会自动继承驱动 shell 的全部环境。应通过运行级 `ray_kwargs.ray_init.runtime_env.env_vars` 显式传递并在 worker 内核验；延长超时本身不修复错配。

前次 CPU 核验发现的数据缺 `reward_model.ground_truth`、奖励函数 `response` 与框架 `solution_str` 参数不兼容，本次为保持原配置未修正，也没有执行到奖励阶段。这两项仍是后续阻塞，不能把它们写成本次新发生的异常。

建议修正顺序：

1. 按 rollout 父进程的完整设备列表，把每个 DP 子进程物理 chip 映射回 rollout 节点内逻辑 rank；覆盖整机、非零起点、非连续设备及多 replica，保留 job 隔离。不要简单写死减 8。
2. 在广播之前校验单张量容量，避免空桶广播掩盖容量错误；确定同步桶或分块方案，并核验显存预算。
3. 修正数据与奖励合同，再逐项验证首次权重同步、一次 rollout、奖励计算、真实 optimizer 更新及检查点。

本次范围是复现定位；尚未部署上述正式修复，也没有把临时地址别名当成已交付修复。

## 证据与收尾

原始驱动日志、`stack_sync_*.txt`、`stack_local_*.txt`、设备映射、干预记录及汇总保留在服务器独立目录。Git 仅记录诊断结果，不回传样本正文或全部运行日志。

| 证据 | SHA256 |
|---|---|
| `driver.log` | `9f97270f7893d7c08bb3a6b7c709dc7f491cde5d0a5d6b10edcd1ab1591a477c` |
| `stack_sync_3448958.txt` | `5f4e1d8744c7cdb80e90633280134d25365aa93dfe6f296fcd9620302a867636` |
| `stack_sync_3448959.txt` | `3c6c393ad4aef2275e8b33b7a2a19c6856e9e3ec1e3c40cb5e6ba36df7576980` |
| `stack_sync_3450598.txt` | `e74322ce4f83349a15aedfb420f1108a14d5836f7ddb5de5c0a258bcedd9459e` |
| 容器 `vllm_rollout/utils.py` | `198f2877dc9752df1e78338a4fec06b6d511c228d650e097fbaaa8f608919b14` |

10:10 左右收尾检查：无 `training/global_step` 指标、无 `global_step_*` 检查点；本次训练与推理 worker 均已消失，NPU 全部健康、无计算进程，HBM 回到约 2.8–3.1 GiB。未重启机器，未修改共享容器源码；历史报告原文保留。
