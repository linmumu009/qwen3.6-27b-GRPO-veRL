# 单机 Qwen3.5-9B GRPO 三步训练跑通记录

日期：2026-09-15。对应 [故障复现报告](single_machine_9b_grpo_reproduction_20260915.md)。

## 实际结果

五号机原容器内，单机 16 chip 按 Trainer 8 / Rollout 8 分配，FSDP2 + vLLM TP4/DP2 完成 **3 个真实优化步，退出码 0**。不是只加载模型或推进空计数。

| 步 | 平均结果奖励 | 梯度范数 | policy loss | 训练侧分配显存峰值（GiB） |
|---|---:|---:|---:|---:|
| 1 | 0.25 | 1.237004 | 0.031070 | 46.4843 |
| 2 | 0.125 | 0.709090 | -0.038381 | 55.2360 |
| 3 | 0.1875 | 0.485253 | -0.036316 | 55.2360 |

原 100 条数学题中按训练加载器采样，batch 4、每题 4 响应、共 3 步；学习率 `1e-6`，prompt/response 上限 1024/2048，rollout 上下文 3072。奖励仅为数值或严格文本匹配的工程烟测评分，不是完整 MATH 符号等价评测；本次不证明泛化收益。

首次及训练过程中重复权重同步均通过，每次记录 760 个参数接收完成。该 one-step-off-policy 入口在下一批训练前同步已有参数；本次没有另做最终 Step3 权重的独立推理/断点续训验证。

## 必需修正

1. **IPC 编号转换。** 在 vLLM 父进程保存完整 rollout 设备列表 `VERL_ROLLOUT_LOCAL_DEVICES`；DP 子进程先得到物理 chip，再查其在父列表中的位置作为逻辑 IPC rank。支持非零起点和非连续设备，不写死减 8，保留 job/replica 隔离。
2. **同步桶由 2560 改为 4096 MiB。** 当前最大 FP32 张量为 embedding，3880 MiB。准备脚本读取 safetensors 头校验完整 FP32 参数尺寸，超出桶时在加载模型前失败。
3. **关闭训练侧 CPU offload。** 首个修复尝试已跨过通信和奖励，但在 `offload_fsdp_optimizer` 触发主机锁页分配 `207001`；最终将 actor 的 `param_offload`、`optimizer_offload` 均设 False。9B 此配置的实测训练显存峰值为 55.236 GiB；不要未经验证套用到更大模型/上下文。
4. **数据与奖励接口适配。** 新数据副本补齐 `reward_model.ground_truth`；函数接收 `data_source, solution_str, ground_truth, extra_info`。同时修正负分数、boxed/fbox 全文先后顺序与空回答评分。
5. **显式传入 Ray worker 环境。** `HCCL_EXEC_TIMEOUT=600`、`HCCL_CONNECT_TIMEOUT=300` 通过 runtime_env 传递，已从初始化日志确认；并非只在 driver shell 导出。

保留 FSDP2，不需要为此改为 Megatron。原报告和原始启动脚本保留；所有 veRL 修改仅应用于运行目录的 11 MB 源码副本，未覆盖共享 `/verl/verl`。

## 可复用启动方法

已将以下入口与适配模块部署到五号机项目目录，并纳入 Git：

- [启动入口](../scripts/run_single_host_math_smoke.sh)：检查 NPU 与单机 Ray 空闲后创建新运行目录，保存驱动日志与退出码。
- [准备脚本](../scripts/prepare_single_host_math_smoke.py)：复制当前容器 veRL 源码、检查最大张量、修补地址、转换数据并生成完整参数脚本。
- [地址补丁](../scripts/patch_verl_vllm_device_namespace.py)、[奖励函数](../llin_verl/math_smoke_reward.py)。

在五号机宿主执行（Ray head 需为当前单节点 16 NPU，带 `llin_trainer` 与 `llin_rollout` 标签；入口检查不满足会退出）：

```bash
docker exec llin-verl-trainer-m05-20260730 \
  bash /workspace/llin-verl-grpo/scripts/run_single_host_math_smoke.sh
```

默认每次创建 `runs/math-single-host-时间戳`，不会覆盖成功记录。也可通过 `docker exec -e OUTPUT_DIR=/workspace/llin-verl-grpo/runs/自定新目录 ...` 指定目录。准备脚本要求目录不存在，且仍依赖已核验的原 `scripts/run_math_grpo_smoke.sh`、模型及 100 题数据；不将它包装成通用任意模型启动器。

本次成功运行的完整展开参数保存在 `runs/math-fixed-20260915-v2/run.sh`。最终准备脚本另在 `runs/math-ready-20260915` 完成仅 CPU 的生成与 shell 语法核验，没有额外启动一轮训练。

## 检查点与证据

容器内成功目录：`/workspace/llin-verl-grpo/runs/math-fixed-20260915-v2`。

宿主成功目录：`/data3/llin/qwen3.6-27b-verl-grpo/runs/math-fixed-20260915-v2`。

检查点：`checkpoints/global_step_3/actor`。保存 8 份模型、8 份优化器、8 份 extra_state，另有数据与配置状态，合计 **112,950,688,681 bytes（约 105.2 GiB）**。在 CPU 上逐 rank 以 `torch.load` 读取元数据和优化器 step：

- 每 rank 模型条目 760、优化器状态 760；所有 optimizer step 的最小值/最大值均为 3。
- 所有 extra_state 可读取。模型与优化器大文件采用 mmap 读取，此项不是逐字节 CRC 或逐张量数值完整性扫描。
- 主日志 SHA256：`78ba81f61e78251ec075e23b1a7de659533719085f38b992e58fdad81803026d`。
- 服务器 `verified_summary.json` 保存三步指标、8 rank 读取结果、体积与日志校验值。

13 项针对性单元测试通过，覆盖多种设备映射、补丁幂等、无效父设备列表、奖励接口及负分数等边界。最终脚本通过容器内实际生成与 `bash -n` 检查。首轮保留在 `math-fixed-20260915-v1`，退出码 1，记录 offload 故障，不用于恢复。

结束检查：16 chip 全部健康，8 个设备条目均显示无计算进程。没有重启机器、重置 NPU、删除历史实验或修改其他模型的训练配置。
