---
报告日期: 2026-09-13
更新类型: 单机多卡 GRPO 训练情况报告（待 llin 分析）
更新人: renjunxiang
状态: 待 llin 分析解答
---

# 单机多卡 GRPO 训练情况报告

## 一、背景

我们希望在单台服务器（huawei-5）上实现 GRPO 训练，验证单机多卡架构的可行性。理想配置是：
- 单机 16 chip Ascend910
- 8 chip 用于 Trainer（FSDP2）
- 8 chip 用于 Rollout（vLLM）
- 使用 Qwen3.5-9B 基座模型
- 使用数学推理数据（GSM8K + MATH）进行冒烟测试

## 二、已完成的工作

### 2.1 基础架构搭建

✅ **Ray 集群配置**
- 启动 Ray head（huawei-5）
- 添加 `llin_trainer` 和 `llin_rollout` 资源标签
- Ray 集群正确识别 16 NPU 资源

✅ **训练脚本编写**
- 创建 `run_math_grpo_smoke.sh`
- 配置 FSDP2 并行策略
- 实现数学 reward 函数（`math_reward.py`）
- 准备训练数据（100 条数学题）

✅ **环境验证**
- 使用与 llin 相同的容器镜像：`llin-verl-a3:20260730`
- NPU 健康状态检查：全部 OK
- 模型权重加载成功（Qwen3.5-9B, 19GB）
- vLLM 服务启动成功

### 2.2 训练脚本核心配置

```bash
# NPU 分配
export TRAIN_NPUS=8
export ROLLOUT_NPUS=8
export ROLLOUT_NNODES=1
export TRAIN_TP=4
export ROLLOUT_TP=4

# 关键参数
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.actor.fsdp_config.fsdp_size=${TRAIN_NPUS}
actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
actor_rollout_ref.rollout.data_parallel_size=$((ROLLOUT_NPUS / ROLLOUT_TP))
trainer.n_gpus_per_node=${TRAIN_NPUS}
rollout.n_gpus_per_node=${ROLLOUT_NPUS}

# Reward 函数
reward.custom_reward_function.path="${PROJECT_ROOT}/llin_verl/math_reward.py"
reward.custom_reward_function.name=compute_score
```

## 三、遇到的问题

### 3.1 问题描述

训练能够成功启动，但**无法完成实际训练步骤**。具体表现：

1. **初始化阶段卡住**：训练进程运行，但一直停留在初始化阶段（vLLM 加载模型权重后无进展）
2. **间歇性 NPU 错误**：有时会遇到 NPU 内部错误（error code 507001）

### 3.2 NPU 错误详情

```
[Error]: An internal error occurs in the task scheduler module on the device.
EE9999: Inner Error!
EE9999[PID: 3124257] 2026-09-13-11:22:04.971.538 (EE9999):  
rtEventQueryStatus execution failed, reason=tsfw unknown error
[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]
[Query][Status]query event recorded status failed, runtime result = 507001
```

**错误分析**（来自 Kimi 调研）：
- `507001`：Host 侧调用同步/查询类接口失败的通用失败码
- `EE9999`：CANN 的兜底错误，本身不含根因
- `tsfw unknown error`：任务调度固件上报未分类错误，真实原因被归为"未知"

**可能的根因**（按概率排序）：
1. HCCL 集合通信超时/对端退出
2. 设备侧 OOM / 显存碎片
3. sleep/wake（free_cache_engine）已知缺陷
4. 版本不匹配
5. 硬件/驱动层问题

### 3.3 已尝试的解决方案

- ✅ 设置 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
- ✅ 使用诊断模式（`ASCEND_LAUNCH_BLOCKING=1`）
- ✅ 检查 NPU 健康状态
- ✅ 清理残留进程后重试
- ❌ 问题仍然存在

## 四、关键发现：并行策略差异

### 4.1 我们的配置（FSDP2）

```bash
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.actor.fsdp_config.fsdp_size=8
```

### 4.2 llin 的配置（Megatron）

从 llin 的训练脚本 `run_pi_grpo_fully_async_tp4_pp2_cp2.sh` 中提取：

```bash
# 隐含的 strategy 设置（通过 megatron.* 参数推断）
actor_rollout_ref.actor.strategy=megatron

# Megatron 并行拓扑
TRAIN_TP=4
TRAIN_PP=2
TRAIN_CP=2
TRAIN_NPUS=16  # 4 * 2 * 2 = 16

# Megatron 特定配置
actor_rollout_ref.actor.megatron.use_mbridge=True
actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}"
actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}"
actor_rollout_ref.actor.megatron.context_parallel_size="${TRAIN_CP}"
actor_rollout_ref.actor.megatron.param_offload=False
actor_rollout_ref.actor.megatron.optimizer_offload=True
actor_rollout_ref.actor.megatron.grad_offload=True
actor_rollout_ref.actor.megatron.dtype=bfloat16
actor_rollout_ref.actor.megatron.use_distributed_optimizer=True

# 额外的 Transformer 配置
+actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
+actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=kvallgather_cp_algo
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
+actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True
```

### 4.3 核心差异

| 维度 | 我们的配置 | llin 的配置 |
|------|----------|------------|
| 并行策略 | FSDP2 | Megatron |
| 并行拓扑 | fsdp_size=8 | TP=4, PP=2, CP=2 |
| NPU 分配 | 8 chip trainer + 8 chip rollout | 16 chip trainer + 16 chip rollout |
| Megatron 特性 | 无 | use_mbridge, optimizer_offload, grad_offload, recompute, flash_attn, sequence_parallel |

## 五、待解答的问题

### 问题 1：verl 在 NPU 上是否只支持 Megatron 后端？

我们的 FSDP2 配置能够启动但无法完成训练，而 llin 的 Megatron 配置在多机环境下成功运行。

**问题**：
- verl 的 GRPO 实现在 NPU 上是否只完整支持 Megatron 后端？
- FSDP2 后端在 NPU 上是否有已知问题或未充分测试？
- 是否有 verl 官方文档说明 NPU 上的推荐后端？

### 问题 2：单机架构是否可行？

llin 的多机架构（huawei-5 trainer + huawei-6 rollout）已成功运行。我们希望在单机上实现类似功能。

**问题**：
- 单机 16 chip 是否可以配置为：8 chip trainer + 8 chip rollout？
- 如果可以，应该使用什么并行拓扑（TP/PP/CP 如何分配）？
- 单机架构是否需要特殊的 HCCL 或 Ray 配置？

### 问题 3：NPU 507001 错误的根因

我们遇到的 NPU 内部错误（507001 + tsfw unknown error）是偶发的，有时训练会卡在初始化，有时会报错退出。

**问题**：
- llin 在多机训练时是否遇到过类似的 NPU 错误？
- 如果有，是如何解决的？
- 这个错误是否与并行策略（FSDP2 vs Megatron）相关？

### 问题 4：单机 Megatron 配置建议

如果 verl 在 NPU 上确实需要 Megatron 后端，我们需要在单机上配置 Megatron。

**问题**：
- 单机 16 chip 如何配置 Megatron 并行（TP/PP/CP）？
- 是否可以使用 TP=2, PP=2, CP=2（共 8 chip）用于 trainer？
- Rollout 的 TP=4, DP=2 配置在单机上是否需要调整？
- 是否需要特殊的 Megatron 配置（如 use_mbridge, optimizer_offload 等）？

## 六、期望的输出

请 llin 帮助解答以上问题，并提供：

1. **verl 在 NPU 上的后端支持情况**：FSDP2 vs Megatron
3. **单机架构的可行性分析**：是否可以在单机上运行 GRPO
3. **推荐的单机配置**：如果可行，提供完整的训练脚本配置
4. **NPU 错误的排查建议**：如果是我们配置问题，指出具体问题点

## 七、附录

### A. 完整的训练脚本

文件位置：`/workspace/llin-verl-grpo/scripts/run_math_grpo_smoke.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# NPU 分配（单机 16 chip，8 chip 训练 + 8 chip rollout）
export TRAIN_NPUS=8
export ROLLOUT_NPUS=8
export ROLLOUT_NNODES=1
export TRAIN_TP=4
export ROLLOUT_TP=4

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
VERL_ROOT="${VERL_ROOT:-/verl}"
MODEL_PATH="${MODEL_PATH:-/workspace/llin-verl-grpo/models/Qwen3.5-9B}"
DATA_FILE="${DATA_FILE:-${PROJECT_ROOT}/data/math_smoke.parquet}"
RUN_NAME="${RUN_NAME:-math-grpo-smoke-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

export PYTHONPATH="/vllm-ascend:/vllm:${PROJECT_ROOT}:${PYTHONPATH:-}"
export RAY_ADDRESS=192.168.202.5:26379
export LLIN_PIN_RAY_ROLES=1
export LLIN_TRAINER_RESOURCE=llin_trainer
export HCCL_EXEC_TIMEOUT=60000
export HCCL_CONNECT_TIMEOUT=7200
export TOKENIZERS_PARALLELISM=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

cd "${VERL_ROOT}"

echo "=== Math GRPO Smoke Test ==="
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Training NPUs: ${TRAIN_NPUS}"
echo "Rollout NPUs: ${ROLLOUT_NPUS}"
echo ""

python3 -m verl.experimental.one_step_off_policy.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA_FILE}" \
  data.val_files="${DATA_FILE}" \
  data.train_batch_size=4 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.dataloader_num_workers=4 \
  data.return_raw_chat=True \
  data.return_multi_modal_inputs=False \
  data.truncation=error \
  actor_rollout_ref.actor.strategy=fsdp2 \
  critic.strategy=fsdp2 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${TRAIN_NPUS}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.data_parallel_size="$((ROLLOUT_NPUS / ROLLOUT_TP))" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_model_len=3072 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.enable=False \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2560 \
  reward.custom_reward_function.path="${PROJECT_ROOT}/llin_verl/math_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger='["console"]' \
  trainer.project_name=math-qwen35-verl-grpo \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
  trainer.rollout_data_dir=null \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=3 \
  trainer.resume_mode=disable \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${TRAIN_NPUS}" \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node="${ROLLOUT_NPUS}" \
  "$@"
```

### B. 数学 Reward 函数

文件位置：`/workspace/llin-verl-grpo/llin_verl/math_reward.py`

```python
"""Math reward function for GRPO training.

Extracts the final answer from model response and compares with golden answer.
Returns 1.0 for correct, 0.0 for incorrect.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any


def _extract_braced(text: str, marker: str) -> list[str]:
    values: list[str] = []
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            break
        cursor = index + len(marker)
        depth = 1
        while cursor < len(text) and depth:
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            values.append(text[index + len(marker) : cursor - 1])
        start = max(cursor, index + len(marker))
    return values


def extract_explicit_answer(text: str) -> tuple[str, bool]:
    """Return the last explicit final answer and whether one was present."""
    if not text:
        return "", False
    
    boxed = _extract_braced(text, r"\boxed{") + _extract_braced(text, r"\fbox{")
    if boxed:
        return boxed[-1].strip(), True
    
    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if tagged:
        return tagged[-1].strip(), True
    
    cues = re.findall(
        r"(?:final\s+answer|answer|答案)\s*(?:is|为|[:：=])?\s*(.+)",
        text[-1024:],
        re.IGNORECASE,
    )
    if cues:
        return cues[-1].strip().splitlines()[0].strip(), True
    
    return "", False


def normalize_math_text(value: str) -> str:
    value = value.strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if value.startswith(left) and value.endswith(right) and len(value) >= len(left) + len(right):
            value = value[len(left) : -len(right)].strip()
    
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\,", "").replace(r"\!", "")
    value = value.replace(" ", "").replace("\n", "")
    return value.strip(".;。")


def parse_number(value: str) -> float | None:
    value = normalize_math_text(value).replace(",", "")
    
    percent = value.endswith(r"\%") or value.endswith("%")
    if value.endswith(r"\%"):
        value = value[:-2]
    elif value.endswith("%"):
        value = value[:-1]
    
    frac = re.fullmatch(r"[-+]?\s*\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", value)
    try:
        if frac:
            result = float(Fraction(int(frac.group(1)), int(frac.group(2))))
        elif re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", value):
            result = float(Fraction(value))
        else:
            result = float(value)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    
    return result / 100.0 if percent else result


def compute_score(response: str, ground_truth: dict[str, Any], **kwargs) -> float:
    """Compute reward score for math problem.
    
    Args:
        response: Model's response text
        ground_truth: Dict with 'golden_answer' key containing the correct answer
    
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    golden_answer = ground_truth.get("golden_answer", "")
    if not golden_answer:
        return 0.0
    
    extracted, has_answer = extract_explicit_answer(response)
    if not has_answer:
        return 0.0
    
    extracted_num = parse_number(extracted)
    golden_num = parse_number(golden_answer)
    
    if extracted_num is not None and golden_num is not None:
        if abs(extracted_num - golden_num) < 1e-3:
            return 1.0
        return 0.0
    
    extracted_norm = normalize_math_text(extracted)
    golden_norm = normalize_math_text(golden_answer)
    
    if extracted_norm == golden_norm:
        return 1.0
    
    return 0.0
```

### C. 训练日志样例

**成功启动但卡住**：
```
=== Math GRPO Smoke Test ===
Model: /workspace/llin-verl-grpo/models/Qwen3.5-9B
Data: /workspace/llin-verl-grpo/data/math_smoke.parquet
Output: /workspace/llin-verl-grpo/runs/math-grpo-smoke-20260913-140323
Training NPUs: 8
Rollout NPUs: 8

Using dataset class: RLHFDataset
Total training steps: 3
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.77it/s]
[transformers] `Qwen2VLImageProcessorFast` is deprecated...
# 之后无进展，一直停留在初始化阶段
```

**NPU 错误**：
```
[rank:2]:[3124257] [2026-09-13 11:22:04:973] torch.distributed: 
[ERROR] [3124979] Find exception when finishedNPUExecutionInternal, 
query:../torch_npu/csrc/core/npu/NPUEvent.cpp:94 
NPU function error: acl::AclQueryEventRecordedStatus(event_, &currStatus), 
error code is 507001

[ERROR] 2026-09-13-11:22:04 (PID:3124257, Device:0, RankID:2) ERR00100 PTA call acl api failed
[Error]: An internal error occurs in the task scheduler module on the device.
EE9999: Inner Error!
EE9999[PID: 3124257] 2026-09-13-11:22:04.971.538 (EE9999):  
rtEventQueryStatus execution failed, reason=tsfw unknown error
```

---

**报告完成时间**：2026-09-13  
**期望反馈**：请 llin 分析以上问题并提供建议
