# Step-100 checkpoint 与 Hugging Face 导出交付

日期：2026-08-06

## 交付结果

可继续训练的 Megatron checkpoint 保持原样：

```text
/data3/llin/qwen3.6-27b-verl-grpo/runs/llin-v15-dwh-bossreward-12groups-100step-20260805-03/checkpoints/global_step_100
```

- 格式：`megatron_dist_checkpoint`
- step：`100`
- model shards：`32`
- model shard bytes：`54,720,369,973`
- 完整性门禁：通过

供老板直接评测的独立 Hugging Face 目录：

```text
/data3/llin/qwen3.6-27b-verl-grpo/exports/llin-qwen3.6-27b-grpo-step100-hf-20260806
```

- safetensors shards：`15/15`
- safetensors bytes：`56,484,503,104`
- tensor keys：`1199/1199`
- language layers：`0–63`
- 缺失/多余 tensor：`0/0`
- shape mismatch：`0`
- Qwen3.6 GDN 的 `A_log`、`conv1d`、`in_proj_qkv`、`linear_attn.norm` 均存在
- 配置、tokenizer、chat template、generation config 和导出清单均已包含

## 转换边界

通用 veRL merger 不包含 Qwen3.6 GDN 的完整映射，不能用于本模型。训练时使用的 Megatron Bridge 对 64 层主模型映射完整，但当前 vendored 版本明确未实现 MTP 导出映射。

本次训练配置没有启用 MTP，Megatron checkpoint 元数据中也没有 MTP 参数。因此导出时：

- `1184` 个主模型 tensor 来自 step-100 checkpoint；
- `15` 个未参与训练的 `mtp.*` tensor 原样继承基础模型；
- 若未来 checkpoint 包含 MTP 参数，导出器会拒绝使用该 fallback，避免覆盖训练权重。

输出中 `414` 个 tensor 的 dtype 与基础 BF16 文件不同，主要是 Megatron 以 FP32 保存的归一化等参数；所有 shape 完全一致。TP8 vLLM 已实际加载成功，因此这些 FP32 参数不是格式兼容问题，也不应为追求文件 dtype 一致而降低训练后精度。

## 独立加载验收

在一个全新的 vLLM 进程中使用 5 号机 0–7 号 NPU、TP8、BF16、`max_model_len=4096` 加载导出目录：

- 15/15 safetensors 分片全部读取；
- 每个 TP rank 加载约 `6.5443 GB` 模型权重；
- KV cache 可用约 `40.69 GiB/卡`；
- 最小请求成功生成 `HF export works`；
- 进程退出码 `0`；
- 验收结束后 0–7 号 NPU 无运行进程。

老板可直接将上述 HF 目录传给 Transformers 或 vLLM。若使用与正式训练相同的长上下文评测，应按现有容量配置显式设置所需 `max_model_len`、TP 和显存利用率；本次最小门禁只用 4096，目的是验证模型目录可独立加载和前向生成。

## 可复现入口

- `scripts/export_megatron_dist_to_hf.py`：只读恢复 Megatron distributed checkpoint，写入独立临时目录，严格校验后才原子发布 HF 目录。
- `scripts/smoke_test_hf_vllm.py`：全新 vLLM 加载和最小生成门禁。
- HF 目录内 `llin_export_manifest.json`：记录源 checkpoint、基础模型、MTP fallback 和完整 tensor 校验结果。

本次转换失败产生的四个 `.incomplete-*` 目录已清理，释放约 49 GiB；正式 HF 模型、转换日志、vLLM 验收日志与原 Megatron checkpoint 均保留。
