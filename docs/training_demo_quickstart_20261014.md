# SFT GRPO CPT 汇报操作速查

用于2026年10月14日工程演示：先分别完成1步训练，确认日志、检查点和服务器负载。使用5号机现成交付包及包内物流算术样例，每类32条训练、8条验证数据。

**执行位置：5号机宿主机终端，目录 `/data3/llin/business_training_delivery`。** 参考环境为单机16个逻辑NPU、每个64GB，主机约2TiB内存。三项依次运行；开始前用 `npu-smi info` 确认设备空闲、`df -h .` 确认输出盘至少90GB空余。每项检查点约55GB，三项结果都会占空间。

入口自动启动容器、将JSONL转成Parquet、加载模型自带的tokenizer，**不用手动分词或启动Ray**。新服务器须先部署完整交付包及导入镜像，见[运维说明](../delivery/business_training_delivery/docs/运维部署与验收说明.md)；只拉Git仓库不能训练。

## 1 SFT

作用：给定问题和示范答案，学习答案。首次直接用已有数据；换数据时按 `datasets/SFT/example.jsonl` 的 `messages` 格式准备。

|需要确认或修改|位置和值|
|---|---|
|数据路径|`datasets/SFT/train.jsonl`、`val.jsonl`；直接替换内容，保留文件名|
|模型路径|下方 `DELIVERY_MODEL_DIR`；默认 `model/qwen3.6-27b`，可改为完整HF模型目录，需含权重、配置及tokenizer|
|训练步数|`Train/SFT/config.yaml` 中 `trainer.total_training_steps=1`、`trainer.save_freq=1`，首次保持默认|

```bash
cd /data3/llin/business_training_delivery
export DELIVERY_MODEL_DIR=model/qwen3.6-27b
bash Train/SFT/run.sh --check && bash Train/SFT/run.sh
```

**看什么：** `--check` 只预检；真实训练日志出现train loss、梯度范数，结束后有 `outputs/SFT/本次编号/checkpoints/global_step_1/`。SFT只监督assistant答案。

## 2 GRPO

作用：同一道题生成多个答案，用奖励评分，再走训练流程。首次用包内算术题，避免临时适配物流沙箱工具和奖励。

|需要确认或修改|位置和值|
|---|---|
|数据路径|`datasets/GRPO/train.jsonl`、`val.jsonl`；照现有train文件的完整行保留 `prompt`、`reward_model.ground_truth` 等字段；该类 `example.jsonl` 仅作示意，不能直接用于训练|
|模型路径|下方 `DELIVERY_MODEL_DIR`，与SFT相同|
|训练步数|`Train/GRPO/config.yaml` 中 `trainer.total_training_steps=1`、`trainer.save_freq=1`，首次保持默认|

```bash
cd /data3/llin/business_training_delivery
export DELIVERY_MODEL_DIR=model/qwen3.6-27b
bash Train/GRPO/run.sh --check && bash Train/GRPO/run.sh
```

**看什么：** 日志完成生成、奖励计算和训练step；检查点在 `outputs/GRPO/本次编号/checkpoints/global_step_1/actor/`。当前每题生成4个答案，奖励函数为 `tools/reward.py`。包内样例历史运行奖励全为1、梯度为0，只能说明工程流程走通；换成沙箱业务题须同步适配奖励和工具，不能只换数据路径。

## 3 CPT

作用：直接用物流原文继续训练。首次用包内原文；换数据时一行一条 `{"text":"物流业务原文……"}`，长文先切段。

|需要确认或修改|位置和值|
|---|---|
|数据路径|`datasets/CPT/train.jsonl`、`val.jsonl`；格式参照 `example.jsonl`|
|模型路径|下方 `DELIVERY_MODEL_DIR`，与SFT相同；三项各自从基础模型启动|
|训练步数|`Train/CPT/config.yaml` 中 `trainer.total_training_steps=1`、`trainer.save_freq=1`，首次保持默认|

```bash
cd /data3/llin/business_training_delivery
export DELIVERY_MODEL_DIR=model/qwen3.6-27b
bash Train/CPT/run.sh --check && bash Train/CPT/run.sh
```

**看什么：** 日志出现train loss和梯度范数；检查点在 `outputs/CPT/本次编号/checkpoints/global_step_1/`。CPT监督原文token，入口自动加结束符，不需要制作问答。

**三项通用：** SFT完整对话、CPT单条原文和GRPO输入上限均为1024 token，超长时先切短；SFT、GRPO训练数据至少4条，建议4的倍数，验证集保留非空且不与训练集重复。模型并行和设备数保持默认。

想多观察几步，在对应 `config.yaml` 中把 `trainer.total_training_steps=1`、`trainer.save_freq=1` 都改为 `3`，保留 `trainer.total_epochs=1`，结果看 `global_step_3`。文件实际是JSON，保留引号、逗号。默认超时1800秒；需延长时在最外层 `{` 后添加 `"timeout_seconds": 3600,`，已有该项则改值。

**训练时另开一个宿主机终端观察负载：**

```bash
watch -n 2 npu-smi info   # NPU利用率、显存、进程；Ctrl+C仅结束监控
```

CPU和内存用 `top`、`free -h`；训练日志和检查点与负载一起展示。加载模型、生成、反向传播、保存时负载会变化，不要求始终满载。

**三项统一验收：** 使用训练结束时打印的结果目录，确认 `exit_code` 为 `0`、`status.json` 中 `state` 为 `completed`、`checkpoints/` 下有对应step。`--check` 成功不算训练完成。失败优先看该目录的 `driver.log`。历史五号机单步通过记录见[验收状态](../delivery/business_training_delivery/docs/验收状态.md)，本次重跑结果以新日志为准。
