# SFT GRPO CPT 汇报操作速查

只在**5号机宿主机**操作，使用包内已有数据，三项依次运行。**不操作6号机，不启动或停止旧多机Ray集群。** 数据转换、tokenizer加载和容器启动均自动完成。

## SFT

进入business_training_delivery：`cd /data3/llin/business_training_delivery`

1. 打开配置文件：`vim Train/SFT/config.yaml`
2. 确认以下字段，改完保存退出：
   - `trainer.nnodes=1`：参与训练的服务器数量；1表示只用5号机。
   - `trainer.n_gpus_per_node=16`：每台服务器使用的设备数；这里是16个逻辑NPU，字段名沿用GPU命名。
   - `trainer.total_training_steps=1`：训练步数上限；1表示只跑1步，不是把全部数据训练1轮。
   - `trainer.save_freq=1`：保存间隔；1表示每训练1步保存一次检查点（模型训练状态文件）。
3. 确认数据：`datasets/SFT/train.jsonl`、`datasets/SFT/val.jsonl`。首次用现成数据，不改路径。
4. 指定模型：`export DELIVERY_MODEL_DIR=model/qwen3.6-27b`。换模型时只改这里的目录，配置中的 `{MODEL}` 保留。
5. 检查数据和配置：`bash Train/SFT/run.sh --check`，成功后继续。
6. 开始训练：`bash Train/SFT/run.sh`

## GRPO

进入business_training_delivery：`cd /data3/llin/business_training_delivery`

1. 打开配置文件：`vim Train/GRPO/config.yaml`
2. 确认以下字段，改完保存退出：
   - `trainer.nnodes=1`：参与训练的服务器数量；1表示只用5号机。
   - `trainer.n_gpus_per_node=16`：每台服务器使用的设备数；这里是16个逻辑NPU，字段名沿用GPU命名。
   - `trainer.total_training_steps=1`：训练步数上限；1表示只跑1步，不是把全部数据训练1轮。
   - `trainer.save_freq=1`：保存间隔；1表示每训练1步保存一次检查点（模型训练状态文件）。
3. 确认数据：`datasets/GRPO/train.jsonl`、`datasets/GRPO/val.jsonl`。首次用现成算术题；换数据参照train文件完整行，不能直接用简化的example文件。
4. 指定模型：`export DELIVERY_MODEL_DIR=model/qwen3.6-27b`。配置中的 `{MODEL}` 保留。交付入口已设置 `RAY_ADDRESS=local`，只使用5号机本地Ray。
5. 检查数据和配置：`bash Train/GRPO/run.sh --check`，成功后继续。
6. 开始训练：`bash Train/GRPO/run.sh`

简单样例可能奖励全为1、梯度为0，本次只演示流程跑通。

## CPT

进入business_training_delivery：`cd /data3/llin/business_training_delivery`

1. 打开配置文件：`vim Train/CPT/config.yaml`
2. 确认以下字段，改完保存退出：
   - `trainer.nnodes=1`：参与训练的服务器数量；1表示只用5号机。
   - `trainer.n_gpus_per_node=16`：每台服务器使用的设备数；这里是16个逻辑NPU，字段名沿用GPU命名。
   - `trainer.total_training_steps=1`：训练步数上限；1表示只跑1步，不是把全部数据训练1轮。
   - `trainer.save_freq=1`：保存间隔；1表示每训练1步保存一次检查点（模型训练状态文件）。
3. 确认数据：`datasets/CPT/train.jsonl`、`datasets/CPT/val.jsonl`。首次用现成原文；换数据时每行格式为 `{"text":"物流业务原文"}`，单条不超过1024 token。
4. 指定模型：`export DELIVERY_MODEL_DIR=model/qwen3.6-27b`。配置中的 `{MODEL}` 保留。
5. 检查数据和配置：`bash Train/CPT/run.sh --check`，成功后继续。
6. 开始训练：`bash Train/CPT/run.sh`

**三项通用：** 开始前用 `npu-smi info` 确认5号机设备空闲，`df -h .` 确认输出盘至少90GB空余；三项检查点会累积占空间。想观察3步，将对应配置中的 `trainer.total_training_steps` 和 `trainer.save_freq` 都改为 `3`，其余保持默认。

**看负载：** 另开5号机终端执行 `watch -n 2 npu-smi info`；CPU和内存用 `top`、`free -h`。

**看结果：** 训练结束会打印 `outputs/类型/本次编号/`；确认退出码0、`status.json` 中 `state` 为 `completed`，且 `checkpoints/` 下有对应step。失败看该目录的 `driver.log`。预检成功不算训练完成。
