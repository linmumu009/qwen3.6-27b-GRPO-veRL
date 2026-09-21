# 业务方训练交付包（目录整理版）

版本：0.1.0；日期：2026-09-21。
5号机位置：`/data3/llin/business_training_delivery`。
当前只完成目录、格式示例、历史实现索引和交付方案；没有验收五类训练一行跑通。

## 目录
- `model/qwen3.6-27b`：5号机链接到 `/data/models/Qwen3.6-27B`，尚非可搬走的独立模型副本。
- `model/CPT`：预留，等待用户提供CPT模型。
- `datasets/{SFT,trajory-SFT,GRPO,agentic-GRPO,CPT}`：每类一份自编格式示例。
- `Train/{SFT,trajory-SFT,GRPO,agentic-GRPO,CPT}`：每类启动入口、配置、说明的预留位置。
- `frameworks`：共享框架，仅保留实际依赖版本，优先veRL。
- `environment`：环境版本、容器和安装说明。
- `tools`：Agent训练需要的工具、沙箱与奖励函数。
- `outputs`：训练日志、检查点、导出模型和验收结果。
- `docs`：交付方案与历史材料索引。

沿用用户指定的 `trajory-SFT` 目录名，其含义是 trajectory SFT（轨迹监督微调）。
请先阅读 `docs/交付方案.md`。本版不提供伪装成可训练的占位启动脚本。
