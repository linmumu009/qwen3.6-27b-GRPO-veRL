# 业务方训练交付包

版本：0.2.2；日期：2026-09-21。
5号机位置：`/data3/llin/business_training_delivery`。
面向Ascend 910 A3单机16个逻辑NPU，优先使用veRL。CPT模型已作为独立实体副本放入model/CPT。

首次部署和完整操作见[业务方运行手册](docs/业务方运行手册.md)，日常操作见[快速开始](docs/业务方快速开始.md)，逐项完成情况见[验收状态](docs/验收状态.md)。

本次验收的临时产物已按要求清理，保留验证汇总；如需训练检查点或HF导出模型，请重新运行相应入口。

## 启动
在本目录中选择一条命令：
```bash
bash Train/SFT/run.sh
bash Train/trajory-SFT/run.sh
bash Train/GRPO/run.sh
bash Train/agentic-GRPO/run.sh
bash Train/CPT/run.sh
```
每次默认仅训练1步并保存检查点，依次运行。正式训练参数在各自config.yaml中修改。
训练前先执行例如 `bash Train/SFT/run.sh --check`，只检查数据、工具和配置，不训练。
真实验收状态以 `docs/验收状态.md` 为准，不能将入口存在或预检成功等同于完整训练通过。

## 内容
- model：原始模型与CPT模型；模型整理状态见model/README.md。
- datasets：五类自编样例，每类32条训练数据、8条验证数据；JSONL可读，入口自动生成Parquet。
- Train：五类run.sh、config.yaml、README.md。沿用用户指定的trajory-SFT名称。
- frameworks：共享源码及补丁，第三方源码只在5号机归档，不提交到项目Git。
- environment：独立容器入口、依赖清单、源码版本与离线镜像导出命令。
- tools：数据转换、监督加载器、算术奖励、计算器及模型导出。
- outputs：按训练类型和时间保存配置、日志、预检结果、退出码与检查点。
- docs：交付方案、来源索引与验收状态。

算术数据仅用于证明工程流程，不用于衡量业务能力或训练收益。原始业务数据、历史运行结果和登录凭据不在包内。
