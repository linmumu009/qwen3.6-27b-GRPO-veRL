# CPT 语料与运行说明

数据与训练方法说明  2026年9月16日

本材料说明物流原文 CPT 的来源、输入格式和启动方法。运行示例对应 2026 年 9 月 11 日的完整知识语料：从 Qwen3.6-27B 的开源续训 Step120 出发，训练一遍。

## 一 原始语料来源

原始材料由用户提供，共 13 份：12 份 PDF、4,050 页，以及 1 个 HTML 教材包。正文经 MinerU 或本地文本层提取，保留章节、页码及来源关系。

| 内容方向 | 原始材料 |
| --- | --- |
| 港口与运输 | Ports and Waterways；Container Terminals and Automated Transport Systems |
| 配送与仓储 | Distribution Logistics；Warehouse & Distribution Science |
| 供应链管理 | JSI Supply Chain Manager’s Handbook；Managing Supply Chain Risk；MITx SCM Key Concepts |
| 交通与统计 | Traffic Flow Theory；Transport Statistics Glossary |
| 贸易与战略 | International Trade: Theory and Policy；Fundamentals of Global Strategy；Challenges and Opportunities in International Business |
| 仿真 | Beyond Lean: Simulation in Practice |

在教材正文基础上，追加 205 条经核对的知识单元，来源包括 UNECE、FAA、UNCTAD、IFRS、IATA 及百科等公开资料。补充内容保留适用范围与出处。

## 二 CPT 语料形态

下面摘录实际 train.jsonl 的第 1 条记录。展示 text 字段全文，未改写正文；省略 id、指纹、token 数及来源索引等管理字段。

```
{
  "text": "Source: 43 Book Manuscript 398 1 10 20230209\nSection: I General introduction > 1 Ports and waterways systems > 1.1 On the importance of waterborne transport and its facilities\n\nPorts and waterways are parts of a coherent system enabling supply chains over water. Their functions, design, operation and maintenance influence the performance of these supply chains and the transport system as a whole. This chapter gives a general orientation, terminology and essential definitions, as well as an introduction into how the elements of the transport system interact."
}
```

这条数据是一段港口与水运知识正文，包含来源提示和章节标题，没有问题、标准答案或对话角色。CPT 直接学习这段连续文本。

样本来源：CPT_resources/llin-knowledge-complete-20260911/release/train.jsonl，第 1 行。正式训练使用对应 Parquet 的 text 字段，每条追加一次 EOS，最大长度 4,096，不套用聊天模板。

完整语料共 4,912 条，其中教材正文 4,707 条、补充知识 205 条。该版本包含评测来源教材，属于定向学习实验，相关成绩不代表独立泛化。

## 三 CPT 脚本运行指令

在 5 号机 llin-verl-trainer-m05-20260730 容器内执行。需已有 Step120 模型、分布式权重、固定 Megatron Bridge 及训练文件。脚本固定本批 4,912 条、batch 8、614 步，不能直接用于其他规模语料。

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/llin-verl-grpo
export PROJECT_ROOT="$PWD"
export TRAIN_FILE="$PWD/runs/llin-knowledge-complete-20260911/train.parquet"
export EXPECTED_CONTENT_TOKENS=1868012
sha=6ffa16684e617f657293918bf1d89ae8930
sha+=ec24e0e08da346f175c4710c1f1c8
export EXPECTED_TRAIN_SHA="$sha"
export RUN_NAME="llin-cpt-knowledge-replay-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_DIR="$PWD/runs/$RUN_NAME"
bash scripts/run_cpt_knowledge_complete_one_epoch.sh
```

默认起点为 llin-step120-opensource-20260825-02。16 张 NPU，TP4/PP2/CP2；学习率 5e-7 至 1e-7。第 614 步保存检查点，导出后用于推理。

此处复用 veRL 的 sft_trainer 入口，通过原文加载器执行 CPT。新输出目录须不存在；脚本自动检查文件指纹与完整加载。

## 依据

本仓库 docs/cpt_complete_package_20260911.md、docs/cpt_knowledge_complete_one_epoch_20260911.md；scripts/run_cpt_knowledge_complete_one_epoch.sh、scripts/qwen36_causal_lm_dataset.py。
