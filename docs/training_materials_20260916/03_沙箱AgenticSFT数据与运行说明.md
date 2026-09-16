# 沙箱 Agentic SFT 数据与运行说明

数据与训练方法说明  2026年9月16日

本材料对应已完成的 Qwen3.5-9B 沙箱轨迹 SFT：2,028 条 16K 内训练候选，150 步。它与 Qwen3.6-27B 的 16 条纠错诊断 SFT 是不同实验。

## 一 数据构成

数据来自 PI agent 在物流沙箱中的多轮交互。环境包含物流数据库、业务文档及任务；轨迹保留问题、模型思考、工具调用、执行结果和最终回答。

| 数据层 | 数量与范围 |
| --- | --- |
| 原始 OpenAI 轨迹副本 | 24 个 JSONL，共 36,000 条 |
| 训练候选合并文件 | 2,028 条，完整轨迹不超过 16,384 token |
| 其中 SQL 结果强验证 | 27 条 |
| 其中弱证据待复核 | 2,001 条 |
| 隔离数据 | 250 条 validation/test；1,981 条 16–32K 长轨迹未混入上述文件 |

处理链路：读取完整事件消息 → 对齐任务身份 → 检查工具调用与返回闭环 → 按判分和证据筛选 → 按任务切分 → 按真实 token 长度筛选 → 合并训练文件。

2,028 条是历史实际使用的候选集，其中 27 条达到当时的强验证标准；不能把整批数据表述为已逐条完成高质量审核。训练完成与数据质量是两个不同结论。

## 二 数据形态

原始层使用 OpenAI messages，包含 assistant.reasoning_content、tool_calls 和 tool 消息。适配层由 prepare_trajectory_sft.py 转为 ms-swift 的 assistant、tool_call、tool_response 角色。

| 字段或消息 | 作用 |
| --- | --- |
| messages | 按真实顺序保存完整交互 |
| tools | bash、read、write、edit 的工具定义；实际保存为 JSON 字符串 |
| assistant | 思考与回答；参与训练 |
| tool_call | 工具名和参数；参与训练 |
| tool_response | 工具执行结果；只提供上下文 |

以下仅展示角色与字段，内容为演示；真实 tools 需包含完整参数定义。

```
{"tools":"[完整工具 schema]", "messages":[
  {"role":"user","content":"读取业务说明并回答问题。"},
  {"role":"assistant","content":"先查阅说明。", "loss":true},
  {"role":"tool_call",
   "content":"{\"name\":\"read\",\"arguments\":{\"path\":\"rules.md\"}}",
   "loss":true},
  {"role":"tool_response","content":"业务说明正文", "loss":false},
  {"role":"assistant","content":"依据说明得出的答案。", "loss":true}
]}
```

训练保留整条工具交互。推理写入 think 块，工具返回不作为模型应生成的答案。超出 16K 的记录不混入本次训练。

## 三 训练脚本运行指令

在 5 号机 qwen3.5-9b-msswift-trajory-sft 容器内执行。宿主机 /data3/llin/trajory_sft 对应容器 /workspace/sft。需使用该项目已验证的 ms-swift、Megatron 和 MindSpeed 环境。

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/sft
export MODEL=/models/Qwen3.5-9B
data_dir="$PWD/data/screened/upstream_correct_screen_v1"
export DATASET="$data_dir/train_candidates_16k_2028.jsonl"
export RUN_DIR="$PWD/runs/agentic-sft-$(date +%Y%m%d-%H%M%S)"
export TRAIN_ITERS=150
export SAVE_STEPS=15
export SAVE_TOTAL_LIMIT=10
export NO_SAVE_OPTIM=false
export NO_SAVE_RNG=false
bash scripts/run_qwen35_megatron_tp4_pp2_dp2.sh
```

脚本默认只跑两步测试，因此正式复现必须显式设置 TRAIN_ITERS=150。既定参数为 16 张 NPU、TP4/PP2/DP2、micro batch 1、global batch 16、长度 16,384、学习率 1e-6 至 1e-7。

历史运行已完成 150 步并导出 10 个 HF 模型。本启动脚本保存 MCore 分布式检查点，推理前需使用对应导出流程；视觉模块不参与本次文本轨迹训练。

## 依据

相邻项目 qwen9b-trajory-SFT：scripts/prepare_trajectory_sft.py、scripts/run_qwen35_megatron_tp4_pp2_dp2.sh；updates 中 v0.5.1 数据合并、v0.6.0 训练配置及 v0.7.0 完成记录。
