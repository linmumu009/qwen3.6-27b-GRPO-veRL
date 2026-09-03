# Qwen3.6-27B Step120 单书物流 CPT 实测结论

日期：2026-09-03

## 结论

**原 Step120 veRL 容器可以完成继续预训练，但《The Handbook of Logistics and Distribution Management》第八版单书、单次曝光没有带来可证明的物流知识增益，本候选不得晋级交付。**

这次验证把“环境是否可用”和“方案是否有效”分开回答了：

- 工程上可用。原 `llin-verl-a3:20260730` 容器、Megatron TP4/PP2/CP2 拓扑和 Step120 `dist_ckpt` 已完成最长 4,096-token 一步门禁及 29 步正式训练，均退出码 0；正式 checkpoint 和 HF 导出均通过完整性检查。
- 训练只能从 Step120 **模型态**继续，不能恢复旧 GRPO 优化器。Step120 交付物没有完整 optimizer/RNG 状态，因此本轮按新阶段初始化 fresh Adam；MTP 与 Step120 一致不参与训练，HF 导出时继承原有 15 个冻结 MTP tensor。
- 效果上不成立。在同一台 5 号机、同一 vLLM-Ascend 镜像、同一 BF16 TP4×DP2 启动参数和同一 1,672 题协议下，Step120 为 `81.88%`，单书 CPT 为 `81.70%`，变化 `-0.18` 个百分点；5 题由错变对、8 题由对变错，McNemar 精确检验 `p=0.581`。
- 这个结果不能表述为显著退化，因为 Step120 自身两次零温度并行推理也出现 4 个 0→1 和 4 个 1→0 的逐题波动，总分保持不变。稳健结论是：**未观察到增益，且远低于预登记的 +3 点实用门槛。**

## 正式训练

数据只使用取得用户书面 AI/ML 训练授权确认的书籍正文；授权文件未由执行方审阅，不构成法律判断。制数排除了扉页、目录、参考文献和索引，保留 44 章、7 个部分，经保守清洗后形成 116 个章节内块。

| 项目 | 实际值 |
|---|---:|
| 训练内容 token | 336,586 |
| 加每块 EOS 后 sequence token | 336,702 |
| 最大序列长度 | 4,096 |
| 全局 batch | 4 |
| optimizer step | 29 |
| 曝光次数 | 1 |
| 学习率 | `5e-7 → 1e-7` |
| 拓扑 | 16 NPU，TP4/PP2/CP2 |
| 首步 / 末步 loss | `1.9778 / 2.0787` |
| 全程最低 / 平均 loss | `1.7403 / 2.0504` |
| 最大裁剪前 grad norm | `116.97` |
| 单卡峰值分配 / 保留显存 | `38.87 / 39.68 GiB` |

正式 checkpoint 为 `global_step_29`，包含 32 个 Megatron 分片、约 54.72 GB，只保存 model+extra，不保存 optimizer。HF 严格导出得到 15 个 safetensors 分片、1,199 个 tensor；相对 Step120 HF 参考，missing/extra/shape/dtype mismatch 均为 0。

## 物流知识评测

评测冻结 LogistikaBench 1,446 题与 SC-bench 知识题 226 题，统一使用零基选项 JSON、集合完全匹配、`temperature=0`、关闭 thinking、最大输出 128 token、并发 64。Step120 与 CPT 都使用同一 vLLM-Ascend `v0.22.1rc1-a3` 镜像、BF16、TP4×DP2、相同 seed 与相同服务参数；两次 API 和解析失败均为 0。

| 范围 | Step120 同配置复跑 | 单书 CPT | 变化 | 0→1 | 1→0 |
|---|---:|---:|---:|---:|---:|
| LogistikaBench（1,446） | 81.54% | 81.40% | -0.14 点 | 5 | 7 |
| SC-bench 知识题（226） | 84.07% | 83.63% | -0.44 点 | 0 | 1 |
| 合计（1,672） | 81.88% | 81.70% | -0.18 点 | 5 | 8 |

合计配对 `p=0.5811`，类别宏平均变化 `-0.10` 点。作为辅助参照，CPT 相对更早的同机原生模型总分为 `+0.18` 点（7 个 0→1、4 个 1→0，`p=0.549`），仍与 Step120 训练前已存在的零点几分波动处于同一量级，不能证明单书 CPT 建立了新物流能力。

为检查服务侧波动，用本轮同一启动参数复跑 Step120。两次 Step120 总分都为 1,369/1,672，但逐题有 4 个 0→1 和 4 个 1→0；因此决策以实用幅度门槛和配对方向为主，不对个位数题目的变化作知识归因。

## 决策与下一步

本候选保持 `promotion_allowed=false`。由于首要物流知识门禁已经失败，本轮不再消耗算力运行原 Agent 与开源密封回归；这不等于这些能力已验证无回归。

下一步不应重复曝光同一本书，也不应把 SC-bench 或 LogistikaBench 错题制成训练数据。建议继续使用 Step120，而不是本次 CPT checkpoint，按以下顺序推进：

1. 建设 20M–50M token 的多来源、已授权物流语料；书籍作为骨架之一，加入 Wikipedia 概念、政府/国际组织公开资料、企业内部知识和一般能力回放，并做来源级去重和评测污染隔离。
2. 先取 5M token 做混合 CPT 金丝雀，而非直接全量长训；沿用本轮已跑通的原 veRL 容器、模型态初始化 fresh Adam 和最长序列门禁。
3. 另建 300–500 道来源隔离的隐藏物流解释题。公开选择题只做辅助观测，不能承担唯一晋级标准。
4. 若隐藏题显示“知道但不会组织回答”，再补少量阅读理解/场景问答 SFT；若是事实更新与企业知识，优先 RAG。
5. 只有物流隐藏集和公开集达到实用增益、且原 Agent/开源/工具门禁均不下降超过约定阈值，才导出交付候选。

## 证据位置

- 制数安全清单：`docs/handbook8e_cpt_4096_20260903.safe.json`
- 训练与 HF 导出安全汇总：`docs/logistics_cpt_book_one_epoch_20260903.safe.json`
- 评测与配对安全汇总：`docs/logistics_cpt_handbook8e_evaluation_20260903.safe.json`
- 数据和训练程序：`scripts/prepare_logistics_book_cpt.py`、`scripts/qwen36_causal_lm_dataset.py`、`scripts/run_logistics_cpt_book_one_epoch.sh`

5 号机保留可复核大产物：

```text
/data3/llin/qwen3.6-27b-verl-grpo/runs/logistics-cpt-book-one-epoch-20260903-01/
/data/llin/logistics_cpt_20260903/eval/
```

仓库不包含书籍正文、评测题面、标准答案、逐题模型原始输出或密钥。临时推理容器已经删除，NPU 进程为 0。`/data3` 仅余约 35 GB，删除或迁移本次约 100+ GB checkpoint/HF 产物前应单独确认保留策略。
