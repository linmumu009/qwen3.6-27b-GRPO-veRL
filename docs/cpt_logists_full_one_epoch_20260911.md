# 全部13份材料单遍 CPT 与双评测（2026-09-11）

## 范围纠正与当前状态

用户明确要求其提供的全部教材用于训练，并纠正此前擅自隔离评测来源材料的做法。本轮**全部13份材料均贡献训练内容**，不按评测同源或文字重合排除；保留正常质量清洗与去重。此前留出的开发验证内容也纳入训练，不另留开发验证集。评测题目/答案文件本身不作为训练输入。

本轮状态为 `training`，协调PID **2416944**；实际加载器已逐条通过4,707条的分词、EOS和损失掩码检查，最大长度4,085、零截断。已确认完成2/523步，训练正常运行，尚无本轮完成或评测成绩。[启动与全量语料核验清单](cpt_logists_full_launch_20260911.safe.json)。

服务器huawei-05，容器`llin-verl-trainer-m05-20260730`。运行目录：`/workspace/llin-verl-grpo/runs/cpt-stage2-storage-20260910/llin/llin-cpt-logists-full-one-epoch-20260911`。独立`code/`快照64个文件及哈希清单，沿用已修复的固定Megatron Bridge环境。启动前NPU无运行任务，空闲空间约1.93 TB。前一轮及其结果保留，不恢复旧实验。

## 语料清单

正式语料`llin-logists-v5-full`：4,707条，1,844,881正文token，1,849,588含EOS输入token。原13份资料合计12份PDF、4,050页及一个HTML教材包；复用已完成的MinerU和本地提取结果。

| 材料 | 训练样本 | 正文token |
|---|---:|---:|
| Ports and Waterways | 504 | 109,777 |
| Beyond Lean | 238 | 186,130 |
| Challenges and Opportunities in International Business | 348 | 334,645 |
| Container Terminals and Automated Transport Systems | 294 | 211,957 |
| Distribution Logistics | 183 | 145,816 |
| Fundamentals of Global Strategy | 543 | 91,588 |
| International Trade: Theory and Policy | 1103 | 159,662 |
| JSI物流手册 | 157 | 94,717 |
| Traffic Flow Theory | 233 | 116,091 |
| Transport Statistics Glossary | 278 | 37,032 |
| Managing Supply Chain Risk | 219 | 171,184 |
| MITx SCM Key Concepts | 400 | 63,183 |
| Warehouse & Distribution Science | 207 | 123,099 |

已取消来源族排除、段内评测指纹排除、跨块指纹排除和开发集留出四项行为。117条样本命中既有评测整段/13词窗口指纹，**仅记录，全部保留**。原文出处、页码/HTML成员、质量排除原因仍可追溯；“全部材料”指每份材料的合格内容都进入训练，不是把乱码、重复页或无法可靠转换的图表塞入训练。

本地：`CPT_resources/llin-logists-v5-full-20260911/`；服务器语料：`/workspace/llin-verl-grpo/runs/llin-logists-v5-full-20260911/`。训练文件SHA256：`898dada4de8c7274ee0b18bf03f685998bd9cf25739af2502f1eed67c3cb9733`。已核验原书/产物哈希、JSONL与Parquet一致、真实分词长度、13来源覆盖、块来源/页码/HTML成员和去重。原书和训练正文不提交Git。

## 冻结训练与评测

- 起点：原始Step120模型态，fresh Adam，重新进行完整材料的一遍训练；不接在上一轮CPT模型上。
- 全局batch9、micro batch1、523步一遍，4,707=9×523，恰好完整覆盖，不丢尾批、不补重复。TP4/PP2/CP2/DP1，共16张NPU。批量为适应完整条数从上一轮8调整为9，应在比较中保留这一差别。
- 学习率5e-7→1e-7余弦，11步预热；Adam beta=(0.9,0.999)、weight decay0、clip1。原文causal LM，无聊天模板，长度4096，超长报错。
- 第523步保存完整训练检查点，再导出 `llin-step120-logists-full-cpt-1epoch-20260911`。
- 随后SC-bench知识226题及LogistikaBench1446题各3轮，共5,016请求。原提示、thinking关闭、temperature0、seed1024、最大输出96、上下文8192、TP8和并发32不变。对照仍为原Step120：189/226与1180/1446，核验案例/提示身份后报告改对改错。

本轮是**包含评测来源教材的学习实验**。评测成绩可以用于这个问题，但不描述为独立泛化结果；这一记录不会触发删除教材或再次隔离同源内容。

## 接续与验证

协调器自动训练→预算审核→完整检查点→导出→双评测→配对比较。`cpt`每30分钟自动跟进已恢复，先核对`status.safe.json`、`coordinator.pid`和日志，不能重复启动。正常中间状态保持安静，只有完成、实质失败或需介入时通知；全部完成后暂停。仅授权一遍，没有额外遍数。

11项测试通过，包括全13来源与匹配文本实际保留、开发集为空、523步预算和缺尾步拒绝，以及已有文本清洗测试。训练开始前实际4,707条分词门禁通过；远端快照Python编译和训练脚本语法检查通过。训练完必须核对523步、1,849,588输入token、有限loss/梯度/LR、完整检查点与HF导出，再独立复核全部5,016条评测输出。不能仅凭协调器状态宣称正确完成。

故障时保留现有检查点、代码及日志，不盲目重跑训练。`--after-training`只在第523步已完整完成时接续；`--retry-export`会使用独立导出日志且仍拒绝覆盖正式模型目录。不得恢复已取消的旧实验、额外训练或重新排除同源教材。

复现语料时使用：

```powershell
python scripts/prepare_llin_logists_corpus.py --out CPT_resources/llin-logists-full-rebuild --mineru-dir CPT_resources/llin-mineru-results --all-materials
python scripts/verify_llin_logists_corpus.py --root CPT_resources/llin-logists-full-rebuild --report CPT_resources/llin-logists-full-rebuild/verification.safe.json
```

`--all-materials`是本轮必须的开关。旧默认模式保留以复现历史结果，不用于本轮。构建器强制要求13个来源全部贡献训练内容，否则失败，不会默默输出缺来源的训练文件。
