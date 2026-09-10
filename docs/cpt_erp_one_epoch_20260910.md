# 新语料单遍 CPT 与双评测执行账本

## 最新状态：训练完成，接续导出（2026-09-10）

47/47步已完成，32,515输入token，逐步长度多重集合与47篇一致，平均逐步loss=1.8917238255764575；完整第47步模型/优化器/调度器/RNG及数据加载检查点门禁通过。原协调器在训练后因独立快照缺少scripts.verify_checkpoint_integrity依赖停止，尚未导出或评测。

已在独立code_v2补齐依赖并通过真实训练日志审核和检查点检查；新PID **2020874** 使用 `--after-training` 仅接续导出和双评测，不再调用训练。运行目录及llin模型名称不变，日志coordinator_v2.log，快照code_v2_manifest.safe.json。3项定向测试通过。后续检查status.safe.json、export.log和evaluation进度，尚无新评测成绩。

## 最新执行状态：立即启动（2026-09-10）

用户明确要求停止旧实验剩余验证，并要求模型名称以llin开头。本节取代下文历史等待规则。已终止旧验证PID1723046及旧等待PID1831246的完整进程组，核验公共锁已释放；保留原模型、结果和日志，不再恢复通用/Agent验证。旧等待目录未产生training目录，无重复曝光。

新运行目录：`/workspace/llin-verl-grpo/runs/cpt-stage2-storage-20260910/llin/llin-cpt-erp-one-epoch-20260910`，协调PID **1832784**，已核验状态 **training**。训练目录为`llin-training`；HF模型输出名 **llin-step120-erp-cpt-1epoch-20260910**。正式47条分词门禁通过，32,468监督token和32,515输入token匹配；16进程训练正在初始化，尚无评测成绩。

自动跟进cpt已改为仅新任务，禁止恢复旧实验。新任务全部完成后通知并暂停。下文的旧等待进程、路径及旧回归义务仅是历史记录；新代码不再依赖confirmation状态。训练日志位于llin-training/torchrun_logs，协议仍为Step120起点、47步一遍、双评测各3轮。

2026-09-10，用户明确要求“CPT一遍，然后在两个评测集上进行验证”。已部署独立运行并排队；**尚未开始训练，尚无新评测成绩**。

## 冻结方案

- 模型起点：原始 Step120 Open Source 模型态，fresh Adam，不从旧 CPT 候选继续。
- 语料：已发布 `cpt-corpus-v1.0.2` 的 `train.parquet`，SHA256 `7918321f1b66f62c4239d419f2b7093b483748c31fae4fdfc850c4aa34065172`。47篇只训练一遍，32,468有效监督token、32,515含EOS输入token。3篇语料验证集不参与训练。
- 为避免训练器 `drop_last=True` 丢弃47条中的尾部3条，采用全局batch1、DP1，共47步；不补重复样本。TP4/PP2/CP2，16张NPU，最大长度4096，禁止截断，原始文本因果语言建模。
- 学习率5e-7→1e-7余弦、2步预热；显式Adam beta=(0.9,0.999)，weight decay0，clip1；视觉与MTP沿用现有冻结机制。保存第47步模型、优化器、调度器、RNG及数据加载状态，再完整导出HF模型。
- 评测：SC-bench知识226题、LogistikaBench1446题，各3轮，共5,016请求；原提示、thinking关闭、temperature0、seed1024、最大输出96、上下文8192、TP8、并发32。无效与截断按原规则计错，逐题多数票。
- 对照复用已审核的 Step120 `curve8x/eval_epoch_0`，SC189/226、Logistika1180/1446。新旧题目哈希、提示token哈希和生成参数必须一致；分别报告正确数、改对和改错。因选材使用了评测知识结构，仍属开发评测，不宣称独立泛化或自动晋级。

## 运行与衔接

- SSH huawei-05，容器 `llin-verl-trainer-m05-20260730`。
- 运行目录：`/workspace/llin-verl-grpo/runs/cpt-stage2-storage-20260910/llin/cpt-erp-one-epoch-20260910`。
- 对应宿主机 `/data1/llin/cpt-erp-one-epoch-20260910`；已核验约2.81TB空闲，训练前再次要求550GB以上。
- 协调进程PID **1831246**；已确认 `status.safe.json=waiting_for_existing_confirmation`，协调日志无错误。
- 使用39个文件的独立代码快照及SHA清单，未覆盖旧实验运行代码。先等旧流程 `confirmation_general_complete_agent_pending`，再获得公共 `.logistics-exam-cpt.lock`，自动执行训练、训练预算审核、完整检查点门禁、导出、双评测及对比。
- 原实验的Agent回归仍须继续，不能被新任务取消；新流水线与原任务共用资源锁。旧确认失败则新队列停止并报告，不盲目抢卡。
- 现有每30分钟自动跟进 `cpt` 已扩充本实验，同时保留原实验全部验证义务；新实验完整结束后通知。原实验及新实验均完成前不暂停自动跟进。

## 继续检查

检查PID与 `status.safe.json` 后再行动，不重复启动。状态依次为 waiting_for_existing_confirmation、training、export、evaluation、complete；任何异常为failed。查看对应日志，`training/torchrun_logs`保留逐rank记录。

训练完成必须核对 `training_audit.safe.json`：恰好47步、输入总量32,515、逐步长度多重集合与47篇记录一致、loss/梯度/LR有限。该检查结合DP1、batch1单遍采样配置约束曝光，不把token长度本身当成内容身份认证。再检查完整第47步检查点与导出结果。

最终读取 `comparison.safe.json`、`evaluation/summary.safe.json`，核查5,016条结果和全部题目×repeat身份以及新旧提示一致；同步安全结果到仓库、更新本账本和README、提交推送。不可仅凭协调进程结束宣称成功。

## 已验证与未验证

- 语料全量实际加载器CPU验证在上一版本已通过。
- 本次3项定向测试通过：47步正确预算、缺步拒绝、相同总token但单步曝光不同拒绝；远端所有快照Python编译与训练脚本语法检查通过。
- 实装训练器使用DP采样及drop_last，DP1/batch1可完整覆盖47条；实际NPU训练、导出与双评测等待资源后执行，尚不能报告通过或收益。
