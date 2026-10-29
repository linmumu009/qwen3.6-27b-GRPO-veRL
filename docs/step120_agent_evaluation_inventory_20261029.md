# 垂域 Step120 Agent 轨迹评测总索引

核查日期：2026-10-29。服务器：5号机。

已经找到多批完整的多轮 Agent 轨迹评测。最适合从头查看的一批是 **2026-08-21 的100题、每题8条，共800个采样槽位**：输入、17个原始轨迹分片、运行合同、日志、逐题评分和后续奖励审计都保留在5号机。另有300题双模型对照、281题多沙箱评测、500题开放任务系列和PI运行时对照。

本文 Step120 专指 **8月10日 dense-correctness 训练得到的垂域模型**，不包含8月25日的 `opensource` 续训分支。`open-dwh` 中的 open 指业务任务开放度，这批评测使用的仍是上述垂域 Step120。

完整运行索引与逐分片指纹见 [机器可读核验清单](step120_agent_evaluation_inventory_20261029.safe.json)。清单保留51个运行、合并或阶段记录；其中24个有原始分片的目录完成逐行结构核验，共21,967行。这些行包含基座、补跑和阶段性重复来源，**不能相加当作独立Step120评测总量**。

## 模型身份

5号机项目根目录为：

```text
/data3/llin/qwen3.6-27b-verl-grpo
```

下文的 `runs/`、`exports/` 和 `evaluations/` 均相对于该目录。

| 项目 | 已核对内容 |
| --- | --- |
| 本次目标HF模型 | `exports/llin-qwen3.6-27b-grpo-step120-hf-20260813` |
| 导出清单中的源checkpoint | `/workspace/llin-verl-grpo/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120/actor` |
| 应排除的开源续训模型 | `runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource` |
| 开源续训清单中的源checkpoint | `/workspace/llin-verl-grpo/runs/llin-step120-opensource-20260825-02/checkpoints/global_step_120/actor` |
| 权重文件存在性 | 索引引用15个safetensors文件，全部存在，共56,484,503,104字节 |
| 权重核查范围 | 核对导出清单、配置、权重索引指纹和文件存在性；本次未对全部大权重重算SHA256 |

目标模型的三个小文件SHA256，与100题评测及后续 `open-dwh` 运行合同记录一致：

```text
llin_export_manifest.json
bf363a04d460a389021cf5e6e0c7013552ea88df6c5bd7bdb5ffd43300057208
config.json
43eee23b0489ed1769bf2d49438015728ad686bef55d4aecb18a3a1ce840d6da
model.safetensors.index.json
747b0b46e0334189cf0eb02ecc860f6de1e5f156f7115d5f76c20b1bc9ae03aa
```

早期281题合同尚未记录这些身份哈希；其5号机driver日志指向同一HF导出路径。64题完整工具协议评测使用原生分布式checkpoint，日志出现目标dense训练目录和强制权重同步标记。

## 主要评测及完整性

“轨迹槽位齐全”表示每题的采样位置落盘，不表示每条轨迹成功完成。表中完成和超时分别沿用各自运行的统计字段，不把它们强行互补。

| 评测 | 题数和采样 | 轨迹槽位 | 有最终回答 | 超时 | 历史评分结果和状态 |
| --- | --- | ---: | ---: | ---: | --- |
| v15 Codex Model2 100题 | 100×8 | 800 | 783 | 7 | 原始结果评分344正确；后续严格终值shadow为289正确，口径见下节 |
| plan-first v3 Step120 原始轮 | 300×8 | 2,400 | 1,675 | 725 | 1,584正确；存在大量超时，已有原位回填版 |
| plan-first v3 Step120 超时回填版 | 300×8 | 2,400 | 2,400 | 0 | 2,178正确；基座同版2,201/2,400，两臂共4,800个槽位 |
| 老板多沙箱281题 | 281×8，5/6号机分担题目 | 2,248 | 1,771 | 475 | 911正确，49个技术mixed题组；槽位齐全，成功轨迹不齐 |
| 开放DWH v15 runtimefixed | 500×8，双机合并 | 4,000 | 133 | 3,861 | 1正确；严重超时，不能作为完整成功能力评测 |
| 开放DWH v20 runtimefixed | 500×8，双机合并 | 4,000 | 158 | 3,841 | 14正确；严重超时，技术mixed为1组 |
| PI和veRL运行时对照 | 10×8×2个运行时 | 160 | PI 69，veRL 56 | 最终PI 1，veRL 0 | 两臂均0/80正确，运行时等价门未通过 |
| 原生和Step120完整工具协议 | 64×1×2个模型 | 128 | 原生40，Step120 35 | 原生24，Step120 29 | Step120 numeric correct为7/64、严格correct verdict为5/64 |

以上评分分属不同题库、预算和解析器，不能跨行合并为一个准确率。300题回填版只替换原超时位置，不能把它称为两侧全部轨迹统一96K预算的重新独立评测。5号机保留其双模型合并结果和Step120逐题缓存；本次没有连接6号机重新核验该侧原始轨迹。

## 100题每题8条的完整材料

主目录：

```text
runs/llin-v15-codex-model2-100-step120-8x-20260821-01
```

北京时间2026-08-21 21:32:08开始，22:55:33结束；`exit_code=0`。

| 目录内文件 | 内容和用途 |
| --- | --- |
| `standalone_contract.json` | 模型身份、采样参数、上下文、超时和资源合同 |
| `driver.log`、`started_at`、`finished_at`、`exit_code` | 实际启动、运行过程和退出证据 |
| `data/tasks.jsonl`、`data/rollout_100.sensitive.parquet` | 原始任务定义及评测输入；正文保留服务器 |
| `data/logistics.sqlite`、`data/schema_dictionary.md` | 对应数据库和schema |
| `shards/tasks_00000_00006.jsonl` 至 `shards/tasks_00096_00100.jsonl` | 17个原始轨迹分片，合计23,470,319字节 |
| `outcomes/per_task.sensitive.jsonl` | 100题逐题结果和8条采样统计 |
| `outcomes/safe_summary.json` | 原始聚合结果和正确数分布 |
| `outcomes/mixed_review_queue.sensitive.jsonl` | 原始21个mixed题组的审阅队列 |
| `outcomes/mixed_groups.sensitive.parquet` | 原始技术筛选候选 |
| `grpo_readiness_audit_20260822-05/` | 后续语义、SQL/gold重放及奖励假阴性审计 |
| `grpo_reward_shadow_20260822-01/` | 第一版过程奖励shadow，复用相同800条轨迹 |
| `grpo_reward_shadow_20260822-02/` | 严格终值解析及outcome-gated奖励shadow，复用相同800条轨迹 |
| `grpo_grounded_tristate_calibration_20260822-01/` | 从43题子集抽取344条历史轨迹的PASS/FAIL/UNKNOWN校准 |

运行合同为 `temperature=1.0`、`top_p=0.95`、`top_k=20`，每题独立8条；16 NPU、TP8×DP2、16个Agent worker。输入预算4,096、response预算90,112、总上下文94,208 token，每轨迹超时1,800秒。该运行是冻结模型评测，合同记录没有初始化optimizer、没有保存新checkpoint。

逐行核验800个位置全部存在：100题均恰有 `sample_index=0..7`，重复槽位0、损坏JSON行0、空输入0。7条超时的输出为空，不能伪称800条均为完整成功轨迹。解码输出保留5,093个 `<tool_call>` 和5,081个 `<tool_response>`；历史分片没有单独保存完整结构化工具事件流。读取原始过程时应使用 `shards/` 中的 `input` 和 `output`，不能只看逐题正确数。

98题的8条解码输出全部不同；另两题分别有4种、7种输出，包含空超时输出。独立采样身份已核验，但不能声称这批每个题组8条文本全部互异。

### 同一800条轨迹的评分版本

| 评分版本 | 覆盖 | 结果 | 解释 |
| --- | --- | --- | --- |
| 原始 `outcomes` | 100题800条 | 344正确；全对30、全错47、mixed21、超时题组2 | 采样结束时保存的原始口径 |
| readiness `-05` | 原始21 mixed和47全错题，共68题544条 | 27题存在奖励假阴性；原mixed21题加修复后的mixed22题，形成43题候选 | 不是新增轨迹，也不是全100题统一重新评分 |
| reward shadow `-01` | 同一800条 | 483正确 | 修复表格路线后的旧宽松终值口径；不能与原344混称同一评分器 |
| reward shadow `-02` | 同一800条 | 289正确、511错误；全对12、mixed54、全错34 | 相对宽松版194个标签变化，含118条数值歧义；硬门另有774通过、26失败 |
| grounded三态校准 | 43题344条子集 | PASS 0、FAIL 253、UNKNOWN 91；final parser正确128 | 增加真实证据和语义合同要求；人工标签当时为0/344，不能把PASS 0解释成模型总体正确率0 |

评分版本必须连同文件名保留。344、483、289是同批历史轨迹的不同判定口径，344条三态子集又有不同分母；这些均不代表模型在各版本之间重新训练或重新生成。后两个目录虽用于设计Qwen3.8训练奖励，**输入仍是本次Step120产生的历史轨迹**。

已有说明：[43题readiness安全摘要](v15_codex_model2_100_grpo_readiness_audit_20260822.safe.json)、[过程奖励shadow](qwen38_trajectory_process_reward_shadow_20260822.md)、[严格终值shadow v2](qwen38_outcome_gated_process_shadow_20260822.md)、[三态校准](qwen38_grounded_tristate_calibration_20260822.md)。

## 300题双模型对照

5号机上应优先查看最终超时回填版：

```text
runs/llin-planfirst-v3-base-vs-step120-reconciled-30m-96k-20260814-02-finalizer/
  comparison/safe_summary.json
  comparison/candidate_hashes.sensitive.jsonl
  step120_remote/safe_summary.json
  step120_remote/per_task.sensitive.jsonl
  finalizer_safe_status.json
  finalizer_exit_code
```

最终比较记录两臂退出0、300题各8条：Step120正确2,178、全对240题、全错4题、mixed56题；基座正确2,201、全对244题、全错3题、mixed53题。Step120全部56个mixed中，排除冻结评估题后，比较摘要列出46个训练候选；56和46分母不同。

回填前的300题对照位于 `runs/llin-planfirst-v3-base-vs-step120-300x8-20260814-02-finalizer/`。原始Step120为1,584正确、725超时，基座为1,129正确、1,227超时；不能用这个未解决超时的差异宣称稳定训练收益。原位回填复用了已完成槽位，不能把前后各2,400再计为4,800个独立Step120样本。

基座本地原始轮及回填轨迹分别在 `runs/llin-planfirst-v3-native-300x8-20260814-02/shards/` 和 `runs/llin-planfirst-v3-native-reconciled-30m-96k-20260814-01/shards/`。Step120侧在5号机查到的是结果缓存，不能把 `step120_remote/per_task.sensitive.jsonl` 当作完整多轮文本。

## 281题多沙箱筛选

```text
runs/llin-multisandbox-dwh-full281-m05-step120-20260814-02/
  standalone_contract.json
  driver.log
  shards/
  outcomes/per_task.sensitive.jsonl
  outcomes/safe_summary.json
  incremental_review/
runs/llin-multisandbox-dwh-full281-dual-final-20260814-02/
  merged/safe_summary.json
  remote_arm/safe_summary.json
  finalizer_safe_status.json
```

5号机自身141题、1,128条分片全部在场，另一侧缓存140题、1,120条；合并为281题、2,248条。技术分桶为全对65、全错49、mixed49、含超时118题组。49个mixed后来只有13个通过语义审阅，36个拒绝；技术mixed不等于可训练。

前次 `llin-multisandbox-dwh-full281-m05-step120-20260814-01` 退出1，不能优先使用。全量合并目录曾有失败finalizer尝试，应以 `dual-final-20260814-02/merged/` 为汇总入口。背景筛选见 [多沙箱安全摘要](boss_multisandbox_dwh_screening_20260813_summary.json)，最终49→13审阅记录也保留于README对应版本说明。

## 500题开放任务系列及补采

| 版本 | 5号机入口 | 当前可用范围 |
| --- | --- | --- |
| v15修正运行时全量版 | `runs/llin-open-dwh-v15-step120-runtimefixed-500x8-dual-summary-20260815-02/merged/safe_summary.json` | 500题4,000槽位，3,861超时；本机原始轨迹在对应 `runtimefixed-500x8-h05-20260815-01/shards/` |
| v20修正运行时全量版 | `runs/llin-open-dwh-v20-step120-runtimefixed-500x8-dual-summary-20260815-01/merged/safe_summary.json` | 500题4,000槽位，3,841超时；本机250题2,000条已逐行核验 |
| v15和v20旧运行时版 | 各自 `runs/llin-open-dwh-v15-step120-500x8-dual-summary-20260815-01/` 和 `v20` 同名目录 | 均有4,000槽位旧汇总、正确数0；属于旧运行时结果，不能替代修正版本 |
| v21旧版快照 | `runs/llin-open-dwh-v21-step120-500x8-dual-partial-snapshot/merged/safe_summary.json` | 只含384题3,072条，不能写成500题完成 |
| v21修正运行时版 | `runs/llin-open-dwh-v21-step120-runtimefixed-500x8-h05-20260815-01/` | 本机仅查到48题384条分片，其中382超时；未找到完整全量收尾汇总 |
| v15超时补跑 | `runs/llin-open-dwh-v15-step120-timeout-retry-admission-v2-h05-20260815-01/` 和 `timeout-retry-rolling-v3-h05-20260816-01/` | 分片分别1,200和1,584行；局部摘要滞后，只覆盖1,152和1,248，不能据旧摘要认定全部重试已收尾 |
| v20两阶段补6条 | `runs/llin-open-dwh-v20-step120-adaptive-final-earlystop-v3-h05-20260816-02/` | 22题合为176条、8条一组；17正确、36超时，技术mixed5题 |
| v20分波2+2+2 | `runs/llin-open-dwh-v20-step120-2plus2plus2-final-h05-20260816-01/adaptive_final_safe_summary.json` | 215题累计1,278条；11个方差候选，204题未决；属于可变采样数筛题 |
| v21分波筛题 | `runs/llin-open-dwh-v21-step120-2plus2plus2plus2-work-h05-20260816-01/` 与对应 `window125` wave目录 | wave2为250题500条，wave4为243题486条，wave6有235题470条；未查到wave6退出码和最终全量收尾摘要 |

`adaptive`、`wave4`、`wave6` 的数据会复用前序两条，既不是每题独立8条的统一测试，也不能逐目录简单累计。生成原则及等级定义见 [开放业务DWH策略](open_multisandbox_dwh_generation_strategy_20260815.md)。

## PI运行时和完整工具协议的补充材料

PI本机完整原始文件位于：

```text
runs/llin-runtime-parity-10x8-step120-20260813-01/pi_native_matched48k/
```

该目录查到80个 `*_s00.jsonl` 至 `*_s07.jsonl` 形式的采样文件，共662,052,907字节，另有 `pi_arm_manifest.sensitive.jsonl`、`runtime_parity_analysis.sensitive.json`、`runtime_parity_summary.safe.json`。原生PI session JSONL与veRL解码后的轨迹分片是不同格式。缺少工具parser的 `pi_native_invalid_missing_tool_parser/` 明确无效；较早的 `pi_native/` 也不能代替最终 `pi_native_matched48k/`。

最终PI/veRL两臂各80条，temperature均为1.0，但单次生成上限、上下文压缩、墙钟和工具实现不完全相同；两臂均0正确不能证明运行时等价。PI首轮12条超出30分钟，恢复后的最终1条超时不能覆盖这个首次超时事实。完整解释见 [运行时对照](runtime_parity_10x8_step120_20260813.md) 和 [采样配置审计](runtime_parity_sampling_config_audit_20260813_summary.json)。

64题完整工具协议的两个本机输入输出文件为：

```text
runs/llin-disjoint-pair-step120-first-query-20260812-05-full25/validation/0.jsonl
runs/llin-disjoint-native-full25-20260813-01/validation/0.jsonl
```

两文件各64行、两运行退出0。Step120日志有force同步标记且无skip同步标记；原生侧按设计跳过actor同步。这里“full25”表示最多25次工具反馈，**不是同题重复25次**。结果和配对归因见 [完整64题归因](native_vs_step120_full25_attribution_20260813.md)。此前16题自由回放有Step120权重未同步问题，已在 [8月12日更正说明](native_vs_step120_reward_behavior_attribution_20260812.md) 撤回，不能继续作为Step120自然行为证据。

另外，固定val20的老板原版评分文件还在 `evaluations/boss_exact_step120_20260810/`：`step120.validation.jsonl`、`step120.openai.jsonl`、`step120.manifest.jsonl`、`step120.reward.jsonl`、`step120.adapter_summary.json`。这批是20题单次贪心评估，不是8次采样；报告见 [dense试验汇总](step120_dense_trial_20260810_summary.json)，老板总奖励均值0.563745、完整回答16/20、numeric correct 2/20。

## 使用这些材料时的边界

查看完整多轮过程，优先100题 `shards/` 和PI的80份session文件；查看最终结果与候选分桶，使用同一运行的 `outcomes/`；判断训练适用性，还需读取对应语义审阅、严格解析和三态校准。过程奖励、答案正确、可验证证据和可训练性是不同结论。

本次51条索引覆盖上述主要评测及预检、失败、补跑阶段；不是服务器所有含Step120字样目录的穷尽清单。知识选择题、教材CPT/SFT和开源续训Step120均不并入本次Agent目录。所有本次操作为服务器只读核验，没有启动新推理或训练。原题、gold、SQL、工具结果和轨迹正文保留服务器；仓库仅保存汇总、定位说明和分片指纹。
