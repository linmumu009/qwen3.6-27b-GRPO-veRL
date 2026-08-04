# 老板 PI 数据对齐逐项更正记录（2026-08-04）

## 结论

旧 formal V2 不可继续用于正式 GRPO。它虽然能完成工程训练，但数据链路把 Qwen3.7-Max 跑批 manifest 与 Qwen3.6 conversation 混在一起，system prompt 使用项目 fallback，工具 schema/runtime 也不是老板源定义的精确副本；更严重的是，未经 instruction/gold 语义复核的 hidden SQL 被直接提升为在线奖励目标。

本次已将这些问题改成代码级硬门禁。新的正式入口只接受 `boss-pi-aligned-grpo-v1` 契约；在 277 条 v15 Qwen3.6 候选完成显式对齐审核并生成 train/val Parquet 前，正式训练会在模型加载前失败。没有启动新训练。

## 1. 为什么不再重新筛固定 200 条

旧构建器默认 `160/20/20`，无论源数据实际有多少都固定选择 200 条。这个数字适合作为 pilot，但不应冒充完整正式数据。

更正如下：

- `scripts/prepare_boss_aligned_dataset.py` 默认保留全部已审核、可执行 verifier 样本。
- 只有显式传入 `--pilot-size SPLIT=COUNT` 时才截取，并把数据契约标记为 `mode=pilot`。
- `scripts/check_boss_alignment_contract.py` 和正式启动脚本默认拒绝 pilot 数据。
- 旧 `scripts/prepare_pi_formal_dataset.py` 已标记为历史复现工具，必须显式传 `--allow-legacy-v2` 才能运行。

## 2. system prompt 如何更正

老板源工程的权威定义位于：

`/data/renjunxiang/coding/huawei_train/scripts/data/pi_to_openai.py`

源文件 SHA256：

`f665fa5f0b5fc355dc2b178ab90169753a3ef864ea9231c0adf2e0a00f9c270f`

本项目把其中 `DEFAULT_SYSTEM` 原样冻结在 `configs/boss_pi_contract.json`，其内容 SHA256 为：

`a60460aa18bade618df488754b523f68d7b610c7bceddcc6450bd6a84d2ce3d7`

新构建器不再存在项目 system fallback。历史 converted 文件若带有旧的“物流分析师”提示，会严格按老板自己的 `fix_sft_data_system.py` 语义替换成上述 `DEFAULT_SYSTEM`，并记录源脚本与哈希。契约中的 `project_system_fallback_count` 必须为 0。

## 3. 四工具 schema 与 runtime 如何更正

老板 `DEFAULT_TOOLS` 的 `bash/read/edit/write` 完整描述、参数、required 字段和 `additionalProperties` 已原样冻结。tool schema 的规范化 SHA256 为：

`5aac873b3d92213ef43ba90f210f65931b0aea60f747bbd0c519800ec711cb55`

`configs/pi_workspace_tools.yaml` 现在与该 JSON schema 做测试级精确比较。runtime 同步修正了以下行为：

- `read.offset` 改为老板定义的 1-based 行号。
- `read/bash` 输出按 2,000 行或 50 KiB 截断。
- bash 截断时把完整输出保存在轨迹工作区，再把路径返回给 Agent。
- 单工具上限放宽到与老板 agent 任务相同的 900 秒边界。
- 每条轨迹仍复制独立 `/workspace`，文件布局保持 `logistics.sqlite`、`documents/` 和 `schema_dictionary.md`。

唯一保留的差异是安全边界：GRPO replica 禁止访问宿主机、网络和破坏性命令，而老板 runner 使用 host network。该差异写进契约，不能再宣称“runtime 完全无差异”；对当前离线物流 sandbox 的任务可见文件和工具 schema 已对齐。

## 4. hidden gold 与奖励来源如何更正

### 4.1 发现的真实血缘错误

旧 formal V2 的首条 manifest 记录中，`output` 明确指向：

`trajectories_qwen37max/...`

而当时用于讨论“老板 Qwen3.6 原轨迹”的 conversation 来自 Qwen3.6 converted 文件。两个跑批都会随机选 instruction variant，不能按行号或模糊文本混用。

首次严格文本连接的结果也证明了问题：4,500 条 Qwen3.6 conversation 与旧 V2 manifests 只有 291 条能按 user instruction 精确连接。

### 4.2 新的源身份链

本次改用老板 v15 Qwen3.6 原始事件目录：

`/data/renjunxiang/coding/huawei_train/archives/trajectories_v15_27B_table`

使用老板自己的 `pi_to_openai.py` 重新转换 1,500/1,500 条事件轨迹，并保留文件名中的真实 `task_id`。随后从同一个 sandbox：

`/data/renjunxiang/pi/sandbox/sft/20260628_v15`

无损导出 1,500 条 task 定义。连接键是原始事件文件 `task_id`，不是 row order、相似文本或另一个模型的 manifest。

### 4.3 当前严格分流结果

1,500 条 v15 Qwen3.6 源轨迹全部进入 SFT/reference 导出。在线 GRPO 只考虑当前 reward 能严格执行的 DWH numeric/table：

| 分流 | 数量 | 处理 |
| --- | ---: | --- |
| KB/Hybrid | 1,000 | SFT/reference only |
| DWH 报告型或无严格 numeric/table verifier | 220 | SFT/reference only |
| gold SQL 执行结果与 gold 值不一致 | 3 | 拒绝 |
| task_id 同源且 verifier 可执行 | 277 | 进入 alignment review queue，尚未进入 GRPO |

277 条候选必须逐条审核以下两个固定哈希：

- `instruction_sha256`：真实 Qwen3.6 事件里的 user instruction；
- `gold_sha256`：同一 boss task 的 answer type、value 与 verification SQL。

审核记录必须包含 `approved_for_grpo=true`、`reviewer`、`reviewed_at` 和 split。任一哈希变化都会使批准失效。没有审核的样本只留在 review queue；不会生成 GRPO Parquet。

老板历史 `judge_trajectory.py/reward_judge.py` 是对完整历史轨迹做判分的实现证据，但它不自动等价于本次在线 GRPO verifier。新链路只复用同源 task 的可执行 gold，并额外要求 instruction/gold 对齐审核。

## 5. 老板完整 response 如何使用

GRPO 输入只保留权威 system 和真实 user instruction；历史 assistant、tool call、tool result、最终回答的输入计数强制为 0。这符合 GRPO 在线重新采样 response 的要求。

老板完整轨迹没有丢弃，而是单独写入 `boss_pi_sft_reference.jsonl`：

- 可用于 SFT warm start；
- 可用于工具行为、轮数、输出长度和最终答案回归；
- 不会作为 GRPO 的预生成 response；
- 不会把历史答案泄漏给 rollout。

## 6. 正式启动门禁

`scripts/run_pi_formal_50step.sh` 的默认数据目录已改为：

`data/boss_pi_aligned_v1`

启动前必须通过 `scripts/check_boss_alignment_contract.py`。它会验证：

- full 而非 pilot；
- source system/tool/contract 三个哈希一致；
- 无 fallback、无自造 instruction/gold/SQL、无未审核 GRPO 行；
- GRPO prompt 不含历史 assistant/tool messages；
- train/val 无 task/instruction 泄漏；
- Parquet 文件 SHA256 与契约一致。

旧 `formal_pi_v2_20260803` 即使文件仍在，也不能再通过正式入口。

## 7. 当前服务器产物与状态

5 号机只读审计产物位于：

`/data3/llin/qwen3.6-27b-verl-grpo/data/boss_pi_alignment_stage_20260804`

关键产物：

- v15 原始 Qwen3.6 task-id conversation：1,500 条；
- 同源 boss task manifest：1,500 条；
- SFT/reference：1,500 条；
- 严格 GRPO alignment review queue：277 条；
- 当前已批准 GRPO：0 条；
- 当前新训练：未启动。

v20/v21 目前只有不带 `task_id` 的 Qwen3.6 converted conversation，未发现与 v15 同等级的 Qwen3.6 原始事件身份档案。因此它们可以保留为 SFT/reference，但在找到原始事件文件或权威 task-id 映射前，不进入 boss-aligned GRPO。

## 8. 下一步

下一步不是继续调训练参数，而是完成 277 条 alignment review：

1. 优先复核“最新/最近”、`LIMIT` 无确定排序、宽泛分析问题单一 hidden target 等已知高风险项。
2. 给批准行分配 train/val/test，保持 task ID 和 instruction hash 隔离。
3. 用全量批准行生成 `boss_pi_train/val/test.parquet` 与最终契约。
4. 先跑冻结模型基线和小规模奖励重放；通过后再恢复正式 GRPO。

在这四步完成前，正式入口会保持阻断。
