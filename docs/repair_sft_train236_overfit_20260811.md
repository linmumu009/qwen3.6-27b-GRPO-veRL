# Step 120 真实纠错 SFT 小样本门禁

日期：2026-08-11

## 结论边界

本轮不是 held-out 准确率实验，而是效率优先的两段门禁：先验证 16 条真实、机械可核验的 train236 纠错轨迹能否通过 veRL 官方 SFT 链路改变 Step 120 模型；再在完全相同的 16 条训练题上做训练前后老板原始评分器回放。即使同题回放通过，也只能证明模型学到了这批纠错轨迹，不能直接宣称泛化准确率提升。

## 数据门禁

- 来源仅为正式 `boss_pi_train.parquet` 的 train236；同时加载 val20/test20 做 task-id 隔离检查，重叠为 0。
- 16 条均为 `approved_for_grpo=true`、`review_status=approved`，且 instruction 来自当前任务定义。
- 排除 later-task-definition drift、宽泛任务压成隐藏单值、`LIMIT` 无 `ORDER BY`、numeric gold 无聚合等高风险候选。
- 最终 4 条无启发式语义警告，12 条仅有 `latest_instruction_without_temporal_sql` 单一低风险警告；13 条 numeric、3 条 table。
- 每条 verification SQL 都在 `/pi_sandbox/sft/20260628_v15/logistics.sqlite` 以只读模式执行，结果非空且支持 expected value。
- system prompt 与 `bash/read/edit/write` schema 使用冻结老板合同，哈希分别为 `a60460aa...d2ce3d7` 和 `5aac873b...711cb55`。
- Qwen3.6 实际 chat template 门禁 16/16 通过：总长度 1,513–1,675 tokens，assistant loss 78–171 tokens；system/user/tool response 全部 mask。

门禁还拦截并修正了一处真实格式问题：历史 OpenAI JSONL 把 `function.arguments` 存成 JSON 字符串，但当前 Qwen3.6 chat template 训练入口要求 mapping。最终 parquet 使用 mapping，渲染后的工具语义仍是老板 `bash` 调用。

## 训练配置

- Trainer：veRL 官方 `verl.trainer.sft_trainer`。
- 初始化：Step 120 `actor/model/dist_ckpt`，只加载模型参数；fresh Adam 与 fresh dataloader。
- 拓扑：TP4 / PP2 / CP2，5 号机 16 张 Ascend NPU。
- 全参数 SFT，`lr=1e-6`、constant schedule、无 warmup、无 weight decay、clip grad 1.0。
- 每次更新消费全部 16 条，micro batch 1；5 epochs = 5 optimizer steps，总计 80 条样本曝光。
- 最大长度 2,048；无中途 validation；仅结束保存 model + extra，不保存 optimizer。
- 运行：`llin-repair-sft-train236-overfit-step120-20260811-01`。

## 训练结果

| Step | loss | pre-clip grad norm | 用时 |
|---:|---:|---:|---:|
| 1 | 1.873814 | 65.6652 | 179.22s |
| 2 | 1.645420 | 52.0092 | 74.03s |
| 3 | 1.217984 | 42.0899 | 90.12s |
| 4 | 1.077590 | 40.1865 | 91.52s |
| 5 | 0.576376 | 32.3587 | 约 123s |

loss 从 1.873814 降到 0.576376，下降约 69.2%；梯度范数有限、非零且持续下降。整轮从 `08:51:43 UTC` 到 `09:02:47 UTC`，墙钟 11 分 04 秒。

单卡峰值 allocated HBM 为 26.344 GiB、reserved 26.861 GiB。CPU Adam 峰值达到 1,072.063 GiB，接近机器内存上沿；因此本配置不能安全扩成 data-parallel 双副本，也不应并行启动第二个全参 optimizer。

## Checkpoint

- 最终目录：`runs/llin-repair-sft-train236-overfit-step120-20260811-01/checkpoints/global_step_5`。
- `ckpt_contents.json` 声明且实际仅有 `model`、`extra`，没有 optimizer。
- model dist checkpoint 验证通过：`.metadata` 与 `metadata.json` 均存在，32 个非空 `.distcp` 分片，总字节数 54,720,369,973（磁盘显示约 51 GiB）。
- 该 checkpoint 用于同题回放；是否进入后续 48–64 条 canary，取决于老板原始评分器门禁。

## 工程观察

- 本轮 `driver.log` 约 315 MiB，主要由已知 `mstx.range_end()` 性能标记噪声膨胀；不影响训练结果，但下次应在镜像层修正或抑制，避免长训日志无效增长。
- veRL 的 `optim.betas=[0.9,0.95]` 配置在打印的外层 optimizer config 中存在，但底层 Megatron OptimizerConfig 仍显示 beta2=0.999；本轮仅 5 步未据此重跑，后续正式扩量前应显式下沉 `adam_beta2` 并加启动合同断言。
- 16 条同题 replay 使用老板四工具、48K、25 个工具反馈回合、greedy n=1；val-only 从任意 Megatron dist checkpoint 初始化时强制 actor→vLLM 权重同步，避免误用基础 HF 权重。

## 同题回放

回放由 `llin-repair-sft-prepost-20260811-01` 无人值守流水线执行。它先跑原 Step 120，再跑 SFT Step 5，用老板原始 `reward_judge.py` 重评并做逐题配对。门禁要求训练后至少 14/16 条 exact result success；该门禁通过也不允许产生 held-out 泛化声明。

Step 120 基线回放于 `2026-08-11 17:13:58 +08:00` 启动。状态复核时 Ray 两个节点均 active、无 pending demand 与 recent failure，AgentLoop 仍在生成长轨迹；流水线会在基线退出后自动执行训练后回放、老板原始评分和配对比较，无需人工值守。
