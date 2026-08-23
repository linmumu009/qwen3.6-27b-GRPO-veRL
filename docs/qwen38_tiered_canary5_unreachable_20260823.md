# Qwen3.8 tiered-v1 五步金丝雀结果：活性修复通过，但训练信号不可用

## 结论

本轮不能放行完整训练。

活性修复在真实 fully-async 运行中生效：前两批连续被严格门禁跳过后，第三批仍能在同一 policy version 下继续产生，原先“跳过后永远不再采样”的确定性死锁已经消除。但新的金丝雀在 6 个名义批次、12 个题组、96 条完整轨迹后仍为 `PASS=0`、strict-mixed 题组 `0`、实际 optimizer 更新 `0`。此时剩余最多 4 个名义批次，即使每批都更新也只能新增 4 步，无法达到 5 步目标，因此在 `2026-08-23 15:50 +08:00` 按合同安全停机。

奖励边界在这 96 条上没有被突破：错误奖励大于 `0.2`、有证据正确奖励小于 `0.8`、猜对无查询获得非零奖励、unsafe/超预算获得非零奖励、UNKNOWN 或 uniform 产生非零优势、错误奖励高于正确有证据奖励均为 `0`。但这是“没有误放高分”的安全证据，不是“奖励可用于训练”的有效性证据；96 条奖励和优势全部为 `0`，KL、grad norm、actor loss 与 Adam 变化均没有机会被实测。

因此判断是：tiered-v1 的安全边界目前有效，在线训练效用未成立；没有观察到 reward hacking 获利，但也没有任何正奖励或可训练组，不能推荐后续全量。

## 运行与资产

- run：`/data3/llin/qwen3.6-27b-verl-grpo/runs/llin-qwen38-approved43-tiered-v1-canary5-20260823-03`
- 运行时间：`13:40:40 → 15:50:43`，约 2 小时 10 分钟。
- 角色：5 号机 actor/ref，6 号机 rollout；专用 Ray 端口 `36379`。
- actor/ref 起点均为 `/models/Qwen3.8-27B`，未复用 Qwen3.6、Step120、旧 Qwen3.8 checkpoint 或金丝雀权重。
- Qwen3.8 config SHA256：`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`。
- 18 分片复合 SHA256：训练前与停机后两机均为 `e2c3b44e4e198e94fcd74903983fc8997f8e504a21575e397f9d59db1cc2fc8f`。
- approved43 Parquet/manifest SHA256：`d86b53…d944` / `1426bc…f424`，没有扫描 100 题扩充训练集。
- canary20 与 sealed8 在两机的 SHA256 分别为 `d07b03…86f3` / `5fad9c…c698`，私有文件保持 `0600`。
- 冻结合同：tiered-query-cost-v1、LR `5e-8` constant、entropy `0`、KL 不进 reward、actor KL loss `0.001 low_var_kl`、硬陈旧度 `0`、每个名义批次 `2×8`。

## 活性修复的真实证据

修复 commit 为 `4652637`。uniform/UNKNOWN skip 继续保持以下状态不变：optimizer、actor 参数、Adam state、policy version、权重广播；唯一新增动作是调用 `rollouter.reset_staleness()`，重新开放同一 policy version 的采样许可。

真实运行中，批次 1 和 2 都没有 strict-mixed 组，全部跳过；随后批次 3 成功产生并落盘。最终连续生成 6 个 JSONL、每个恰好 16 行，共 96 行。该事实直接证明“两组许可耗尽并连续 skip 后，第三组仍可产生”。本地既有合同测试 `43` 项、5/6 号机实际容器各 `29` 项通过；新增安全汇总器 `4` 项测试通过，三处 runtime SHA 一致。

## 逐批奖励与耗时

| 名义批次 | PASS / FAIL / UNKNOWN | mixed组 | q均值 / 最大 | E均值 | 输出tokens均值 | queue wait | ref | actor update | 整步 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 / 2 / 14 | 0 | 0.188 / 1 | 0.006 | 4,883 | 524.47s | 111.94s | 0.00s | 636.71s |
| 2 | 0 / 0 / 16 | 0 | 0.875 / 6 | 0.024 | 4,903 | 846.88s | 81.96s | 0.00s | 929.10s |
| 3 | 0 / 8 / 8 | 0 | 12.875 / 39 | 0.256 | 14,389 | 1,367.32s | 107.74s | 0.00s | 1,475.37s |
| 4 | 0 / 0 / 16 | 0 | 0.313 / 1 | 0.000 | 4,460 | 477.94s | 77.93s | 0.00s | 556.16s |
| 5 | 0 / 3 / 13 | 0 | 0.063 / 1 | 0.000 | 4,724 | 644.32s | 77.56s | 0.00s | 722.15s |
| 6 | 0 / 1 / 15 | 0 | 0.188 / 2 | 0.000 | 4,555 | 1,168.27s | 83.25s | 0.00s | 1,251.88s |

12 个题组都恰有 8 条，题目 SHA256 身份 96/96 完整；轨迹 SHA256 身份只有 48/96 可形成，另外 48 条因 request identity 缺失而 fail-closed。每组的 8 个 reward 和 8 个 advantage 均为全零，组内没有可排序信号；所有组都被严格门禁跳过。

## 为什么没有训练信号

总计状态为 `PASS=0 / FAIL=14 / UNKNOWN=82`。原因可加总为：

- `database_unavailable=43`
- `unsupported_table_presentation=22`
- `tool_response_cost_unobservable=17`
- `budget_exceeded=6`
- `unsafe=5`
- `no_relevant_readonly_attempt=3`

96 条中有 22 条相关只读 SQL attempt、20 条被事件层识别为成功相关查询，但 `tool_response_tokens` 在全部 96 条上都是 `0`；成本不可观测或数据库/表格判别异常将大量样本变为 UNKNOWN。唯一一条 final-correct 同时是无相关查询的 guess，正确地被阻断为 reward `0`。第 3 批出现最明显的暴力查询行为：q 均值 `12.875`、最大 `39`、输出 tokens 最大 `47,874`，其中 6 条命中 hard budget；它们全部 reward `0`，没有通过暴力查库获得高分。

最高奖励错误轨迹的奖励仍为 `0`；不存在正确有证据轨迹，因此“最高成本正确”队列为空。reward hacking 没有得到正回报，但由于没有 PASS，也不能用本轮证明奖励能在正确轨迹之间形成有用排序。

## 参数、优化器、KL 与 sealed

- 六个 `[LLIN_TRAIN_STAGE]` 的 `update_actor_s` 全部为 `0.0`；实际 optimizer step 为 `0`。
- parameter audit 文件 `0`、checkpoint 文件 `0`、Adam state 变化 `0`。
- 基座 18 分片复合哈希在两机停机后与训练前完全一致。
- 因没有 actor 更新，actor loss、grad norm、训练 KL、entropy 与 LR 的逐步数值没有产生；不能把“无 NaN/OOM”误写为优化健康。
- sealed8 已按固定哈希分发，但本轮没有生成 sealed validation 文件；同时模型参数从未改变，因此只有“模型文件状态相同”的证据，没有可报告的 grounded-correct 前后性能差。sealed 方向门没有完成，不能作为放行证据。

## 停机与资源

目标不可达证据在停机前冻结：完成 6 个名义批次、剩余 4 个、实际更新 0、目标 5。监督器退出后，5/6 号机专用 Ray/VLLM 进程均为 `0`，两机 NPU process rows 均为 `0`，端口 `36379` 无监听；没有最终模型或临时 checkpoint。

服务器完整安全摘要保留在 run 的 `audit/canary_rollouts.safe.json`，SHA256 为 `5cebe04d5350deb06d5234f47a7ab1195233b837575e4bc5f950f6d1f27c6564`；敏感 rollout 与工具返回只留服务器。仓库只保存本报告与去标识摘要。

## 建议

不建议全量训练。下一轮只应做最小在线证据链修复：确保 runtime 原生持久化 request identity、数据库可用状态、成功工具响应及其 token count，并补齐合法 table final 的解析覆盖；UNKNOWN 仍必须 mask，不得把缺日志当模型错误，也不得放宽猜测、unsafe 或 hard budget 的零奖励规则。修复后仍从原始 Qwen3.8 基座重跑五步实际更新金丝雀；只有出现可靠 PASS/FAIL mixed、真实 optimizer/参数变化、有限 KL/grad/loss，以及 sealed 前后结果后，才有资格讨论全量。
