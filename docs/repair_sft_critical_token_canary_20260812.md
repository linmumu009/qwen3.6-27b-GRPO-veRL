# Semantic critical-token 一步金丝雀结论

日期：2026-08-12
结论：训练与前向诊断均成功，但完整纠正 SQL 概率门禁失败；停止，不做短回放、不追加训练、不晋级。

## 目标与冻结变量

本轮验证一个严格单变量问题：在 Step 120、相同 16 条“模型首错 SQL + 已观察工具结果”状态、相同一步更新和相同 `tool/sql/final = 0.25/8/1` 基础权重下，只把每题 semantic-mask v3 的首个非 greedy SQL token 权重从 `8×` 提升到 `32×`，能否让整条纠正 SQL 进入可靠概率区间。

没有加入人工解释或提示，没有改变目标 SQL、最终答案、数据行、学习率、训练步数和并行拓扑。晋级条件预先固定为：16 题中至少 12 题的完整纠正 SQL 几何平均目标概率大于 `0.5`。只改善被加权 token、平均 NLL 或 token rank 均不足以启动 replay。

## 训练前门禁

- 数据：16 条；Step 120 同题开发门禁；held-out overlap 为 0。
- semantic critical-token 家族：aggregation function `9/16`、query start `3/16`、identifier/literal `3/16`、clause keyword `1/16`。
- 每条样本恰有 1 个 critical token，目标 token ID 与冻结诊断一致，权重恰为 `32×`。
- 首错 assistant 与其真实工具结果只作上下文，loss mass 为 0；所有序列在 `8192` token 上限内无截断。
- CPU mask 门禁通过后才占用 NPU。

## 一步训练结果

运行 `llin-repair-sft-critical-token-step120-1step-20260812-01` 于 2026-08-12 07:35:07–07:38:51 UTC 完成，退出码为 0。

| 指标 | 结果 |
| --- | ---: |
| train loss | 1.8237778 |
| grad norm | 171.6451 |
| learning rate | 1e-6 |
| global tokens | 35,277 |
| 单卡峰值 HBM | 26.5908 GiB |
| checkpoint | 32 个 model distcp + 8 个 extra distcp，约 51 GiB |
| optimizer 文件 | 0 |

checkpoint 仅用于本轮诊断；它不是可晋级模型。

## 同数据 forward-only 结果

Step 120 与训练后 checkpoint 均在同一 critical Parquet 上以 TP4/PP2/CP2、8K、forward-only 运行，数据 SHA256 和 16 个 task ID 完全相同，二者均未初始化 optimizer。

| 指标 | Step 120 | Critical-token Step 1 | 变化 |
| --- | ---: | ---: | ---: |
| official assistant loss | 1.623356 | 1.576742 | -0.046614 |
| semantic SQL mean NLL | 1.292942 | 1.143501 | -0.149441（-11.56%） |
| semantic SQL 几何平均目标概率 | 0.274462 | 0.318701 | 1.1612× |
| SQL greedy tokens | 221/300 | 225/300 | +4 |
| SQL top-5 tokens | 266/300 | 270/300 | +4 |
| SQL mean rank | 21.54 | 16.76 | -4.78 |
| 全 SQL token greedy 的任务 | 0/16 | 0/16 | 0 |
| 完整 SQL 概率 > 0.5 | 2/16 | 2/16 | 0 |

SQL NLL 在 `16/16` 题上改善；tool structure 与 final answer NLL 也分别在 `16/16` 题改善。它们说明更新方向有效，但不能替代完整 SQL 概率门禁。

## 原 critical token 归因

- `3/16` 个冻结 critical token 转为 greedy，且全部是 query-start 家族的 `3/3`。
- `13/16` 仍是该题首个 non-greedy token；其中 rank 改善 `11`、持平 `2`、恶化 `0`。
- aggregation function `9/9`、identifier/literal `3/3`、clause keyword `1/1` 均没有转为 greedy。
- 没有任务出现比冻结位置更早的新 non-greedy 分叉。

因此本轮并非完全无效：它精确修复了 query-start 局部障碍，并普遍提高剩余 critical token 的 rank；但 aggregation 等语义选择仍未跨过 greedy 边界，且整条 SQL 的可靠性没有改善到可回放水平。

## 决策与下一步

主门禁结果为 `2/16 < 12/16`，故执行 `stop_no_replay_probability_gate_failed`：

- 不做短 replay、48K 完整 replay、held-out 或老板完整评分；
- 不在该 checkpoint 上继续增加步数或继续放大单 token 权重；
- 不作泛化、准确率或可部署声明。

下一训练目标应从“首个 token 放大”切换为“机械抽取的语义差异 span”：仍保留实际首错状态与同一正确目标，但只对错误 SQL 与纠正 SQL 之间经审计的聚合/select/group/filter/table 等差异 span 加权。第一阶段仅实现无 NPU 的 span 提取、token 对齐和 fail-closed mask 门禁；其后若再做一步金丝雀，仍以完整 SQL `>0.5 ≥ 12/16` 为主门禁，并额外要求 aggregation 家族不出现更早分叉。

## 安全证据链

仓库不保存原始问题、SQL、答案、Parquet、checkpoint 或服务器绝对路径。关键服务器侧证据 SHA256：

- critical mask gate：`1f2d130206547bbff822d3b1d4fe540caebac41c17d2f2c941806fe4f08b8e92`
- Step 120 forward result：`b883a858923fc4bf3780ad4f0cfe5e6dbb389c71856155c81b6f0b446b1961af`
- Step 1 forward result：`266d716e14a7239586799c5e695965756a795b58cf8c4a0386472a32ae3d7c8f`
- pre/post comparison：`d0cbba5cdd65e421c5ef022b7e64cc5b2b13b580a45fca8551c8d1efa9c34710`
- critical-token attribution：`9454af6b578597340a5356f59ed54706501b5ab6be62771bcdc3d7626d291879`

实验结束后 huawei-05 与 huawei-06 均为 8/8 张 NPU 空闲。
