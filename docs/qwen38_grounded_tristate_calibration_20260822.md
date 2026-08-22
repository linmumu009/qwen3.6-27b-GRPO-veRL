# Qwen3.8 grounded 三态奖励 CPU 校准

日期：2026-08-22
结论：**不放行正式训练**。本轮只完成代码、CPU shadow、逐题 verifier casepack 与隔离源码副本静态合同检查；未加载模型，未启动 Ray rollout 或 optimizer，未占用 NPU。

## 决策摘要

- 冻结输入仍为独立批准包中的 43 题（21 numeric + 22 table），每题恰有 8 条历史轨迹，共 344 条；43 题和 344 条轨迹身份均唯一。
- 新奖励是完整多轮轨迹结束后的单个标量，不是逐 turn/token credit assignment：
  - `PASS`：final 正确、真实证据可重放且满足题目语义合同、过程安全；`train_mask=1, success=1, R=1`。
  - `FAIL`：证据完整且能确定 final、grounding 或安全过程失败；`train_mask=1, success=0, R=0`。
  - `UNKNOWN`：日志/基础设施/解析/证据路线无法可靠裁决；`train_mask=0`，必须重采，不能当负样本。
- 第一轮不使用 table/field/fit/efficiency bonus，也不使用答案长度、数值接近、报告长度、LLM judge 或答案文字自报证据。
- EvidencePlan 已变成逐题绑定的可审计合同，覆盖 filters/time window、JOIN 条件、grouping、order/TopN、metric、ratio/unit。候选 SQL 只有在保守地证明等价时才可 PASS；合法但不能证明的路线为 UNKNOWN；明确漏过滤、改过滤值、漏 JOIN、改粒度、错排序/TopN、错单位则 FAIL。
- 历史文本重建事件的 `call_parse_valid=false` 不再自动归因模型；只有 runtime 结构化事件能把 malformed 判为 FAIL，否则为 UNKNOWN。

## 冻结输入

| 项目 | 结果 |
| --- | --- |
| 批准题 | 43/43，43 个唯一 instruction |
| 历史轨迹 | 344/344，344 个唯一 trajectory identity |
| 批准 Parquet SHA256 | `d86b53d906806b150d43a508dce9b0dd6d05105c07e03961e8e7bf9439ccd944` |
| 批准 manifest SHA256 | `1426bc09a3dbaf4709fd89227790603afb7a2bf11beeba80946057d490e0f424` |
| dataset SHA256 | `c0befda32166340bf68e6b948a1e8fcc6f8f0887d7a5f38a4e6b1051b8f9f7af` |
| tasks SHA256 | `096e6be41c10c4b4c340e941d34533a76d181eb3879dfd2c4ea72639b3ae1e7f` |
| database SHA256 | `6d9c90cb5869dca751ba4865d4e682578105f984b52837a7f75adfdc8d9ef5f8` |

服务器私有产物位于：

`/data3/llin/qwen3.6-27b-verl-grpo/runs/llin-v15-codex-model2-100-step120-8x-20260821-01/grpo_grounded_tristate_calibration_20260822-01`

题面、gold、SQL、工具响应和逐轨迹人工审核包只留服务器；所有私有 JSONL 权限均为 `0600`。

## 344 条自动 shadow

| 指标 | 结果 |
| --- | ---: |
| PASS | 0 |
| FAIL | 253 |
| UNKNOWN | 91 |
| UNKNOWN 率 | 26.45% |
| 新 final parser 正确 | 128 |
| EvidencePlan 合同无法建立 | 0 |
| guess-correct 候选 | 126 |
| guess-correct 被阻断 | 126/126 |
| 自动 grounded 覆盖（相对 final-correct） | 0/128 |
| 历史 43 组中可直接训练 mixed 组 | 0 |

旧 v2 的 344 条奖励均值为 `0.37209302`，126 条为正；新三态二值奖励在当前历史文本重建上全部为 0。旧过程分量均值为：`P_sql=0.11046512`、`P_table=0.95639535`、`P_field=0.72868217`、`P_fit=0.96220930`、`E=0.99808140`。这再次说明 table/field/fit/efficiency 大量饱和，不能作为本轮训练 bonus。

`PASS=0` 不是训练信号，而是当前自动判别器覆盖不足的停止证据。历史轨迹没有完整 runtime 结构化事件，且模型 SQL 常与 gold 使用不同写法；在没有人工混淆矩阵前，不能靠“同库结果相同”放宽为 PASS，否则会重新引入漏日期/区域/状态过滤等假阳性。

## 43 题 verifier casepack

- 43/43 题均执行逐题 casepack；21 numeric、22 table。
- 共生成 946 个 case 行，其中 725 个适用，725/725 通过，失败 0。
- 每题覆盖 gold 直查、两个受证明的等价包装、正确答案无工具、错 SQL 猜中、正确 SQL + 错 final、正确证据后追加无关 1×1、确定性 Python 合成、缺工具响应、unsafe/malformed，以及 table 调序/漏行/漏列/重复行。
- 新增 7 类语义对抗变异：删除时间过滤、改变过滤值、遗漏 JOIN 条件、改变聚合粒度、错误排序、错误 TopN、错误单位/比例。
- 164 个适用的错误 SQL 语义变异中，自动 PASS 为 **0**。
- casepack safe summary SHA256：`d7349d0edc4ef105da8ca5f2f13dbc4d8257a0bd56b27c565aac561360acba0b`。

## 人工 344 条校准状态

服务器已生成恰好 344 行的私有审核 packet 和 344 行人工标签模板，并冻结 identity；但本轮没有可接受的人工标签文件：

- 人工完成：`0/344`
- 人工—自动混淆矩阵：未建立
- 自动 false negative / false positive：未获人工证据
- 可用轨迹被自动 UNKNOWN：未获人工证据
- direct/composed/table precision、recall、unknown rate：未获人工证据
- guess-correct 被自动 PASS：自动 shadow 为 0，但尚未由人工 gold 证明

自动结果不会被标记为“人工审核”。在 344 条人工标签齐全、全部分歧逐条解决前，`formal_training_allowed=false`。

## veRL 静态合同

- 当前 live veRL 容器的 `agent_loop.py` 尚未安装三态 UNKNOWN 重采补丁，因此 live 静态检查按预期失败。
- 在同一容器源码的隔离副本上，补丁器可成功适配：UNKNOWN 最多物理采样 16 条以补齐 8 条 PASS/FAIL，补不齐则跳组；uniform/不可训练组跳 optimizer；hard staleness=0；actor KL loss 使用 active response mask；KL 不进入 reward。
- 隔离副本静态检查通过，但明确记录 `validated_source_mode=staged_copy`、`live_container_patch_installed=false`。正式授权前还必须重新核验并在维护窗口安装 live 补丁。
- staged container contract SHA256：`dfb2f5a9f634af513512df8adb555a43d045f155e5d4d144f04dcd95b4da37b8`。

## 当前阻断与下一门

1. 完成 344/344 条私有人工标注，逐条解决人工—自动分歧；不得用自动 judge 回填人工字段。
2. 混淆矩阵必须证明：原人工错误被自动 PASS 为 0、guess-correct 被 PASS 为 0，并报告 direct/composed/table 各自 precision/recall/UNKNOWN 率。
3. 自动 grounded 覆盖必须足以形成 PASS/FAIL mixed 组；当前 `0` 个 mixed 组无法训练。
4. live veRL 三态重采与严格组门尚未安装，正式启动前必须再次静态/容器核验。
5. 只有上述门均通过且另行收到正式训练授权，才允许加载原始 Qwen3.8-27B、启动 rollout 或 optimizer。

本轮安全摘要 SHA256：`ce420579faf5a54a11e55392d289bae58672c061e36bf86278099d35011fca3a`。
