# Qwen3.8 Outcome-gated 多轮轨迹奖励 Shadow v2

## 结论

正式训练继续暂停。本轮完成了修订实现、冻结 800 条轨迹的 CPU-only shadow、实际 veRL 容器的 CPU 合同验证和 43×4 派生 schedule 构建测试；没有加载 Qwen3.8、没有新 rollout、没有调用 optimizer，也没有新增 NPU 占用。

审核后的第一轮训练奖励应退回纯二值：

\[
R=H\cdot C
\]

候选过程奖励公式 `R=H·C·(1+0.10·P_verified)` 在数学安全性上通过：36 个严格 mixed 组中，所有错误轨迹的 GRPO advantage 都严格小于 0，错误正 advantage 计数为 `0`。但 `P_verified` 在硬门通过且最终正确轨迹中的覆盖率只有 `34/287=11.85%`；20 条私有阳性抽样的自动结构核验为 20/20，但受敏感数据不出服务器约束，本轮没有建立独立人工语义精度。因此过程 bonus 不晋级，正式 launcher 默认 `alpha=0`，只有未来覆盖与人工精度同时单独放行后才允许改为 `0.10`。

严格终值解析使 800 条历史轨迹的正确标签从 483 降到 289，共 194 条变化，其中 118 条为明确数值歧义。原批准 43 组按新口径有 36 组 mixed、7 组全错；这 7 组会零优势跳过，不会为了凑步数进入更新。22 道修复表格假阴性全部保持 22/22 与审计正确数一致且仍为 mixed。

## 奖励与解析合同

硬门 `H` 同时要求：gold SQL 与 gold 自洽、数据库可用、工具协议完整、所有工具安全只读、EvidencePlan/verification SQL/required tables/must-use fields 的绑定哈希匹配。任一失败，轨迹不可训练。

最终正确性 `C∈{0,1}`：

- numeric：仅解析唯一显式 `final answer/final result/最终答案/最终结果` 字段；若没有显式字段，只读取最后一个非空行；多字段、多数字或矛盾候选失败关闭。`abs_tol=1e-3`、`rel_tol=1e-5`。
- table：比较完整行列和重复行计数；只有 verification SQL/EvidencePlan 明确包含 ORDER BY、TopN、ranking 或 trend 语义时按序，否则按完整行多重集合。

`P_verified=1` 只在最后一个 answer-bearing 的成功只读查询，其完整结果与严格最终结果一致时成立。SQL 必须来自结构化工具事件中的真实执行器；`echo/cat/grep` 里的 SQL 文本不能产生证据。Python `sqlite3.execute`、`read_sql/read_sql_query` 和多轮 sqlite3 查询已纳入抽取。`P_table/P_field/P_fit/E` 继续作为观测指标，全部不进入训练 reward。

当前仍是完整多轮轨迹结束后得到一个 scalar，不是逐 turn/token credit assignment。若以后做真正逐轮 shaping，必须额外持久化 turn/token span 并实现 return-to-go，本轮没有伪称已实现。

## 800 条 Shadow 结果

| 指标 | 结果 |
| --- | ---: |
| 历史宽松 numeric 正确 | 483/800 |
| 严格终值正确 | 289/800 |
| 标签变化 | 194 |
| numeric 歧义 | 118 |
| 全部题组：全对 / mixed / 全错 | 12 / 54 / 34 |
| 批准 43：mixed / uniform | 36 / 7 |
| mixed 中错误正 advantage | **0** |
| mixed 中错误非负 advantage | **0** |
| `P_verified` 覆盖 | 34/287 = 11.85% |
| `P_verified=1` 且最终正确 | 31 |
| 硬门通过 / 失败 | 774 / 26 |

新奖励的错误轨迹均值和最大值都为 `0`；正确轨迹均值 `1.0038`，范围 `0–1.1`。正确轨迹出现 0 仅来自 `H=0`，这些轨迹会在重采层被拒绝，而不是作为错误样本训练。旧 boss shadow 下，按新严格标签划分的错误轨迹均值仍为 `0.7899`、最大 `1.0`。

批准 43 组的匿名严格正确数如下，顺序只用于复核、不带题目身份映射：

`0, 6, 2, 2, 2, 1, 3, 6, 1, 0, 0, 1, 0, 1, 1, 1, 2, 0, 1, 0, 0, 3, 6, 2, 3, 6, 6, 1, 3, 7, 6, 3, 7, 3, 6, 7, 2, 3, 5, 7, 5, 5, 2`。

过程观测项仍缺少有效区分：`P_sql` 在正确/错误轨迹的均值分别为 `0.1073/0.1076`，几乎相同；表命中、任务适配和效率接近饱和。这个结果进一步支持第一轮只使用 `H·C`，而不是强行加入稀疏过程 bonus。

## 硬门重采

正式补丁在每个名义组最多生成 16 个物理候选，只允许 `online_eligible=H=1` 的轨迹填入 8 条 GRPO 组：

- 先得到 8 条 H=1：组就绪；
- 16 次尝试后不足 8 条：用 H=0 占位保持张量形状，组门随后清空整组 `advantages/returns/response_mask`，不调用 optimizer；
- 不会把 H=0 当作 C=0 的正常错误轨迹训练。

按每题历史 8 条的 H 通过率做独立二项 bootstrap，16 次上限下预计可填满 `42.892/43` 组，单组最低成功概率 `90.01%`。这是历史率假设下的容量模拟，不是未来在线成功率保证。

## veRL 容器合同

实际 veRL 容器中只对源码副本应用补丁并编译，没有改动 `/verl`。24 项容器 CPU 测试通过：

- 全错/全对/硬门失败/陈旧组同时清零 `advantages`、`returns`、`response_mask`；
- 整批无严格 mixed 不进入 actor update/optimizer.step，Adam 参数和状态哈希不变；
- mixed+uniform 的 policy+KL 梯度与 mixed-only 在容器 CPU 容差内一致；
- 每批显式要求 `min_global_steps=max_global_steps=current_policy_version`，实现硬陈旧度 0；
- KL 不进入 reward，actor 使用冻结 reference 的 `low_var_kl`，系数 `0.001`，实际 veRL loss 以 active `response_mask` 聚合 KL；
- 记录 `nominal_group_step` 与真实 `optimizer_step`，跳过更新时也跳过无意义权重同步。

新的 launcher 固定原始 `/models/Qwen3.8-27B` 同时作为 actor/ref，5/6 号机 config SHA256 均为 `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`；冻结 18 shard 复合哈希为 `e94e58ab1b25c6bbbff809b8af9f57a5d42eeff7db7b077a8248d359a20b7325`。合同为 43×4=172 个名义组、1376 条被接受轨迹、LR `5e-8 constant`、entropy 0、KL 不进 reward、最终只保存 `model,extra` 一份持久 checkpoint。launcher 仍要求 post-shadow 明确令牌，当前不能误启动。

真实批准包的 CPU 构建测试得到 172 行 schedule、43 个唯一 EvidencePlan/field 绑定哈希，schedule SHA256 为 `c0ad4fa8e2d049bf049c9de9dc2861e171cc29cd778a7b1981223bceb9fbafb5`。成员只来自冻结 43 行批准包，不扫描原 100 题扩充。

## 门禁结论与保留证据

`formal_training_allowed=false`，原因有三项：

1. `P_verified` 覆盖只有 11.85%，未达到采用过程 bonus 的最低可信度；
2. 人工语义精度未建立，只完成了不导出敏感内容的 20/20 自动结构核验；
3. 严格终值解析改变 194 个历史标签，需要老板审核新解析口径及 7 个 approved43 uniform 组后再决定是否放行纯二值金丝雀。

服务器私有目录：`.../grpo_reward_shadow_20260822-02`。安全摘要 SHA256 `1e8a9cb48975bbabc0ce938872de97c2c3ec8b03824b0bf8f8d73e72b30628b0`；结构核验 SHA256 `d4182f7b49736ff91167b1aa32f470eb94874300c33719cc3f4f56c8fecf07b8`；容器合同 SHA256 `8932f38017c995bb251c392ba3e29f10548fcecc7c7a81223ba4a89b9bb03feb`。敏感题面、答案、SQL、工具结果和身份映射未写入 Git。

机器检查时 5、6 号机均无 NPU 模型进程和 AICore 负载；5 号机仍保留上一轮空闲 Ray 基础服务，这不等于训练已启动。
