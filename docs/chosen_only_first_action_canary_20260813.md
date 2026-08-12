# Chosen-only schema-conditioned 首动作一步金丝雀

## 结论

这一步证明了梯度方向有效，但没有证明策略边界已经被可靠改变。对 48 条不重叠训练样本只做一次 `tool structure / SQL = 0.25 / 8` 全参更新后，相同 calibration16 的 SQL NLL 从 `1.2913` 降到 `1.1267`（相对改善 `12.75%`），`16/16` 逐题改善；然而 greedy SQL token 只从 `277` 增至 `282`，低于训练前冻结的 `+12` 门槛，且整条 SQL 全 greedy 仍为 `0/16`。

因此预注册的 7 项门中 6 项通过、1 项失败。最终决策是：**停止该 chosen-only 金丝雀，不跑自由回放，不追加训练，不晋级 checkpoint。**

## 1. 因果边界

- 起点固定为 Step 120 model-only 状态；optimizer 与 dataloader 状态全新。
- 训练只读取 train48；calibration16、冻结 16 题、val20 和 test20 均未进入更新。
- 每条样本只监督一个正确 bash/SQLite assistant 首动作，不包含工具结果或最终答案 loss。
- prompt 只含 oracle 选中相关表的 schema，因此这是 task-specific schema 上界诊断，不是可部署输入。
- 训练后只运行同一 calibration16 的 teacher-forced 纯前向；在门通过前不允许自由生成。

## 2. 训练前强门

train48 再次通过完整 Qwen3.6 chat template 的 CPU 门：48/48 无截断、loss 恰好覆盖唯一 assistant tool action、system/user loss 全为 0，tool structure 与 SQL mask 均非空、互斥且完整。序列长度为 `1,639–2,035` tokens，首动作 `61–112` tokens，SQL `6–57` tokens；训练文件 SHA-256 仍为 `bcb6e2f5d9c20318a81f35f00b668c1926af71666a165d885527bfa99db5ed07`。

首个 run `-01` 在 CPU 门导入阶段因容器 `/verl/scripts` 遮蔽项目同名 namespace 而退出，未初始化 optimizer、未占用 NPU、未更新参数、未保存 checkpoint。v1.4.1 调整 Python 搜索路径并加入回归锁定后，使用新目录 `-02` 重新执行。

## 3. 一步训练

| 项目 | 实际值 |
| --- | ---: |
| 训练行数 / optimizer steps | `48 / 1` |
| 拓扑 | `TP4 / PP2 / CP2` |
| 学习率 | `1e-6` |
| loss 权重 | `tool=0.25, SQL=8` |
| train loss | `1.347718` |
| grad norm | `60.4354` |
| global tokens | `85,754` |
| 单芯片峰值 HBM | `26.53 GiB` |
| 整机 CPU 内存峰值 | `834.41 GiB` |
| 端到端墙钟 | `358s` |
| 退出码 | `0` |

最终 checkpoint 只包含 model 与 extra：model 为 32 个 `.distcp` 主分片加 3 个 metadata/common 文件，extra 为 8 个 `.distcp` 主分片加 3 个 metadata/common 文件，二者均有 `.metadata`；optimizer 文件为 `0`。模型目录约 `54.72 GB`。该 checkpoint 仅保留作诊断证据，不具备 promotion 资格。

## 4. 相同 calibration16 训练前后

训练后纯前向退出码为 0，墙钟 `110s`，未初始化 optimizer、未保存 checkpoint。

| 指标 | Step 120 | 一步后 | 变化 | 门槛 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| SQL mean NLL | `1.291303` | `1.126706` | `-12.75%` | 相对改善 `≥5%` | 通过 |
| SQL NLL 改善题数 | — | `16/16` | — | `≥12/16` | 通过 |
| greedy SQL tokens | `277` | `282` | `+5` | `≥+12` | **失败** |
| top-5 SQL tokens | `344` | `348` | `+4` | `≥344` | 通过 |
| SQL mean rank | `18.7795` | `15.8661` | 改善 | 必须改善 | 通过 |
| tool structure NLL | `1.423074` | `1.385288` | `-2.66%` | 相对退化 `≤5%` | 通过 |
| 更早 SQL 边界退化 | `0` | `0` | — | `0` | 通过 |
| 整条 SQL 全 greedy | `0/16` | `0/16` | `0` | 观察项 | 未改善 |

## 5. 为什么 NLL 改善仍不足以开放回放

首个 non-greedy SQL 边界显示，更新主要降低了整段教师路径的平均损失，却没有普遍跨过局部 argmax：

| 首分叉家族 | Step 120 | 一步后 | 原边界清除 |
| --- | ---: | ---: | ---: |
| aggregation function | `9` | `10` | `0/9` |
| query start | `3` | `1` | `2/3`，但仅移动到更后的分叉 |
| clause keyword | `3` | `3` | `0/3` |
| identifier / literal | `1` | `2` | `0/1` |

合计只有 `2/16` 的首分叉向后移动，`14/16` 保持同一 offset；没有更早退化，但也没有任何一题达到整段 SQL 全 greedy。尤其原有 aggregation 障碍 `9/9` 全部保留，说明当前瓶颈仍是聚合/operator 语义选择，而不是工具 JSON、SQLite 命令外壳或一般 schema 可见性。

## 6. 与“奖励黑客来自哪里”的整体归因

此前原生 Qwen3.6-27B 与 Step 120 的公平对照已经表明，高过程分但答案错的模式在原生模型中更常见，训练没有新制造或放大该模式；Step 120 另外引入的是查询覆盖与完成率下降、正确数小幅上升的权衡。本轮 chosen-only 金丝雀进一步说明：给出 task-specific schema 并直接监督正确首动作，可以一致提高正确 SQL 的教师概率，但一次更新仍不足以越过主要 semantic decision boundary。因此不能把现象归因于单纯的工具格式学习，也不能用继续堆相同 chosen-only 步数来替代因果诊断。

## 7. 下一步

下一阶段不继续使用本 checkpoint，也不降低门槛。最有价值的无训练动作是构建一个新的、与所有冻结集合不重叠的 aggregation/operator 对比数据审计：从经过语义复核的任务中生成只改变 `COUNT/SUM/AVG/MIN/MAX`、`DISTINCT`、grouping 或 select target 的最小反事实 rejected action；要求 chosen/rejected 均只读可执行、chosen 继续支持 verifier、rejected 结果机械不等价，并按 operator 家族分层。

在至少 48 个严格可用、来源清楚且无重叠的 pair 之前，训练保持关闭。现有 current-definition review-required 池不能仅凭 SQL 可执行和 gold 自洽自动放行。
