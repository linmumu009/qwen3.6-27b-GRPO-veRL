# XTuner、veRL、ms-swift 与 MindSpeed 调研及选型结论

日期：2026-08-14

## 结论

当前项目不应从 veRL 迁出。veRL 仍是唯一已经在本项目的 Qwen3.6-27B、Ascend A3、`TP4/PP2/CP2` 全参训练、双机训练/rollout 分离、48K 多轮四工具、完整 checkpoint 恢复和自定义奖励链路上得到实跑证明的框架。

XTuner 值得持续跟踪并做隔离技术验证，但不适合现在直接替换 veRL。XTuner V1 的当前主干已经包含真正的 GRPO/DAPO、Ray、vLLM/SGLang/LMDeploy、同步/异步 replay buffer、partial rollout、工具调用和 agentic RL 代码，不再只是传统 SFT 工具；然而官方入门仍把 RL 标为 Beta，当前训推分离示例还明确使用 mock 权重同步，且没有发现 Qwen3.6-27B 的明确支持条目。这三个事实对本项目都是迁移硬风险。

ms-swift 是最值得做第二套小规模 POC 的候选。它是本次四者中唯一明确注册 `Qwen/Qwen3.6-27B` 的框架，同时具备 Megatron TP/PP/CP/EP、Ray 角色分离、vLLM server/colocate、多轮调度器、自定义环境、奖励插件和 loss mask。限制是其 `async_generate` 官方说明为使用上一轮模型的异步采样，并明确不支持多轮；因此它现成能力不能同时复刻本项目的“多轮工具 + bounded fully-async + 完整 GRPO group 原子性”语义。

MindSpeed 需要分层理解：MindSpeed Core 是昇腾 Megatron 加速层，MindSpeed-LLM 是模型/预训练/SFT 套件，真正与 veRL 同层的是 MindSpeed-RL。MindSpeed-RL 有 GRPO、PPO、DAPO、DPO、训推共卡/分离、异步流水、partial rollout 和 ReTool/Search 多轮能力，但官方已在 2026 年 4 月宣布暂停新增功能集成，并直接建议新方案使用 veRL 的昇腾实践。它适合保留为昇腾实现参考，不适合作为新迁移目标。

## 调研边界与证据等级

本报告比较的是 2026-08-14 拉取到 `reference/` 的官方主干源码，而不是历史版本宣传材料。引用仓库的远端和提交号另见 [`framework_reference_manifest_20260814.json`](framework_reference_manifest_20260814.json)。

证据按以下顺序解释：

1. **本项目实跑**：已在 5/6 号机、Qwen3.6-27B 和当前数据/工具合同上运行；只有 veRL 达到这一等级。
2. **上游实现与官方示例**：上游源码、测试、配置或文档存在，但未在本项目硬件和合同上复验。
3. **路线图或宣传**：只能证明方向，不能证明当前可用。

本次没有在四个框架之间运行统一性能 benchmark，因此不对吞吐、显存或最终效果做跨框架数值排名。

## 框架层级

| 项目 | 实际定位 | 与当前项目的关系 |
| --- | --- | --- |
| veRL | 端到端 RL 编排和训练数据流；训练后端可选 FSDP/FSDP2/Megatron，rollout 可选 vLLM/SGLang | 当前生产主框架，层级和能力完全匹配 |
| XTuner V1 | FSDP 为核心的大模型训练引擎，正在快速扩展 Beta RL、异步调度和 agent loop | 可作为新一代训练/RL 引擎候选，但迁移面很大 |
| ms-swift | 模型、模板、数据、SFT/RLHF/GRPO、Megatron、推理、部署的一体化产品层 | 模型兼容和快速实验最强，深度调度控制弱于当前定制 veRL |
| MindSpeed-RL | 昇腾原生 RL 编排层，底层依赖 MindSpeed/Megatron/vLLM-Ascend | 同层候选，但已停止新增功能集成 |
| MindSpeed Core | Megatron-LM 的昇腾适配、并行、算子、内存和通信加速层 | 是后端组件，不是 veRL 替代品 |
| MindSpeed-LLM | 基于 MindSpeed 的模型、预训练、SFT、评测和工具套件 | 可提供模型实现参考，不负责当前完整 RL 数据流 |

## 面向本项目的能力对比

| 维度 | veRL | XTuner V1 | ms-swift | MindSpeed-RL |
| --- | --- | --- | --- | --- |
| Qwen3.6-27B | 上游未列出精确型号，但本项目已通过自定义数据集/模板和 Megatron Bridge 实跑 | 未发现精确型号条目；通用 HF 配置不等于已验证 | **明确注册并列入支持表** | 未发现精确型号条目 |
| GRPO/RL | 成熟主能力，算法和 worker 扩展面广 | 已实现 GRPO/DAPO 等，但官方入门标为 **Beta** | GRPO 族丰富，HF 与 Megatron 两条路径 | GRPO Released；部分 Qwen3/DAPO/PPO/DPO 为 Preview |
| 训练后端 | FSDP/FSDP2/Megatron 等 | 当前 RL 配置以 FSDP 为核心 | HF/FSDP/DeepSpeed/Megatron；Megatron 支持 TP/PP/CP/EP | MindSpeed/Megatron，昇腾原生 |
| rollout | vLLM/SGLang/HF | 当前源码有 vLLM/SGLang/LMDeploy | vLLM server/colocate，另有 SGLang/LMDeploy 推理能力 | vLLM-Ascend 为主，仓内另含 veRL NPU 适配 |
| 训推分离 | 当前项目已验证双机 Ray 自定义资源和权重同步 | 有 disaggregated trainer，但示例说明真实跨设备权重同步尚未落地 | Megatron-Ray 支持 train/rollout 独立资源组；非 Ray 也可连外部 server | 支持训推共卡和分离、异构重切分 |
| 异步/长尾 | one-step off-policy、fully async、separation、partial rollout 等；当前项目已有定制 bounded fully-async | 有 async replay、staleness、oversampling、partial rollout 源码，仍处快速演进期 | `async_generate` 为上一轮模型采样，且**不支持多轮**；未发现等价 bounded group queue | 异步流水和 partial rollout 存在，部分能力为 Preview |
| 多轮工具 Agent | 通用 agent loop/tool config；本项目四工具与 25 次反馈已实跑 | 有 tool/agentic RL 配置和 agent loop；README 与当前源码状态不完全同步 | 多轮 Scheduler、Gym Env、工具插件、奖励信息和 token 级 loss mask 可扩展 | 支持 ReTool/Search Tool；当前解析格式仅 Hermes，官方示例偏窄 |
| Ascend A3 | 官方 A3 quickstart；本项目 32 NPU 双机链路已验证 | 官方 NPU RL 配置和 A3 优化方向明确 | 官方 NPU 指南、vLLM-Ascend 和 16-NPU Megatron-GRPO 示例 | 原生优势最强 |
| 当前迁移成本 | 无 | **高**：训练并行、checkpoint、权重同步、工具/奖励和调度均需移植 | **中高**：模型/模板省力，但调度和工具合同仍需重做 | **高且无长期收益** |

## XTuner 重点判断

### 已经具备的能力

- 官方 README 将 XTuner V1 定位为面向超大 MoE、长序列和 Ascend A3 的新一代训练引擎，并列出 GRPO 已实现。[XTuner README（固定提交）](https://github.com/InternLM/xtuner/blob/575d7e058040baa7f609b3d5d3f397653877bc25/README.md)
- 当前源码有独立的 RL trainer、GRPO loss/advantage、agent loop manager、同步/异步 replay buffer，以及 colocate 和 disaggregated trainer。
- NPU GRPO 官方配置使用 16 个 NPU worker、vLLM rollout 和 FSDP 训练；这证明上游明确在做 Ascend RL，而不是只把 NPU 写进路线图。[NPU GRPO 配置](https://github.com/InternLM/xtuner/blob/575d7e058040baa7f609b3d5d3f397653877bc25/examples/v1/config/rl_qwen3_30B_grpo_npu.py)
- 当前配置已经表达 async stale、oversampling、partial rollout 和 max staleness 等概念，方向上与本项目处理长尾 rollout 的需求接近。
- 仓内已有带工具的 GSM8K 和 agentic code RL 示例，说明多轮/工具能力已经进入源码，只是公开入门和成熟度标注仍落后于实现。

### 当前不能替换 veRL 的原因

1. 官方 GRPO 入门页明确标为 Beta。[XTuner GRPO 入门](https://github.com/InternLM/xtuner/blob/575d7e058040baa7f609b3d5d3f397653877bc25/docs/en/get_started/grpo.md)
2. `rl_disagg_single.py` 首行说明目前使用 mock disaggregated weight-sync hook，等待真实跨设备权重更新模块落地；这直接阻断本项目 5 号机训练、6 号机 rollout 的生产迁移。[训推分离示例](https://github.com/InternLM/xtuner/blob/575d7e058040baa7f609b3d5d3f397653877bc25/examples/v1/config/rl_disagg_single.py)
3. 当前官方 NPU RL 示例是 colocate、Qwen3-30B、短上下文、单轮数学任务，不能外推为已支持 Qwen3.6-27B、48K、25 次工具反馈或跨机 HCCL 权重同步。
4. XTuner RL 训练配置以 FSDP 为中心；本项目现有容量和稳定性依赖 Megatron `TP4/PP2/CP2`、CPU Adam/gradient offload 和既有 checkpoint 语义，迁移不是配置文件翻译。
5. 未发现 Qwen3.6-27B 的明确模型或模板条目。通用 Hugging Face 自动配置可能使模型可加载，但 chat template、工具 token、loss mask、vLLM 权重名和 NPU 算子都必须单独验证。

因此，XTuner 的正确定位是“未来架构候选”，不是“当前 veRL 的直接升级包”。

## ms-swift 重点判断

### 优势

- 当前主干明确注册 `Qwen/Qwen3.6-27B` 和 FP8 变体，复用 Qwen3.5 模型/模板实现。[模型注册](https://github.com/modelscope/ms-swift/blob/c08a1ca7b443bb63cf2597781edf69fa638db54e/swift/model/models/qwen.py)；[支持表](https://github.com/modelscope/ms-swift/blob/c08a1ca7b443bb63cf2597781edf69fa638db54e/docs/source/Instruction/Supported-models-and-datasets.md)
- Megatron-GRPO 明确支持全参/LoRA、TP/PP/CP/EP 和 vLLM server/colocate。[Megatron-GRPO 文档](https://github.com/modelscope/ms-swift/blob/c08a1ca7b443bb63cf2597781edf69fa638db54e/docs/source/Megatron-SWIFT/GRPO.md)
- Megatron-Ray 可把 train 与 rollout 放到独立资源组并跨节点自动调度，拓扑表达比普通 HF Trainer 更接近本项目。[Ray 文档](https://github.com/modelscope/ms-swift/blob/c08a1ca7b443bb63cf2597781edf69fa638db54e/docs/source/Instruction/Ray.md)
- 多轮 Scheduler 支持自定义环境推进、工具返回、每轮 token ids、response loss mask、rollout 信息传给 reward，以及训练/推理 log-prob 不一致修正。[多轮开发指南](https://github.com/modelscope/ms-swift/blob/c08a1ca7b443bb63cf2597781edf69fa638db54e/docs/source/Instruction/GRPO/DeveloperGuide/multi_turn.md)
- 官方已有 16-NPU Qwen3.5 全参 Megatron-GRPO 配置，使用 TP4/PP2、vLLM、模型/优化器卸载和 padding-free，说明本项目所需的主要底层组件并非空白。[Ascend 全参 GRPO 示例](https://github.com/modelscope/ms-swift/blob/c08a1ca7b443bb63cf2597781edf69fa638db54e/examples/ascend/train/qwen3_5/qwen3_5_full_grpo_megatron.sh)

### 限制

- 官方参数文档明确说明 `async_generate` 使用上一轮更新模型采样，并且不支持多轮。[Megatron 参数说明](https://github.com/modelscope/ms-swift/blob/c08a1ca7b443bb63cf2597781edf69fa638db54e/docs/source/Megatron-SWIFT/Command-line-parameters.md)
- Ray separate 证明角色可以分配到不同设备，但没有证明项目当前的自定义 Ray resource pinning、跨机权重广播、连续 token、Fastest-K 取消和 group 原子队列能原样保留。
- 明确注册 Qwen3.6-27B 只证明加载/模板入口存在，不等于 16-NPU 全参 GRPO、48K 工具 agent 和 checkpoint 恢复已经由上游验证。

ms-swift 因此适合做“最小兼容 POC”，不适合立即接管生产训练。

## MindSpeed 重点判断

MindSpeed-RL 的官方 README 同时给出了两个最重要的信息：它确实是端到端昇腾 RL 框架；但 2026 年 4 月已经完成既定开发目标并暂停新增功能集成，最新昇腾 RL 方案指向 veRL。[MindSpeed-RL README（固定提交）](https://github.com/Ascend/MindSpeed-RL/blob/26c21e64f84ae8cd26de9483292349a1850099cf/README.md)

它现有能力并不弱：多轮工具、异步引擎、partial rollout、权重重切分、训推共卡/分离都存在。多轮文档显示当前主要提供 ReTool 和 Search Tool，工具解析只支持 Hermes，默认示例只调用一次工具；扩展成本高于 veRL 和 ms-swift 的通用 scheduler/agent loop。[MindSpeed-RL 多轮文档](https://github.com/Ascend/MindSpeed-RL/blob/26c21e64f84ae8cd26de9483292349a1850099cf/docs/zh/features/multi_turn.md)

MindSpeed Core 和 MindSpeed-LLM 仍然值得作为昇腾算子、Megatron 适配、模型实现、通信和性能问题的 reference，但不应被计入端到端 RL 框架排名。[MindSpeed Core](https://github.com/Ascend/MindSpeed/blob/99c7a37b7466947975d69ccfb9c8d31f2ab06134/README.md)；[MindSpeed-LLM](https://github.com/Ascend/MindSpeed-LLM/blob/eabc35035634e15b854ff3a76c49061b0ef3c5c3/README.md)

## veRL 主干更新风险

本地 `reference/verl` 已从 `922af88` 快进到 `5ba5f2f`。这只是调研引用更新，不代表服务器运行时应该升级。

最新主干在提交 [`859c712`](https://github.com/verl-project/verl/commit/859c712) 移除了 MindSpeedLLM backend engine，并将昇腾路线集中到直接 Megatron/FSDP2/VeOmni 与 vLLM-Ascend/SGLang 等路径。当前官方 A3 quickstart 仍列出 vLLM/SGLang 搭配 Megatron 或 FSDP2 的组合。[veRL Ascend quickstart](https://github.com/verl-project/verl/blob/5ba5f2f51281522150a6c1c87e3f841d1d220042/docs/ascend_tutorial/get_start/quick_start.rst)

由于本项目对 veRL experimental 调度、Megatron Bridge、Continuous Token、checkpoint 和权重同步有大量定制，后续如要升级，只能建立独立分支按已冻结合同回归，不能把 `reference/verl` 的最新代码直接覆盖服务器运行环境。

## 建议路线

### P0：继续使用并锁定 veRL

- 生产训练保持当前已验证运行时、上游 SHA、Megatron Bridge SHA、vLLM-Ascend 版本和项目补丁集。
- `reference/verl` 只用于阅读和选择性 backport，不作为部署源。
- 下一次 veRL 升级必须覆盖：Qwen3.6 完整模板、四工具共享状态、48K/25 feedback、GRPO group 原子性、staleness/背压、权重同步、optimizer 恢复和密封评测。

### P1：给 ms-swift 一个受限 POC

建议先做 CPU/单步工程门禁，不直接占用正式训练窗口：

1. 用现有 Qwen3.6-27B tokenizer 检查 system/tools/user 的 byte/token 等价性。
2. 验证完整四工具多轮轨迹的 response token ids 与 loss mask，工具结果必须全遮罩。
3. 验证 Megatron TP4/PP2/CP2 模型导入、单步全参更新和 model/optimizer 恢复。
4. 只用同步多轮 + separate rollout 建立正确性基线；不要开启官方明确不支持多轮的 `async_generate`。
5. 若上述全部通过，再评估是否值得移植 bounded fully-async/group queue，而不是提前重写调度器。

### P2：把 XTuner 作为观察性 POC

只有以下条件满足后，才值得投入双机验证：

1. disaggregated trainer 使用真实跨设备权重同步，不再是 mock hook；
2. Qwen3.6-27B 在 NPU 上通过模板、模型加载、权重同步和完整 checkpoint 门禁；
3. 工具 agent 支持本项目四工具共享沙箱、至少 25 次反馈和 48K 上下文；
4. async replay 对 prompt group 的原子性、staleness 和取消语义可以精确说明并测试。

如果未来任务转向 200B+ MoE、超长序列和 A3 超节点，XTuner 的优先级会显著提高；对当前 27B dense agent GRPO，它的潜在训练引擎收益不足以抵消迁移风险。

### 不启动 MindSpeed-RL 迁移

保留 MindSpeed-RL/Core/LLM 作为 Ascend 实现参考；遇到算子、HCCL、Megatron、模型转换或 vLLM-Ascend 问题时选择性借鉴。新的端到端 RL 投入继续落在 veRL，符合 MindSpeed-RL 官方给出的方向。

## 最终排序

| 目标 | 首选 | 次选 | 不建议 |
| --- | --- | --- | --- |
| 当前项目继续训练 | **veRL** | — | 直接迁移到其他框架 |
| Qwen3.6 快速兼容 POC | **ms-swift** | XTuner | MindSpeed-RL 新建生产线 |
| 未来超大 MoE/A3 训练引擎 | **XTuner（持续观察）** | ms-swift Megatron | 用 MindSpeed Core 单独承担 RL 编排 |
| 昇腾底层实现参考 | **MindSpeed Core/LLM** | veRL Ascend | — |

一句话决策：**保留 veRL；先小试 ms-swift 的 Qwen3.6/Megatron 兼容性；观察 XTuner 的真实训推分离成熟度；不向已停止新增功能的 MindSpeed-RL 迁移。**
