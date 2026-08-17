# Qwen3.8 Reasoning Effort 调研报告

**调研对象：** 0号机自部署 Qwen3.8-27B、PI 0.82.1、vLLM 0.23.0，以及千问云端 Qwen3.8-Max 公开接口
**调研日期：** 2026-08-17
**文档性质：** 实机代码审计与公开资料核对

## 1. 执行摘要

本次调研得到五个可确定结论：

- 0号机 Qwen3.8-27B 的 `low / medium / xhigh` 不对应三套权重、Adapter 或专用 effort token；其公开推理实现是 chat template 中的系统提示词分支。
- `medium` 是原生 thinking 基线：开启 `<think>`，但不增加额外的深浅指令。`low` 加入“简短、聚焦”的提示，`xhigh` 加入“仔细验证、考虑替代方案”的提示。
- 模型没有 `<reasoning_low>`、`<reasoning_medium>`、`<reasoning_xhigh>` 这类档位 special token；但 `<think>` 与 `</think>` 是词表中的单独 token，用于标记思考边界。
- 当前27B部署没有为三档设置独立思考 token 预算；模型自行决定何时输出 `</think>`，最终受请求输出上限约束。本次 PI 评测的 `maxTokens` 为8,192。
- 云端 Qwen3.8-Max 多了一层明确的服务端预算控制：`low / medium / xhigh` 分别映射到4,096、16,384和262,144个思考 token。该差异属于公开接口与服务实现，不能反推出 Max 的训练数据或内部 special token。

最稳妥的训练侧结论是：Qwen3.8-27B 在服务接口上明显依赖较强的指令遵循能力；官方未披露是否为同一道 instruction 准备三份不同长度思考、是否执行 effort-conditioned SFT/RL，或三种档位的数据配比。

## 2. 调研范围与证据等级

### 2.1 调研范围

- 0号机容器：`llin-qwen3.8-27b-server`
- 模型目录：`/data3/models/Qwen3.8-27b`
- 服务框架：vLLM 0.23.0 / vLLM-Ascend
- 评测框架：PI 0.82.1，OpenAI-compatible Chat Completions
- 云端对照：Qwen3.8-Max / Qwen3.8-Max-Preview 官方 API 文档

### 2.2 证据等级

- **已证实：** 直接读取0号机模型文件、tokenizer、容器配置和 vLLM/PI 运行代码所得结果。
- **官方公开：** 千问模型卡或官方 API 文档明确说明的行为。
- **合理推断：** 能由已证实机制解释，但官方没有公布训练细节。
- **未知：** 无公开证据，不能写成事实。

## 3. 27B 的 reasoning_effort 实现链路

完整调用链如下：

1. 评测包装器向 PI 注入 `--thinking medium`。
2. PI 将其规范化为内部 `thinkingLevel=medium`。
3. 模型配置声明 `reasoning=true`、`supportsReasoningEffort=true`，且没有 `thinkingLevelMap`，所以 PI 原样发送 `reasoning_effort="medium"`。
4. vLLM 的 Chat Completions 协议接收 `reasoning_effort`，将其放入 chat template kwargs，并在 effort 不为 `none` 时启用 thinking。
5. Qwen3.8 的 chat template 根据档位选择系统提示词，然后以 `<think>` 开始生成。
6. `qwen3` reasoning parser 把 `</think>` 之前的内容返回为 `reasoning_content`，之后的内容返回为普通 `content`；未闭合且被截断时，生成内容会被视为尚未结束的 reasoning。

对应代码位置：

- PI请求组装：`.../@earendil-works/pi-ai/dist/api/openai-completions.js`
- vLLM协议：`/vllm-workspace/vllm/vllm/entrypoints/openai/chat_completion/protocol.py`
- 模型模板：`/data3/models/Qwen3.8-27b/tokenizer_config.json` 的 `chat_template`
- 思考解析：`/vllm-workspace/vllm/vllm/reasoning/qwen3_reasoning_parser.py`

## 4. 三档究竟改变了什么

模型模板的核心逻辑是：

```jinja2
{% set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{% if resolved_reasoning_effort == 'xhigh' %}
  {% set reasoning_instructions = 'Reasoning effort is set to xhigh. ...' %}
{% elif resolved_reasoning_effort == 'low' %}
  {% set reasoning_instructions = 'Reasoning effort is set to low. ...' %}
{% endif %}
```

其中没有 `medium` 分支。实际语义是：

| 档位 | 模板动作 | 控制性质 |
|---|---|---|
| `off` | 预填空的 `<think>\n\n</think>` | 关闭显式思考 |
| `low` | 开启 thinking，并注入“保持简短、直接得出结论”的系统提示 | 软提示控制 |
| `medium` | 开启 thinking，不注入额外 reasoning 指令 | 原生 thinking 基线 |
| `xhigh` | 开启 thinking，并注入“仔细验证假设、考虑替代方案”的系统提示 | 软提示控制 |
| 未指定 | 模板默认采用 `xhigh` | 默认深思考提示 |

同一条测试消息的实机模板 token 数为：`off=27`、`low=55`、`medium=25`、`xhigh=67`、未指定`=67`。这说明档位首先改变输入提示，而不是为采样器设置三组固定预算。

## 5. special token 调查结论

### 5.1 不存在 effort 档位 special token

实机 tokenizer 结果：

| 字符串 | tokenizer结果 | 结论 |
|---|---|---|
| `<reasoning_low>` | 5个普通token | 不是专用token |
| `<reasoning_medium>` | 5个普通token | 不是专用token |
| `<reasoning_xhigh>` | 6个普通token | 不是专用token |
| `reasoning_effort` | 4个普通token | 不是专用token |

因此没有证据表明27B通过隐藏的 effort token 选择不同推理策略。

### 5.2 存在思考边界 token

- `<think>`：token ID 248068
- `</think>`：token ID 248069

它们负责结构化区分“思考过程”和“最终回答”，不表示 low、medium 或 xhigh。

## 6. Preserved Thinking

`preserve_thinking` 指多轮对话时，把历史 assistant 的 `reasoning_content` 原样放回下一轮输入。它不是模型内部永久记忆，也不保存隐藏状态；本质是客户端重发历史文本。

开启后的历史形态：

```text
用户上一轮问题
<think>
上一轮完整思考
</think>
上一轮最终答案
用户当前问题
```

关闭后只保留用户消息和历史最终答案，不再携带旧思考。

主要收益：

- 保持多步Agent计划和工具调用的推理连续性。
- 减少重复推导，并可能提高相同前缀的KV Cache复用。
- 让模型知道上一轮结论依据，而不只看到结论。

主要代价：

- 历史思考持续占用上下文，增加prefill、延迟与成本。
- 错误推理会形成锚定并传播到后续轮次。
- 敏感或无价值的中间推理被长期携带。

Qwen3.8-27B模板默认保留历史thinking，但只有上游客户端实际保存并回传 `reasoning_content` 时才生效。Qwen3.8-Max官方接口同样默认开启 preserved thinking，并要求完整、原序回传历史 `reasoning_content`。

## 7. 可行的 reasoning-effort 训练范式

### 7.1 长短混合或配对监督

为相同或相近任务提供不同推理长度与验证深度。严格的同题三轨迹能提供最干净的因果监督，但成本很高；实践中可以采用少量同题配对校准，加大量不同题但共享难度分布的档位标注数据。

风险是模型把“长”误学成“好”，产生重复、复述和无效展开。因此监督目标应强调有效验证与纠错，而不是机械token数量。

### 7.2 强指令遵循

把模型训练成能够可靠服从“简短推理”“充分验证”等自然语言约束。Qwen3.8-27B公开模板明确依赖这一能力：medium保持基础策略，low/xhigh用普通英文系统提示施加偏移。

仅靠通用指令遵循可以产生档位差异，但不保证推理长度严格单调，也不能提供硬成本上限。

### 7.3 显式控制标记

训练 `<reasoning_low>` 等专用控制token，使档位成为模型输入中的稳定离散条件。优点是控制信号稳定；缺点是需要修改tokenizer和训练数据，并增加部署兼容成本。本次调查确认Qwen3.8-27B没有采用这种公开实现。

### 7.4 成本感知强化学习

使用按档位变化的token成本系数：

```text
R = R_correct - lambda_effort * N_thinking_tokens
lambda_low > lambda_medium > lambda_xhigh
```

模型学习的是“在当前成本约束下，多少计算值得”，而不是模仿固定长度。还可以加入验证、工具正确性、重复惩罚与完整收尾奖励。

### 7.5 偏好优化

为同一道题构造短而正确、短但遗漏、长且充分、长但重复等候选，并按档位改变排序偏好。可使用DPO、GRPO或pairwise ranking，使low偏好高效正确，xhigh偏好充分验证，同时共同反对无意义冗长。

### 7.6 多教师或多预算蒸馏

用不同教师、搜索次数、Best-of-N规模或工具验证强度生成不同档位轨迹，再蒸馏进同一学生模型。这样不需要人工逐题撰写三份思考。

### 7.7 停止策略训练

专门训练何时输出 `</think>`：简单问题尽早结束，不确定时继续检查，完成关键验证后停止。它直接决定实际思考长度，是比“要求写短/写长”更接近计算分配本质的目标。

### 7.8 数值预算条件化

以连续或离散预算作为训练条件，例如 `thinking_budget=512/2048/8192`。推理时可以把low/medium/xhigh映射为预算。优点是可控性强；难点是预算耗尽后仍需保证模型能平滑转入最终答案。

### 7.9 Adapter、多策略头或路由

不同档位使用不同LoRA、Adapter、策略头或MoE路由。控制更强，但会引入多权重部署、缓存复用和多轮一致性问题。Qwen3.8-27B开源权重没有显示这种实现。

### 7.10 纯推理时控制

不改变训练，使用硬token budget、强制结束thinking、不同采样次数、自一致性或verifier搜索深度控制计算量。这类方法可以与任何训练范式叠加。

## 8. Qwen3.8-27B 与 Qwen3.8-Max 对比

| 维度 | 0号机 Qwen3.8-27B | 云端 Qwen3.8-Max |
|---|---|---|
| 权重与后端 | 开源权重，自部署vLLM | 闭源云端模型与服务 |
| effort档位 | low/medium/xhigh | low/medium/xhigh |
| 已知控制信号 | chat template系统提示词 | API参数与服务端预算映射 |
| effort专用special token | 已确认没有 | 内部不可见，无法确认 |
| `<think>`边界 | 单独token | 返回`reasoning_content`，内部token不可见 |
| low | 简短思考提示词 | 4,096 thinking tokens |
| medium | 无额外提示词 | 16,384 thinking tokens |
| xhigh | 深入验证提示词 | 262,144 thinking tokens |
| 未指定 | 模板默认xhigh提示 | 默认effort为xhigh、默认预算131,072 |
| 是否硬限制思考长度 | 当前部署没有独立预算 | 有明确预算层 |
| preserved thinking | 模板默认开启 | 默认开启，要求完整原序回传 |
| 训练方法是否公开 | 只说明Pre-training与Post-training | 未公开内部训练方法 |

Max的官方API规定 `reasoning_effort` 与 `thinking_budget` 不能同时设置；二者会互相映射。该机制至少证明Max服务层不是“只有一句系统提示词”，但不能证明其模型内部采用effort special token、三份数据或条件化RL。

## 9. 对我们训练设计的建议

如果目标是训练出可控且有价值的reasoning effort，不建议只追求三种平均长度。推荐采用四层组合：

- **控制条件：** 保留自然语言指令，必要时增加内部档位字段，但不急于修改tokenizer。
- **数据层：** 少量同题low/medium/xhigh配对数据用于校准，大量同分布非配对数据用于覆盖任务。
- **优化层：** 正确性、工具证据、完整收尾与token成本联合奖励；不同档位使用不同成本权重。
- **运行层：** 增加thinking budget和超限转答机制，避免仅靠模型自觉停止。

建议重点监测：

- 各档位正确率、partial率和incomplete率。
- thinking token分布及P50/P90/P99，而不只看均值。
- 有效验证步骤、重复工具调用和无效自我复述。
- 达到预算但没有最终答案的比例。
- preserved thinking对多轮准确率、上下文增长和延迟的影响。
- low→medium→xhigh是否形成单调的质量—成本曲线。

## 10. 已知边界与不可下结论事项

下列结论当前不能成立：

- “Qwen3.8训练时一定为每个instruction准备了三份think。”
- “Max一定使用了隐藏effort special token。”
- “27B的medium等于固定16K预算。”
- “xhigh更长就必然更准确。”
- “preserved thinking一定提高多轮质量。”

公开证据只能确认27B的模板与解析机制、当前部署参数，以及Max的API预算行为。训练数据、损失函数、RL奖励和内部服务提示均未完整披露。

## 11. 主要证据与参考资料

### 11.1 0号机实机证据

- `/data3/models/Qwen3.8-27b/tokenizer_config.json`
- `/data3/models/Qwen3.8-27b/README.md`
- `/data3/models/qwen3.8_27b_docker_compose/docker-compose.yaml`
- `/vllm-workspace/vllm/vllm/entrypoints/openai/chat_completion/protocol.py`
- `/vllm-workspace/vllm/vllm/reasoning/qwen3_reasoning_parser.py`
- PI 0.82.1：`.../@earendil-works/pi-ai/dist/api/openai-completions.js`

### 11.2 官方公开资料

- Qwen3.8-27B ModelScope模型页：https://modelscope.cn/models/Qwen/Qwen3.8-27B/summary
- 千问思考模式说明：https://platform.qianwenai.com/docs/developer-guides/text-generation/thinking
- 千问DashScope API参考：https://platform.qianwenai.com/docs/api-reference/chat/dashscope
- Qwen3技术报告：https://arxiv.org/abs/2505.09388
