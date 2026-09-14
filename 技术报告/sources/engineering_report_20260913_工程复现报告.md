---
报告日期: 2026-09-06
更新类型: 工程复现报告——新人对照操作即可复现完整训练流程
更新人: renjunxiang
状态: 优化版——修复章节编号、增加实际路线声明
---

# Qwen3.6-27B 物流运营智能体 —— 工程复现报告

> **本报告定位**：指导新人对照操作，完整复现本项目的训练流程。
>
> 与技术报告（`technical_report_20260824_151236_端到端技术报告.md`）的关系：
> - 技术报告讲"为什么这么做"（算法逻辑、设计决策）
> - 本报告讲"怎么做"（操作步骤、代码、验证点）
>
> 本报告不重复技术报告的算法解释，聚焦可执行的操作细节。

---

> **⚠️ 实际训练路线说明**：本报告的章节按"SFT → GRPO → RM → 推理部署 → 评测"的逻辑顺序组织，但**项目的实际训练路线跳过了基座模型的 SFT**，直接从 GRPO 开始（原因见技术报告"本报告的设计意图：正确路线 vs 实际执行"一节）。实际路线为：基座模型 → GRPO（跳过 SFT）→ Step120 最佳 → CPT 注入教材 → 评测迭代。**新人接手时应按技术报告的章节顺序执行**（基座评测 → CPT → SFT → GRPO），基础设施已齐全，不需要重走弯路。

---

## 第 0 章 项目部署与数据同步

> **本章目标**：指导如何完整获取本项目（代码 + 数据），以及如何在本地与服务器之间同步数据。
>
> **适用场景**：
> - 新客户/新人需要完整拷贝项目
> - 在新服务器上部署项目
> - 本地与服务器之间的数据同步

### 1. 项目主体结构

```
huawei_train/
├── scripts/              # 训练、评测、数据处理脚本（~150 个文件）
├── configs/              # 配置文件（评测、数据集配置等）
├── dashboard/            # 评测看板（后端 + 前端）
├── report/               # 技术报告、工程报告、更新报告等
├── reference/            # 参考论文、开源仓库地址清单
├── datasets/             # 数据集（详见下文）
├── eval_results/         # 评测结果
└── .gitignore            # 定义了哪些文件不被 Git 跟踪
```

**datasets/ 目录结构**：
```
datasets/
├── open_source/          # 开源数据集
│   ├── raw/              # 原始数据（从 HuggingFace/ModelScope 下载）
│   └── processed/        # 处理后的数据（四字段标准格式）
├── PI/                   # PI 原始轨迹（蒸馏数据溯源，~24G）
├── trajectories/         # OpenAI 格式的轨迹数据（~50G）
├── sandboxes/            # 沙箱环境（sqlite + 任务定义，~3G）
└── judgement/            # 评测模型训练数据（judge_train.jsonl）
```

### 2. 被 Git 跟踪的文件（可通过 git clone 获取）

**Git 仓库地址**：`git@gitcode.com:renjunxiang1215/huawei_train.git`

**获取代码**：
```bash
git clone git@gitcode.com:renjunxiang1215/huawei_train.git
cd huawei_train
```

**Git 跟踪的内容**：
- ✅ `scripts/` - 所有脚本代码
- ✅ `configs/` - 配置文件
- ✅ `dashboard/` - 看板代码
- ✅ `report/` - 所有报告文档
- ✅ `reference/` - 参考论文和仓库地址清单
- ✅ `datasets/open_source/processed/` - 处理后的开源数据（~7G）
- ✅ `datasets/judgement/` - 评测模型训练数据（~300M）
- ✅ 部分小文件和数据集

### 3. 被 Git Ignore 的文件（需从 huawei-5 手动拉取）

以下数据**不在 Git 中**（太大或不常变更），需要从服务器手动同步：

| 目录 | 大小 | 说明 | 位置 |
|------|------|------|------|
| `datasets/PI/` | ~24G | PI 原始轨迹（蒸馏数据溯源） | huawei-5 + 本地 |
| `datasets/trajectories/` | ~50G | OpenAI 格式的轨迹数据 | huawei-5 + 本地 |
| `datasets/sandboxes/` | ~3G | 沙箱环境（sqlite + 任务定义） | huawei-5 + 本地 |
| `datasets/open_source/raw/` | ~32G | 完整的原始开源数据 | huawei-5 |
| `eval_results/` | ~10G | 评测结果 | huawei-5 + 本地 |
| `/data/models/` | ~100G | 模型权重（Qwen3.6-27B 等） | huawei-5 |
| `/data3/llin/` | ~50G | GRPO 训练产物 | huawei-5 |

### 4. 数据同步命令

#### 场景 1：从 huawei-5 拉取数据到本地（Mac）

**适用场景**：本地开发，需要从服务器获取最新数据。

```bash
# 同步整个 datasets 目录
rsync -avz huawei-5:/data/renjunxiang/coding/huawei_train/datasets/ datasets/

# 同步评测结果
rsync -avz huawei-5:/data/renjunxiang/coding/huawei_train/eval_results/ eval_results/

# 同步特定目录（如 PI 轨迹）
rsync -avz huawei-5:/data/renjunxiang/coding/huawei_train/datasets/PI/ datasets/PI/
```

#### 场景 2：从本地上传数据到 huawei-5

**适用场景**：本地跑完实验，需要上传结果到服务器。

```bash
# 同步整个 datasets 目录
rsync -avz datasets/ huawei-5:/data/renjunxiang/coding/huawei_train/datasets/

# 同步评测结果
rsync -avz eval_results/ huawei-5:/data/renjunxiang/coding/huawei_train/eval_results/
```

#### 场景 3：从本地推送到新服务器（如 huawei-6）

**适用场景**：在新服务器（如 huawei-6）上部署项目，从本地推送数据。

```bash
# 步骤 1：在新服务器上克隆代码
ssh huawei-6
git clone git@gitcode.com:renjunxiang1215/huawei_train.git
cd huawei_train

# 步骤 2：在本地 Mac 上执行，推送数据到新服务器
rsync -avz datasets/ huawei-6:/data/renjunxiang/coding/huawei_train/datasets/
rsync -avz eval_results/ huawei-6:/data/renjunxiang/coding/huawei_train/eval_results/

# 步骤 3：推送模型权重（大文件，约 100G）
rsync -avz huawei-5:/data/models/ huawei-6:/data/models/
```

#### 场景 4：本地与 huawei-5 数据并集同步

**适用场景**：本地和 huawei-5 分别跑过不同的实验，需要合并数据（保留并集）。

```bash
# 步骤 1：从 huawei-5 下载到本地（获取 huawei-5 独有的数据）
rsync -avz huawei-5:/data/renjunxiang/coding/huawei_train/datasets/ datasets/

# 步骤 2：从本地上传到 huawei-5（上传本地独有的数据）
rsync -avz datasets/ huawei-5:/data/renjunxiang/coding/huawei_train/datasets/

# 最终两边都有完整的数据并集
```

### 5. SSH 配置

上述 rsync 命令依赖 SSH 别名。确保在 `~/.ssh/config` 中配置：

```
Host huawei-5
    HostName 182.151.17.158
    Port 22205
    User root
    IdentityFile ~/.ssh/sf_renjunxiang

Host huawei-6
    HostName 182.151.17.158
    Port 22206
    User root
    IdentityFile ~/.ssh/sf_renjunxiang
```

**验证**：
```bash
ssh huawei-5 'hostname'  # 应返回服务器主机名
ssh huawei-6 'hostname'  # 应返回服务器主机名
```

### 6. 注意事项

1. **首次同步耗时较长**：大数据集（如 datasets/PI 约 24G）首次同步可能需要 30-60 分钟
2. **增量同步**：rsync 会自动跳过已存在的文件，后续同步很快
3. **网络稳定性**：大数据同步建议使用 `screen` 或 `tmux`，避免网络中断导致任务终止
4. **磁盘空间**：确保目标服务器有足够的磁盘空间（完整项目约 300G+）
5. **数据并集**：本地和服务器可能分别跑过不同的实验，同步时是并集关系，不会覆盖

---

## 第 1 章 环境准备（前置依赖）

> **本章目标**：把服务器、容器、驱动、模型、推理服务全部准备到可以开始训练的状态。
>
> **本章状态**：✅ 已完成

### 0.1 服务器集群

项目有三台 Atlas 800 A3 服务器 + 一台 CPU 服务器，硬件配置相同：

| 服务器 | SSH 别名 | 用途 | 公网地址 |
|--------|----------|------|---------|
| huawei-0 | `huawei-0` | 推理服务器（vLLM 推理、PI 跑轨迹、评测） | 110.188.22.101:12202 |
| huawei-5（别名 huawei-llm） | `huawei-llm` | 训练服务器（MindSpeed-MM SFT、ms-swift DPO、verl GRPO） | 182.151.17.158:22205 |
| huawei-6 | `huawei-6` | 训练备份、并行实验 | 182.151.17.158:22206 |
| L40S-ps-sf | `L40S-ps-sf` | 商用模型轨迹采集（CPU 服务器） | 30.144.109.13:22 |

**单机硬件**：4 路鲲鹏 920（640 vCPU）+ 2.0 TiB 内存 + 8 张昇腾 910（16 chip，整机 1 TiB HBM）+ 17 TiB NVMe + 双口 200G 网络

### 0.2 SSH 配置

三台昇腾服务器共用同一份 SSH 密钥。在 `~/.ssh/config` 中添加：

```
Host huawei-0
    HostName 110.188.22.101
    Port 12202
    User root
    IdentityFile ~/.ssh/sf_renjunxiang

Host huawei-llm
    HostName 182.151.17.158
    Port 22205
    User root
    IdentityFile ~/.ssh/sf_renjunxiang

Host huawei-6
    HostName 182.151.17.158
    Port 22206
    User root
    IdentityFile ~/.ssh/sf_renjunxiang
```

**验证**：
```bash
ssh huawei-0 'hostname'     # 期望：Server910C-002
ssh huawei-llm 'hostname'   # 期望：XXZX2JL-501-F-04-A1P1-SEV-HWA800A3-10U05
```

**服务器间互通**：huawei-llm → huawei-0 内网免密（`ssh root@10.10.2.2`），用于训练机访问推理服务。

### 0.3 NPU 环境验证

登录后第一步验证 NPU 可用：

```bash
# 查看 NPU 状态
npu-smi info

# 期望输出：8 张卡 x 2 chip = 16 chip，全部健康
# 每个 chip HBM 容量 65536 MB (64 GiB)
```

**关键概念**：昇腾 910 是单卡双芯片架构。一张物理卡有 2 个 chip，每个 chip 有独立的 64 GiB HBM。环境变量有两种：
- `ASCEND_VISIBLE_DEVICES`：**chip 级别**（0-15），分布式训练用这个
- `ASCEND_RT_VISIBLE_DEVICES`：**卡级别**（0-7），容器设备挂载用

### 0.4 核心容器镜像

| 容器名 | 镜像 | 用途 | 所在服务器 |
|--------|------|------|-----------|
| vLLM 推理 | `qwen3.6-27b:vllm-ascend-v0.23.0rc1-a3` | Qwen3.6-27B 推理服务（端口 8011） | huawei-0 |
| MindSpeed-MM | `mindspeed-mm:26.0.0-a3-openeuler24.03-py3.11-aarch64` | Qwen3.6-27B Full SFT / LoRA SFT | huawei-llm |
| ms-swift (DPO/RM) | `mindspeed-llm:26.0.0-a3` + pip ms-swift | Qwen3.6-27B DPO + Qwen3.5-9B RM 训练 | huawei-llm / huawei-0 |
| verl-grpo | `llin-verl-a3:20260730` | Qwen3.6-27B 全参数 GRPO 训练 | huawei-llm |
| PI 沙箱 | `pi-sandbox:latest`（511MB） | 在沙箱中采集 agent 轨迹 | huawei-llm |

### 0.5 启动容器

**训练容器（huawei-llm）**：

```bash
# MindSpeed-MM 容器（Full SFT / LoRA SFT）
docker run -dit \
  --ipc=host --network host --name mindspeed_mm_rjx --privileged \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
  -v /usr/local/sbin/:/usr/local/sbin/ \
  -v /home/:/home/ \
  -v /data:/data \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-mm:26.0.0-a3-openeuler24.03-py3.11-aarch64 \
  sleep infinity

# 进入容器并初始化 CANN 环境
docker exec -it mindspeed_mm_rjx bash
echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc
source ~/.bashrc
python -c "import torch; import torch_npu; print(torch.npu.device_count())"
# 期望：16

# ms-swift 容器（DPO / RM 训练）—— 基于 mindspeed-llm 镜像 + pip 装 ms-swift
docker run -d --name llin-rl-dpo \
  --network host --ipc=host --privileged \
  -e ASCEND_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v /data/liulin/llin-rl-dpo:/workspace/llin-rl-dpo \
  -v /data/models/Qwen3.6-27B:/models/Qwen3.6-27B:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64 \
  sleep infinity
```

**推理容器（huawei-0）**：通过 docker compose 管理，配置在 `/data3/models/qwen3.6_27b_docker_compose/`。

```bash
cd /data3/models/qwen3.6_27b_docker_compose
docker compose --env-file .env up -d --build

# 验证服务
curl -s http://110.188.22.101:8011/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
# 期望返回：{"id":"Qwen3.6-27B",...}
```

### 0.6 模型权重下载

在 huawei-llm 上（能访问 modelscope.cn）：

```bash
# Qwen3.6-27B（主力模型，27B Dense）
modelscope download --model Qwen/Qwen3.6-27B --local_dir /data/models/Qwen3.6-27B

# Qwen3.5-9B（评估模型基座）
modelscope download --model Qwen/Qwen3.5-9B --local_dir /data/models/Qwen3.5-9B

# 其他模型（按需）
modelscope download --model Qwen/Qwen3.5-27B --local_dir /data/models/Qwen3.5-27B
modelscope download --model Qwen/Qwen3.8-27B --local_dir /data/models/Qwen3.8-27B
```

在 huawei-0 上，大模型权重在 `/data3/models/` 下（`Qwen3.5-9B` / `Qwen3.6-27B` / `Qwen3.8-27b`）。

**检查点**：
- ✅ `ssh huawei-0 'npu-smi info'` 显示 16 chip 健康
- ✅ `ssh huawei-llm 'npu-smi info'` 显示 16 chip 健康
- ✅ 训练容器和推理容器都在运行
- ✅ 模型权重已下载到对应路径

### 0.7 路径速查

| 用途 | huawei-0 路径 | huawei-llm 路径 |
|------|-------------|----------------|
| 模型权重 | `/data3/models/Qwen3.6-27B/` | `/data/models/Qwen3.6-27B/` |
| vLLM compose | `/data3/models/qwen3.6_27b_docker_compose/` | — |
| 项目代码 | — | `/data/renjunxiang/coding/huawei_train/` |
| GRPO 产物 | — | `/data3/llin/qwen3.6-27b-verl-grpo/` |
| PI 轨迹 | — | `/data/renjunxiang/pi/` |
| RM 训练产物 | `/data/liuxr/RM/` | `/data/liuxr/RM/`（开发机副本） |

---

## 第 2 章 数据准备（训练原料）

> **本章目标**：把通用配比数据、沙箱轨迹数据、LLM judge 训练数据准备到可以喂进训练的状态。
>
> **本章状态**：✅ 已完成

### 1.1 本章目标

训练需要三类数据：

| 数据线 | 用途 | 来源 | 对应章节 |
|--------|------|------|---------|
| 通用配比数据 | SFT 防通用能力遗忘 | 开源数据集（MATH/GSM8K 等 25K 条） | 第 2 章 SFT |
| 沙箱轨迹数据 | SFT/GRPO 教垂域能力 | PI agent 在沙箱中采集 | 第 2、3 章 |
| LLM judge 训练数据 | 蒸馏评估模型 | qwen3.7-plus 对轨迹打 5 维分 | 第 4 章 |

### 1.2 数据处理流水线总览

```
原始数据集（HuggingFace 格式）
    │
    ▼  scripts/data/convert.py + converters/（27 个转换器）
四字段标准格式（system_prompt / task_input / thinking / golden_answer）
    │  按 thinking 非空/为空分流
    ├── with_golden_thinking/    → 7 个数据集（训练用）
    └── without_golden_thinking/ → 42 个数据集（评测用）
    │
    ▼  scripts/data/to_openai.py
OpenAI ChatCompletion 格式（messages + reasoning_content + tools）
    │
    ▼  合并到 datasets/sft/final/20260702_openai.jsonl（25,167 条）
SFT 训练集
```

### 1.3 通用配比数据转换

#### 1.3.1 四字段标准格式

所有开源数据统一为四字段 JSONL：

```json
{
  "system_prompt": "你是一个数学推理助手...",
  "task_input": "求解方程 2x + 3 = 7",
  "thinking": "首先将 3 移到等号右边...",
  "golden_answer": "x = 2"
}
```

**转换脚本**：`scripts/data/convert.py` 自动扫描 `scripts/data/converters/` 下的 27 个数据集转换器，批量转换。

```bash
# 在本地执行
cd /Users/renjunxiang/coding/huawei_train
python scripts/data/convert.py

# 产出：
# datasets/open_source/sft/processed/with_golden_thinking/*.jsonl    （7 个）
# datasets/open_source/sft/processed/without_golden_thinking/*.jsonl （42 个）
```

每个转换器（如 `converters/math.py`、`converters/gsm8k.py`）负责：
1. 读取原始数据集（从 `datasets/open_source/raw/<DatasetName>/`）
2. 提取字段映射到四字段格式
3. 按 `configs/data/dataset_config.json` 标注 `extract_method` + `score_method`（双维度路由，第 5 章用）

#### 1.3.2 转为 OpenAI 格式

```bash
python scripts/data/to_openai.py

# 产出：datasets/sft/final/20260702_openai.jsonl（25,167 条）
```

转换逻辑：
- 有 thinking → `assistant.content = "<answer>\n" + golden + "\n</answer>"`, `assistant.reasoning_content = thinking`
- 无 thinking → `assistant.content = golden`
- 组装 `{"messages": [system, user, assistant]}` 格式

#### 1.3.3 数据配比

| 数据集 | 数量 | 占比 | 涉及能力 |
|--------|------|------|---------|
| MATH | 12,500 | 49.7% | C2 推理 + C1 知识 |
| GSM8K | 7,473 | 29.7% | C2 推理 + C1 知识 |
| Omni-MATH | 4,428 | 17.6% | C2 推理 + C1 知识 |
| AMO-Bench | 50 | 0.2% | C2 推理（IMO 级别） |
| PHYBench | 200 | 0.8% | C2 推理（物理） |
| ATLAS | 256 | 1.0% | C2 推理（PhD 级跨学科） |
| C-Eval-dev | 260 | 1.0% | C1 知识（中文） |
| **合计** | **25,167** | **100%** | |

数学主导（97.2%）是有意为之——数学好的模型推理能力通常不差。

### 1.4 沙箱轨迹数据

> 管道设计逻辑见技术报告 §5.3（含 mermaid 架构图、判分方式表格、数据清单）。本节聚焦工程实现细节。

#### 1.4.1 沙箱环境

沙箱是合成的物流运营场景，包含：
- **数仓**：`logistics.sqlite`（含 orders/deliveries/claims 等表）
- **知识库**：`documents/`（含业务规则文档）
- **任务**：`dwh_tasks.jsonl` / `kb_tasks.jsonl` / `hybrid_tasks.jsonl`

沙箱分组：
| 组 | 用途 | 沙箱数量 |
|----|------|---------|
| sft | SFT 训练 | 19 个（v15~v50） |
| rl | GRPO 训练 | 高难度任务 |
| dev | 开发调试 | 若干 |
| test | 评测（未见沙箱） | 10 个（v30~v36） |

#### 1.4.2 轨迹采集（PI agent）

**PI**（https://github.com/anthropics/pi-coding-agent）是开源 coding agent，在沙箱中执行多轮交互（思考→执行 SQL/文档检索→观察→循环），捕获完整的思考-动作-观察事件流。

**采集方式**（三种）：

| 方式 | 运行位置 | 模型类型 |
|------|----------|---------|
| 本机模式 | Mac 本机 | 商用模型（qwen3.7-max 等） |
| 服务器模式 | L40S-ps-sf | 商用模型 |
| Docker 模式 | huawei-5 | 私有化模型（Qwen3.6-27B via vLLM） |

##### Docker 镜像构建

PI 在服务器上以 Docker 镜像方式部署，镜像名 `pi-sandbox:latest`。

**Dockerfile.pi**（位于 `/data/renjunxiang/pi/Dockerfile.pi`）：

```dockerfile
FROM docker.m.daocloud.io/library/node:24-bookworm-slim

# 替换 Debian 软件源为阿里云镜像（服务器无法访问国外源）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
  && sed -i 's|https://|http://|g' /etc/apt/sources.list.d/debian.sources \
  && rm -f /etc/apt/sources.list

# 安装系统工具：bash/ca-certificates/git/ripgrep 是 pi 基础依赖
# sqlite3/python3 用于查询数据库（沙箱任务需要）
RUN apt-get update \
  && apt-get install -y --ignore-recommends bash ca-certificates git ripgrep sqlite3 python3 \
  && rm -rf /var/lib/apt/lists/*

# 安装 pi（使用官方 npm registry，加 --fetch-retries=5 重试）
RUN npm config set strict-ssl false \
  && npm install -g --ignore-scripts @earendil-works/pi-coding-agent --registry=https://registry.npmjs.org --fetch-retries=5 --fetch-retry-mintimeout=10000 --fetch-retry-maxtimeout=60000

# 设置工作目录，挂载数据时映射到这里
WORKDIR /workspace

# 容器启动时运行 pi
ENTRYPOINT ["pi"]
```

**构建命令**：

```bash
ssh huawei-5 'cd /data/renjunxiang/pi && docker build -t pi-sandbox:latest -f Dockerfile.pi .'
```

##### 模型配置（models.json）

三种采集模式的核心区别在于 `~/.pi/agent/models.json` 的 provider 配置：

**商用模型配置（本机模式 + 服务器模式共用）**——通过顺丰内网 LLM Hub 访问：

```json
{
  "providers": {
    "sf-llm": {
      "name": "SF Express LLM Hub",
      "baseUrl": "https://claudecode.sf-express.com/ccr/v1",
      "apiKey": "<your-api-key>",
      "api": "openai-completions",
      "models": [
        {
          "id": "aliyun/qwen3.7-max",
          "name": "Qwen3.7 Max",
          "reasoning": true,
          "contextWindow": 131072,
          "maxTokens": 16384,
          "compat": { "thinkingFormat": "qwen", "maxTokensField": "max_tokens" }
        },
        {
          "id": "aliyun/glm-5.2",
          "name": "GLM 5.2",
          "reasoning": true,
          "contextWindow": 131072,
          "maxTokens": 16384,
          "compat": { "thinkingFormat": "qwen", "maxTokensField": "max_tokens" }
        }
      ]
    }
  }
}
```

**私有化模型配置（Docker 模式）**——访问本地 vLLM 推理服务：

```json
{
  "providers": {
    "my-local": {
      "baseUrl": "http://127.0.0.1:8010/v1",
      "api": "openai-completions",
      "apiKey": "<VLLM_API_KEY>",
      "models": [
        {
          "id": "Qwen3.6-27B",
          "name": "Qwen3.6 27B",
          "reasoning": true,
          "contextWindow": 262144,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

##### 批量采集命令

**本机模式 / 服务器模式**（裸装 pi CLI，不需要 Docker）：

```bash
# 商用模型跑批：sf-llm provider + aliyun/qwen3.7-max
python3 scripts/data/batch_run_pi_local.py \
  --sandbox-dir outputs/sandbox_v21 \
  --sandboxes 20260628_v21 \
  --tdir outputs/trajectories_qwen37max_v21 \
  --provider sf-llm \
  --model aliyun/qwen3.7-max \
  --per-type 500 \
  --max-workers 5 \
  --timeout 600
```

> 服务器模式（L40S-ps-sf）与本机模式命令相同，只需改 `--sandbox-dir` 和 `--tdir` 为服务器上的绝对路径（如 `/home/ps/pi/sandbox_v21`）。

**Docker 模式**（每个任务启动独立容器）：

核心 `docker run` 命令：

```bash
docker run --rm --network host \
  -v {sandbox_dir}:/workspace \                    # 沙箱目录（sqlite + documents + 字典）
  -v {tmp_db}:/workspace/logistics.sqlite \        # sqlite 副本（防 agent 写污染原库）
  -v {BASE}/sessions:/root/.pi/agent/sessions \    # 会话持久化
  -v {BASE}/models.json:/root/.pi/agent/models.json \  # 模型配置
  pi-sandbox:latest \
  --provider my-local --model Qwen3.6-27B --mode json -p "{instruction}"
```

挂载说明：
- `{sandbox_dir}` → `/workspace`：沙箱完整目录（sqlite + documents + schema 字典）
- `{tmp_db}` → `/workspace/logistics.sqlite`：**sqlite 副本**，防止 agent 的 bash 写操作污染原库（v10 sqlite 污染事件的修复）
- `sessions` → 会话持久化：断点续跑时恢复上下文
- `models.json` → 模型配置

批量跑批脚本 `batch_run_pi.py` 封装了上述 docker run + 并发控制：

```bash
# 小批压测：test 组每类前 5 条，8 并发
python3 scripts/data/batch_run_pi.py --group test --per-type 5 --max-workers 8 --model-name qwen3.6-27B

# 正式跑批：sft 组每沙箱每类前 12 条，32 并发
python3 scripts/data/batch_run_pi.py --group sft --per-type 12 --max-workers 32 --model-name qwen3.6-27B
```

##### 服务器信息速查

| 项 | 值 |
|----|-----|
| 主机名 | L40S-ps-sf |
| IP | 30.144.109.13 |
| 用户 | ps |
| 端口 | 22 |
| 用途 | CPU 服务器，调用商用模型 API |
| 推理后端 | 无本地 vLLM，走顺丰内网 LLM Hub |

#### 1.4.3 轨迹格式转换

##### PI 原始输出格式

PI 跑批后，每条轨迹输出一个独立的 JSONL 文件（按 task_id 命名），包含完整的多轮交互事件流：

```
/data/renjunxiang/pi/trajectories/{group}_{sandbox}_{model}_{variant}/
├── sft_20260628_v15_dwh_task_001.jsonl          # 单条轨迹（JSONL 事件流）
├── sft_20260628_v15_dwh_task_001.stderr.log     # 错误日志
├── sft_20260628_v15_dwh_task_002.jsonl
├── ...
└── manifest.jsonl                                # 任务元数据（gold_answer/verification 等）
```

**单条轨迹 JSONL 文件结构**（按事件顺序，每行一个事件）：

```jsonl
{"type":"session", "version":3, "id":"uuid", "cwd":"/workspace"}       // 会话元信息
{"type":"agent_start"}                                                  // agent 开始执行
{"type":"turn_start"}                                                   // 开始新一轮对话
{"type":"message_start", "message":{"role":"user", "content":[{"type":"text", "text":"任务：分析..."}]}}
{"type":"message_end", "message":{"role":"user", "content":[{"type":"text", "text":"任务：分析..."}]}}
{"type":"message_start", "message":{"role":"assistant", "content":[]}}  // assistant 消息开始（content 为空）
{"type":"message_update", "delta":"The"}                                // 流式输出：thinking 第一个 token
{"type":"message_update", "delta":" user"}                              // 流式输出：第二个 token
{"type":"message_update", "delta":" is asking"}                         // 流式输出：第三个 token
...（几十上百个 message_update，逐个 token 输出 thinking）
{"type":"message_end", "message":{"role":"assistant", "content":[{"type":"thinking", "thinking":"完整的思考内容..."}, {"type":"toolCall", "id":"toolu_xxx", "name":"read", "arguments":{...}}]}}  // assistant 消息结束（content 是最终完整内容）
{"type":"tool_result", "toolCallId":"toolu_xxx", "content":"表结构：..."}  // 工具执行结果
{"type":"turn_start"}                                                   // 开始下一轮对话
...（重复多轮：assistant → tool_result → assistant）
```

**事件类型说明**：

- **session**：会话元信息（id、工作目录等）
- **agent_start / turn_start**：标记 agent 执行开始和每轮对话开始
- **message_start**：一条消息开始（此时 content 通常为空）
- **message_update**：流式输出的增量内容（thinking_delta / text_delta / tool_call_delta），逐个 token 输出
- **message_end**：一条消息结束（content 是最终完整内容，包含所有累积的 thinking / text / tool_calls）
- **tool_result**：工具执行结果

##### 转换逻辑

`scripts/data/pi_to_openai.py` 读取 JSONL 事件流，**只取 message_end 的完整内容**（忽略中间的 message_update 流式事件），按 user / assistant / tool 角色组装成 OpenAI messages 格式：

| PI 事件 | OpenAI 格式 |
|---------|------------|
| `message_end role=assistant` 中的 `type=thinking` | `reasoning_content` 字段 |
| `type=toolCall` | `tool_calls` 数组 |
| `message_end role=toolResult` | `role=tool` 消息 |
| 多轮交互循环 | 保留完整 user → assistant(tool_calls) → tool → assistant |
| — | 附带 `tools` 字段（bash/read/edit/write 四个工具 schema） |

**转换命令**：

```bash
# 单条转换
python3 scripts/data/pi_to_openai.py --input trajectory.jsonl --output trajectory_openai.jsonl

# 批量转换（从 tar.gz 读取）
python3 scripts/data/pi_to_openai.py --batch --input-dir datasets/PI/sft/ --output-dir datasets/trajectories/converted/
```

##### 转换后格式（OpenAI ChatCompletion）

每条轨迹一个 JSON 对象：

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert coding assistant... Available tools: bash/read/edit/write..."},
    {"role": "user", "content": "任务：分析华东区上周派送延误原因..."},
    {
      "role": "assistant",
      "reasoning_content": "我需要先查表结构...",
      "content": "",
      "tool_calls": [
        {"id": "toolu_xxx", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"/workspace/schema_dictionary.md\"}"}}
      ]
    },
    {"role": "tool", "tool_call_id": "toolu_xxx", "content": "表结构：fact_damage_record(id, damage_date, region, ...)"},
    {"role": "assistant", "reasoning_content": "...", "content": "根据查询结果...", "tool_calls": null}
  ],
  "tools": [...],
  "task_id": "sft_20260628_v15_dwh_task_001"
}
```

**关键特性**：
- **多轮对话**：完整保留 user → assistant(tool_calls) → tool → assistant 的交互循环
- **思考过程**：`reasoning_content` 字段保存完整 CoT（从 PI 的 thinking 事件提取）
- **工具调用**：`tool_calls` 字段包含工具名和 JSON 格式参数，训练框架自动序列化为 Qwen3.6 XML
- **工具定义**：`tools` 字段定义可用工具（bash/read/edit/write）

**转换后存储**：

```
datasets/trajectories/{group}/{sandbox}/{model}/converted/
├── qwen3.6-27B_20260628_v15_openai.jsonl        # 该沙箱下所有轨迹（每行一个 JSON）
└── ...
```

> **与单步数据的区别**：单步数据（MATH/GSM8K 等）是 system_prompt/task_input/thinking/golden_answer 四字段；多轮轨迹数据是 OpenAI messages 格式，包含完整的多轮 agent 交互过程。

#### 1.4.4 gzip 归档

原始 JSONL 总量 ~213G，通过 `scripts/data/pack_raw_gzip.sh` 按 `(group,sandbox,model)` 分组打包为 tar.gz：

```bash
# 服务器打包
bash scripts/data/pack_raw_gzip.sh
# 产出：archives/gzip/{group}_{sandbox}_{model}.tar.gz
# 压缩比：213G → 3.5G（~60:1）
```

**归档后目录结构**：

```
datasets/PI/sft/{v}/{model}/raw/
├── 20260628_v15_qwen3.6-27B_baseline.tar.gz     # 包含该沙箱下所有轨迹的 JSONL
├── 20260628_v15_qwen3.7-max_dict.tar.gz
└── ...
```

> **存储策略**：服务器保留解压的 JSONL（物理源，用于跑批/judge）；本机保留 tar.gz 压缩包（备份，~3.5G）。

### 1.5 数据路径速查

| 用途 | 本地路径 | 服务器路径 |
|------|----------|-----------|
| 开源数据 raw | `datasets/open_source/raw/`（27 集） | — |
| 四字段 processed | `datasets/sft/processed/{with,without}_golden_thinking/` | — |
| SFT 训练集 | `datasets/sft/final/20260702_openai.jsonl` | 同路径（scp 同步） |
| 沙箱源 | `datasets/sandboxes/raw/{group}/{v}/` | `/data/renjunxiang/pi/sandbox/{group}/{v}/` |
| 轨迹数据 | `datasets/trajectories/{group}/{v}/{model}/` | `/data/renjunxiang/pi/trajectories/{group}/{v}/{model}/raw/` |
| PI gzip 归档 | `datasets/PI/sft/{v}/{model}/raw/*.tar.gz` | `/data/renjunxiang/pi/archives/` |

**检查点**：
- ✅ `datasets/sft/final/20260702_openai.jsonl` 存在（25,167 条）
- ✅ 服务器 `/data/renjunxiang/pi/sandbox/sft/` 有 19 个沙箱
- ✅ 轨迹数据已采集并转换为 OpenAI 格式

---

## 第 3 章 监督微调（SFT）——技术储备（实际路线跳过此阶段）

> **本章目标**：从零开始，在 huawei-5 上用 MindSpeed-MM 和 ms-swift 框架对 Qwen3.6-27B 做 SFT（Full SFT + LoRA），完整记录从镜像到训练跑通的全过程。
>
> **本章状态**：✅ 已完成（2026-09-13 冒烟验证通过）
>
> **重要说明**：本项目实际路线**跳过了基座模型的 SFT**，直接走 GRPO。原因是 SFT 用开源短数据训练后导致思维变短（技术报告 §7.4.4）。但本章的框架验证作为技术储备保留，确保需要时能快速启动。
>
> **章节结构**：本章两个框架完全独立，新人可根据需要只读对应章节：
> - **§3.1 MindSpeed-MM SFT**：完整端到端流程（环境→修复→权重转换→数据→配置→训练→验证→踩坑→checkpoint转换），含 Full SFT 和 LoRA
> - **§3.2 ms-swift SFT**：完整端到端流程（环境安装→数据→训练命令→验证→超参→对比），含 LoRA 和 Full SFT
> - **§3.3 为什么不继续 SFT**：项目决策记录
>
> **框架选择速查**：
>
> | 维度 | MindSpeed-MM（§3.1） | ms-swift（§3.2） |
> |------|---------------------|------------------|
> | 训练速度 | 3.4s/iter（快 19 倍） | 65s/iter |
> | 易用性 | 需写 YAML + 修复代码 | 一行 `swift sft` 命令 |
> | 权重格式 | 需 HF→DCP 转换 | 直接读 HF |
> | 数据要求 | 需 `images: null` 字段 | 标准 OpenAI 格式即可 |
> | 峰值显存（LoRA） | 33 GB/chip | 7.86 GB/chip |
> | **推荐场景** | 正式训练、追求效率 | 快速实验、超参扫描 |

---

### 3.1 MindSpeed-MM SFT（完整端到端流程）

> 本节从零开始，在 huawei-5 上用 MindSpeed-MM 框架完成 Qwen3.6-27B 的 Full SFT 和 LoRA SFT。按时间顺序分 8 个步骤，每步有验证点。照着做就能跑通。

#### 3.1.1 步骤 1：环境准备（硬件 + 镜像 + 软件版本）

**硬件环境**（huawei-5）：

| 项 | 值 |
|----|-----|
| 服务器 | huawei-5（Atlas 800 A3） |
| NPU | 8 张昇腾 910 = 16 chip（每卡 2 chip） |
| 单 chip HBM | 64 GiB（整机 1 TiB） |
| CPU | 4 路鲲鹏 920（640 vCPU） |
| 内存 | 2.0 TiB DDR4 |
| 存储 | 17 TiB NVMe |

**容器镜像**：

| 项 | 值 |
|----|-----|
| 镜像名 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-mm:26.0.0-a3-openeuler24.03-py3.11-aarch64` |
| 大小 | 19.3 GB |
| 容器名 | `mindspeed_mm_rjx` |

**启动容器**：

```bash
# 如果是全新服务器
docker run -dit \
  --ipc=host --network host --name mindspeed_mm_rjx --privileged \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
  -v /usr/local/sbin/:/usr/local/sbin/ \
  -v /home/:/home/ \
  -v /data:/data \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-mm:26.0.0-a3-openeuler24.03-py3.11-aarch64 \
  sleep infinity

# 如果容器已存在但已停止
docker start mindspeed_mm_rjx
```

**挂载说明**：

| 宿主机路径 | 容器路径 | 用途 |
|-----------|---------|------|
| `/usr/local/Ascend/driver` | 同 | NPU 驱动（硬件层） |
| `/usr/local/Ascend/add-ons` | 同 | NPU 附加组件 |
| `/usr/local/sbin/` | 同 | 系统工具 |
| `/home/` | 同 | 用户目录 |
| `/data/` | 同 | **关键**：模型权重、训练数据、代码都在此 |

> ⚠️ 容器内 `/workspace/MindSpeed-MM/` 是镜像内置的（不挂载宿主机），容器重启后修改会丢失。需要持久化的代码改动要写到 `/data/` 下。

**软件版本**：

| 组件 | 版本 | 备注 |
|------|------|------|
| Python | 3.11.14 | 容器内置 |
| PyTorch | 2.7.1+cpu | 配合 torch_npu |
| torch_npu | 2.7.1 | 昇腾 NPU 后端 |
| transformers | 5.12.1 | ⚠️ 与 MindSpeed-MM 有 cache_position 兼容问题（步骤 2 修复） |
| MindSpeed-MM | 0.1 | 容器内置（`/workspace/MindSpeed-MM/`） |

**模型权重路径**：

| 模型 | 路径 | 格式 | 大小 |
|------|------|------|------|
| Qwen3.6-27B（HF 原始） | `/data/models/Qwen3.6-27B/` | 15 个 safetensors | 52 GB |

> ⚠️ MindSpeed-MM 训练需要 DCP 格式权重，HF 原始格式无法直接使用。转换命令见步骤 3.1.3。

**验证 NPU 可用**：

```bash
docker exec mindspeed_mm_rjx bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python -c "import torch; import torch_npu; print(torch.npu.device_count())"
'
# 期望输出：16
```

#### 3.1.2 步骤 2：修复 cache_position 兼容性问题（必须！）

**问题**：transformers 5.12.1 的 `create_causal_mask()` 不再接受 `cache_position` 参数，但 MindSpeed-MM 的 `modeling_qwen3_5.py` 还在传，导致 `TypeError: create_causal_mask() got an unexpected keyword argument 'cache_position'`。

**修复**：

```bash
docker exec mindspeed_mm_rjx bash -c '
  cd /workspace/MindSpeed-MM
  sed -i "/cache_position=cache_position,/d" mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py
  echo "修复完成，剩余 cache_position=cache_position 行数："
  grep -c "cache_position=cache_position," mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py || echo "0"
'
```

> ⚠️ 此修改在容器内 `/workspace`，容器重启会丢失。如果重建容器，需重新执行。

#### 3.1.3 步骤 3：转换模型权重 HF → DCP

MindSpeed-MM 训练要求 DCP 格式的权重（`init_model_with_meta_device: true` 要求）。如果 `/data/models/Qwen3.6-27B-dcp/` 已存在可跳过。

```bash
docker exec mindspeed_mm_rjx bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  mm-convert Qwen35Converter hf_to_dcp \
    --hf_dir /data/models/Qwen3.6-27B \
    --dcp_dir /data/models/Qwen3.6-27B-dcp
'
# 耗时约 70 秒，产出 15 个 .distcp 文件共 52GB
```

**转换后模型权重路径**：

| 模型 | 路径 | 格式 | 大小 |
|------|------|------|------|
| Qwen3.6-27B（HF 原始） | `/data/models/Qwen3.6-27B/` | 15 个 safetensors | 52 GB |
| Qwen3.6-27B（DCP 转换后） | `/data/models/Qwen3.6-27B-dcp/` | MindSpeed DCP 格式 | 52 GB |

**验证**：`ls /data/models/Qwen3.6-27B-dcp/` 应有 `release/` 目录和 `latest_checkpointed_iteration.txt`。

#### 3.1.4 步骤 4：准备训练数据

**数据格式要求**：MindSpeed-MM 使用 OpenAI ChatCompletion 格式（messages 数组），且**必须包含 `images` 字段**（即使纯文本训练也要设为 `null`，因为 qwen3_5 是多模态模型框架）。

```json
{
  "messages": [
    {"role": "system", "content": "你是一个数学推理助手..."},
    {"role": "user", "content": "求解方程 2x + 3 = 7"},
    {"role": "assistant", "content": "<answer>\nx = 2\n</answer>", "reasoning_content": "首先将 3 移到等号右边..."}
  ],
  "images": null
}
```

**数据准备**：使用 `scripts/train/mindspeed_mm/sft/prepare_data.py` 将四字段格式转为 OpenAI 格式（脚本源码见**附录-3.1**）：

```bash
# 默认：从 GSM8K 和 MATH 各取 200 条做冒烟测试（共 400 条）
python3 scripts/train/mindspeed_mm/sft/prepare_data.py
# 产出：/data/renjunxiang/coding/huawei_train/datasets/sft_smoke_400.jsonl

# 自定义输入文件、条数和输出路径
python3 scripts/train/mindspeed_mm/sft/prepare_data.py \
  --input-dir /data/renjunxiang/coding/huawei_train/datasets/open_source/processed/with_golden_thinking/Reasoning/ \
  --files GSM8K.jsonl MATH.jsonl \
  --per-file 200 \
  --output /data/renjunxiang/coding/huawei_train/datasets/sft_smoke_400.jsonl
```

**复制到容器可见路径**（MindSpeed-MM 用相对路径加载数据）：

```bash
docker exec mindspeed_mm_rjx cp \
  /data/renjunxiang/coding/huawei_train/datasets/sft_smoke_400.jsonl \
  /workspace/MindSpeed-MM/data/sft_smoke_400.jsonl
```

#### 3.1.5 步骤 5：编写训练配置（Full SFT 完整 YAML）

**配置文件**：保存为 `/workspace/MindSpeed-MM/examples/qwen3_6/smoke_test_config.yaml`（完整内容见**附录-3.2**）。

```bash
# 创建目录并复制配置文件到容器内
mkdir -p /workspace/MindSpeed-MM/examples/qwen3_6/
cp scripts/train/mindspeed_mm/sft/smoke_test_config.yaml \
   /workspace/MindSpeed-MM/examples/qwen3_6/smoke_test_config.yaml
```

**关键超参说明**：

| 超参 | 值 | 选择依据 |
|------|-----|---------|
| `lr` | 1.0e-5 | Full SFT 标准值，太大容易不稳定 |
| `micro_batch_size` | 1 | 27B 全参 + 长上下文，单卡放不下更大 batch |
| `cutoff_len` | 1024 | ⚠️ 不超过 2048（NPU rotary 算子限制） |
| `recompute` | true | 27B 全参必须开，否则显存不够 |
| `clip_grad` | 0.0 | Full SFT 通常不裁剪，LoRA 建议设为 1.0 |
| `dataloader_mode` | `sampler` | ⚠️ 必填，不加会报错 |
| `sampler_type` | `BaseRandomBatchSampler` | ⚠️ 必填 |
| `dataset` | `preprocess_parameters` 内 | ⚠️ 层级不能错 |

**性能实测**（huawei-5, 400 条数据, cutoff_len=1024）：

| 指标 | 值 |
|------|-----|
| iter 1 耗时 | 31.2s（含模型加载和编译） |
| iter 2+ 耗时 | 3.4-4.4s |
| 峰值显存 | 33,041 MB/chip（~33 GB） |
| global batch size | 16 |

#### 3.1.6 步骤 6：启动训练

```bash
docker exec -d mindspeed_mm_rjx bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd /workspace/MindSpeed-MM

  export NON_MEGATRON=true                # FSDP2 模式（非 Megatron）
  export MULTI_STREAM_MEMORY_REUSE=2
  export TASK_QUEUE_ENABLE=2
  export ASCEND_LAUNCH_BLOCKING=0
  export ACLNN_CACHE_LIMIT=100000         # NPU 算子缓存上限
  export CPU_AFFINITY_CONF=1
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True  # 允许显存动态扩展
  export HCCL_CONNECT_TIMEOUT=7200        # 16 卡初始化可能慢

  torchrun \
    --nproc_per_node 16 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000 \
    mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3_6/smoke_test_config.yaml \
    > /data/renjunxiang/coding/huawei_train/models/smoke_sft_full/train.log 2>&1
'
```

**查看日志**：

```bash
# 实时查看
ssh huawei-5 'tail -f /data/renjunxiang/coding/huawei_train/models/smoke_sft_full/train.log'

# 检查训练是否完成
ssh huawei-5 'grep "iteration" /data/renjunxiang/coding/huawei_train/models/smoke_sft_full/train.log'
```

#### 3.1.7 步骤 7：验证训练结果

```bash
# 检查 checkpoint 是否生成
ls /data/renjunxiang/coding/huawei_train/models/smoke_sft_full/

# 期望输出：
# iter_0000003/  latest_checkpointed_iteration.txt  train.log
```

**训练日志关键指标**（Full SFT 冒烟实测）：

```
iteration 1/3 | loss: 2.073 | grad_norm: 214.8 | elapsed: 31219ms（含模型加载）
iteration 2/3 | loss: 0.400 | grad_norm: 23.3  | elapsed: 4420ms
iteration 3/3 | loss: 0.167 | grad_norm: 16.8  | elapsed: 3367ms
memory: allocated 19353 MB | max allocated 33041 MB | reserved 41866 MB
```

#### 3.1.8 LoRA 变体

LoRA 和 Full SFT 共用框架，只需在配置末尾追加 `lora_args` 段，并修改 `lr` 和 `clip_grad`：

```yaml
# ---- 修改项 ----
training:
  lr: 5.0e-5                       # LoRA 学习率比 Full SFT 大 5 倍
  clip_grad: 1.0                   # LoRA 建议做梯度裁剪
  save: /data/renjunxiang/coding/huawei_train/models/sft_lora

# ---- 新增段 ----
lora_args:
  lora_r: 8                        # LoRA 秩（rank）
  lora_alpha: 16                   # 缩放因子（alpha/rank = 2）
  lora_dropout: 0.0
  lora_target_modules: v_proj,o_proj,k_proj,q_proj
```

**LoRA 超参经验**：

| 超参 | 值 | 选择依据 |
|------|-----|---------|
| `lora_r` | 8 | 4/8/16/32 扫描后 8 是性价比甜点 |
| `lora_alpha` | 16 | alpha/rank = 2，常见比例 |
| `lora_target_modules` | `q/k/v/o_proj` | 注意力层全部注入 |
| `lr` | 5.0e-5 | LoRA 标准值 |
| 可训练参数 | ~58M（0.21%） | 27B 模型的 0.21% |

**性能对比**：

| 指标 | Full SFT | LoRA |
|------|---------|------|
| iter 1 耗时 | 31.2s | 16.5s |
| iter 2+ 耗时 | 3.4-4.4s | 3.4-4.5s |
| 峰值显存 | 33 GB/chip | 33 GB/chip（相同，基础模型都要加载） |
| 可训练参数 | 27.4B | ~58M |

> **注意**：LoRA 和 Full SFT 显存几乎相同（~33 GB/chip），因为 27B 权重都要加载。LoRA 优势在优化器状态小、产物小（adapter 几十 MB）。

#### 3.1.9 踩坑记录（5 个坑）

| 编号 | 现象 | 根因 | 修复 |
|------|------|------|------|
| 坑 1 | `TypeError: create_causal_mask() got unexpected keyword 'cache_position'` | transformers 5.12.1 接口变了 | `sed -i "/cache_position=cache_position,/d"` 删参数行。**容器重建后需重新修复** |
| 坑 2 | `aclnnRotaryPositionEmbeddingGrad error 561002` | NPU rotary 算子 cutoff > 2048 时崩溃 | 强制走 torch 原生 rotary 分支。**规避：cutoff_len ≤ 2048** |
| 坑 3 | `pydantic ValidationError: dataloader_mode/sampler_type Field required` | 配置校验要求这两个字段 | 在 `dataloader_param` 下添加 `dataloader_mode: sampler` + `sampler_type: BaseRandomBatchSampler` |
| 坑 4 | `KeyError: 'images'` | qwen3_5 是多模态框架，数据必须有 images 字段 | 给每条数据添加 `"images": null` |
| 坑 5 | `Failed to load JSON from file '.../train_engine.py'` | dataset 字段放错层级 | 必须放在 `preprocess_parameters.dataset` 内，不能放在 `dataset_param.dataset` |

#### 3.1.10 Checkpoint 转 HF（用于 vLLM 推理）

训练产物是 MindSpeed DCP 格式，需要转回 HF 才能在 vLLM 推理：

```bash
docker exec mindspeed_mm_rjx bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  mm-convert Qwen35Converter dcp_to_hf \
    --save_hf_dir /data/models/Qwen3.6-27B-sft-hf \
    --dcp_dir /data/renjunxiang/coding/huawei_train/models/sft_full/iter_00500 \
    --origin_hf_dir /data/models/Qwen3.6-27B
'
```

---

### 3.2 ms-swift SFT（完整端到端流程）

> ms-swift（ModelScope）是更轻量的训练框架，一行命令即可启动。以下是在 huawei-5 上从零跑通的完整记录。
>
> **与 MindSpeed-MM 的核心差异**：ms-swift 直接读 HF 权重（不需 DCP 转换）、数据不需要 `images` 字段、配置通过命令行参数传入（不需写 YAML）。但训练速度慢 19 倍（NPU 适配深度不如 MindSpeed-MM）。

#### 3.2.1 步骤 1：环境准备（安装 + 验证）

**前提**：使用与 MindSpeed-MM 相同的容器 `mindspeed_mm_rjx`。ms-swift 需要额外安装。

**硬件环境**：与 §3.1.1 相同（huawei-5，16 chip Ascend910）。

**软件安装**：

```bash
# 进入容器
docker exec -it mindspeed_mm_rjx bash

# ⚠️ 必须 source CANN 环境（否则 torch_npu 报 undefined symbol）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装 ms-swift
pip install ms-swift -q

# 验证安装
swift --version
# 期望：4.5.3

# 验证 NPU 可用
python -c "import torch; import torch_npu; print(f'NPU: {torch.npu.device_count()}')"
# 期望：NPU: 16
```

> ⚠️ **关键**：每次新开 shell 都要 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。用 `docker exec` 运行 swift 时，必须在同一条命令中 source：
> ```bash
> docker exec mindspeed_mm_rjx bash -c '
>   source /usr/local/Ascend/ascend-toolkit/set_env.sh
>   swift sft ...
> '
> ```

#### 3.2.2 步骤 2：准备数据

ms-swift 直接读取 JSONL 格式，使用标准 OpenAI ChatCompletion 格式。**不需要 `images` 字段**（与 MindSpeed-MM 不同）。

```json
{
  "messages": [
    {"role": "system", "content": "你是一个数学推理助手..."},
    {"role": "user", "content": "求解方程 2x + 3 = 7"},
    {"role": "assistant", "content": "<answer>\nx = 2\n</answer>", "reasoning_content": "首先将 3 移到等号右边..."}
  ]
}
```

**数据准备**：使用 `scripts/train/mindspeed_mm/sft/prepare_data.py`，指定 `--framework swift`（脚本源码见**附录-3.1**）：

```bash
# ms-swift 格式（不需要 images 字段）
python3 scripts/train/mindspeed_mm/sft/prepare_data.py --framework swift
# 产出：/data/renjunxiang/coding/huawei_train/datasets/sft_smoke_400_swift.jsonl
```

> 如果已有 MindSpeed-MM 格式的数据（带 `images: null`），也可以直接用，ms-swift 会忽略 `images` 字段。

#### 3.2.3 步骤 3：启动训练（完整命令）

```bash
# ⚠️ 必须在同一条命令中 source CANN 环境
docker exec -d mindspeed_mm_rjx bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

  swift sft \
    --model /data/models/Qwen3.6-27B \
    --dataset /data/renjunxiang/coding/huawei_train/datasets/sft_smoke_400.jsonl \
    --output_dir /data/renjunxiang/coding/huawei_train/models/ms_swift_sft \
    --max_steps 500 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 100 \
    > /data/renjunxiang/coding/huawei_train/models/ms_swift_sft/train.log 2>&1
'
```

**关键参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model` | `/data/models/Qwen3.6-27B` | HF 权重路径（ms-swift 直接读 HF，不需要 DCP 转换） |
| `--dataset` | 数据文件路径 | JSONL 格式，OpenAI messages |
| `--output_dir` | 输出目录 | checkpoint 和日志保存位置 |
| `--max_steps` | 500 | 总训练步数 |
| `--per_device_train_batch_size` | 1 | 单卡 batch |
| `--gradient_accumulation_steps` | 16 | global_batch = 1×16×1 = 16 |
| `--learning_rate` | 5e-5 | LoRA 默认值 |
| `--lr_scheduler_type` | cosine | 学习率调度 |
| `--warmup_ratio` | 0.1 | 热身比例 |
| `--save_steps` | 100 | 每 100 步保存 |

> ⚠️ **ms-swift 默认使用 LoRA**（rank=8）。如需 Full SFT，需添加 `--train_type full`。

#### 3.2.4 步骤 4：查看训练进度

```bash
# 实时查看日志
ssh huawei-5 'tail -f /data/renjunxiang/coding/huawei_train/models/ms_swift_sft/train.log'

# 查看训练指标（JSONL 格式）
ssh huawei-5 'tail -5 /data/renjunxiang/coding/huawei_train/models/ms_swift_sft/v0-*/logging.jsonl'
```

**日志样例**（每行一个 JSON）：

```json
{"loss": 1.045, "grad_norm": 18.47, "learning_rate": 7.5e-06, "token_acc": 0.813,
 "global_step/max_steps": "1/3", "memory(GiB)": 7.85, "train_speed(s/it)": 68.19}
```

#### 3.2.5 步骤 5：验证训练结果

```bash
# 检查 checkpoint 目录
ssh huawei-5 'ls /data/renjunxiang/coding/huawei_train/models/ms_swift_sft/v0-*/'
# 期望：checkpoint-100/  checkpoint-200/  ...  logging.jsonl  train_args.json
```

**训练日志关键指标**（ms-swift LoRA 冒烟实测）：

| 指标 | 值 |
|------|-----|
| 模型类型 | PeftModelForCausalLM（LoRA） |
| 总参数 | 27,415 M |
| 可训练参数 | 58.36 M（0.21%） |
| iter 1 耗时 | 68s |
| iter 2+ 耗时 | 65-67s |
| 峰值显存 | 7.86 GB/chip |
| token_acc | ~0.80 |

#### 3.2.6 超参经验总结

| 超参 | 默认值（LoRA） | 说明 |
|------|--------------|------|
| `train_type` | LoRA（默认） | ms-swift 默认行为 |
| `lora_rank` | 8 | ms-swift 默认值 |
| `lora_alpha` | 32 | ms-swift 默认值（alpha/rank = 4） |
| `lora_target_modules` | `all-linear` | 默认注入所有 Linear 层 |
| `learning_rate` | 5e-5 | LoRA 默认值 |
| `per_device_train_batch_size` | 1 | NPU 显存限制 |
| `gradient_accumulation_steps` | 16 | global_batch = 16 |

**与 MindSpeed-MM 对比**：

| 维度 | MindSpeed-MM | ms-swift |
|------|-------------|----------|
| iter 时间 | 3.4s | 65s（**慢 19 倍**） |
| 峰值显存 | 33 GB/chip | 7.86 GB/chip |
| 数据格式 | 需 `images: null` | 标准 OpenAI 格式 |
| 权重格式 | 需 DCP 转换 | 直接读 HF |
| 配置方式 | YAML 文件 | 命令行参数 |
| NPU 适配 | 极深（triton 算子） | 一般 |
| **推荐** | 正式训练 | 快速实验 |

#### 3.2.7 踩坑记录（1 个坑）

| 编号 | 现象 | 根因 | 修复 |
|------|------|------|------|
| 坑 1 | `ImportError: undefined symbol` | 未 source CANN 环境 | 运行 swift 前必须 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |

---

### 3.3 为什么不继续 SFT（关键决策）

**踩坑：SFT 训练导致思维变短**（技术报告 §7.4.4）

用开源数据（MATH/GSM8K 等 25K 条）SFT 后：

| 指标 | 训练前（基座） | 训练后（SFT） |
|------|--------------|-------------|
| AIME24 准确率 | 93.3%（28/30） | **13.3%（4/30）** |
| 推理长度中位数 | 15,863 token | **484 token（缩短 32.8 倍）** |
| 终止 `</think>` 概率 | 基线 | 放大 605~270,518 倍 |

**根因**：开源数学数据答案普遍很短（几十到几百 token），而 AIME 竞赛题需要长链推理（上万 token）。交叉熵损失在短轨迹上反复优化，把解码策略压向短路径和早终止。

**决策**：基座在沙箱已有非零得分，不存在冷启动问题。SFT 的风险（思维变短）大于收益，直接走 GRPO——在保持长链推理能力的基础上，通过 RL 提升垂域表现。

---

## 第 4 章 GRPO 强化学习训练

> **本章目标**：复现我们负责的 GRPO 工程环节：单机多卡以 Qwen3.5-9B 做数学冒烟，多机多卡以 Qwen3.6-27B 做物流工具任务训练。
>
> **本章状态**：⏳ 单机 9B 已尝试，训练闭环尚未通过；✅ 双机 27B 有历史训练、权重同步、Step120 检查点及 HF 导出记录。
>
> **核验日期**：2026-09-14。本章根据 5 号机上的脚本、数据契约、历史日志和产物清单补齐；本次编写没有启动训练。6 号机的历史参与情况由 5 号机保存的远端 Rollouter 日志确认，不代表其当前运行状态。
>
> **章节结构**：沿用第三章的端到端操作组织方式，两条路线分别阅读：
> - **§4.1 路线与路径约定**：先明确模型、设备口径和两套入口。
> - **§4.2 单机多卡 9B**：环境→数据→奖励接口→配置→资源→启动→验证→踩坑。
> - **§4.3 多机多卡 27B**：环境→数据与工具→奖励→配置→Ray→恢复→启动→验证→导出。
> - **§4.4 验收与路径速查**：区分训练完成、产物完整和能力提升。

### 4.1 路线与路径约定

**方案速查**：

| 维度 | 单机多卡（§4.2） | 多机多卡（§4.3） |
|------|------|------|
| 模型 | Qwen3.5-9B | Qwen3.6-27B |
| 任务 | GSM8K/MATH 单轮数学题 | 物流 DWH 多轮工具任务 |
| 服务器 | 5 号机，一机分配 8+8 chip | 5 号机训练、6 号机 rollout，各 16 chip |
| Trainer | FSDP2，分片组大小 8 | Megatron，TP4×PP2×CP2，训练 DP1 |
| Rollout | vLLM，TP4×DP2 | vLLM，TP8×DP2 |
| veRL 入口 | `experimental.one_step_off_policy.main_ppo` | `experimental.fully_async_policy.fully_async_main` |
| 权重同步 | 配置 backend 为 `nccl`，NPU 实际进入 HCCL 实现 | 同样使用 NPU checkpoint engine 同步 |
| 验证状态 | 权重同步阶段失败，无完成训练步证据 | Step100→120 续训、保存与 HF 导出有记录 |

**设备口径**：Atlas 800 A3 单机 8 个 NPU ID，每个包含 2 个计算 chip，共 16 chip，每 chip 64 GiB HBM。本章统一按 **chip** 计数，双机共 32 chip。FSDP 分片组不等于张量并行；推理 DP2 也不等于训练 DP2。

**命令执行位置**：本章中未标注“宿主机”的命令均在训练容器内执行。沿用当前本地 SSH 别名 `huawei-05`、`huawei-06`，原报告中的 `huawei-5` / `huawei-llm` 对应同一台 5 号机；实际 SSH 端口使用自己的配置，不从内网地址推测公网端口。

| 用途 | 5 号机宿主机路径 | 容器内路径 |
|------|------|------|
| GRPO 项目 | `/data3/llin/qwen3.6-27b-verl-grpo` | `/workspace/llin-verl-grpo` |
| 27B 基座 | `/data3/llin/base_model/Qwen3.6-27B` | `/models/Qwen3.6-27B`（只读挂载） |
| 9B 基座 | 项目下 `models/Qwen3.5-9B` | `/workspace/llin-verl-grpo/models/Qwen3.5-9B` |
| 沙箱源 | `/data/renjunxiang/pi/sandbox` | `/pi_sandbox`（只读挂载） |
| 框架源码 | 容器内资产 | `/verl`、`/vllm`、`/vllm-ascend` |
| 日志与检查点 | 项目下 `runs/` | 项目下 `runs/` |

> **复现版本说明**：以下命令复用项目已有运行时及补丁。镜像标签不能覆盖容器后续修改；要复现历史实验，需同时保存脚本、实际展开配置、模型身份、数据契约和依赖版本。不要仅按镜像标签重新安装最新依赖。

---

### 4.2 单机多卡 GRPO：Qwen3.5-9B

> 本节记录 2026-09-13 的实际尝试与 9 月 14 日核验结果。8+8 是已有配置，**尚不是验证通过的训练配方**。步骤中的启动命令用于修复阻塞后的复测，不能把执行到模型加载视作 GRPO 已跑通。

#### 4.2.1 步骤 1：环境准备（硬件、容器与软件）

| 项目 | 实际配置 |
|------|------|
| 服务器 | 5 号机 Atlas 800 A3，16 chip |
| 容器 | `llin-verl-trainer-m05-20260730` |
| 镜像标签 | `llin-verl-a3:20260730` |
| 模型 | Qwen3.5-9B，配置架构 `Qwen3_5ForConditionalGeneration` |
| Trainer 预算 | 8 chip，FSDP2 |
| Rollout 预算 | 8 chip，TP4×DP2 |

**进入已有环境**（5 号机宿主机）：

```bash
ssh huawei-05
npu-smi info
docker ps --filter name=llin-verl-trainer-m05-20260730
docker exec -it llin-verl-trainer-m05-20260730 bash
```

**容器内检查**：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/llin-verl-grpo
python3 - <<'PY'
import importlib.metadata as md
import torch
import torch_npu
for package in ('verl', 'torch', 'torch-npu', 'ray', 'vllm', 'transformers'):
    print(package, md.version(package))
print('NPU count:', torch.npu.device_count())
PY
test -f models/Qwen3.5-9B/config.json
test -f scripts/run_math_grpo_smoke.sh
test -f llin_verl/math_reward.py
```

9 月 14 日此前现场核验版本为 veRL `0.9.0.dev0`、PyTorch `2.9.0`、torch-npu `2.9.0.post2`、Ray `2.56.1`、vLLM `0.18.0+empty`、Transformers `5.10.4`。这是本 GRPO 容器的版本记录，不能套用第三章 MindSpeed-MM 或第六章独立推理服务的版本表。

**检查点**：容器能看到模型完整权重、框架能导入、当前 chip 资源可用。`config.json` 存在只证明配置可见，还需检查权重索引引用的分片。

#### 4.2.2 步骤 2：准备数学任务数据

**已存在的数据**：`data/math_smoke.parquet`，100 条，来源标签为 GSM8K 50 条、MATH 50 条。

**已有转换脚本**：`scripts/prepare_math_dataset.py`，从四字段 JSONL 读取 `system_prompt`、`task_input`、`golden_answer`。如需重建，先将下列输入变量设为实际文件路径，另存新文件：

```bash
python3 scripts/prepare_math_dataset.py \
  --gsm8k "${GSM8K_JSONL:?设置 GSM8K 四字段 JSONL 路径}" \
  --math "${MATH_JSONL:?设置 MATH 四字段 JSONL 路径}" \
  --max-samples 50 \
  --output data/math_smoke_rebuilt.parquet
```

**当前产物结构**：

```json
{
  "data_source": "gsm8k",
  "prompt": [
    {"role": "system", "content": "Solve the math problem..."},
    {"role": "user", "content": "题目正文"}
  ],
  "ground_truth": {"golden_answer": "42"}
}
```

**读取检查**：

```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_parquet('data/math_smoke.parquet')
print('rows:', len(df))
print('columns:', df.columns.tolist())
print(df['data_source'].value_counts().to_dict())
PY
```

> **待修复项**：现有文件的答案在顶层 `ground_truth`，而当前容器 `naive` reward manager 读取 `reward_model.ground_truth`。下一次试跑前要转换数据并经过真实 dataset/reward manager 检查；上面的现有制备脚本仍会产出旧结构，重新运行它不会自动解决问题。

与当前 manager 对齐的目标结构应包含：

```json
{
  "data_source": "gsm8k",
  "prompt": [{"role": "user", "content": "题目正文"}],
  "reward_model": {
    "style": "rule",
    "ground_truth": {"golden_answer": "42"}
  },
  "extra_info": {}
}
```

这是接口修复目标，尚未作为新训练数据验证。训练和验证复用这 100 条只适合冒烟，不能据此报告泛化准确率。

#### 4.2.3 步骤 3：对齐数学 Reward 接口

**文件**：`llin_verl/math_reward.py`。逻辑为提取 `boxed` / `fbox` / `<answer>` / 答案提示后的内容，进行数值近似或字符串比较，正确返回 1.0，错误返回 0.0。

**当前阻塞**：manager 使用 `solution_str=...` 调用，现有函数要求 `response` 参数；已有 CPU 合成调用复现了接口错误。修复后的接口应接受实际调用关键字，例如：

```python
# 接口示意：内部答案解析仍需同步使用 solution_str，并验证评分用例。
def compute_score(data_source, solution_str, ground_truth,
                  extra_info=None, **kwargs):
    ...
```

**验证点**：通过真实 manager 检查正确答案、错误答案、空回答、负分数与多个答案标记。现有负分数正则以及 `boxed/fbox` 混合排序也有待修正，不能将该评分器描述为完整数学符号等价判定。

GRPO 需要同一题多条采样的奖励差异。整组奖励相同时，该组的相对优势通常为零；**是否跳过 optimizer 要看训练实现和实际日志**，不能仅凭全错组就写“不调 optimizer”。

#### 4.2.4 步骤 4：训练脚本与关键配置

**完整入口**：`scripts/run_math_grpo_smoke.sh`。以下为关键参数摘录，不替代原脚本：

```bash
python3 -m verl.experimental.one_step_off_policy.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.data_parallel_size=2 \
  actor_rollout_ref.rollout.multi_turn.enable=False \
  trainer.nnodes=1 trainer.n_gpus_per_node=8 \
  rollout.nnodes=1 rollout.n_gpus_per_node=8
```

| 参数 | 现有脚本值 | 用途 |
|------|------|------|
| `data.train_batch_size` | 4 | 每批 4 个 prompt |
| `rollout.n` | 4 | 每题 4 条回答，名义上每批 16 条轨迹 |
| `actor.optim.lr` | `1e-6` | 学习率 |
| prompt / response 上限 | 1024 / 2048 | 总长度 3072 |
| `ppo_mini_batch_size` | 4 | PPO mini-batch 配置 |
| `ppo_micro_batch_size_per_gpu` | 1 | 微批设置 |
| `actor.use_dynamic_bsz` | True | 动态 token 批次 |
| `rollout.gpu_memory_utilization` | 0.60 | 推理显存预算 |
| `rollout.enforce_eager` | True | eager 执行 |
| `checkpoint_engine` bucket | 2560 MB | 权重同步分桶 |
| reward KL / actor KL loss | 均关闭 | 本次没有 KL 惩罚项 |
| `trainer.total_training_steps` | 3 | 计划冒烟步数 |
| `trainer.save_freq` | 1 | 每步保存 |
| `trainer.resume_mode` | disable | 不恢复历史检查点 |

脚本虽然导出 `TRAIN_TP=4`，却没有用它配置训练 TP；此处 Trainer 仍应写 FSDP size 8。`critic.strategy=fsdp2` 是配置项，GRPO 路线没有因此多出一个独立价值网络。

#### 4.2.5 步骤 5：Ray 资源与设备隔离

```text
5 号机：16 chip
  Ray 节点资源与设备调度
    ├─ Trainer：8 chip，FSDP2
    └─ Rollout：8 chip，vLLM TP4×DP2
           ↑ 接收 Trainer 更新后的权重
```

**先检查现有集群**：

```bash
export RAY_ADDRESS=192.168.202.5:26379
ray status --address="$RAY_ADDRESS"
python3 - <<'PY'
import os, ray
ray.init(address=os.environ['RAY_ADDRESS'])
print('cluster:', ray.cluster_resources())
print('available:', ray.available_resources())
for node in ray.nodes():
    if node['Alive']:
        print(node['NodeManagerAddress'], node['Resources'])
PY
```

`llin_trainer` / `llin_rollout` 是节点角色标签，不能当成 chip 掩码。本节不沿用原文“在同机再启动一个 Ray worker 即可划出 rollout 卡”的做法：同机隔离必须结合实际资源池、可见设备和 worker 映射验证；双机 `start_ray_m05.sh` 单独使用也不会自动完成 8+8 配置。

现有 v11 的 CANN plog 已记录训练组 chip 0–7、权重接收端 chip 8–15；同步组包含 Trainer rank 0，属于发送设计，不是抢卡证据。新运行仍应保存自己的 worker→chip 映射。当前没有独立验证通过的单机集群初始化脚本，因此此步骤及权重同步验收仍是单机复测前置工作。

#### 4.2.6 步骤 6：启动冒烟与查看日志

**使用条件**：先完成 §4.2.2–§4.2.5 的接口修复、资源准备及权重同步排查。以下是现有入口的调用方式，本次没有执行：

```bash
cd /workspace/llin-verl-grpo
export MODEL_PATH=/workspace/llin-verl-grpo/models/Qwen3.5-9B
export DATA_FILE="${MATH_FIXED_PARQUET:?设置已通过接口检查的数据路径}"
export RUN_NAME="math-grpo-smoke-9b-$(date +%Y%m%d-%H%M%S)"
export OUTPUT_DIR="/workspace/llin-verl-grpo/runs/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"
bash scripts/run_math_grpo_smoke.sh > "$OUTPUT_DIR/driver.log" 2>&1
```

在另一个容器 shell 查看：

```bash
tail -f '/workspace/llin-verl-grpo/runs/<本次运行名>/driver.log'
```

`Total training steps: 3` 表示计划预算；要看到实际训练指标和检查点，才说明对应步骤执行完成。

#### 4.2.7 步骤 7：验证结果与失败定位

**历史事实**：12 份 `runs/math_grpo_smoke*.log` 未检出 `training/global_step` 完成指标，相关 `runs/math*/checkpoints/global_step_*` 未发现检查点。v10/v11 已进入 `actor_update_weights()`，随后失败。

```text
actor_update_weights
  → HCCL checkpoint engine / send_weights
  → FSDP DTensor 参数恢复与 full_tensor()
  → AllGather 等待 remote rank 0
  → EI0002 / Communication_Error_Timeout，1836 秒
  → 507001 / EE9999 / 设备事件查询失败
```

**已定位到的直接故障**是权重同步期间的 HCCL 集合通信超时。rank 0 为什么停住仍缺少当时 Python 栈和完整算子轨迹，不能写成已证硬件故障、显存不足或 FSDP2 不支持 NPU。

**下一次通过标准**：实际完成 3 个训练更新，至少观察到有奖励差异的组及有效梯度；参数版本推进，更新后的 rollout 可运行；保存并读回检查点。仅有加载进度条、资源分配或非零 loss 不足以代替这些证据。

#### 4.2.8 踩坑记录

| 现象 | 核验定位 | 处理与验证点 |
|------|------|------|
| 模型路径被当作 HF repo ID | 模型路径/容器可见性失败 | 检查挂载与完整权重，不先改并行策略 |
| processor 无 chat template，最终样本数 0 | 预处理失败 | 用实际 tokenizer/processor 检查保留样本数 |
| available GPUs 0，desired GPUs 8 | 资源池或设备识别失败 | 核对 Ray 节点资源、可用资源和运行占用 |
| `cannot import name 'LLM' from 'vllm'` | Python 包路径问题 | 对照容器的 `/vllm`、`/vllm-ascend` 与 PYTHONPATH |
| `TRAIN_NPUS: unbound variable` | shell 变量未定义 | 在入口中显式设置训练/推理设备数 |
| EI0002 后出现 507001 | 权重聚合等待 rank 0 超时 | 排查 rank 0 聚合/广播顺序、offload 与通信；尚无已验证修复 |
| 导出超时 60000，worker 仍用 1836 秒 | plog 显示 worker 使用默认执行超时 | 检查 Ray 环境传递；延长等待不能修复阻塞本身 |
| 奖励 manager 找不到字段/参数 | 数据结构与函数签名不匹配 | 同时修复 `reward_model.ground_truth` 与 `solution_str` |

**产物状态**：当前没有可交付的 9B GRPO checkpoint，因此本节不提供虚构的 HF 导出结果。详细日志定位见[单机 9B 核验报告](../single_machine_9b_grpo_report_20260914.md)。

---

### 4.3 多机多卡 GRPO：Qwen3.6-27B

> 本节以 **Step100→120 的 dense-correctness 续训**作为可追溯实例，依次说明环境、数据、启动、检查点及导出。它从已有 Step100 开始，目标累计 Step120，新增更新预算 20 步；不是从基座新训 120 步。本项目跳过基座 SFT 的总体决策见 §3.3。

#### 4.3.1 步骤 1：环境准备（两机角色与挂载）

| 角色 | 机器 | 内网地址 | 设备 / 并行 |
|------|------|------|------|
| Ray head + Trainer | 5 号机 | `192.168.202.5` | 16 chip，Megatron TP4×PP2×CP2 |
| Ray worker + Rollout | 6 号机 | `192.168.202.4` | 16 chip，vLLM TP8×DP2 |

```text
5 号机：Trainer（TP4 / PP2 / CP2，DP1）
     │ HCCL checkpoint engine：同步策略参数
     ▼
6 号机：vLLM（TP8 / DP2）→ 多轮工具执行 → 轨迹与奖励队列
     └──────────────────────────────────→ Trainer 更新
```

`trainer.nnodes=1` 与 `rollout.nnodes=1` 分别指两个角色池各一台机器。不要为了“双机”把两项都设为 2；Trainer 的 TP/PP/CP 在 5 号机内展开。

5 号机使用 §4.2.1 的 GRPO 容器与软件栈。6 号机需要对应的 veRL/vLLM 运行时、项目代码、模型路径及 `/pi_sandbox`；实际 rollout 容器名先用 `docker ps` 确认，不把 5 号机容器名直接搬过去。

**两机分别在宿主机检查**：

```bash
npu-smi info
docker ps --format '{{.Names}}  {{.Image}}'
```

**进入各自 GRPO 容器后检查**：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/llin-verl-grpo
test -d /models/Qwen3.6-27B
test -d /pi_sandbox
test -d reference/Megatron-Bridge-de93536e/src/megatron/bridge
test -f configs/pi_workspace_tools.yaml
test -f configs/pi_agent_loops.yaml
```

5 号机实查项目为读写挂载，基座和沙箱源为只读挂载。Rollout 侧还需可写任务工作区（工具配置为 `/workspace/grpo_run/pi_workspaces`）。同名镜像之外，应同步与目标训练匹配的项目代码及框架补丁。

#### 4.3.2 步骤 2：准备正式任务与数据契约

**本次配方数据目录**：

```text
/workspace/llin-verl-grpo/data/boss_v15_dwh_full276_20260806/dataset/
├── boss_pi_train.parquet              # 236 条
├── boss_pi_val.parquet                # 20 条
├── boss_pi_test.parquet               # 20 条
├── boss_alignment_contract.json       # 数据身份与来源约束
├── boss_pi_alignment_review_queue.jsonl
└── boss_pi_sft_reference.jsonl         # 1500 条，仅作 SFT/回归参考
```

GRPO 输入提供任务 prompt、工具/环境信息和判分 ground truth；**不把源轨迹中的 assistant 回答或 tool 消息作为 prompt 喂入**。源回答另存 SFT 参考文件。数据契约记录任务/指令跨 train、val、test 的交集为零。

**两机容器内校验**：

```bash
cd /workspace/llin-verl-grpo
export PYTHONPATH="/workspace/llin-verl-grpo/runtime:/workspace/llin-verl-grpo:${PYTHONPATH:-}"
python3 scripts/check_boss_alignment_contract.py \
  --data-dir data/boss_v15_dwh_full276_20260806/dataset
```

**预期**：`status: passed`，`selected` 为 train 236、val 20、test 20。检查器核对来源契约、样本隔离和训练/验证文件哈希；若发生漂移，应恢复目标版本，而不是删除检查步骤继续启动。

数据目录与历史文件可用于复现实验。要重新生成新任务，走 `scripts/prepare_boss_aligned_dataset.py` 及人工审核流程，并产生新的契约；不能用新数据冒充此处 236 条历史训练集。

#### 4.3.3 步骤 3：配置多轮工具与奖励

**工具与执行循环**：

| 文件 | 作用 |
|------|------|
| `configs/pi_agent_loops.yaml` | 注册 `pi_agent`，实现为 `llin_verl.pi_agent_loop.PiAgentLoop` |
| `configs/pi_workspace_tools.yaml` | 注册 bash/read/edit/write 四个工作区工具 |
| `llin_verl/pi_workspace_tools.py` | 在任务工作区执行工具并记录事件 |
| `llin_verl/boss_pi_contract.py` | 保持任务 prompt、工具定义与来源契约一致 |
| `llin_verl/pi_reward.py` | 按回答、SQL/工具证据、协议与安全条件计算奖励 |

Rollout 执行“生成工具调用→执行工具→返回观察→继续生成→最终回答”。原始沙箱从 `/pi_sandbox` 读取，每条轨迹使用自己的工作区；新运行要检查工作区可写和任务间隔离。

**本次奖励入口**：`compute_score_dense30`，固定加入 30% 的最终答案稠密正确性分数。按所核对实现，满足安全、工具协议、gold SQL 可验证条件时：

```text
evidence_reward = 0.60 × 答案正确
                + 0.25 × (SQL 证据正确且 bash 执行成功)
                + 0.10 × 使用要求表
                + 0.05 × 有最终回答
base_score = 0.70 × boss_reward + 0.30 × evidence_reward
最终奖励 = 0.70 × base_score + 0.30 × dense_final_answer_correctness
不满足准入条件时：最终奖励为 0
```

`boss_reward` 是代码实现的结果/过程/效率组合分。上述公式对应 blend 模式，复现时还要确认 worker 环境没有遗留 `PI_REWARD_MODE=banded_v1/banded_v2` 覆盖。稠密答案匹配不是完整语义正确性证明，不能把 reward mean 当作任务通过率。重点观察返回的 `acc`、`final_answer_correct`、`sql_evidence_correct`、`gold_sql_verified` 等字段。

第五章蒸馏的 LLM Judge **没有接入这条历史训练循环**。后续 banded/grounded 等奖励实验也不能倒填到本次 `dense30` 配方。

#### 4.3.4 步骤 4：确认完整训练配置（通用入口与续训覆盖）

**调用链**：

```text
launch_pi_dense_correctness_step100_to_step120.sh
  → run_pi_dense_correctness_step100_to_step120.sh
    → run_pi_formal_step100_to_step200_12groups.sh（最终步覆盖为 120）
      → run_pi_grpo_fully_async_tp4_pp2_cp2.sh
        → verl.experimental.fully_async_policy.fully_async_main
```

最内层使用 `fully_async_ppo_megatron_trainer.yaml`，显式设置 `actor.strategy=megatron`、`model.lora_rank=0`，为全参数训练。Bridge、分布式检查点、vLLM DP 权重同步、连续 token、多轮 agent 与异步队列补丁由入口脚本应用。

**Step100→120 配方关键参数**：

| 参数 | 本次值 | 含义 |
|------|------|------|
| Trainer TP / PP / CP | 4 / 2 / 2 | 16 chip，训练 DP1 |
| Rollout TP / DP | 8 / 2 | 16 chip，两套推理副本 |
| `LEARNING_RATE` | `1e-7` | 覆盖通用脚本的 `1e-6` |
| `GROUPS_PER_STEP` | 4 | 每次更新 4 个任务组 |
| `RESPONSES_PER_GROUP` | 4 | 每组 4 条回答，名义每次更新 16 条轨迹 |
| `FASTEST_K` / candidates | 4 / 4 | 该配方没有 6 选 4 超采样 |
| 目标并发组 | 12 | 容量参数；不是每次更新 12 组 |
| 预热组 / 队列组预算 | 8 / 8 | 续训脚本设置 |
| `STALENESS_THRESHOLD` | 2.0 | 按实现处理轨迹陈旧度，非通用默认 0.5 |
| prompt / response / context | 4096 / 45056 / 49152 | token 长度预算 |
| assistant / user turn 上限 | 26 / 25 | 多轮交互预算 |
| rollout memory utilization | 0.80 | 覆盖通用默认 0.60 |
| batched tokens / max seqs | 16384 / 24 | 每个 rollout 配置的容量 |
| `AGENT_WORKERS` | 12 | agent 并发工作进程配置 |
| 权重同步 bucket | 2560 MB | 覆盖通用默认 3072 |
| `PI_DENSE_CORRECTNESS_WEIGHT` | 0.30 | 对应固定 `compute_score_dense30` |
| reward KL / actor KL loss | 均关闭 | 不是 KL 正则训练配方 |
| 保存内容 | `[model,optimizer,extra]` | 支持续训所需的状态保存 |
| 读取内容 | `[model,extra]` | 源 Step100 缺 optimizer，见步骤 6 |
| 累计最终步 | 120 | 起点 100，新增更新预算 20 |

**offload 要分层阅读**：通用脚本设置 Megatron 参数不 offload、梯度 offload、重计算和分布式优化器，并配置 CPU optimizer offload；续训脚本另外覆盖 `actor.megatron.optimizer_offload=False`。优化器配置中的 CPU offload 与 engine 层的 optimizer offload 不是同一个开关，不能只写一句“offload 全开/全关”。

完整配置以目标运行 `driver.log` 的展开值为准；上表为配方关键项，不是替代入口的一条最小命令。

#### 4.3.5 步骤 5：启动双机 Ray 集群

**前提**：两机训练窗口已空闲，运行时和数据已准备。现有同地址集群若已符合配置，先查看并复用，不重复启动或停止他人作业。

**5 号机 GRPO 容器**：

```bash
cd /workspace/llin-verl-grpo
bash scripts/start_ray_m05.sh
```

**6 号机 GRPO 容器**：

```bash
cd /workspace/llin-verl-grpo
bash scripts/start_ray_m06.sh
```

`start_ray_m06.sh` 包装 `start_ray_rollout_node.sh`，设置本机地址 `192.168.202.4`。关键配置如下：

| 项 | 5 号机 | 6 号机 |
|------|------|------|
| Ray 角色 | head | worker |
| Head 地址 | `192.168.202.5:26379` | 加入同一地址 |
| 自定义资源 | `llin_trainer:1` | `llin_rollout:1` |
| `HCCL_IF_IP` | `192.168.202.5` | `192.168.202.4` |
| `HCCL_SOCKET_IFNAME` | `eno0` | `eno0` |
| HCCL broadcast 算法 | `broadcast=level0:NA;level1:NHR` | 相同 |
| Ray worker 端口 | 27000–27999 | 27000–27999 |
| HCCL host / NPU socket 端口 | 60100–60163 / 60200–60263 | 相同 |

这些 IP、网卡和端口是既有机器配置，换机器时需同步修改；不要与其他模型实验的 Ray 端口混用。

**验证**：

```bash
export RAY_ADDRESS=192.168.202.5:26379
ray status --address="$RAY_ADDRESS"
```

确认两个存活节点、Trainer 与 Rollout 角色分别位于对应节点，设备预算各 16 chip。再对照各 worker 实际设备与通信日志；角色标签本身不是设备健康证明。

#### 4.3.6 步骤 6：准备 Step100 恢复视图

**源检查点**：

```text
runs/llin-v15-dwh-bossreward-12groups-100step-20260805-03/
  checkpoints/global_step_100/
```

本次源 Step100 缺少 optimizer 状态，且原数据游标来自 train237，当前数据已修正为 train236。恢复策略为：加载模型与 extra，重置优化器和数据游标，保留累计策略步口径。**这不是完整优化器状态连续恢复。**

**5 号机容器**：

```bash
cd /workspace/llin-verl-grpo
bash scripts/prepare_pi_step100_resume_view.sh trainer
```

**6 号机容器**：

```bash
cd /workspace/llin-verl-grpo
bash scripts/prepare_pi_step100_resume_view.sh rollout
```

默认视图是 `runs/resume-views/llin-v15-step100-train236/global_step_100`。训练侧通过链接引用源 actor；rollout 侧不暴露旧 `data.pt`。脚本会拒绝错误链接或残留旧数据游标，不要绕过检查。

启动后 `resume_contract.txt` 应记录：

```text
source_policy_step=100
final_policy_step=120
new_optimizer_updates=20
checkpoint_load_contents=model,extra
optimizer_state=reset_missing_from_source
dataloader_state=reset_for_corrected_train236
```

#### 4.3.7 步骤 7：启动训练与记录运行身份

**5 号机容器内执行**，给新运行单独命名：

```bash
cd /workspace/llin-verl-grpo
export RUN_NAME="llin-pi-dense-correctness-step100-to-step120-repro-$(date +%Y%m%d-%H%M%S)"
mkdir -p "runs/${RUN_NAME}"
nohup bash scripts/launch_pi_dense_correctness_step100_to_step120.sh \
  > "runs/${RUN_NAME}/launcher.log" 2>&1 &
```

包装脚本保存 `driver.pid`、`started_at`、`driver.log`，结束后检查最终保存步和检查点完整性，再写 `exit_code`、`finished_at`。保存失败会写 `CHECKPOINT_INVALID`，不能只看 nohup 启动成功。

**另一个容器 shell 查看**（设置同一个运行名）：

```bash
RUN_DIR="/workspace/llin-verl-grpo/runs/<本次运行名>"
tail -f "$RUN_DIR/driver.log"
```

**训练闭环观察顺序**：模型与 Bridge 加载→6 号机 Rollouter 启动→工具轨迹生成→同题组奖励与优势→Trainer 更新→参数版本同步→检查点保存。fully-async 调度还会受到队列等待、工具耗时和陈旧样本处理影响，不能从“双机”直接推出固定加速比。

#### 4.3.8 步骤 8：验证历史训练结果

**已核验的运行目录**：

```text
runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/
```

**检查命令**（只读已有结果）：

```bash
RUN_DIR=/workspace/llin-verl-grpo/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01
cat "$RUN_DIR/exit_code"
cat "$RUN_DIR/finished_at"
cat "$RUN_DIR/resume_contract.txt"
cat "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt"
cat "$RUN_DIR/checkpoint_integrity.json"
grep 'training/global_step:120' "$RUN_DIR/driver.log"
grep '_fit_update_weights' "$RUN_DIR/driver.log" | tail -3
```

**历史结果**：

| 指标 | 实际记录 |
|------|------|
| 最终训练步 | `training/global_step:120.0` |
| 最终参数同步 | `current_param_version:120`，该条同步耗时 7.7294 秒 |
| 最终步学习率 | `1e-7` |
| 最终步 actor loss | 约 `-0.01461108` |
| 最终步 grad norm | 约 `1.08248` |
| 最终步 reward mean | 约 `0.197386` |
| 最终步 advantage min / max | 约 `-1.49994` / `1.48262` |
| 最终步记录的 max allocated memory | 约 `33.26 GiB` |
| 包装进程退出码 | 0 |
| 完成时间 | 2026-08-10 16:16:14（北京时间） |
| 保存步指针 | 120 |
| 模型检查点完整性记录 | `valid=true`，32 个分片 |
| 优化器完整性记录 | `valid=true`，32 个分片 |

以上是指定历史运行的日志与既有完整性记录，本次没有重新读取数百 GB 张量。最终步数据是单个批次指标，不是全程平均吞吐或独立评测成绩。日志中的 `critic/score`、`critic/advantages` 是指标命名，不表示本 GRPO 训练了独立 critic。

**能力验收另行执行**：把 Step120 部署到与基座一致的工具环境，固定 prompt、采样、超时和评分口径，比较未见任务与通用 benchmark。不能由 reward mean、checkpoint 存在直接推出“Step120 能力最佳”。

#### 4.3.9 步骤 9：Checkpoint 转 HF 与导出验证

Megatron 分布式 actor checkpoint 不能直接作为普通 HF 模型交给 vLLM。本项目采用单 CPU/Gloo rank 恢复到 TP1/PP1/CP1，再由模型对应 Bridge 导出，以避免 PP2 在线导出只写出部分流水线层。

**实际脚本**：`scripts/export_megatron_dist_to_hf.py`。在已准备好 Bridge 的 5 号机 GRPO 容器中运行，输出目录必须不存在：

```bash
cd /workspace/llin-verl-grpo
export PYTHONPATH="/workspace/llin-verl-grpo/reference/Megatron-Bridge-de93536e/src:/workspace/llin-verl-grpo/runtime:/workspace/llin-verl-grpo:${PYTHONPATH:-}"
ACTOR_DIR=/workspace/llin-verl-grpo/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120/actor
HF_DIR="/workspace/llin-verl-grpo/exports/step120-hf-repro-$(date +%Y%m%d-%H%M%S)"
python3 scripts/export_megatron_dist_to_hf.py \
  --actor-checkpoint "$ACTOR_DIR" \
  --base-model /models/Qwen3.6-27B \
  --output-dir "$HF_DIR"
```

导出需读取大模型分片并重建完整模型，要留出 CPU 内存和磁盘空间。它不覆盖原始可恢复 checkpoint。新运行应将 `ACTOR_DIR` 改为新运行自己的产物。

**已有交付路径**：

```text
/workspace/llin-verl-grpo/exports/llin-qwen3.6-27b-grpo-step120-hf-20260813/
├── model-00001-of-00015.safetensors
├── ...
├── model-00015-of-00015.safetensors
├── model.safetensors.index.json
├── config.json / generation_config.json
├── tokenizer.json / tokenizer_config.json / chat_template.jinja
└── llin_export_manifest.json
```

**重新检查已有导出**：

```bash
python3 scripts/export_megatron_dist_to_hf.py \
  --actor-checkpoint "$ACTOR_DIR" \
  --base-model /models/Qwen3.6-27B \
  --output-dir /workspace/llin-verl-grpo/exports/llin-qwen3.6-27b-grpo-step120-hf-20260813 \
  --verify-only
```

8 月 13 日导出清单记录：1,199 个张量、15 个 safetensors 分片、语言层 0–63，缺失/额外张量与形状不匹配均为 0。另有 414 个张量 dtype 与基座不同，清单已记录；冻结的 MTP 参数从基座补入。不能把该产物描述为“每个张量都经过 GRPO 更新且全为 BF16”。导出后按第六章部署，以独立模型名标识 Step120，再按第七章评测。

#### 4.3.10 踩坑记录与复现边界

| 现象或误区 | 工程处理 |
|------|------|
| 直接调用底层 main，遗漏容器补丁 | 通过完整项目入口，核对 Bridge、权重同步、多轮与异步队列补丁 |
| 双机都写 `nnodes=2` | Trainer 和 Rollout 各自为 1，合计两台机器 |
| 看到脚本名 `12groups` 就写每步 12 组 | 12 是目标并发组；该配方每次更新为 4 组×4 条 |
| 用通用默认参数描述历史续训 | 逐层读 wrapper overrides，再核对展开日志；学习率、陈旧度和 bucket 都有覆盖 |
| Step100 恢复后声称优化器连续 | 原检查点缺 optimizer，本配方明确重置；Step120 保存时补齐 optimizer |
| 训练集修正后沿用旧 data.pt | 通过恢复视图重置数据游标，防止 train237 游标用于 train236 |
| PP2 在线导出缺后半层 | 使用 CPU 单 rank 完整重建与 HF 张量清单核验 |
| 只保存 `[model,extra]` 却要求完整续训 | 正式配方显式保存 `[model,optimizer,extra]` 并检查分片 |
| 双机成功就认定单机 9B 可直接换后端 | 模型、FSDP/Bridge 路径及调度入口不同，9B 仍需独立验证 |

---

### 4.4 验收与路径速查

#### 4.4.1 两条路线的交付状态

| 验收项 | 9B 单机 | 27B 双机历史实例 |
|------|------|------|
| 环境、脚本与数据定位 | 已有 | 已有 |
| 数据/奖励接口 | 已发现待修复项 | 历史链路已运行 |
| 权重同步 | AllGather 超时，待定位 rank 0 阻塞 | 参数版本推进到 120 |
| 实际训练完成记录 | 未获得 | 有 Step120 指标、梯度及退出记录 |
| 检查点 | 未获得 | Step120 模型与 optimizer 完整性记录通过 |
| HF 产物 | 未获得 | 1,199 张量 / 15 分片，已有导出清单 |
| 能力提升 | 尚不能评价 | 需按独立评测结果判断，不由工程完成推出 |

#### 4.4.2 文件路径速查

下表相对路径均以 `/workspace/llin-verl-grpo` 为容器根目录，宿主机对应 `/data3/llin/qwen3.6-27b-verl-grpo`。

| 路径 | 用途 |
|------|------|
| `scripts/prepare_math_dataset.py` | 9B 数学数据制备，现有接口问题见 §4.2.2 |
| `scripts/run_math_grpo_smoke.sh` | 9B 单机冒烟入口 |
| `llin_verl/math_reward.py` | 数学评分，待接口与边界用例修复 |
| `runs/math_grpo_smoke_v11.log` | 9B 权重同步失败日志 |
| `scripts/start_ray_m05.sh` | 双机 Trainer 的 Ray head 启动 |
| `scripts/start_ray_m06.sh` | 6 号机 rollout 启动包装 |
| `scripts/start_ray_rollout_node.sh` | Rollout 节点通用入口 |
| `scripts/check_boss_alignment_contract.py` | 正式训练数据契约检查 |
| `scripts/prepare_pi_step100_resume_view.sh` | 构建 Step100 恢复视图 |
| `scripts/launch_pi_dense_correctness_step100_to_step120.sh` | 续训包装与结束验收 |
| `scripts/run_pi_dense_correctness_step100_to_step120.sh` | Step120 与 dense30 覆盖 |
| `scripts/run_pi_formal_step100_to_step200_12groups.sh` | 续训配方与恢复配置 |
| `scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh` | Megatron fully-async 完整工程入口 |
| `llin_verl/pi_reward.py` | 多轮物流轨迹奖励 |
| `scripts/verify_checkpoint_integrity.py` | 分布式检查点完整性核验 |
| `scripts/export_megatron_dist_to_hf.py` | Megatron → HF 导出及验证 |

#### 4.4.3 本次核验来源

2026-09-14 再次只读访问 5 号机，确认两条入口、Step120 续训包装、数据契约、恢复记录、最终步日志、检查点完整性记录、HF 清单和容器挂载。以下文件 SHA256 与既有核验报告一致：

| 文件 | SHA256 |
|------|------|
| `scripts/run_math_grpo_smoke.sh` | `dd6c480b7d2fa08e258f8fc96ba70d1a9d9eb71a56e8c247f9d220660327ee66` |
| `scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh` | `65bfbe1c90d963844ab02e8847aa92f1b027706c7856c05f5c48cf52aa3b28af` |
| `runs/math_grpo_smoke_v11.log` | `c63725e1681b0df8a443c385168e7fc42642b9dffac45c5f604ba93176097458` |
| Step100→120 `driver.log` | `d29189e71341741f32de88cc564e1b33892089d128fc3ef533f1ac6ed8d6c5b1` |

更详细的历史故障与证据边界见[单机 9B 报告](../single_machine_9b_grpo_report_20260914.md)和[多机 27B 报告](../multi_machine_27b_grpo_report_20260914.md)。本章提供操作路径，不把尚未通过的单机尝试写成已完成结果。

---


## 第 5 章 评估模型训练（LLM Judge 蒸馏）

> **本章目标**：用 Qwen3.5-9B 基座 + LoRA，蒸馏商业模型（qwen3.7-plus）的判分能力，训一个私有化的评估模型，drop-in 替换商业模型做 LLM judge。
>
> **本章状态**：✅ 已完成

### 5.1 本章目标

本章回答：**怎么训一个私有化的 LLM judge，替代商业模型做轨迹评估？**

**背景**：项目的轨迹评估体系（技术报告 §9.2）用商业模型 qwen3.7-plus 做语义判分（5 个维度：result_correct_report / answer_relevance / answer_grounding / report_quality / process_reasonableness）。但商业模型有三个问题：

1. **成本高**：每次评测 14757 条轨迹 × 5 维，API 费用不小
2. **速度慢**：qwen3.7-plus 单条轨迹打分 ~40s，大批量评测耗时长
3. **不可私有化**：商业模型无法本地部署，数据出域有合规风险

**方案**：用 Qwen3.5-9B 基座 + LoRA，蒸馏 teacher（qwen3.7-plus）的判分输出。训练数据是 teacher 已经打过的 12.5 万条轨迹，本质是"LLM 当标注员"的合成数据路线。

**验收指标**：对齐率（alignment rate）= student 与 teacher 在 val 上 5 维 0/1 一致比例，分维度算。不是绝对质量，是"便宜模型和贵模型判得像不像"。

**交付物**：LoRA adapter（82.6MB），merge 成完整权重后部署 vLLM 服务，drop-in 替换 qwen3.7-plus。

### 5.2 前置条件

开始本章前，确认以下环境已就绪：

| 条件 | 检查命令 | 期望输出 |
|------|---------|---------|
| 能 SSH 到 huawei-0（训练机） | `ssh huawei-0 'hostname'` | `Server910C-002` |
| 能 SSH 到 huawei-llm（开发机） | `ssh huawei-llm 'hostname'` | `huawei-llm` |
| huawei-0 上 NPU 可用 | `ssh huawei-0 'npu-smi info'` | 8 卡 16 chip 全空 |
| Qwen3.5-9B 模型权重已下载 | `ssh huawei-0 'ls /data/liuxr/models/Qwen3.5-9B/'` | 4 个 safetensors（共 19G） |
| ms-swift 容器已启动 | `ssh huawei-0 'docker ps \| grep swift_rm'` | 容器 `swift_rm_liuxr` 运行中 |

如果以上条件不满足，先完成第 0 章环境准备。

### 5.3 数据准备

训练数据来自 teacher（qwen3.7-plus）对物流 agent 轨迹的判分输出，已经以 `*_judge_train.jsonl` 形式存在 `data/judge_train/` 下。

#### 5.3.1 原始数据分布

```bash
# 在 huawei-llm 上查看
ssh huawei-llm 'ls /data/liuxr/RM/data/judge_train/sft/'
# 输出：多个版本目录，共 73 个 *_judge_train.jsonl 文件，385M
```

数据按 `prompt_version` 分三类：

| prompt_version | 条数 | 占比 | 输入（待判内容） | 输出（teacher 判分） |
|---|---|---|---|---|
| `process_reasonableness_v1` | ~108k | 78% | agent 执行轨迹 | 0/1 + 理由 |
| `simple_batch_v1` | ~15k | 11% | 批次数据 | JSON（多维 0/1） |
| `report_quality_v1` | ~15k | 11% | 报告 | 0/1 + 理由 |

三类不均衡约 7:1。第一版不做平衡（15k 条 simple/report 已够学格式），先跑出对齐率 baseline，若某类对齐率明显低再加权。

#### 5.3.2 数据格式

每条样本是 messages 格式，自包含（prompt 已拼好，训练时不需要判分代码/门控/沙箱）：

```json
{
  "messages": [
    {"role": "system", "content": "<判分角色设定>"},
    {"role": "user", "content": "<判分 prompt + 待判内容>"},
    {"role": "assistant", "content": "<teacher(qwen3.7-plus)的判分输出>"}
  ],
  "metadata": {
    "prompt_version": "process_reasonableness_v1",
    "teacher_model": "qwen3.7-plus",
    "task_id": "xxx"
  }
}
```

#### 5.3.3 数据制备脚本

数据制备分两步：合并+去重+清洗 → 分层留验证集。

**脚本 1：合并+去重+清洗**（`scripts/prep_stage1.py`）

```python
#!/usr/bin/env python3
"""阶段1:训练数据准备 —— 合并 + 精确去重 + 清洗。

输入: data/judge_train/ 下 73 个 *_judge_train.jsonl
输出: data/prepared/train_merged.jsonl  (合并+去重+清洗后的总集)

清洗规则(复用 mentor 的 parse 函数,保证与部署时解析口径一致):
  - simple_batch_v1            : parse_json_dims 必须解析出 3 维全非 None
  - report_quality_v1          : parse_score 必须解析出 0/1
  - process_reasonableness_v1   : parse_score 必须解析出 0/1
  - ERR: 开头的 assistant 输出剔除(parse 函数本身返回 None)
去重规则: 整条记录内容做 canonical JSON hash 去绝不用 task_id)。
"""
import json
import hashlib
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
from llm_judge import parse_score, parse_json_dims, _SIMPLE_DIMS  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "data", "judge_train")
OUT_DIR = os.path.join(ROOT, "data", "prepared")
OUT_PATH = os.path.join(OUT_DIR, "train_merged.jsonl")
DROP_LOG = os.path.join(OUT_DIR, "dropped_samples.jsonl")


def find_files():
    files = []
    for dirpath, _, filenames in os.walk(SRC_DIR):
        for fn in filenames:
            if fn.endswith("_judge_train.jsonl"):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def rec_hash(obj):
    """整条记录 canonical 序列化后取 sha1,内容相同即同一样本。"""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def is_cleanable(rec):
    """返回 (ok: bool, reason: str)。ok=False 表示该样本应剔除。"""
    meta = rec.get("metadata", {})
    pv = meta.get("prompt_version")
    msgs = rec.get("messages", [])
    asst = ""
    for m in msgs:
        if m.get("role") == "assistant":
            asst = m.get("content", "")
            break
    if pv == "simple_batch_v1":
        dims = parse_json_dims(asst, _SIMPLE_DIMS)
        if any(v is None for v in dims.values()):
            return False, f"simple_batch_parse_fail: {dims}"
        return True, ""
    elif pv in ("report_quality_v1", "process_reasonableness_v1"):
        s = parse_score(asst)
        if s is None:
            return False, f"score_parse_fail: pv={pv}"
        return True, ""
    else:
        return False, f"unknown_prompt_version: {pv}"


def main():
    files = find_files()
    print(f"[1] 发现 {len(files)} 个 jsonl 文件")
    os.makedirs(OUT_DIR, exist_ok=True)

    n_lines = 0
    n_bad_json = 0
    seen = set()
    n_dup = 0
    clean_recs = []
    drop_reasons = Counter()
    per_pv_total = Counter()
    per_pv_clean = Counter()

    with open(DROP_LOG, "w", encoding="utf-8") as dlog:
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for ln in f:
                    n_lines += 1
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rec = json.loads(ln)
                    except Exception:
                        n_bad_json += 1
                        dlog.write(json.dumps({"file": fp, "reason": "bad_json", "line": ln[:200]}, ensure_ascii=False) + "\n")
                        continue
                    pv = rec.get("metadata", {}).get("prompt_version", "?")
                    per_pv_total[pv] += 1
                    # 去重
                    h = rec_hash(rec)
                    if h in seen:
                        n_dup += 1
                        continue
                    seen.add(h)
                    # 清洗
                    ok, reason = is_cleanable(rec)
                    if not ok:
                        drop_reasons[reason] += 1
                        dlog.write(json.dumps({"file": fp, "reason": reason, "prompt_version": pv, "task_id": rec.get("metadata", {}).get("task_id")}, ensure_ascii=False) + "\n")
                        continue
                    per_pv_clean[pv] += 1
                    clean_recs.append(rec)

    # 写出
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for rec in clean_recs:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 统计
    print("\n========== 阶段1 统计 ==========")
    print(f"原始行数(73 文件累计): {n_lines}")
    print(f"  其中 JSON 解析失败:   {n_bad_json}")
    print(f"去重: 重复记录 {n_dup} 条 (按整条内容 sha1)")
    print(f"清洗: 剔除 {sum(drop_reasons.values())} 条 (teacher 输出 parse 失败)")
    for r, c in drop_reasons.most_common():
        print(f"    - {r}: {c}")
    print(f"最终保留(写入 {OUT_PATH}): {len(clean_recs)} 条")
    print("\n各 prompt_version (原始 / 清洗后):")
    for pv in sorted(per_pv_total.keys()):
        print(f"  {pv:35s} {per_pv_total[pv]:>7d}  ->  {per_pv_clean[pv]:>7d}")
    print(f"\n丢弃明细写入: {DROP_LOG}")
    print(f"输出文件: {OUT_PATH}")


if __name__ == "__main__":
    main()
```

**脚本 2：分层留验证集**（`scripts/prep_split_val.py`）

```python
#!/usr/bin/env python3
"""阶段1-步5: 分层留验证集。

按 prompt_version 各 90/10 切(分层随机抽样),在唯一原始样本上切,
val 不参与训练,用于测 student vs teacher 的分维度对齐率。

输入: data/prepared/train_merged.jsonl  (138915 条,已去重+清洗)
输出:
  data/prepared/train_raw.jsonl   (90%, 各类 ~90%)
  data/prepared/val.jsonl         (10%, 各类 ~10%, 不再动)

seed=42 固定,可复现。
"""
import json
import os
import random
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(ROOT, "data", "prepared", "train_merged.jsonl")
TRAIN_OUT = os.path.join(ROOT, "data", "prepared", "train_raw.jsonl")
VAL_OUT = os.path.join(ROOT, "data", "prepared", "val.jsonl")
SEED = 42
VAL_RATIO = 0.10


def main():
    by_pv = defaultdict(list)
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            pv = r["metadata"]["prompt_version"]
            by_pv[pv].append(r)

    rng = random.Random(SEED)
    train_recs, val_recs = [], []
    cnt_total, cnt_train, cnt_val = Counter(), Counter(), Counter()

    for pv in sorted(by_pv.keys()):
        recs = by_pv[pv]
        rng.shuffle(recs)
        n_val = int(round(len(recs) * VAL_RATIO))
        val = recs[:n_val]
        train = recs[n_val:]
        cnt_total[pv] = len(recs)
        cnt_train[pv] = len(train)
        cnt_val[pv] = len(val)
        train_recs.extend(train)
        val_recs.extend(val)

    # 整体打散(切完再 shuffle,避免同类聚集成块)
    rng.shuffle(train_recs)
    rng.shuffle(val_recs)

    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(VAL_OUT, "w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("========== 分层切验证集结果 ==========")
    print(f"{'prompt_version':35s} {'total':>7s} {'train':>7s} {'val':>6s}")
    for pv in sorted(cnt_total.keys()):
        print(f"{pv:35s} {cnt_total[pv]:>7d} {cnt_train[pv]:>7d} {cnt_val[pv]:>6d}")
    print(f"{'合计':35s} {sum(cnt_total.values()):>7d} {sum(cnt_train.values()):>7d} {sum(cnt_val.values()):>6d}")
    print(f"\nseed={SEED}, val_ratio={VAL_RATIO}")
    print(f"train -> {TRAIN_OUT}")
    print(f"val   -> {VAL_OUT}")


if __name__ == "__main__":
    main()
```

#### 5.3.4 执行数据制备

```bash
# 在 huawei-llm 上执行
ssh huawei-llm
cd /data/liuxr/RM

# 步骤 1：合并+去重+清洗
python3 scripts/prep_stage1.py
# 期望输出：
# [1] 发现 73 个 jsonl 文件
# ========== 阶段1 统计 ==========
# 原始行数(73 文件累计): 139012
#   其中 JSON 解析失败:   0
# 去重: 重复记录 97 条 (按整条内容 sha1)
# 清洗: 剔除 0 条 (teacher 输出 parse 失败)
# 最终保留(写入 data/prepared/train_merged.jsonl): 138915 条

# 步骤 2：分层留验证集
python3 scripts/prep_split_val.py
# 期望输出：
# ========== 分层切验证集结果 ==========
# prompt_version                            total   train    val
# process_reasonableness_v1                108518   97666  10852
# report_quality_v1                         15240   13716   1524
# simple_batch_v1                           15157   13641   1516
# 合计                                     138915  125023  13892

# 验证数据
wc -l data/prepared/train_raw.jsonl  # 125023
wc -l data/prepared/val.jsonl        # 13892
ls -lh data/prepared/                # train_raw.jsonl (346M), val.jsonl (39M)
```

**检查点**：
- ✅ `train_raw.jsonl` 有 125023 条
- ✅ `val.jsonl` 有 13892 条
- ✅ 两类文件无交集（val 不参与训练）

### 5.4 训练

#### 5.4.1 训练框架与镜像

| 项 | 值 |
|---|---|
| 训练框架 | **ms-swift 4.3.0**（基于 transformers Trainer） |
| 镜像 | `quay.io/ascend/ms-swift:v4.3.0-A3-py311-CANN9.0.0-ubuntu22.04` |
| NPU 运行时 | torch_npu 2.9.0.post2 |
| 微调方式 | **LoRA SFT**（`--tuner_type lora`） |
| 底座模型 | Qwen3.5-9B（`/data/liuxr/models/Qwen3.5-9B`，19G，bf16） |

**参数名实测（4.3.0）**：`--tuner_type`（非旧版 `train_type`）、`--target_modules`（非 `lora_target_modules`）。`--learning_rate` 默认 None→LoRA 自动 1e-4（显式写防歧义）。

#### 5.4.2 硬件环境

| 项 | 值 |
|---|---|
| 机器 | 0号机 Server910C-002（内网 10.10.2.2） |
| NPU | NPU 0-3 = **8 芯片**（ASCEND_RT_VISIBLE_DEVICES=0-7） |
| 容器 | `swift_rm_liuxr`（带卡，`--runtime=ascend`） |
| 并行策略 | 数据并行（DDP），8 芯片各放完整 9B 副本，只同步 LoRA 梯度 |

#### 5.4.3 训练脚本

完整脚本：`scripts/train_lora_sft.sh`

```bash
#!/usr/bin/env bash
# ============================================================
# RM 判分器蒸馏:Qwen3.5-9B LoRA SFT 训练配置(ms-swift 4.3.0)
# 路线A:1个混合9B,3类prompt混训,drop-in 替换 qwen3.7-plus
# 执行环境:0号机训练容器 swift_rm_liuxr(填好芯片号后起的带卡容器)
# ============================================================
set -e

# 8芯片数据并行(NPU 0-3,容器内逻辑号0-7)
export NPROC_PER_NODE=8

swift sft \
  --model /data/liuxr/models/Qwen3.5-9B \
  --tuner_type lora \
  --target_modules all-linear \
  --lora_rank 8 \
  --lora_alpha 32 \
  --dataset /data/liuxr/RM/data/prepared/train_raw.jsonl \
  --val_dataset /data/liuxr/RM/data/prepared/val.jsonl \
  --enable_thinking false \
  --torch_dtype bfloat16 \
  --max_length 2048 \
  --learning_rate 1e-4 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing true \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --save_strategy epoch \
  --save_total_limit 3 \
  --eval_strategy epoch \
  --logging_steps 10 \
  --report_to tensorboard \
  --output_dir /data/liuxr/RM/output
```

#### 5.4.4 训练超参

| 超参 | 值 | 说明 |
|---|---|---|
| `--model` | `/data/liuxr/models/Qwen3.5-9B` | 底座 |
| `--tuner_type` | `lora` | LoRA 微调 |
| `--target_modules` | `all-linear` | 所有 Linear 层加 LoRA |
| `--lora_rank` | `8` | LoRA 秩 |
| `--lora_alpha` | `32` | 缩放因子（alpha/rank=4） |
| `--dataset` | `train_raw.jsonl`（125023 条） | 训练集 |
| `--val_dataset` | `val.jsonl`（13892 条） | 验证集 |
| `--enable_thinking` | `false` | 关思维链 |
| `--torch_dtype` | `bfloat16` | 精度 |
| `--max_length` | `2048` | 实测 train_raw 各类 max token: process 1227 / simple_batch 1771 / report 1587，**0% 截断** |
| `--learning_rate` | `1e-4` | LoRA 惯例 |
| `--num_train_epochs` | `2` | 数据量大（12.5万）用 2 epoch |
| `--per_device_train_batch_size` | `4` | 单卡每步样本 |
| `--gradient_accumulation_steps` | `4` | 梯度累积 |
| **有效 batch** | **4 × 4 × 8 = 128** | 每次更新用 128 样本算梯度 |
| `--gradient_checkpointing` | `true` | 省显存（换计算） |
| `--warmup_ratio` | `0.03` | 前 3% 步线性升温 |
| `--lr_scheduler_type` | `cosine` | 余弦衰减 |
| `--save_strategy` | `epoch` | 每 epoch 存 checkpoint |
| `--save_total_limit` | `3` | 最多保留 3 个 |
| `--eval_strategy` | `epoch` | 每 epoch eval |
| `--logging_steps` | `10` | 每 10 步打日志 |
| `--report_to` | `tensorboard` | 可视化 |
| `NPROC_PER_NODE` | `8` | 8 芯片数据并行 |

#### 5.4.5 可训练参数

LoRA 只训适配器，冻结 9B 全部原始参数：
- `target_modules = all-linear`（展开到注意力的 q/k/v/o_proj + FFN 的 up/down/gate_proj 等）
- **可训练参数 ≈ 21.6M / 9431M ≈ 0.23%**（99.77% 冻结，LoRA 天然强正则）

#### 5.4.6 执行训练

```bash
# 在 huawei-0 上执行
ssh huawei-0

# 进入训练容器
docker exec -it swift_rm_liuxr bash

# 初始化 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 启动训练（后台跑）
cd /data/liuxr/RM
nohup bash scripts/train_lora_sft.sh > train.log 2>&1 &

# 查看训练进度
tail -f train.log
```

**训练耗时**：约 15.9 小时（3908 步）

#### 5.4.7 训练结果

**收敛指标**：

| 指标 | 值 |
|---|---|
| 总步数 | 3908 / 3908（全跑完） |
| train_runtime | 57260s ≈ 15.9h |
| train_loss（平均） | 0.4485 |
| train token_acc | ≈ 0.86 |
| eval_loss（epoch1） | 0.446 |
| **eval_loss（epoch2）** | **0.4248** |
| eval_token_acc | 0.8459 |
| 显存占用 | ~59.83 GiB / 芯片 |

**健康性判读**：
- ✅ eval_loss 单调下降 0.446 → 0.4248，无剪刀差（train 降、eval 也降），**无过拟合**
- ✅ 全程无报错，健康收敛
- ✅ best = last = epoch2（最后一步 eval 最低）

#### 5.4.8 训练产物

```
/data/liuxr/RM/output/v0-20260804-090154/
├── checkpoint-1954/   (epoch1 备份)
│   └── adapter_model.safetensors 等
└── checkpoint-3908/   (epoch2 = 交付版，best=last)
    ├── adapter_model.safetensors   (82.6 MB  ← LoRA adapter，交付用)
    ├── adapter_config.json          (LoRA 配置：r=8/alpha=32/target_modules)
    ├── optimizer.pt                (166 MB，训练优化器状态，resume 续训用，推理不需要)
    ├── trainer_state.json          (训练状态：step/loss 历史)
    ├── scheduler.pt / rng_state_*.pth 等
    └── args.json                   (完整超参)
```

**交付物**：`checkpoint-3908/adapter_model.safetensors`（82.6MB）+ `adapter_config.json`

**检查点**：
- ✅ `adapter_model.safetensors` 文件存在，大小 82.6MB
- ✅ `adapter_config.json` 存在，`r=8, alpha=32`
- ✅ `trainer_state.json` 里有完整训练历史

### 5.5 部署（LoRA merge + vLLM 服务）

vLLM 不支持直接加载 LoRA（昇腾兼容性未知），需要先 merge 成完整权重。

#### 5.5.1 LoRA merge

```bash
# 在 huawei-0 的 ms-swift 容器内执行
docker exec -it swift_rm_liuxr bash

# merge epoch1
swift export \
  --model /data/liuxr/models/Qwen3.5-9B \
  --adapters /data/liuxr/RM/output/v0-20260804-090154/checkpoint-1954 \
  --merge_lora true \
  --output_dir /data/liuxr/RM/output/merged-1epoch-0807

# merge epoch2
swift export \
  --model /data/liuxr/models/Qwen3.5-9B \
  --adapters /data/liuxr/RM/output/v0-20260804-090154/checkpoint-3908 \
  --merge_lora true \
  --output_dir /data/liuxr/RM/output/merged-2epoch-0807

# 验证产物
ls -lh /data/liuxr/RM/output/merged-1epoch-0807/  # 18G, 4 个 safetensors
ls -lh /data/liuxr/RM/output/merged-2epoch-0807/  # 18G, 4 个 safetensors
```

#### 5.5.2 vLLM 部署（双服务并行）

两个 checkpoint 同时部署，方便对比评估：

**NPU 分配**：8 卡 16 chip 分两组（每组 4 卡 8 chip）：
- 1epoch：NPU 0-3（chip 0-7），端口 8012
- 2epoch：NPU 4-7（chip 8-15），端口 8013

**关键**：`ASCEND_VISIBLE_DEVICES` 在 vLLM-Ascend v0.23.0rc1 是 **chip 序号（0-15）**，不是 NPU 卡号（0-7）。

**compose 配置**（以 1epoch 为例）：

```yaml
# /data3/models/liuxr_lora_1epoch_compose/docker-compose.yaml
services:
  vllm:
    image: qwen3.5-9b:vllm-ascend-v0.23.0rc1-a3
    ports:
      - "8012:8000"
    environment:
      - ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      - VLLM_API_KEY=${VLLM_API_KEY}
    volumes:
      - ./merged-1epoch-0807:/model
    command: >
      python3 -m vllm.entrypoints.openai.api_server
      --model /model
      --served-model-name liuxr-lora-0807-1epoch
      --tensor-parallel-size 1
      --data-parallel-size 8
      --enable-prefix-caching
      --default-chat-template-kwargs '{"enable_thinking": false}'
      --max-model-len 262144
```

**启动服务**：

```bash
# 启动 1epoch 服务
cd /data3/models/liuxr_lora_1epoch_compose
docker compose --env-file .env up -d

# 启动 2epoch 服务
cd /data3/models/liuxr_lora_2epoch_compose
docker compose --env-file .env up -d

# 验证服务
curl -s http://110.188.22.101:8012/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
# 期望返回：{"id":"liuxr-lora-0807-1epoch",...}

curl -s http://110.188.22.101:8013/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
# 期望返回：{"id":"liuxr-lora-0807-2epoch",...}
```

#### 5.5.3 thinking 关闭（重要）

judge 任务是结构化输出（0/1 或 JSON），不需要 CoT。**必须关 thinking**，原因：

1. Qwen3 系列 overthinking 通病：开 thinking 跑满 max_tokens 不收敛，content 空
2. judge 关 thinking 后 content 直接返回，快且稳（9B 实测 3.5s/次 vs 开 thinking 8.7s/次）

**配置**：`--default-chat-template-kwargs '{"enable_thinking": false}'`（服务级关 thinking）

### 5.6 评测（对齐率验证）

#### 5.6.1 切换 judge 配置

修改 `configs/eval/llm_judge_config.yaml`，把 `active` 指向本地 9B：

```yaml
active: qwen3.5-9b-local

models:
  qwen3.5-9b-local:
    base_url: http://110.188.22.101:8012  # 或 8013（2epoch）
    model: liuxr-lora-0807-1epoch         # 或 2epoch
    description: huawei-0 本地 9B 评估模型（LoRA 蒸馏）
    qpm: 10000
    max_workers: 50
```

#### 5.6.2 跑评估

```bash
# 在 huawei-llm 上执行
cd /data/renjunxiang/coding/huawei_train

# 跑 sft/dev/test 三组轨迹评估
python3 scripts/data/run_all_reward.py --group sft,dev,test

# 产出：eval_results/rewards/{group}/{v}/{policy_class}/{policy_name}/{judge}/xxx_llm.jsonl
```

#### 5.6.3 对齐率结果

**三 group 完整对齐率**：

| | Qwen3.5-9B 基座 | liuxr-lora-1epoch | liuxr-lora-2epoch |
|---|---|---|---|
| **sft（训练集）** | 62.4% | **84.6%** | **85.3%** |
| **dev（验证集）** | 56.9% | 73.3% | 73.8% |
| **test** | 61.6% | 76.7% | 77.0% |

**分维度对齐率（sft 组）**：

| 维度 | 9B 基座 | 1epoch | 2epoch | 提升 |
|------|---------|--------|--------|------|
| result_correct_report | 70.3% | 87.7% | 87.5% | +17pp |
| answer_relevance | 80.9% | 93.0% | 92.9% | +12pp |
| answer_grounding | 70.5% | 78.6% | 79.9% | +9pp |
| report_quality | 67.3% | 90.2% | 90.7% | +23pp |
| **process_reasonableness** | 53.9% | **82.2%** | **83.2%** | **+29pp（最大）** |

**关键发现**：

1. ✅ **训练有效**：三 group 都提升明显（sft +23pp, dev +17pp, test +15pp）
2. ✅ **2epoch 略好于 1epoch**（sft 85.3% vs 84.6%），但提升很小（接近收敛）
3. ✅ **process_reasonableness 提升最大**（+29pp）：9B 最弱维度，训练后大幅改善
4. ⚠️ **dev 的 answer_grounding 2epoch 降到 48.9%**（过拟合信号，需关注）
5. ✅ **9B 偏严 74.3%**：关 thinking 后判断保守，是 shortcut 检测的好事

### 5.7 验收发现（实习生工作审查）

在完成本章工程报告时，对照实习生原始报告和实际代码，发现以下差异：

#### 5.7.1 梯度累积步数不一致

**实习生报告**（`update_report_20260811_RM训练完成报告.md`）第 5 节：
> `--gradient_accumulation_steps 2`
> 有效 batch = 4 × 2 × 8 = 64

**实际脚本**（`scripts/train_lora_sft.sh`）：
> `--gradient_accumulation_steps 4`
> 有效 batch = 4 × 4 × 8 = **128**

**影响**：实际训练用的是 batch 128，不是报告里写的 64。这不影响训练结果（脚本是正确的），但报告描述与实际不符。

**建议**：报告应修正为 `gradient_accumulation_steps 4`，有效 batch 128。

#### 5.7.2 其他实验脚本

实习生做了多组超参消融实验，脚本都在 `scripts/` 下：

| 实验 | 脚本 | 变量 |
|------|------|------|
| 学习率 | `lr_test.sh` | 5e-5, 1e-4, 2e-4, 5e-4 |
| LoRA rank | `rank_test.sh` | 4, 8, 16, 32（alpha/rank=4） |
| warmup | `warmup_test.sh` | 0, 0.03, 0.1 |
| epoch | `epoch_test.sh` | 1, 2, 3 |
| batch size | `batch128.sh` | 64 vs 128 |
| 去理由 | `train_lora_sft_noreason.sh` | 有理由 vs 无理由 |

这些实验的结论在实习生的多份技术报告里有详细分析（见 `technical_report_20260813_RM超参实验现状报告.md` 等），本章不重复。

### 5.8 路径速查

#### huawei-0（训练机）

```
/data/liuxr/models/Qwen3.5-9B/          # 9B base 底座（19G）
/data/liuxr/RM/data/prepared/
    train_raw.jsonl   (346M, 125023条, 训练)
    val.jsonl          (39M,  13892条, 验证)
/data/liuxr/RM/scripts/train_lora_sft.sh  # 训练配置脚本
/data/liuxr/RM/output/v0-20260804-090154/
    checkpoint-3908/   # ★交付版 LoRA adapter（82.6M）
/data/liuxr/RM/output/merged-2epoch-0807/ # merge 后的完整权重（18G）
/data3/models/liuxr_lora_2epoch_compose/  # vLLM 部署目录（端口 8013）
```

#### huawei-llm（开发机）

```
/data/liuxr/RM/                          # 本报告所在
/data/liuxr/RM/scripts/prep_stage1.py    # 合并+去重+清洗
/data/liuxr/RM/scripts/prep_split_val.py # 分层留验证集
/data/liuxr/RM/README_RM.md              # RM 训练工作区说明
```

#### 连接

- 笔记本 → huawei-0：VSCode Remote-SSH，Host `huawei-0`（110.188.22.101:12202）
- huawei-llm → huawei-0：`ssh root@10.10.2.2`，免密

### 5.9 常见问题

#### Q1：训练报错 `TypeError: create_causal_mask() got an unexpected keyword argument 'cache_position'`

**原因**：ms-swift 镜像的 transformers 版本与 Qwen3.5 模型代码不兼容。

**修复**：升级到 ms-swift 4.3.0 镜像（已修复），或用 MindSpeed-MM 框架（见第 2 章）。

#### Q2：部署后推理返回 content=None

**原因**：Qwen3 系列 overthinking 通病，开 thinking 时内容写入 `reasoning` 字段，`content` 为空。

**修复**：vLLM 启动参数加 `--default-chat-template-kwargs '{"enable_thinking": false}'`，服务级关 thinking。

#### Q3：`ASCEND_VISIBLE_DEVICES` 设置后 NPU 占用不符合预期

**原因**：vLLM-Ascend v0.23.0rc1 的 `ASCEND_VISIBLE_DEVICES` 是 **chip 序号（0-15）**，不是 NPU 卡号（0-7）。每张卡 2 个 chip：chip 0,1 = NPU 卡 0；chip 2,3 = NPU 卡 1；...

**修复**：部署后必须 `npu-smi info | grep VLLMWorker` 确认进程分布在预期的 NPU 卡上。

### 5.10 本章小结

本章完整复现了评估模型训练流程：

1. ✅ **数据准备**：12.5 万条 teacher 判分数据，分层切验证集
2. ✅ **训练**：ms-swift 4.3.0 + LoRA（rank=8），2 epoch，15.9h 健康收敛
3. ✅ **部署**：LoRA merge 成完整权重，vLLM 双服务并行（端口 8012/8013）
4. ✅ **评测**：对齐率从基线 62.4% 提升到 85.3%（sft 组），process_reasonableness 提升最大（+29pp）

**交付物**：
- LoRA adapter：`/data/liuxr/RM/output/v0-20260804-090154/checkpoint-3908/`（82.6MB）
- 完整权重：`/data/liuxr/RM/output/merged-2epoch-0807/`（18G）
- vLLM 服务：端口 8012（1epoch）/ 8013（2epoch）

**下一步**：把评估模型的判分结果接入 GRPO 奖励函数（技术报告 §7.3），但当前 GRPO 用的是规则 verdict（RLVR 思路），模型评分暂未进 RL 循环。

---

## 第 6 章 推理服务部署（vLLM）

> **本章目标**：在 huawei-0 上部署 vLLM 推理服务，支撑 PI 轨迹采集、GRPO rollout、模型评测。
>
> **本章定位**：技术报告 §4.3.1 讲"为什么选 vLLM、架构设计"，本章讲"怎么部署、每个参数什么意思"——操作手册级别，照着做就能跑起来。

### 6.1 镜像版本

| 镜像 | 版本 | 说明 |
|------|------|------|
| vllm-ascend | v0.23.0rc1 | 当前生产版本，支持 prefix caching + session 亲和 |

**版本选择**：v0.23.0rc1 修复了混合 Mamba KV block 的 prefix caching 问题（v0.22.1rc1 命中率为 0%），实测命中率 76.68%。

### 6.2 Dense 模型推理（Qwen3.6-27B）

#### 6.2.1 启动命令

```bash
# vLLM 推理服务容器（huawei-0）
docker run -d --name qwen3.6-27b-server \
  --network host --ipc=host --privileged \
  -e VLLM_API_KEY=<api-key> \
  -v /data/models:/models \
  -v /data3/models:/data3/models \
  quay.io/ascend/vllm-ascend:v0.23.0rc1 \
  --model /models/Qwen3.6-27B \
  --served-model-name Qwen3.6-27B \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --max-model-len 262144 \
  --enable-prefix-caching \
  --enable-thinking \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}'
```

#### 6.2.2 Docker 参数逐行解释

| 参数 | 值 | 含义 |
|------|-----|------|
| `-d` | — | 后台运行（detached mode） |
| `--name` | `qwen3.6-27b-server` | 容器名称，方便 `docker exec` / `docker logs` |
| `--network host` | — | 使用宿主机网络栈（不走 Docker bridge），性能更好，端口直接暴露在宿主机 |
| `--ipc=host` | — | 使用宿主机 IPC 命名空间。NPU 训练/推理需要共享内存，Docker 默认的 `/dev/shm` 只有 64MB 不够 |
| `--privileged` | — | 特权模式，允许访问 NPU 设备（`/dev/davinci*`） |
| `-e VLLM_API_KEY` | `<api-key>` | API 鉴权密钥。客户端请求需带 `Authorization: Bearer <api-key>` |
| `-v /data/models:/models` | 宿主机:容器 | 挂载模型权重目录。容器内路径 `/models/Qwen3.6-27B` |
| `-v /data3/models:/data3/models` | 宿主机:容器 | 挂载备用模型目录 |

#### 6.2.3 vLLM 参数逐行解释

| 参数 | 值 | 含义 |
|------|-----|------|
| `--model` | `/models/Qwen3.6-27B` | 模型权重路径（容器内路径） |
| `--served-model-name` | `Qwen3.6-27B` | API 返回的模型名。客户端用这个名字调用 `/v1/chat/completions` |
| `--tensor-parallel-size` | `4` | **TP=4**：模型权重切分到 4 张卡。27B BF16 权重 ≈ 54GB，单卡 64GB HBM 放不下，必须切分。TP=4 意味着每张卡存 1/4 权重 |
| `--data-parallel-size` | `4` | **DP=4**：4 个独立副本并发处理请求。16 chip NPU = TP4 × DP4 = 全机利用。每个副本独立处理请求，吞吐翻 4 倍 |
| `--max-model-len` | `262144` | 最大上下文长度 256K token。Qwen3.6 原生支持的最大长度。KV cache 按这个长度预分配显存 |
| `--enable-prefix-caching` | — | 开启 prefix caching：多轮对话中，相同前缀的 KV cache 可复用，不用重新计算。对 PI 多轮采轨迹至关重要（命中率实测 76.68%） |
| `--enable-thinking` | — | 输出 `reasoning_content` 字段（CoT 思考过程）。PI 采轨迹需要完整思考链 |
| `--speculative-config` | JSON | **MTP 投机解码**配置。`method: qwen3_5_mtp` = 用模型自带的 MTP 头；`num_speculative_tokens: 3` = 一次预测 3 个 token 再验证；`enforce_eager: true` = 禁用 CUDA graph（MTP 与 CUDA graph 不兼容） |

#### 6.2.4 TP × DP 拓扑

```
16 chip NPU（huawei-0 全机）
├── Replica 0 (DP=0)          ├── Replica 1 (DP=1)
│   ├── TP=0 (chip 0)         │   ├── TP=0 (chip 4)
│   ├── TP=1 (chip 1)         │   ├── TP=1 (chip 5)
│   ├── TP=2 (chip 2)         │   ├── TP=2 (chip 6)
│   └── TP=3 (chip 3)         │   └── TP=3 (chip 7)
├── Replica 2 (DP=2)          ├── Replica 3 (DP=3)
│   ├── TP=0 (chip 8)         │   ├── TP=0 (chip 12)
│   ├── TP=1 (chip 9)         │   ├── TP=1 (chip 13)
│   ├── TP=2 (chip 10)        │   ├── TP=2 (chip 14)
│   └── TP=3 (chip 11)        │   └── TP=3 (chip 15)
```

每个 Replica 独立处理请求（DP 并发），每个 Replica 内部 4 卡协同计算（TP 切权重）。

#### 6.2.5 端口与访问

- **服务端口**：8011
- **API 格式**：OpenAI 兼容
- **访问方式**：`http://110.188.22.101:8011/v1`

```bash
# 验证服务
curl -s http://110.188.22.101:8011/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
# 期望返回：{"id":"Qwen3.6-27B",...}
```

### 6.3 MoE 模型推理（Qwen3.6-35B-A3B）

#### 6.3.1 启动命令

```bash
# MoE 模型推理服务（huawei-6）
docker run -d --name vllm-moe-test \
  --network host --ipc=host --privileged \
  -v /data/models:/models \
  llin-vllm-ascend:v0.23.0rc1-a3 \
  --model /data/models/Qwen3.6-35B-A3B \
  --served-model-name Qwen3.6-35B-A3B \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --max-model-len 8192 \
  --trust-remote-code \
  --enforce-eager \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}'
```

#### 6.3.2 与 Dense 模型的关键差异

| 参数 | Dense (Qwen3.6-27B) | MoE (Qwen3.6-35B-A3B) | 原因 |
|------|---------------------|----------------------|------|
| `--data-parallel-size` | 4 | **2** | MoE 总参数 35B（虽只激活 3B），每副本显存占用更大，DP 只能开 2 |
| `--max-model-len` | 262144 | **8192** | 当前评测用 8K 上下文。256K 需要更多 KV cache 显存，MoE 显存紧张 |
| `--trust-remote-code` | 不需要 | **需要** | MoE 架构的 model config 有自定义代码，需信任 |
| `--enforce-eager` | 不需要 | **需要** | MoE 推理在 vllm-ascend 上 CUDA graph 有兼容问题 |
| 端口 | 8011 | **28200** | 避免与 Dense 模型服务冲突 |
| 镜像 | `vllm-ascend:v0.23.0rc1` | `llin-vllm-ascend:v0.23.0rc1-a3` | MoE 需要定制的 vllm-ascend 镜像 |

**MTP 同样有效**：MoE 模型的 MTP 接受率 65-75%，与 Dense 模型相当。

### 6.4 粘性代理（解决 DP 多副本 KV cache miss）

#### 6.4.1 问题背景

DP=4 是 4 个独立副本，KV cache 不共享。PI 多轮采轨迹时，同一 session 的不同轮若被轮询分到不同副本，会触发全量重算 prefill。

**PI 0.83 的改进**：PI 0.83 新增了 `sendSessionAffinityHeaders` 配置，可以在请求头中自动发送 `x-session-affinity`（值为 session ID）。**但 vLLM 0.23 不识别这个 header**，它期望的是 `X-data-parallel-rank`。因此仍需要粘性代理做转译。

#### 6.4.2 解决方案

在 huawei-5 上部署粘性代理（sticky proxy），读取 PI 发送的 `x-session-affinity`，通过 MD5 取模计算出 rank（0-3），注入 `X-data-parallel-rank` header 后转发给 vLLM。

```bash
# 粘性代理容器（huawei-5）
docker run -d --name sticky-proxy \
  --network host --restart unless-stopped \
  -v /data/renjunxiang/proxy/sticky_v3.js:/app/proxy.js \
  node:24-bookworm-slim \
  node /app/proxy.js
```

#### 6.4.3 PI 侧配置

`/data/renjunxiang/pi/models.json`：

```json
"compat": {
  "sendSessionAffinityHeaders": true,
  "sessionAffinityFormat": "openai"
}
```

`baseUrl` 指向 `http://127.0.0.1:8010/v1`（粘性代理），而非 vLLM 直连的 `110.188.22.101:8011`。

#### 6.4.4 效果

| 指标 | 无粘性 | 有粘性 |
|------|--------|--------|
| 加权命中率 | ~57% | **69.5%** |
| engine2/3 利用率 | 闲置 | 均衡 |
| hybrid 任务耗时 | 331s | **153s（降 54%）** |

> **为什么 PI 没有直接解决这个问题**：PI 是通用 agent 框架，不特定于 vLLM。它只能发送通用的 session affinity header，而 vLLM 有自己特定的 DP 路由 header（`X-data-parallel-rank`）。两者协议不匹配，必须通过代理转译。如果未来 vLLM 原生支持 `x-session-affinity`，粘性代理就可以去掉。

### 6.5 路径速查

| 用途 | 路径 | 所在服务器 |
|------|------|------------|
| vLLM 推理服务 | `docker logs qwen3.6-27b-server` | huawei-0 |
| 粘性代理 | `docker logs sticky-proxy` | huawei-5 |
| PI 模型配置 | `/data/renjunxiang/pi/models.json` | huawei-5 |
| 代理脚本 | `/data/renjunxiang/proxy/sticky_v3.js` | huawei-5 |
| MoE 推理服务 | `docker logs vllm-moe-test` | huawei-6 |

---

## 第 7 章 评测跑通

> **本章目标**：跑通 16 套开源 benchmark 评测 + 沙箱轨迹评测，验证模型能力。
>
> **本章状态**：✅ 已完成

### 7.1 本章目标

本章回答：**怎么评测训练后的模型？怎么跑 16 套开源 benchmark？怎么看评测结果？**

评测分两条线：
- **开源 benchmark**（本章重点）：16 套标准化评测集（AIME/CMMLU/BBH/HumanEval 等），跑推理 → 评分 → 汇总
- **沙箱轨迹评测**（第 4 章已涉及）：物流 agent 场景端到端评测，规则 verdict + LLM judge

两条线的关系：开源 benchmark 监控通用能力不退化；沙箱评测监控垂域任务通过率。

### 7.2 评测流水线总览

评测分两阶段：**先跑完全部推理，再统一评分**（不是边跑边评）。

```
数据集 JSONL（四字段 + extract_method + score_method）
    │
    ▼  run_benchmark.py（并发 32 题发 vLLM）
predictions.jsonl（N 条 response）
    │
    ▼  score.py（双维度路由：extract → score）
scores.jsonl（N 条对错） + summary.json（accuracy）
```

### 7.3 双维度路由架构

不同数据集的题目形态差异很大，但评测逻辑可拆解为两个正交维度：

1. **答案提取**（`extract_method`）：从模型输出中怎么提取答案
2. **评判比较**（`score_method`）：提取出来后怎么判对错

每条测试数据标注这两个字段，评测管线统一为：

```
model_output → extract(method) → extracted → score(method, golden_answer) → correct/wrong
```

**5 种 extract_method**：

| 方法 | 适用场景 | 提取逻辑 |
|------|---------|---------|
| `boxed` | 数学/科学数值题 | `\boxed{...}` → `<answer>` tag → raw strip |
| `choice_single` | 单选题 | 5 级正则提取单个字母 A-J |
| `choice_multi` | 多选题 | 提取所有大写字母 → sort + dedup |
| `code_block` | 代码题 | `<answer>` → ` ```python ` 代码块 → def/import 行 |
| `answer_or_raw` | BBH 等混合推理（特别是 3-shot CoT） | `<answer>` tag → `ANSWER:` 前缀 → `</think>` 后最后非标签行（支持从"the answer is X"/"答案：X"等句式提取）→ raw |

**7 种 score_method**：

| 方法 | 适用场景 | 评判逻辑 |
|------|---------|---------|
| `numeric_compare` | 数值题 | float 容差 1e-3 → fallback 字符串精确匹配 |
| `exact_match` | 单选/判断 | strip + 精确字符串比较 |
| `ci_match` | BBH-TF（True/False） | strip + lower 匹配 |
| `ws_ci_match` | BBH-Freeform | 空白符规范化 + lower |
| `set_match` | 多选题 | 字母集合相等 |
| `code_execute` | 代码题 | 拼接测试脚本 → subprocess 执行 |
| `constraint_check` | 指令遵循（IFEval） | 16 个约束 checker 逐条验证 |

### 7.4 跑评测

#### 7.4.1 前置条件

| 条件 | 检查命令 | 期望输出 |
|------|---------|---------|
| vLLM 推理服务已启动 | `curl http://127.0.0.1:8012/v1/models` | 返回模型列表 |
| 测试数据已准备 | `ls datasets/open_source/sft/processed/without_golden_thinking/` | 42 个 jsonl |
| 评测配置已就绪 | `cat configs/eval/benchmark_config.yaml` | 16 个 benchmark 定义 |

#### 7.4.2 跑推理

```bash
# 在 huawei-0 上（vLLM 推理服务所在机器）
cd /data/renjunxiang/coding/huawei_train

# 跑全部 benchmark
python scripts/eval/run_benchmark.py \
  --config configs/eval/benchmark_config.yaml \
  --model qwen3.6-27b \
  --output eval_results/

# 只跑某个 benchmark
python scripts/eval/run_benchmark.py \
  --config configs/eval/benchmark_config.yaml \
  --model qwen3.6-27b \
  --benchmarks AIME24 \
  --output eval_results/
```

**产出**：`eval_results/benchmarks/{base_model}/{tag}/{config_key}/{timestamp}_{benchmark}/predictions.jsonl`

#### 7.4.3 跑评分

```bash
# 评分（读 predictions.jsonl + 测试集 → scores.jsonl + summary.json）
python scripts/eval/score.py \
  --config configs/eval/benchmark_config.yaml \
  --model qwen3.6-27b \
  --benchmarks AIME24
```

**产出**：
- `scores.jsonl`（逐题对错）
- `summary.json`（accuracy/correct/wrong/extract_fail）

### 7.5 16 个 benchmark 配置

| 分类 | Benchmark | type | shot | max_tokens | 样本数 |
|------|-----------|------|------|-----------|--------|
| 数学推理 | AIME24 | math | 0 | 30720 | 30 |
| 数学推理 | AIME25 | math | 0 | 30720 | 30 |
| 数学推理 | OlymMATH | math | 0 | 30720 | 400 |
| 数学推理 | PolyMath | math | 0 | 30720 | 9000 |
| 数学推理 | MGSM-zh | math | 0 | 30720 | 250 |
| 数学推理 | CMATH | math | 0 | 8192 | 600 |
| 知识选择 | CMMLU | multiple_choice | 5 | 30720 | 11582 |
| 知识选择 | MMLU-Pro | multiple_choice | 5 | 30720 | 12032 |
| 知识选择 | C-Eval-val | multiple_choice | 5 | 30720 | 1346 |
| 知识选择 | RACE | multiple_choice | 0 | 8192 | 4934 |
| 安全评测 | SafetyBench-zh | multiple_choice | 0 | 8192 | 11435 |
| 安全评测 | SafetyBench-en | multiple_choice | 0 | 8192 | 11435 |
| 推理 | BBH | bbh | 3 | 4096 | 6511 |
| 代码生成 | HumanEval | code | 0 | 30720 | 164 |
| 代码生成 | MBPP | code | 0 | 8192 | 974 |
| 指令遵循 | IFEval | instruction | 0 | 30720 | 541 |

**上下文长度复测策略**：每个 benchmark 在 8K（`nothink_8192`）和 30K（`nothink_30720`）两个上下文长度下复测。若某 benchmark 在 8K 下基本无截断（截断占比 < 0.1%），则 30K 版本不另行重跑，直接复用 8K 结果。

### 7.6 结果查看

#### 7.6.1 命令行查看

```bash
# 查看某个 benchmark 的结果
cat eval_results/benchmarks/Qwen3.6-27B/qwen3.6-27b/nothink_8192/*/AIME24/summary.json
# {"accuracy": 0.933, "correct": 28, "wrong": 2, ...}
```

#### 7.6.2 看板查看

看板后端（`dashboard/serve.py`）扫描 `eval_results/` 下所有 `summary.json`，前端（`dashboard/web/js/eval_results.js`）渲染表格。

```bash
# 启动看板
cd /Users/renjunxiang/coding/huawei_train/dashboard
python serve.py
# 浏览器打开 http://localhost:5001
```

看板功能：
- 按模型/版本/benchmark 筛选
- 同模型不同版本（v0=8K, v1=30K）并排对比
- 点击单元格展开逐题明细

### 7.7 模型定义文件

每个模型在 `eval_results/models/{tag}.yaml` 中定义：

```yaml
# eval_results/models/qwen3.6-27b.yaml
base_model: Qwen3.6-27B
served_model_name: Qwen3.6-27B
description: Qwen3.6-27B 基座模型
```

已有评测结果的模型：

| 模型 tag | base_model | 说明 |
|----------|-----------|------|
| `qwen3.6-27b` | Qwen3.6-27B | 基座模型 |
| `qwen3.5-9b` | Qwen3.5-9B | 评估模型基座 |
| `llin-grpo-step100` | Qwen3.6-27B | GRPO Step100 |
| `llin-grpo-step120` | Qwen3.6-27B | GRPO Step120（最佳） |
| `qwen3.8-27b` | Qwen3.8-27B | Qwen3.8 对比 |
| `qwen3.6-35b-a3b-moe` | Qwen3.6-35B-A3B | MoE 对比 |

### 7.8 新增 benchmark

新增一个 benchmark 只需两步：

**步骤 1：准备测试数据**

在 `datasets/open_source/sft/processed/without_golden_thinking/` 下创建 `{BenchmarkName}.jsonl`，每条标注 `extract_method` + `score_method`：

```json
{"system_prompt": "...", "task_input": "...", "thinking": "", "golden_answer": "42", "extract_method": "boxed", "score_method": "numeric_compare"}
```

**步骤 2：在 benchmark_config.yaml 添加配置**

```yaml
benchmarks:
  NewBenchmark:
    type: math
    shot: 0
    max_tokens: 8192
    num_samples: null  # null=全量
    scorer: math_scorer  # 旧架构兼容
```

不需要写新 scorer——双维度路由自动处理。

### 7.9 路径速查

| 用途 | 路径 |
|------|------|
| 评测主脚本 | `scripts/eval/run_benchmark.py` |
| 评分脚本 | `scripts/eval/score.py` |
| 答案提取器 | `scripts/eval/extractors.py` |
| 评判器 | `scripts/eval/scorers_v2.py` |
| 评测配置 | `configs/eval/benchmark_config.yaml` |
| 模型定义 | `eval_results/models/{tag}.yaml` |
| 评测结果 | `eval_results/benchmarks/{base_model}/{tag}/{config_key}/` |
| 看板后端 | `dashboard/serve.py` |
| 看板前端 | `dashboard/web/js/eval_results.js` |

## 第 8 章 看板逻辑解析

> **本章目标**：解析评测看板的数据来源、路径规范、页面结构和映射逻辑。
>
> **本章状态**：✅ 已完成

### 8.1 eval_results 目录结构

评测结果统一存储在 `eval_results/` 目录下，按评测类型分三个子目录：

```
eval_results/
├── benchmarks/          # 开源基准评测（16 个 benchmark）
│   └── {base_model}/{tag}/{config_key}/{benchmark}/
│       └── summary.json
├── verdicts/            # 沙箱评测 - 规则判分
│   └── {group}/{model}_{sandbox}_{variant}_openai.jsonl
└── rewards/             # 沙箱评测 - 模型多维度评测
    └── {group}/{sandbox}/{model}/{variant}/{checkpoint}/
        └── {model}_{sandbox}_{variant}_llm.jsonl
```

### 8.2 路径规范

#### 8.2.1 开源基准评测（benchmarks/）

路径结构：`{base_model}/{tag}/{config_key}/{benchmark}/summary.json`

- `base_model`：基座模型名（如 Qwen3.6-27B）
- `tag`：模型标签（如 qwen3.6-27b、llin-grpo-step120）
- `config_key`：配置键（如 nothink_8192、nothink_30720）
- `benchmark`：基准名称（如 AIME24、MMLU-Pro）

#### 8.2.2 沙箱评测 - 规则判分（verdicts/）

路径结构：`{group}/{model}_{sandbox}_{variant}_openai.jsonl`

- `group`：沙箱组（sft/test/dev）
- `model`：模型名（如 qwen3.6-27B）
- `sandbox`：沙箱版本（如 20260628_v15）
- `variant`：变体（base/table/cols/dict）

**示例**：`verdicts/sft/qwen3.6-27B_20260628_v15_openai.jsonl`

#### 8.2.3 沙箱评测 - 模型多维度评测（rewards/）

路径结构（6 层）：`{group}/{sandbox}/{model}/{variant}/{checkpoint}/{file}.jsonl`

| 层级 | 字段 | 含义 | 示例 |
|------|------|------|------|
| 0 | group | 沙箱组 | test |
| 1 | sandbox | 沙箱版本 | 20260627_v4 |
| 2 | model | 模型大类 | qwen37max |
| 3 | variant | 模型小类（运行环境/训练状态） | qwen37max_L40S |
| 4 | checkpoint | 检查点/judge 名 | liuxr-lora-0813-1954 |
| 5 | file | 文件名 | qwen37max_L40S_20260627_v4_table_llm.jsonl |

**variant 命名规范**：
- `{model}_baseline`：纯模型，无额外信息
- `{model}_L40S`：模型在 L40S 服务器运行
- `{model}_GRPO`：模型经过 GRPO 训练
- `{model}_dict`：给表信息
- `{model}_cols`：给列信息

**示例**：
```
rewards/test/20260627_v4/qwen37max/qwen37max_L40S/liuxr-lora-0813-1954/qwen37max_L40S_20260627_v4_table_llm.jsonl
```

解析为：
- group = test
- sandbox = v4（从 20260627_v4 提取）
- model = qwen37max_L40S（policy_name）
- server = liuxr-lora-0813-1954（judge）
- variant = table（从文件名末尾提取）

### 8.3 看板页面结构

看板（`dashboard/serve.py`）提供三个主要页面：

#### 8.3.1 开源基准评测页

**数据来源**：`eval_results/benchmarks/`

**展示内容**：
- 行：benchmark 名称（AIME24、MMLU-Pro 等）
- 列：模型标签（qwen3.6-27b、llin-grpo-step120 等）
- 单元格：准确率（从 summary.json 读取）

**映射逻辑**：
```python
# dashboard/serve.py
summary_files = sorted((eval_dir / "benchmarks").rglob("summary.json"))
for f in summary_files:
    # 解析路径：benchmarks/{base_model}/{tag}/{config_key}/{benchmark}/summary.json
    parts = f.relative_to(eval_dir / "benchmarks").parts
    base_model, tag, config_key, benchmark = parts[:4]
    # 读取准确率
    summary = json.loads(f.read_text())
    accuracy = summary.get("accuracy")
```

#### 8.3.2 沙箱评测页 - verdicts 栏

**数据来源**：`eval_results/verdicts/`

**展示内容**：
- 行：沙箱版本（v15、v20 等）
- 列：模型名（qwen3.6-27B、qwen37max 等）
- 单元格：correct 率（规则判分）

**映射逻辑**：
```python
# dashboard/serve.py
verdict_files = sorted((eval_dir / "verdicts").rglob("*.jsonl"))
for f in verdict_files:
    # 解析文件名：{model}_{sandbox}_{variant}_openai.jsonl
    # 读取每条记录的 verdict 字段
    # 统计 correct 率
```

#### 8.3.3 沙箱评测页 - rewards 栏

**数据来源**：`eval_results/rewards/`

**展示内容**：
- 行：沙箱版本（v4、v15 等）
- 列：模型名（qwen37max_L40S、qwen3.6-27B 等）
- 单元格：correct 率（模型多维度评测）+ 15 维 reward mean

**映射逻辑**：
```python
# dashboard/serve.py
reward_files = sorted((eval_dir / "rewards").rglob("*.jsonl"))
for f in reward_files:
    # 解析路径：rewards/{group}/{sandbox}/{model}/{variant}/{checkpoint}/file.jsonl
    rel_parts = f.relative_to(eval_dir / "rewards").parts
    group = rel_parts[0]
    sandbox = extract_version(rel_parts[1])  # 20260627_v4 → v4
    model = rel_parts[3]  # policy_name
    server = rel_parts[4]  # judge/checkpoint
    variant = extract_variant(f.stem)  # 从文件名末尾提取
    
    # 读取每条记录的 reward 字段
    # 统计 correct 率和 15 维 reward mean
```

### 8.4 verdicts vs rewards 的区别

| 维度 | verdicts（规则判分） | rewards（模型评测） |
|------|---------------------|-------------------|
| **判分方式** | 规则（SQL 比对、数值匹配、表命中） | LLM judge（商业模型或蒸馏模型） |
| **判分维度** | 1 维（correct/incorrect/partial） | 20 维（结果组 7 + 过程组 5 + 效率组 8） |
| **适用场景** | 数值型任务（DWH） | 报告型任务（KB/Hybrid） |
| **脚本** | `judge_trajectory.py` | `reward_judge.py` |
| **路径** | `eval_results/verdicts/` | `eval_results/rewards/` |
| **看板栏** | verdicts 栏 | rewards 栏 |

**核心区别**：
- **verdicts**：规则判分，快速、客观，但只能判数值型任务
- **rewards**：模型判分，慢、贵，但能判报告型任务的语义质量

### 8.5 看板启动与访问

```bash
# 启动看板
cd /Users/renjunxiang/coding/huawei_train/dashboard
python serve.py

# 浏览器打开
http://localhost:5001
```

**看板功能**：
- 按模型/版本/benchmark 筛选
- 同模型不同版本（v0=8K, v1=30K）并排对比
- 点击单元格展开逐题明细
- verdicts 栏和 rewards 栏并排对比（看 LLM judge 对报告型 correct 的影响）

### 8.6 路径速查

| 用途 | 路径 |
|------|------|
| 开源基准结果 | `eval_results/benchmarks/{base_model}/{tag}/{config_key}/{benchmark}/` |
| 规则判分结果 | `eval_results/verdicts/{group}/{model}_{sandbox}_{variant}_openai.jsonl` |
| 模型评测结果 | `eval_results/rewards/{group}/{sandbox}/{model}/{variant}/{checkpoint}/` |
| 看板后端 | `dashboard/serve.py` |
| 看板前端 | `dashboard/web/js/eval_results.js` |

---

## 附录：脚本源码

> 本附录收录报告中引用的关键脚本完整源码，供查阅和二次开发。脚本文件位于 `scripts/` 目录下，可直接执行。

### 附录-3.1 数据准备脚本（SFT 训练数据转换）

**文件路径**：`scripts/train/mindspeed_mm/sft/prepare_data.py`

**功能说明**：将四字段格式（system_prompt / task_input / thinking / golden_answer）转为 OpenAI ChatCompletion 格式，支持 MindSpeed-MM（需要 `images: null`）和 ms-swift（标准 OpenAI 格式）两种框架。

```python
#!/usr/bin/env python3
"""为 SFT 训练准备数据。

将四字段格式（system_prompt/task_input/thinking/golden_answer）转为 OpenAI ChatCompletion 格式。

支持两种框架：
- MindSpeed-MM：添加 images=null 字段（兼容 qwen3_5 多模态框架）
- ms-swift：标准 OpenAI 格式，不需要 images 字段

Usage:
    # MindSpeed-MM 格式（默认）：从 GSM8K 和 MATH 各取 200 条
    python scripts/train/mindspeed_mm/sft/prepare_data.py

    # ms-swift 格式：不需要 images 字段
    python scripts/train/mindspeed_mm/sft/prepare_data.py \\
        --framework swift \\
        --output /data/renjunxiang/coding/huawei_train/datasets/sft_smoke_400_swift.jsonl

    # 自定义输入文件、条数和输出路径
    python scripts/train/mindspeed_mm/sft/prepare_data.py \\
        --framework mindspeed \\
        --input-dir /data/renjunxiang/coding/huawei_train/datasets/open_source/processed/with_golden_thinking/Reasoning/ \\
        --files GSM8K.jsonl MATH.jsonl \\
        --per-file 200 \\
        --output /data/renjunxiang/coding/huawei_train/datasets/sft_smoke_400.jsonl
"""

import argparse
import json
import os
from pathlib import Path


def prepare_data(input_dir: str, files: list[str], per_file: int, output_path: str, framework: str):
    """读取四字段格式数据，转为指定框架所需格式。"""
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for fname in files:
            filepath = os.path.join(input_dir, fname)
            if not os.path.exists(filepath):
                print(f"警告：文件不存在，跳过: {filepath}")
                continue

            with open(filepath, "r", encoding="utf-8") as fin:
                for i, line in enumerate(fin):
                    if i >= per_file:
                        break

                    rec = json.loads(line.strip())
                    msg = {"messages": []}

                    # system prompt（可选）
                    if rec.get("system_prompt"):
                        msg["messages"].append({
                            "role": "system",
                            "content": rec["system_prompt"]
                        })

                    # user input
                    msg["messages"].append({
                        "role": "user",
                        "content": rec["task_input"]
                    })

                    # assistant response（带 thinking 的 reasoning_content）
                    assistant = {
                        "role": "assistant",
                        "content": "<answer>\n" + rec["golden_answer"] + "\n</answer>"
                    }
                    if rec.get("thinking"):
                        assistant["reasoning_content"] = rec["thinking"]
                    msg["messages"].append(assistant)

                    # MindSpeed-MM 需要 images 字段（兼容多模态框架）
                    if framework == "mindspeed":
                        msg["images"] = None

                    out.write(json.dumps(msg, ensure_ascii=False) + "\n")
                    count += 1

    print(f"转换完成: {count} 条 -> {output_path} (框架: {framework})")


def main():
    parser = argparse.ArgumentParser(description="为 SFT 训练准备数据")
    parser.add_argument(
        "--framework",
        choices=["mindspeed", "swift"],
        default="mindspeed",
        help="目标框架：mindspeed（需要 images=null）或 swift（标准 OpenAI 格式）"
    )
    parser.add_argument(
        "--input-dir",
        default="/data/renjunxiang/coding/huawei_train/datasets/open_source/processed/with_golden_thinking/Reasoning/",
        help="四字段格式数据所在目录"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["GSM8K.jsonl", "MATH.jsonl"],
        help="要转换的文件列表"
    )
    parser.add_argument(
        "--per-file",
        type=int,
        default=200,
        help="每个文件取多少条（默认 200）"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件路径（默认根据 framework 自动命名）"
    )

    args = parser.parse_args()

    # 自动生成输出路径
    if args.output is None:
        if args.framework == "mindspeed":
            args.output = "/data/renjunxiang/coding/huawei_train/datasets/sft_smoke_400.jsonl"
        else:
            args.output = "/data/renjunxiang/coding/huawei_train/datasets/sft_smoke_400_swift.jsonl"

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    prepare_data(args.input_dir, args.files, args.per_file, args.output, args.framework)


if __name__ == "__main__":
    main()
```

**使用方法**：

```bash
# MindSpeed-MM 格式（§3.1.4）
python3 scripts/train/mindspeed_mm/sft/prepare_data.py
# 产出：/data/renjunxiang/coding/huawei_train/datasets/sft_smoke_400.jsonl

# ms-swift 格式（§3.2.2）
python3 scripts/train/mindspeed_mm/sft/prepare_data.py --framework swift
# 产出：/data/renjunxiang/coding/huawei_train/datasets/sft_smoke_400_swift.jsonl
```

### 附录-3.2 MindSpeed-MM Full SFT 训练配置

### 附录-3.2 MindSpeed-MM Full SFT 训练配置

**文件路径**：`scripts/train/mindspeed_mm/sft/smoke_test_config.yaml`

**容器内路径**：`/workspace/MindSpeed-MM/examples/qwen3_6/smoke_test_config.yaml`

**功能说明**：Qwen3.6-27B 在 MindSpeed-MM 框架下进行 Full SFT 的完整配置文件，使用 16 chip FSDP2 并行策略。

```yaml
# ============================================================
# MindSpeed-MM Full SFT 配置（Qwen3.6-27B，16 chip FSDP2）
# 验证日期：2026-09-13，huawei-5
# 容器内路径：/workspace/MindSpeed-MM/examples/qwen3_6/smoke_test_config.yaml
# ============================================================

# ---- 并行策略 ----
parallel:
  fully_shard_parallel_size: auto    # FSDP2 自动分片
  fsdp_plan:
    apply_modules:
      - model.visual                 # 视觉模块（纯文本 SFT 时冻结）
      - model.visual.blocks.{*}
      - model.language_model
      - model.language_model.embed_tokens
      - model.language_model.layers.{*}
      - lm_head
    param_dtype: bf16                # 参数精度
    reduce_dtype: fp32               # 梯度归约精度
  ulysses_parallel_size: 1           # 上下文并行（1=不开）

# ---- 数据 ----
data:
  dataset_param:
    dataset_type: huggingface
    attr:
      images: images                 # 图片字段名（纯文本也要声明）
      messages: messages
      role_tag: role
      content_tag: content
      user_tag: user
      assistant_tag: assistant
    preprocess_parameters:
      model_name_or_path: &HF_MODEL_LOAD_PATH /data/models/Qwen3.6-27B
      use_fast_tokenizer: true
      split_special_tokens: false
      cutoff_len: 1024               # ⚠️ 不要超过 2048（NPU rotary 算子限制）
      template: qwen3_vl_nothink
      enable_thinking: false
      train_on_prompt: false         # 不对 prompt 部分算 loss
      mask_history: false
      dataset_dir: ./data
      dataset: &DATASET_PATH ./data/sft_smoke_400.jsonl
      cache_dir: ./cache_dir/
      overwrite_cache: false
      preprocessing_batch_size: 1000
      preprocessing_num_workers: 16
      max_samples: null
  dataloader_param:
    pin_memory: true
    shuffle: false
    dataloader_mode: sampler         # ⚠️ 必填，不加会报 pydantic 验证错误
    sampler_type: BaseRandomBatchSampler  # ⚠️ 必填
    drop_last: true
    num_workers: 8
    collate_param:
      model_name: qwen3vl
      ignore_pad_token_for_loss: true
    enable_preload: false

# ---- 模型 ----
model:
  model_id: qwen3_5
  model_name_or_path: *HF_MODEL_LOAD_PATH
  trust_remote_code: true
  attn_implementation: flash_attention_2
  freeze:
    - model.visual                   # 纯文本 SFT 冻结视觉
  gdn_implementation: triton
  causal_conv1d_implementation: eager

# ---- 优化特性 ----
features:
  loss_cfg:
    loss_type: default
    router_aux_loss_coef: 0.0
  recompute: true                    # 激活重计算（省显存，27B 必须开）
  recompute_plan:
    apply_modules:
      - model.visual.blocks.{*}
      - model.language_model.layers.{*}
  enable_chunk_loss: true
  chunkloss_plan:
    apply_module: lm_head
    chunk_size: 1024
  enable_activation_offload: false

# ---- 训练超参 ----
training:
  micro_batch_size: 1                # 单卡 batch
  gradient_accumulation_steps: 1     # global_batch = 1 × 1 × 16 = 16
  seed: 42
  lr: 1.0e-5                         # Full SFT 标准值
  lr_decay_style: cosine
  lr_warmup_ratio: 0.1
  weight_decay: 0
  train_iters: 500
  clip_grad: 0.0                     # Full SFT 通常不裁剪
  init_model_with_meta_device: true  # DCP 格式要求
  optimizer: adamw
  adam_fused: true
  save_interval: 100
  no_load_optim: true
  no_load_rng: true
  no_save_optim: true
  no_save_rng: true
  load: /data/models/Qwen3.6-27B-dcp
  save: /data/renjunxiang/coding/huawei_train/models/sft_full
  use_deter_comp: false
  plugin:
    - mindspeed_mm/fsdp/models/qwen3_5
    - mindspeed_mm/fsdp/data/datasets/huggingface
```

**使用方法**（§3.1.5）：

```bash
# 复制配置文件到容器内
mkdir -p /workspace/MindSpeed-MM/examples/qwen3_6/
cp scripts/train/mindspeed_mm/sft/smoke_test_config.yaml \
   /workspace/MindSpeed-MM/examples/qwen3_6/smoke_test_config.yaml
```

---

*报告完。*
