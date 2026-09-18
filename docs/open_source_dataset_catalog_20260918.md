# 5号机与0号机开放数据资产清单

盘点日期：2026-09-18
适用目的：训练、评测、物流运营沙箱、数仓沙箱、时序世界建模的数据选型与备份。
机器别名：`huawei-05`（5号机）、`huawei-00`（0号机）。

## 一、使用结论

以后选择数据时，先查本清单，不再从磁盘根目录重新搜索。

优先级定义：

- **P0 必须沉淀**：团队做过筛选、清洗、格式转换或人工核验，或者上游需要授权、以后可能无法重新取得。
- **P1 选择性沉淀**：规模适中、任务价值高，但仍可从公开来源重新下载。
- **P2 只登记来源**：体量巨大、公开可重下或与其他目录高度重复，不建议立即复制到个人网盘。
- **受限资产**：可能包含团队内部、老板数据或未确认授权的数据，只登记位置，不得默认上传个人网盘。

## 二、已经完成的备份

| 资产 | 源位置 | 百度网盘位置 | 原始规模 | 归档规模 | SHA-256 | 状态 |
|---|---|---|---:|---:|---|---|
| LLM RAW 首批19类 | `huawei-05:/data/datasets/llm/raw/` | `全部文件/新建文开源数据集_5号机件夹/llm_raw_20260918.tar.gz` | 5.2G | 1.34GB | `48950BE0590363A2FC70A87AD64DCC8D9E744383903E3A55743C5559FA9087FE` | 已上传并核验；本地临时包已删除 |

这份归档包含：AIME24、AIME25、AMO-Bench、ATLAS、BBH、BIG-bench、C-Eval、CMMLU、GSM8K、MATH、MATH-500、MMLU-Pro、MegaScience、MetaMATH、OlymMATH、Omni-MATH、PHYBench、PolyMath、mmlu。

注意：该5.2G目录是下述40G主库的子集，不是完整开放数据资产。

## 三、P0：下一批必须沉淀

### 1. 5号机开放训练与评测主库

- 路径：`huawei-05:/data/renjunxiang/coding/huawei_train/datasets/open_source/`
- 总规模：40G
- 结构：`raw/` 32G、`processed/` 7.4G
- 价值：同时保存上游原始版本和团队统一转换后的JSONL，重做清洗与格式转换成本高。

已备份RAW之外，主库还有13类新增原始数据：

| 新增数据 | 规模 | 主要用途 |
|---|---:|---|
| `guru-RL-92k` | 8.6G | 代码与强化学习任务 |
| `KodCode-V1` | 7.7G | 代码训练 |
| `SCP_116K` | 5.2G | 推理训练 |
| `DeepMath-103K` | 4.8G | 数学推理 |
| `PKU-SafeRLHF` | 431M | 安全对齐 |
| `safety_tool` | 71M | 安全与工具调用评测 |
| `s1K-mix` | 35M | 小规模高质量推理 |
| `FrontierScience` | 13M | 科学推理 |
| `LogistikaBench` | 2.6M | 物流知识与业务问答 |
| `S1-Bench` | 1.2M | 推理评测 |
| `GPQA-Diamond` | 492K | 高难科学问答 |
| `SC-bench` | 400K | 供应链业务与工具使用 |
| `gpqa` | 8K | GPQA补充文件 |

`processed/without_golden_thinking/` 已按 `Agentic / Alignment / Coding / General / Reasoning / Safety` 分层，包含53个文件；其中最重要的团队加工产物包括：

- `MegaScience_train.jsonl` 约4.3G
- `KodCode-V1_train.jsonl` 约1.1G
- `BIG-bench.jsonl` 约871M
- `MetaMATH_train.jsonl` 约581M
- `SCP_116K_train.jsonl` 约377M
- BFCL、CompToolBench、GeneralFunctionCall、MetaTool、NESTFUL、ToolHop等统一Agentic评测文件

建议：将整个40G目录作为一个“规范主库”归档；现有5.2G归档保留为首批快照，不再作为完整清单。

### 2. 5号机安全与工具调用核验集

- 路径：`huawei-05:/data1/yehairui/eval_datasets/`
- 规模：111M
- 内容：SafetyBench、CValues-MC、CS-Eval、MetaTool/ToolE、ToolHop、BFCL、CompToolBench、GeneralFunctionCall-Test、NESTFUL。
- 独特价值：自带 `评测集清单.md` 与 `核验报告.md`，明确8个数据集可以本地确定性判分、CS-Eval因答案不公开而降级；这是人工调研和抽样核验成果，不只是公开文件副本。

建议：整体备份，优先级高于单独备份其中任一上游数据集。

### 3. 5号机Agentic/Thinking训练集合

- 路径：`huawei-05:/data1/yehairui/agentic_sft_thinking_dataset/`
- 规模：114G，约920个文件
- 分层：代码68G、指令20G、工具17G、中文8.7G、多轮1.3G。
- 独特价值：包含逐项下载状态、teacher核验、许可提示、质量判断和失败记录；部分Toolathlon数据需要同意条款，部分上游已私有或不可访问，重新获取风险高。
- 重要文件：`DOWNLOAD_MANIFEST.md`、下载脚本、抽样脚本。

建议按五个领域拆包并分别生成SHA-256，不建议做单个114G大包。先归档manifest和脚本，再归档 `tool/`、`code/` 等数据目录。

### 4. 5号机开源SFT质量筛选结果

- 路径：`huawei-05:/data3/llin/sft/datasets/open_source/huawei_train/`
- 规模：119M
- 内容：OpenAI格式数据、reasoning合并版、`quality_split_20260713/`。
- 独特价值：团队已经完成格式统一、质量分层与审计，重做成本高。

建议：整体备份，并保留质量报告、审计文件和训练/隔离集的相对目录。

### 5. 物流与供应链专项小资产

| 资产 | 路径 | 规模 | 价值 |
|---|---|---:|---|
| 供应链训练调研包 | `huawei-05:/data1/yehairui/Mianyan-Ascend-LLM/supply_chain_continued_pretraining/` | 60M | LogistikaBench、SC-bench、56M调研材料及CPT/SFT/GRPO行动指南 |
| 车辆路径规划原始基准 | `huawei-05:/data/zzl/mianyang-ldm/data/benchmark/` | 60M | CVRPLib、DIMACS、Loggi、ORTEC等物流路径规划实例 |
| HFVRP转换版 | `huawei-05:/data/zzl/mianyang-ldm/data/benchmark2hfvrp/` | 16M | Solomon等VRP实例的JSON转换版 |
| CPT语料v1.0.2 | `huawei-05:/data3/llin/qwen3.6-27b-verl-grpo/runs/cpt-corpus-v1.0.2-20260910/` | 468K | train/validation双格式、manifest、loader检查与归属声明 |
| Step120开放样本清单 | `huawei-05:/data3/llin/qwen3.6-27b-verl-grpo/data/step120_opensource_20260824/` | 240K | 数据manifest、质量报告和训练样本 |

这些资产总体很小、与物流沙箱直接相关，应打成一个“物流开放数据与方法包”整体保存。

## 四、P1：选择性沉淀

### 1. 0号机时序评测集

| 资产 | 路径 | 规模 | 建议 |
|---|---|---:|---|
| FEV | `huawei-00:/data/datasets/fev_datasets/` | 653M | 整体备份；约50类标准时序任务，体量小、覆盖广 |
| GiftEval | `huawei-00:/data/datasets/GiftEval/` | 1.1G | 整体备份；适合时序预测与世界状态演化评测 |
| Time-300B当前副本 | `huawei-00:/data/datasets/Time-300B/` | 984M | 当前主要是energy与synthetic，先核验完整性再备份 |

### 2. 5号机专项中文与结构化任务集合

- 路径：`huawei-05:/data/liuxr/datasetscom/`
- 规模：37G，其中BIRD-bench约32G。
- 其余约5G包含：法律、医疗、金融、COIG-CQIA、LongBench、HotpotQA、ToolACE、xLAM Function Calling、关税裁定、京东评论等。
- 建议：先备份除BIRD-bench以外的专项集合；BIRD-bench单独登记和归档。
- 合规：许可混合，训练或再分发前必须逐数据集核验README/LICENSE。

## 五、P2：只登记来源，暂不复制

| 资产 | 路径 | 规模 | 暂不备份原因 |
|---|---|---:|---|
| The Pile uncopyrighted | `huawei-05:/data/liuxr/datasets/pile-uncopyrighted/` | 337G | 公开、体量大，可按来源重下 |
| Chronos集合 | `huawei-00:/data/datasets/chronos_datasets/` | 124G | 大量公开时序数据，内部还有97G training_corpus |
| GiftEvalPretrain | `huawei-00:/data/datasets/GiftEvalPretrain/` | 114G | 公开预训练集合，建议保留README和下载脚本 |
| GiftEval气候扩展 | `huawei-00:/data/datasets/GiftEvalPretrain_Append/` | 252G | ERA5/CMIP6年度分片，公开且体量巨大 |
| Alpaca-CoT聚合 | `huawei-00:/data/datasets/nlp/datasets/Alpaca-CoT/` | 240G | 多个旧公开指令集聚合，许可和重复关系复杂 |
| Wikipedia clickstream | `huawei-00:/data2/wiki-clickstream/` | 87G | 公开月度数据，可重下 |
| Wikipedia pageviews原始 | `huawei-00:/data2/wiki-pageviews/` | 715G | 公开小时数据，可重下 |
| Wikipedia pageviews加工版 | `huawei-00:/data3/datasets/wiki_pageviews_daily/` | 258G | 与原始数据有派生关系，需先保存加工脚本和血缘再决定 |

P2不是“没有价值”，而是当前更适合保存数据卡、来源、版本、下载脚本和校验值，而不是复制数百GB文件。

## 六、重复与选择规则

1. `/data/datasets/llm/raw/` 的19类数据全部存在于40G主库 `raw/` 中；以后以40G主库为规范入口。
2. 0号机 `/data/renjunxiang/.../open_source/` 的7.1G主要是加工后副本，不优先于5号机同时具备raw与processed的40G主库。
3. `LogistikaBench`、`SC-bench` 在多个目录存在；物流专项包保留源码、README、LICENSE和转换结果，避免分别重复备份。
4. 0号机Wikipedia的原始GZ与Arrow日级/小时级数据存在血缘关系，不能当成互相独立的数据集。
5. FEV、GiftEval、Chronos之间有M4、M5、天气、电力、交通等重叠主题；用于评测时按数据集版本隔离，避免训练/测试泄漏。

## 七、受限资产与个人网盘边界

以下资产虽然业务价值高，但不应在没有明确授权时上传个人百度网盘：

- `huawei-05:/data3/llin/qwen3.6-27b-verl-grpo/data/boss_v15_dwh_full276_20260806/dataset/`
- `huawei-05:/data3/llin/qwen3.6-27b-verl-grpo/data/boss_v15_dwh_full277_20260804/dataset/`
- 其他老板数据、内部沙箱轨迹、未脱敏业务数据。

它们应进入团队批准的内部存储，并单独记录权限、血缘和保留周期。

## 八、建议的后续备份顺序

1. 40G开放训练与评测主库。
2. 111M安全/工具调用核验集。
3. 119M开源SFT质量筛选结果。
4. 约137M物流开放数据与方法包。
5. 114G Agentic/Thinking集合，按领域分五包归档。
6. 0号机FEV与GiftEval，共约1.8G。
7. 5号机专项中文与结构化任务集合，先排除或单列32G BIRD-bench。

每个归档必须同时保存：源路径、盘点日期、文件数、总字节数、SHA-256、上游来源、许可证、转换脚本及是否包含train/dev/test。
