# 可持续更新的 CPT 语料：v1.0.2（2026-09-10）

首批可用包已完成清洗、去重、评测文本隔离、来源追踪及真实加载检查。它是 **ERPNext 业务流程种子语料**，不能代表整个物流领域，更不能以下载字节数冒充有效训练规模。没有启动新训练，也没有替换正在执行的两臂实验数据。

## 交付位置与规模

- 本地：`CPT_resources/cpt_corpus_v1_0_2_20260910/`。
- 现有训练容器：`/workspace/llin-verl-grpo/runs/cpt-corpus-v1.0.2-20260910/`，已复制正式训练/验证文件、署名和清单；混合许可候选正文只保留本地。
- 正式文件：`train.parquet`、`validation.parquet`，同时提供同内容 JSONL、`ATTRIBUTION.md`、`manifest.safe.json`。
- 本地审计文件：`sources.jsonl`（来源）、`staging_documents.jsonl`（提取后的候选正文）、`quarantine.jsonl`（排除原因）。候选及隔离文件不可整体作为训练输入。

| 划分 | 文档/记录 | 正文 token | 含 EOS 的序列 token |
|---|---:|---:|---:|
| train | 47 | 32,468 | 32,515 |
| validation | 3 | 3,515 | 3,518 |
| 合计 | 50 | 35,983 | 36,033 |

每条112–2,637个序列token。当前每篇可完整放入一条记录，无跨文档拼接。验证集是固定文档隔离的开发 NLL 集，仅3篇，不能用于宣布泛化提升。

## 清洗与准确性边界

1. 核验470个已下载文件的原始 SHA-256；提取11,672个候选块，保留原始文件，任何变换都能追溯。
2. ERP 文档限定正文区域，清除导航、脚本、按钮等；修正跳级标题解析，保留段落顺序、数字、否定、单位、版本条件和操作前提。正文添加 ERPNext 来源前缀，避免把产品规则写成行业普遍规律。
3. 简单表格按行列保存，代码保留缩进；复杂表格、图示依赖、异常字符和疑似凭据示例进入复核。正式包排除15个相关阅读导航小节、4个重复小节、3个图示依赖小节。清洗不调用模型、不进行语义改写。
4. 先按文档做精确及近重复筛查，再做完整小节去重；不单独删除某个条件句。固定文档族哈希分组；本次无重复文档，训练/验证之间最大5词集合 Jaccard 为0.002417。
5. 本地只使用1,672道评测题及选项的哈希指纹：13词连续窗口和至少8词完整文本精确匹配；正式包未命中。评测同源书籍独立隔离。该筛查不能证明不存在释义泄漏，按该评测选材之后，原评测只能视为开发评测。
6. PDF已保留页级文本和重复页眉页脚处理结果，但全部仍标记版面/公式复核，不将未经核验的公式与图表转写投放正式集。170个页面未找到可靠正文区域，明确记录而不回退抓取整页导航。

技术检查保证格式、来源和变换可核验；没有完成逐条专家事实审校，也没有证明这些文档适配企业内部 SOP。供应商产品说明、地区税务和历史版本行为须按其原始语境理解。

## 许可与候选层

470个来源文件的分层：50个正式 CC BY-SA 文档、270个仅非商业研究候选、141个许可待核实来源、5个评测同源来源、4个元数据页面。权利分层互斥；质量排除原因可重叠，不应直接相加。

ERP 文档许可依据 [Frappe 官方许可政策](https://docs.frappe.io/legal/others/license-and-trademark)，页面明确 CC BY-SA，但未给出具体版本。署名、来源链接、变换说明及相同方式共享要求随包保存；品牌图案不在正文包内。MIT 非商业限制及其他来源详情见[材料调查](cpt_material_survey_20260910.md)。公开可下载不等于已经具备训练许可。

## 接入现有训练器

使用已有 `scripts/qwen36_causal_lm_dataset.py` 的 `Qwen36CausalLMDataset`，设置 `text_key=text`、`max_length=4096`、`truncation=error`、`pad_mode=no_padding`，分别指定正式 train/validation Parquet。不得套用聊天模板，也不要手工附加 EOS：实际 Step120 分词器由加载器追加一次 EOS，ID 为248046。`token_count` 包含这一个 EOS，`content_tokens` 不包含。

此包不应通过大量重复或虚构改写凑足预训练预算。新增物流材料应先补运输、仓储、库存公式和跨企业履约知识，再单独确定与现有语料的混合比例。此次不改变训练超参数或运行队列。

## 持续更新流程

原始输入、分词器和评测指纹保留在本地；Git仅提交处理程序与安全汇总。运行环境版本见验证报告。构建目录不可覆盖，新增材料使用新版本目录，并指定上个版本：

```powershell
python scripts/build_cpt_corpus.py --out CPT_resources/cpt_corpus_next --tokenizer CPT_resources/corpus_build_inputs_20260910/tokenizer.json --fingerprints CPT_resources/corpus_build_inputs_20260910/benchmark_fingerprints.json.gz --previous-build CPT_resources/cpt_corpus_v1_0_2_20260910
python scripts/verify_cpt_corpus.py --corpus CPT_resources/cpt_corpus_next --tokenizer-dir CPT_resources/corpus_build_inputs_20260910 --fingerprints CPT_resources/corpus_build_inputs_20260910/benchmark_fingerprints.json.gz --report CPT_resources/cpt_corpus_next/validation_report.safe.json
```

收集程序生成新的 `material_survey_*/manifest.jsonl` 后，构建器会自动读取。现有准入策略只允许已核验来源类别；新增出版物需要补充许可证据及专用解析规则，未知来源默认隔离。该过程是可重复执行的增量流程，尚未创建新的定时采集任务。

每次更新复核清单中的新增/移除/不变记录数、来源版本、隔离原因及固定文档族划分。内容修订产生新记录 ID；同一文档族不因新增材料随机换到另一划分。当前支持全量重建后的增量比较，并非仅处理变化文件的加速器。

## 验证证据

- [发布清单](cpt_corpus_manifest_20260910.safe.json)：文件哈希、来源分层、分词器及评测来源哈希。
- [独立检查](cpt_corpus_validation_20260910.safe.json)：50条全部追溯、JSONL/Parquet一致、真实分词器长度、精确污染及跨划分检查通过。
- [实际加载器检查](cpt_corpus_loader_20260910.safe.json)：现有训练容器 CPU 上读取全部50条；训练监督32,468 token、验证3,515 token，首位环绕损失屏蔽、EOS正确。未执行模型训练。
- [重复构建检查](cpt_corpus_reproducibility_20260910.safe.json)：全部8个产物哈希一致；新增0、移除0、不变50条，已有文档族划分不变。清单中的上次版本指针会随比较对象变化。
- 8项回归测试通过，覆盖正文定位、否定/数值保留、代码缩进、表格复核、标题层级、精确污染和分块边界。
