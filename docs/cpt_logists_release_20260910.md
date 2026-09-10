# CPT-LOGISTS 语料交付（2026-09-10）

本批资料已清洗到可以接入现有原文 CPT 加载器的状态，并已按用户授权复制到训练服务器。未启动新训练。正式版本为 **llin-logists-v4**；v1–v3 是检查过程中的中间产物，不用于训练。

## 规模与位置

输入为用户提供的 13 份资料：12 份 PDF 共 4,050 页，以及一个 HTML 教材压缩包。其中 7 本、2,189 页通过用户已登录的 MinerU 网页解析；按网页每文件 200 页的限制拆为 14 份，全部导出并校验了页数和连续性。其余资料使用本地文本层提取。

| 划分 | 样本 | 正文 token | 包含每条 EOS 的 token |
|---|---:|---:|---:|
| 训练 | 3,016 | 1,180,181 | 1,183,197 |
| 开发验证 | 161 | 120,066 | 120,227 |

训练覆盖 9 本资料。这里的开发验证集仅用于 CPT 的 NLL 检查，不是 SC-bench / LogistikaBench 两个外部评测集。

本地正式目录：`CPT_resources/llin-logists-cpt-v4/`。可直接接入的文件为 `train.parquet` 和 `validation.parquet`，同时提供 JSONL。`sources.jsonl`、`staging.jsonl`、`quarantine.jsonl` 和 `manifest.safe.json` 保存来源、原始页码、处理后的块、排除原因、分组及文件哈希。源文件实际位于 `CPT_resources/CPT-LOGISTS/`。

服务器：`huawei-05`，容器 `llin-verl-trainer-m05-20260730`，目录 `/workspace/llin-verl-grpo/runs/llin-logists-cpt-v4-20260910/`。已复制两个 Parquet 和清单；原书、完整审计正文及 MinerU 导出仍保存在本机。

## 各资料的训练内容

| 材料 | 训练正文 token | 开发验证正文 token |
|---|---:|---:|
| Beyond Lean: Simulation in Practice | 178,564 | 7,566 |
| Container Terminals and Automated Transport Systems | 211,960 | 0 |
| Distribution Logistics | 110,638 | 35,178 |
| Fundamentals of Global Strategy | 91,588 | 0 |
| International Trade: Theory and Policy | 159,662 | 0 |
| JSI Supply Chain Manager's Handbook | 87,754 | 6,963 |
| Traffic Flow Theory | 106,453 | 9,638 |
| Managing Supply Chain Risk | 113,969 | 57,215 |
| Warehouse & Distribution Science | 119,593 | 3,506 |

内容涵盖配送网络、仓储拣选、港口运输、交通流、风险管理、物流信息系统、贸易及仿真。材料主要为英文；没有自动翻译、生成题解或补写知识。JSI 文件名包含 2019，书内版本信息为 2020，不能用文件名代替版本判断。

## 清洗与评测隔离

- 保留来源、章节/论文分组和页码；按完整文本块组织样本，不跨书拼接。断词修复需要同书其他位置的完整词证据，保留数值、单位、否定词及数学符号。
- MinerU 表格保留 HTML 单元格及合并表头，公式保留 LaTeX。图片/图表依赖、代码版式、异常括号公式、超长块及过短上下文进入隔离区，不自动重写。无法提取的内容和具体原因可追溯。
- 去除页眉页脚、目录、参考文献等，并做全文块精确去重和 5 词 shingle Jaccard ≥0.85 的近似去重（适用 ≥40 词的块）。开发验证按完整章节、论文或整本书分组，不能随机拆相邻片段。Distribution Logistics 的论文起始页经过标题核对，Traffic Flow Theory 使用原 PDF 书签确定章节。
- **Ports and Waterways、MITx SCM Key Concepts、Eurostat Transport Statistics Glossary、Challenges and Opportunities in International Business** 四份评测同源材料均保留在审计档案，正式训练及开发验证均排除。
- 使用既有两个评测集共 1,672 个案例的整段及 13 词窗口哈希检查；正式样本无命中。没有把评测题干或答案转换为训练材料。该检查不等于语义污染完全不存在。

隔离原因会重叠，不能将清单中的各项次数相加当作独立删除量。例如 30 个公式括号异常标记、71 个代码版式标记及 2 个超长块均有记录。OCR 处理后的原始导出、正文和书籍不提交 Git；原书条款仍适用，本处理不产生新的公开再分发授权。

## 验证结果与边界

[安全结果清单](cpt_logists_release_20260910.safe.json)记录实际统计及哈希。

- 9 项针对性测试通过，包括断词、否定/单位保留、表格合并表头、公式异常、近似重复、跨文件页码连续性和跨页段落。
- 全量独立验证通过：原文件/产物哈希、JSONL 与 Parquet 一致、分词长度、特殊 token、块来源及页码、分组隔离、评测指纹和跨划分近似重复。跨划分近似重复对为 0。
- 用同一输入再次构建，8 个产物（含清单）逐字节一致。
- 原书抽样对照涵盖仓储 PDF 第 125 页公式、配送 PDF 第 22 页扫描公式、JSI PDF 第 47 页三列表格、仿真 PDF 第 221 页伪代码。后者作为图片存在，保持隔离。表格及公式抽查不是逐式专家认证；OCR 仍可能残留个别符号、空格或版式错误。
- 服务器使用实际 Step120 分词器和仓库 `Qwen36CausalLMDataset`，CPU 遍历全部 3,177 条样本通过：EOS、位置编号、损失掩码正确，零截断。未运行模型训练或两套外部评测。

这批语料比前一批 ERP 种子语料覆盖更直接、规模更大，但准备完成不能证明模型效果会提升。后续仍需控制训练预算，并与同一起点在两个外部评测集上比较。

## 接入与持续更新

现有加载器参数：`text_key=text`、`max_length=4096`、`truncation=error`、`pad_mode=no_padding`。正文不套用聊天模板，不预先添加 EOS；加载器追加实际 EOS `248046`。本批最大序列长度为 4,074，训练与验证正文分别为 1,180,181 和 120,066 token。

后续单遍 CPT 应明确冻结起点、学习率、批量与 token 预算；模型保存名称保持 `llin` 开头，例如 `llin-step120-logists-cpt-1epoch-20260910`。不要沿用旧 ERP 的 47 条/47 步预算，也不要将开发验证文件并回训练。本次只完成准备与加载验证。

复现方式（在仓库根目录，输出目录必须不存在）：

```powershell
python scripts/prepare_llin_logists_corpus.py --out CPT_resources/llin-logists-rebuild --mineru-dir CPT_resources/llin-mineru-results
python scripts/verify_llin_logists_corpus.py --root CPT_resources/llin-logists-rebuild --report CPT_resources/llin-logists-rebuild/verification.safe.json
python -m pytest tests/test_prepare_llin_logists_corpus.py -q --basetemp .pytest-logists-new-check
```

构建依赖 PyMuPDF、beautifulsoup4、lxml、tokenizers、pyarrow；复用本地冻结的分词器、评测哈希和 MinerU JSON 导出。网页的大 JSON 导出曾遇到复制缺段，最终通过编辑器分段复制、完整 JSON 解析和页数校验恢复；不拼补缺失事实。新增资料时建立新输出版本，更新来源及必要的 MinerU 分片映射，重新运行全量验证；若改变本地 PDF 提取算法，应另设 `--cache` 重新提取。新增来源还需人工检查是否与评测同源，现有四项前缀隔离并不覆盖未知的新文件。
