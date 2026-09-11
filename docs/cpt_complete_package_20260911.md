# CPT材料整合交付（2026-09-11）

本次交付一个可供后续CPT使用的完整文件包：原13份材料的4,707条正文，加上205条逐条选定核对的补充知识，共4,912条、1,868,012正文token。没有启动训练。这里的“完整”指训练文件、来源、知识文档、处置索引和验证材料齐备，不表示300道错题的知识要求已经穷尽。

## 内容与范围

- 原13份材料全部保留，无评测同源排除，无重复采样，旧训练文件逐字节不变。
- 补充知识由上一版110条扩展至205条，正文23,131 token。新增重点包括铁路/道路/内河统计定义的限定条件、航空载重与代码、包装层级与运输参与者、库位和拣选关系、船舶尺度、低水位影响、成本与指标、全球价值链、排放、隐私及会计口径。
- 61个可识别主题均有选定材料；第62个主题因题干缺损而无法定位。主题命中不能作为逐题完整覆盖率。
- 补充材料保留标题、正文、适用范围、来源和指纹。121个来源记录包含49个百科记录；页面/术语记录数不是独立来源数量。百科同源表述和译述不冒充独立佐证。
- 8个补充正文已出现在旧训练文本中，本次保留其恢复标题和范围后的知识单元；没有为了增加规模反复复制。合并使用全文精确去重，不宣称完成了全量语义去重。

公开补充依据包括[CTU Code术语](https://wiki.unece.org/spaces/TransportSustainableCTUCode/pages/23101840/Chapter%2B2.%2BDefinitions)、[FAA载重手册](https://www.faa.gov/sites/faa.gov/files/2023-09/Weight_Balance_Handbook.pdf)、[UNCTAD 2020报告](https://unctad.org/system/files/official-document/wir2020_en.pdf)、[运输方式与合同关系](https://transportgeography.org/contents/chapter5/intermodal-transportation-containerization/)、[IAS 2](https://www.ifrs.org/issued-standards/list-of-standards/ias-2-inventories/)及[IATA固定版本代码表](https://github.com/IATA-Cargo/ONE-Record/blob/ad5f40d539b29692881672cf91878a4d2dbe59c2/2024-12-standard/Data-Model/IATA-1R-CL-Ontology.ttl)。仅纳入核对后的事实摘要或选定片段，没有将来源全文自动标为审核通过。法规、历史尺度和产品流程带有版本或适用范围。

## 300道错题如何处置

全部300题有私有处置记录，不改题目标签、不把题目与正确选项拼进新增训练正文。旧教材与评测来源重合仍然存在，因此后续原评测集成绩只能解释为有针对性的学习实验，不能作为独立泛化能力证明。

82题继续登记原始资料或口径问题。按优先级互斥展示：32题缺少题干或原图、27题缺少企业语境、9题有冲突事项、14题需限定版本或定义维度；原始标志允许一题有多项问题。用户不知道内部手册位置，未据此编造企业规则。冲突事项不是9题都已被证明为错标。

其余218题中，77题有本轮选定断言的逐题关联，141题目前只有主题材料关联。这两类均未宣称完整答案覆盖；前一版已有的定义也没有通过自动关联升级成逐题证明。逐题索引适合继续核实和观察训练效果，不是“300题已解决”清单。

## 本地交付位置

根目录：`CPT_resources/llin-knowledge-complete-20260911/`（私有，不入Git）。

| 文件 | 用途 |
| --- | --- |
| `START_HERE.md` | 交付入口、限制、训练接入说明 |
| `release/train.parquet`、`release/train.jsonl` | 全13份材料与补充知识的合并训练正文 |
| `core/核心知识文档.md` | 可阅读的205条知识及来源 |
| `core/reviewed_units.private.jsonl`、`core/sources.private.jsonl` | 核对记录与来源指纹 |
| `逐题材料处置索引.md`、`case_disposition.private.jsonl` | 300题处置及知识关联 |
| `topic_disposition.private.jsonl` | 62主题索引 |
| `release/baseline_provenance.private.jsonl` | 原书、页码、原始块关联 |
| `release/verification.safe.json`、`release/tensor_contract.safe.json` | 文件与读取验证 |

## 验证与训练接入

全量重新分词，最大含结束标记长度4,085，低于4,096；没有截断。JSONL/Parquet往返、文本及来源指纹、13来源保留检查通过。10项定向测试通过。全4,912条使用现有训练张量函数的实际函数体、真实CPU PyTorch和固定分词器检查：单个结束标记、结束标记目标参与监督、跨样本回绕屏蔽均通过。

本机缺少完整veRL依赖，CPU检查采用平坦token列表适配器，不冒充完整训练容器验证。启动训练前仍须在实际容器运行既有数据门禁，并按新文件指纹、条数和token预算更新运行配置。旧单遍脚本固定了旧预算，不能直接当作新包启动命令。模型命名继续使用`llin`开头。

复现命令（只构建与验证，不启动训练）：

```powershell
python scripts/build_cpt_complete_package.py --base CPT_resources/llin-logists-v5-full-20260911 --core CPT_resources/llin-knowledge-complete-20260911/core --tokenizer CPT_resources/corpus_build_inputs_20260910/tokenizer.json --output CPT_resources/llin-knowledge-complete-20260911/release
python scripts/check_cpt_local_tensor_contract.py --parquet CPT_resources/llin-knowledge-complete-20260911/release/train.parquet --tokenizer CPT_resources/corpus_build_inputs_20260910/tokenizer.json --output CPT_resources/llin-knowledge-complete-20260911/release/tensor_contract.safe.json
```

审核结论：达到本地CPT文件交付条件；新增选定断言有依据。旧1.8M token教材没有在本轮被逐句人工重新审校，不能保证每句话都无误。现有证据不足以宣布300题知识需求完全补齐。后续应保留缺口账本，而不是为了“覆盖率100%”补造答案。
