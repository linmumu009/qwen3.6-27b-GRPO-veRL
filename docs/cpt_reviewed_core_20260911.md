# CPT核心补充语料v1（2026-09-11）

从既有候选中完成110个选定知识单元的整理，生成本地可分词的CPT补充集：9,290个正文token、9,400个含EOS的序列token，最长411。51个主题有部分已核对知识，11个主题仍无核心单元。这是一个小规模核心补充版本，不是全部300题、全部候选或全部教材的最终审核完成声明。

本地目录：`CPT_resources/llin-knowledge-core-v1-20260911/`。阅读入口为`核心知识文档.md`和`START_HERE.md`；训练格式为`train.jsonl`、`train.parquet`；另有逐单元证据、来源、522条候选处置记录和62主题缺口表。正文和私有映射不提交仓库。

## 本轮实质处理

- 按选定断言审核百科导言，保留原文版本、署名和许可说明。没有把文章全文或词条命中自动宣布为可训练。
- 恢复教材术语标题与定义的联系，保留2019版铁路、道路车辆、船舶、管道及事故统计的条件与排除项；保留仓库类型、拣选分工和循环授信的完整教学段落。
- 将MAE、MSE、MAPE和加权平均等公式整理为完整文字表达，明确变量、分母和量纲。库位排序的步骤连同两个排序维度一起保存。
- 新核对[IATA集装器定义](https://www.iata.org/en/programs/cargo/cargo-operations/unit-load-devices/)、[WCO HS](https://www.wcoomd.org/en/topics/nomenclature/overview/what-is-the-harmonized-system.aspx)、[日本统计代码](https://www.customs.go.jp/toukei/sankou/code/code_e.htm)和[EPAL 1尺寸](https://www.epal-pallets.org/eu-en/load-carriers/epal-euro-pallet)，只写入短事实归纳，不复制整页。
- 原文与编辑归纳分开标识。翻译、改写及同一书的不同页不算独立来源；未通过机械重复放大数据量。

全部13份教材和522条研究候选仍保留。本版从63条候选中选取了部分断言，其余459条未获得本版批准，不能把这459条称为错误或已完成语义审核。选中段落之外的文章内容也仍待审核。已知错误标签、缺图和企业配置问题继续使用原问题清单；没有改评测标签或写入已知错误知识。主题对应关系不证明题目已经解决。

## 使用边界与下一步

这份约9千token的小集适合作为后续语料的精准补充，不能视为新的完整预训练语料。尚需继续补全订单时间窗、延迟策略、资本存量、存货计价、航空力学、排放和历史船型等材料；缺图及企业流程需要相应原资料。已覆盖主题也可能缺少关系、条件或第二个独立来源。

正文已完成本地格式、分词和长度检查；未完成训练端加载验证、混合权重设计或新的CPT运行。`training_ready`仅表示选定正文可进入后续数据准备流程，不是整轮实验已可直接启动。原13份材料训练文件SHA256仍为`898dada4de8c7274ee0b18bf03f685998bd9cf25739af2502f1eed67c3cb9733`。

## 验证与复现

`scripts/build_cpt_reviewed_core.py`读取本地审核清单，验证证据指纹和摘录原文归属；不自动从候选授予审核状态。它保留短定义、完整标题和适用条件，对超过4096的单元报错，绝不截断。每条追加一个EOS，并验证JSONL与Parquet逐条一致。审核状态由编辑录入，程序检查状态与证据完整性，不能替代语义审核。

5项定向测试覆盖短定义与排除项保留、未审候选拒绝、静默删改摘录拒绝、源文变化拒绝以及超长拒绝。全量110条构建和文件读回通过。交付指纹见配套安全JSON。

```powershell
python scripts/build_cpt_reviewed_core.py --root CPT_resources/llin-knowledge-core-v1-20260911 --tokenizer CPT_resources/corpus_build_inputs_20260910/tokenizer.json
python -m unittest discover -s tests -p test_cpt_reviewed_core.py
```
