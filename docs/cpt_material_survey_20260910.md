# 物流 CPT 训练材料调查与收集

日期：2026-09-10。范围：按现有两套评测的知识结构调查选材，公开资料归档、来源与许可登记；未生成新训练集，未改变正在执行的两臂对照。

## 调查结论

现有材料的领域重心与评测需求存在明显差异，值得把下一轮从“单书反复曝光”改为“多来源、有结构的知识覆盖”。但当前证据不足以断言语料选错是唯一原因：训练规模、知识提取方式、提示形式和评测本身的质量也会影响分数。

现有教材并非没有仓储和运输知识。它的优势是物流与配送管理骨架；评测还大量涉及精确运输定义、运输系统模型、仓库作业算法，以及具体订单系统的状态、异常、数据一致性和跨境业务口径。一本336,586正文token的教材很难同时提供这些深度与业务上下文。固定8遍主曲线最高只净增3题、末遍净减5题，也不支持继续单纯增加同一本书的曝光次数。

## 从评测中看到的优先级

按本轮Step120、原提示、三轮严格多数票重新核算，1,672题中303题错误。[逐类别安全统计](cpt_material_gap_audit_20260910.safe.json)不包含题面、选项或答案。

| 类别 | 题数 | 错题数 | 错误率 | 需要重点补充的材料 |
|---|---:|---:|---:|---|
| Logistika物料搬运 | 696 | 125 | 18.0% | 运输/设备术语、搬运与集装化、作业模型 |
| Logistika运输 | 404 | 71 | 17.6% | 港口航道、交通流、排队、运输统计、网络与运筹 |
| Logistika仓储 | 202 | 51 | 25.2% | 拣选、分区/批量、库位与补货、WMS流程 |
| Logistika供应链 | 102 | 13 | 12.7% | 多级库存、需求预测、协同与系统模型 |
| Logistika采购 | 42 | 6 | 14.3% | 采购流程、供应商与交付条件 |
| SC履约 | 79 | 13 | 16.5% | 订单状态、出入库、撤销与清关规则 |
| SC物流协同 | 73 | 12 | 16.4% | 事件跟踪、异常升级、消息与跨系统一致性 |
| SC其他类别合计 | 74 | 12 | 16.2% | 采购口径、库存调查、计划及财务运营 |

前三类贡献247/303=81.5%的错误，但标签不等于准确知识归因。例如物料搬运标签下也出现运输统计定义。调查对约40道错题做了分层间隔抽样主题阅读，尚未完成303道错题逐题证据归因。

数据质量方面：305道题使用269项大候选池，其中92题错误；正则筛出8道图表引用题，其中3题错误。后者只是筛查，不等于8题都缺图或都应删除。另有抽样可见的残缺文本和题干/内容错位。必须保留官方总分与问题子集分数，不能用删难题制造提升，也不能把这些错误全部记为知识缺失。[数据集作者亦注明类别、转录和答案可能存在问题。](https://huggingface.co/datasets/berdymurad/LogistikaBench)

## 已定位的关键材料

本轮实际尝试483个URL，成功归档470个不同文件，共296.45 MB：113份PDF、357个HTML页面。PDF共5,357页，全部能被解析，提取约115.7万空白分隔词（**不是训练token数**）；179页低文本量可能包含封面、空白、图片或扫描页，需要分辨。已做仓储教材页面渲染抽检，尚未完成全量图表/公式质量审核。另有120个Logistics Operational Guide页面（含入口）及50篇ERPNext业务文档。13个下载失败均保留原因。内容尚未做近重复去除，因此不能把文件字节数视为有效训练规模。

完整收集数、下载失败和各来源文件数见[归档统计](cpt_material_collection_20260910.safe.json)；逐URL、时间、哈希及状态见[下载清单](cpt_material_download_manifest_20260910.safe.json)。文件数包含课程资源页面和目录，不能解释成同等数量的教材。

| 材料 | 用途 | 来源关系及使用状态 |
|---|---|---|
| [Warehouse & Distribution Science](https://warehouse-science.com/book/index.html) | 仓库流动模型、布局、拣选和作业优化 | 337页PDF已下载；允许教育复制，不能视为已有商业模型训练授权 |
| [MIT Logistics Systems](https://ocw.mit.edu/courses/esd-260j-logistics-systems-fall-2006/pages/lecture-notes/) | EOQ扩展、安全库存、网络设计、运输采购 | 成套课件；与评测MIT来源有机构/概念接近，需查重；NC限制 |
| [MIT Logistics and Supply Chain Management](https://ocw.mit.edu/courses/esd-273j-logistics-and-supply-chain-management-fall-2009/download/) | 多期库存、随机需求和合同 | 课件和资源入口；NC限制 |
| [MIT Transportation Systems](https://ocw.mit.edu/courses/1-221j-transportation-systems-fall-2004/resources/lecture-notes/)及其他运输课程 | 交通系统、流量、排队与模式选择 | 同时收集Transportation Flow Systems、Introduction to Transportation Systems；NC限制 |
| [MIT Operations Management](https://ocw.mit.edu/courses/15-760b-introduction-to-operations-management-spring-2004/resources/lecture-notes/)和Systems Optimization | 产能、质量、生产、运输网络优化 | 补公式、假设与推导，必须保留图表关系；NC限制 |
| [Inventory Analytics](https://www.openbookpublishers.com/books/10.11647/obp.0252) | 库存、预测、随机/多级库存、Python例子 | 出版社确认CC BY 4.0；出版社和OAPEN下载未成功，保留链接与失败记录 |
| [Supply Chain Management，Davor Dujak](https://vscht.futurebooks.cz/book/supply-chain-management/) | 供应链概念与运输成本体系 | 页面内容已获取；来源标示CC BY-SA，需按具体书籍内容保留署名 |
| [Logistics Operational Guide](https://log.logcluster.org/en) | 采购、供应商、运输、库存、仓储、通关、KPI、冷链 | 归档英文操作指南各节；人道物流流程需标注适用背景，许可仍待逐来源核验 |
| [ERPNext操作文档](https://docs.frappe.io/erpnext/delivery-note) | 采购收货、销售订单、拣货、出库、退货、库存核对 | 按真实系统流程补SC类缺口；文档为CC BY-SA，系统规则不能当所有企业通用规范 |
| [GS1 EPCIS/CBV 2.0](https://ref.gs1.org/standards/epcis/2.0.0/artefacts) | 物流事件、追踪、业务步骤和状态词汇 | 两份标准PDF已下载；公开标准不等于自动获准训练，许可待审核 |
| [GS1 Web Vocabulary](https://ref.gs1.org/voc/) | 产品、主体、交付及物流属性 | 页面已归档；本体许可与标准正文许可须分别核实 |
| [WCO HS概览与FAQ](https://www.wcoomd.org/en/topics/nomenclature/overview/what-is-the-harmonized-system.aspx) | 商品分类体系与申报层级 | 必须区分国际基础编码和地区申报扩展；时效规则需独立维护 |
| [ICC Incoterms 2020](https://iccwbo.org/business-solutions/incoterms-rules/incoterms-2020/) | 责任、风险、费用和交付条件 | 官方概览与checklist已收集；完整规则正文仍属授权材料 |
| [中国跨境电商监管公告194号](https://www.mofcom.gov.cn/zfxxgk/gkml/art/2021/art_1e97a9dbe17b45398e962927596b16f9.html) | 交易、支付、物流信息及通关流程 | 官方历史版本已归档；入训练前需核查现行修订、日期和适用范围 |
| [AWS事务发件箱与Saga](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/saga-patterns.html) | 订单撤销、补偿、幂等、跨系统一致性 | 补物流系统工程知识；不是专属供应链规范，比例应有限 |
| [RabbitMQ优先级文档](https://www.rabbitmq.com/docs/priority) | 消息与消费者优先级、版本差异 | 官方实例材料，不把某产品行为写成通用标准 |
| [World Bank Port Reform Toolkit](https://openknowledge.worldbank.org/bitstreams/e4efb5b5-2e53-45d1-bfe0-885da9d1e71a/download) | 港口运营与治理 | 本次只获取引言卷，不能称为完整工具包 |

此外登记UNCTAD海运报告、CTU装箱规范、中文物流术语标准、OpenStax信息系统与供应链材料，以及其他开放物流和风险管理教材。被403/405/429或超时拒绝的下载保留失败状态，不绕过访问限制、不冒充已获取。

## 评测同源材料必须单独标记

LogistikaBench公开列出五类来源：MITx SCM Key Concepts、Ivanov等Global Supply Chain and Operations Management、Ports and Waterways、Challenges and Opportunities in International Business、2019 Glossary for Transport Statistics。此次已定位全部五类入口；MITx和Ports and Waterways取得PDF，Ivanov取得作者资源页，国际商务取得目录页，运输词汇官方PDF下载受限。不同入口的版本日期可能不同，不能直接声称与原出题版本逐字一致。

这些材料对理解覆盖范围非常重要，但如果进入训练，LogistikaBench只能是开发集，分数上涨可能包含同源知识记忆。即使不直接喂原题，用其错误与来源选材也已经属于评测驱动开发。本次没有将原题、答案或其改写写入候选训练集，且没有启动新材料训练。后续需要按**整本书/机构/文档家族**隔离一套未用于选材的测试来源，并审核答案歧义。

## 许可与制数优先级

1. 优先处理明确开放的教材正文与技术文档，例如CC BY的Inventory Analytics、CC BY-SA的ERPNext文档；继续保留署名、版本、例外内容和衍生使用条件。下载失败的资料不能计入可用token预算。[出版社许可](https://www.openbookpublishers.com/books/10.11647/obp.0252)、[Frappe文档许可](https://docs.frappe.io/legal/others/license-and-trademark)。
2. MIT OCW和Ports and Waterways不是无条件商业训练材料。MIT现行页面明确要求AI训练遵守非商业及ShareAlike条件；本轮仅做研究归档，不把它们直接加入商业模型训练。[MIT条款](https://ocw.mit.edu/pages/privacy-and-terms-of-use/)、[Delft许可](https://books.open.tudelft.nl/home/catalog/book/204)。
3. 更贴近业务的新增资料是自有SOP、OMS/WMS/TMS状态字典、异常处理矩阵、数据指标口径、系统事件接口、国家/地区生效规则。公开教材无法可靠补出某公司的“应升级给哪个团队”“某指标分母怎么算”等内部事实。已向用户询问可用目录，未假定能够访问这些文档。
4. 收集量与训练量分开。PDF已做逐页文本提取与低文本页计数，仍需去页眉导航、保留定义—条件—公式—图表关系、按整文档去重与近重复审核；表格/公式抽取失真和扫描页应进入人工/OCR复核。当前抽取文本仅供检查，不能直接作为训练样本。
5. 建议下一轮先比较经许可审核后的多源混合语料与当前教材，在相同token预算和起点下跑小规模对照，再决定扩大。暂不按页面数预设20M/50M token，也不把这一建议视为已授权启动第三个训练实验。

## 本地归档

所有公开正文放在仓库工作区的`CPT_resources/material_survey_20260910*`目录，未加入Git。可浏览索引位于`CPT_resources/material_survey_20260910_index/INDEX.md`；PDF检查文本位于该目录的`text_for_review`。Git仅保存源目录、下载元数据、安全统计、调查报告和可复用收集程序。

这是一份广覆盖候选资料库，而非已完成许可、语义去重、污染检查和格式审核的训练集。
