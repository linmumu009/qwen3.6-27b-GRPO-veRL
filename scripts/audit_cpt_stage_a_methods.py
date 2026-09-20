"""Review the registered 19 forecasting, inventory and warehouse cases.

Emit a compact delta and summary; full benchmark records stay private.
"""
import copy
import json
from pathlib import Path

import fitz

from audit_cpt_stage_a import audit, read, write
from audit_cpt_transfer import require, sha

PRIVATE = Path('CPT_resources/llin-stage-a-20260920-03')
MIT_URL = 'https://ocw.mit.edu/courses/1-221j-transportation-systems-fall-2004/85c6999b30a667b59613fb41621d81e7_trans_sys_chap10.pdf'
TARGETS = {
    142: 'MRP 定义与提前期资料未给出当前时间桶的专用标签，维持部分依据。',
    153: '找到一般仿真定义，尚未找到覆盖计算资源、求解时间和最优决策局限的完整原文。',
    165: '现有延迟差异化来源讨论产品阶段，不足以证明题目采用的运输延期策略。',
    168: '产品延期与货运需求紧迫性不是同一命题，需要运输策略原始来源。',
    169: 'MIT 原课件第4、10、13页共同支持模型/框架、预测用途和透明假设的完整集合；已逐页视觉核验。',
    171: 'SAP 检索结果提供波动分类线索，但原网页未返回可审正文，不能只凭摘要放行。',
    177: '官方集合含方法混用疑点；误差指标定义无法证明所述专家与统计组合命题，维持待裁定。',
    203: 'MIT 原页支持库存位置触发及检查周期差异，但题面未明确库存口径、安全系数与比较前提，保持部分依据。',
    209: '一般产品延迟差异化材料不能代替残余运输需求的延期规则。',
    217: '原作者预测教材确认百分比指标的跨单位比较用途及零值限制；未证明其应在其他指标不一致时普遍优先。',
    218: '现有资料未覆盖五个原因及局部/全国不确定性比较的全部条件，未升级。',
    227: '仓储原书支持分工与区域组织原则，但没有核实题目特定订单分解情境的完整定义。',
    237: '原书将80/20作为经验规律；按使用价值和库存水平交叉判断仍缺完整同口径依据。',
    238: '原书支持合并订单以减少行走，但未核实题目采用的子订单、分区与设备分类口径。',
    239: '已有批量/分区来源未区分该题下游分拣与边拣边分的完整条件。',
    240: '原书给出一次行程拣多订单的批量说明，尚未确认子订单集合的专用术语表。',
    246: '与另一术语呈现重复，所需专用术语表仍未核实，不增加独立知识数。',
    261: '分组目的地可能同时涉及批量与波次组织，来源未支持该利弊标签唯一性。',
    262: '与另一处理方式题重复，仍需分拣时点和分区组合的完整原文。',
}


def run():
    parent = Path('docs/cpt_stage_a_followup_review_20260920.safe.json')
    review = copy.deepcopy(read(parent))
    paths = {'mit_models.private.json': 'fb21b555dff31ec34f06b94540897ed7e13c61d39cd122db5db238260151c04c',
             'pages.private.json': '210d2b5dfec3d6eee981753112e8ed2bdda52d6c8b254c015f9be761686c59b9',
             'web_read.private.json': 'ad4ffa3730cce017fb28c91eee3781a073491a287b9f03a3604220672135c795'}
    for name, expected in paths.items():
        require(sha(PRIVATE / name) == expected, 'source archive changed: ' + name)
    # Confirm page extraction against original PDFs, rather than trusting extracts.
    pages = read(PRIVATE / 'pages.private.json')
    for row in pages:
        source = Path(row['source'])
        require(sha(source) == row['source_sha256'], 'PDF changed')
        with fitz.open(source) as doc:
            require(doc[row['page']-1].get_text() == row['text'], 'page extraction changed')
    snapshot = dict(id='mit_models_primary', path=str(PRIVATE / 'mit_models.private.json').replace('\\','/'),
                    sha256=paths['mit_models.private.json'], kind='local_pdf_page_extraction')
    review['primary_source_snapshots'].append(snapshot)
    delta = []
    for i,note in TARGETS.items():
        row = review['items'][i]
        before = row['status']
        row['methods_review_note'] = note
        if i in [227,238,239,240,246,261,262]:
            row.update(training_rows=[0], training_coverage='related_or_partial_supervision')
            row['methods_review_note'] += ' 已检查实际训练第0行的拣选组织权衡任务，只作为相邻监督，不视为专用分类已覆盖。'
        if i == 169:
            require(before == 'B', 'unexpected prior decision')
            row.update(status='A', input_sufficiency='sufficient_in_reviewed_scope', label_uniqueness='supported_by_review',
                       official_url=MIT_URL, official_locator='Chapter 10 PDF physical pages 4, 10, 13',
                       primary_snapshot_id='mit_models_primary', primary_required_spans=['mathematical', 'qualitative', 'assumptions'],
                       note=note)
        delta.append(dict(item_hash=row['item_hash'], review_index_zero_based=i, before=before, after=row['status'],
                          capability_group=row['capability_group'], note=row['methods_review_note'], training_coverage=row['training_coverage']))
    write(PRIVATE / 'review.private.json', review)
    result = audit(Path('.'), PRIVATE / 'review.private.json', PRIVATE)
    write(PRIVATE / 'result.private.json', result)
    summary = dict(date='2026-09-20', parent_review_sha256=sha(parent), reviewed=len(delta),
                   state_changes=sum(x['before'] != x['after'] for x in delta), items=delta,
                   datasets=result['datasets'], source_archives=[dict(file=n,sha256=h) for n,h in paths.items()],
                   pdf_sources=[dict(path=r['source'],sha256=r['source_sha256'],page=r['page']) for r in pages],
                   primary_reference=dict(url=MIT_URL,pages=[4,10,13],visual_review_completed=True),
                   model_calls=0,training_runs=0,semantic_review_independent=False,
                   full_result_private_sha256=sha(PRIVATE/'result.private.json'))
    write(Path('docs/cpt_stage_a_methods_20260920.safe.json'),summary)
    print(json.dumps(dict(reviewed=19,state_changes=1,datasets=result['datasets']),ensure_ascii=False,indent=2))
    return summary


if __name__ == '__main__':
    run()
