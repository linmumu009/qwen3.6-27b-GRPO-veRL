"""Reconcile SC source dependencies without rewriting frozen adjudications."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ledger_path=Path('CPT_resources/llin-stage-a-20260920-03/ledger.private.json')
    previous_path=Path('docs/cpt_structured_candidate_gate_20260922.safe.json')
    snapshot=Path('CPT_resources/llin-sc-source-readiness-20260922-01/web_reads.private.json')
    previous=read(previous_path)
    assert sha(ledger_path)==previous['inputs_sha256'][str(ledger_path)]
    archive=read(snapshot)
    assert archive['retrieved_at']=='2026-09-22' and len(archive['reads'])==3
    # Archive completeness checks; these do not constitute semantic validation.
    text='\n'.join(archive['reads'])
    for marker in ['News-320.aspx','07-%20Public%20Transportation.pdf','PageNotFound.aspx',
                   'supply of the','international transport','Failed to fetch']:
        assert marker in text, marker
    rows=[r['review'] for r in read(ledger_path) if r['review']['dataset']=='SC-bench-knowledge']
    assert len(rows)==32 and Counter(r['status'] for r in rows)==dict(A=1,B=5,C=22,D=4)
    selected=[r for r in rows if r['status'] in ('B','D')]
    assert len(selected)==9
    additions={
        'tax_jurisdiction_date':('public_rules_plus_applicability',
            '取得题目对应期间、辖区及交易性质的依据；区分货物/车辆销售、运输服务、标准税率、零税率和免税，不能用一般税率替代全部条件。'),
        'national_tariff_scope':('jurisdiction_adjudication',
            '补齐申报国家、程序及版本，再核验国际基础码与本国扩展码；通用长度资料不能确立未指定辖区题的唯一标签。'),
        'order_parcel_cardinality':('business_mapping',
            '提供订单—包裹实体映射和取消事件计数口径，包括拆单、合单及一对多处理；算术不构成映射依据。'),
        'ddp_adoption_conditions':('business_context_and_causal_evidence',
            '给出地区、时期、交易门槛及采用动因的原始论述；条款责任定义或税制事实不能单独证明采用率原因。'),
        'customs_fallback_policy':('business_policy',
            '提供兜底策略、触发条件和风险处置规范；编码标准无法决定本系统是否取消兜底。'),
        'customs_architecture_tradeoff':('system_constraints_and_adjudication',
            '提供系统约束、评价目标及允许排除其他权衡的原始依据；一般架构知识不能保证单选唯一。'),
        'code_namespace':('template_and_dictionary',
            '提供目标设计工具、模板版本及代码字典或可核验跨模板例证；数字应用标识符不等于单字母处理代码。'),
        'inbound_count_attribution':('metric_dictionary_and_diagnostic_policy',
            '提供入库指标分母、渠道维度、时间口径及异常排查依据；相关指标不证明唯一归因。'),
    }
    assert {r['capability_group'] for r in selected}==set(additions)
    groups=defaultdict(list)
    for row in rows:
        if row['status']!='A':groups[row['capability_group']].append(row)
    dependencies=[]
    for key,members in sorted(groups.items()):
        route,requirement=additions.get(key,('versioned_business_source',members[0]['note']))
        dependencies.append(dict(capability_group=key,items=len(members),
            retained_statuses=dict(Counter(r['status'] for r in members)),
            evidence_route=route,acceptance_requirement=requirement,
            shared_requirements=['来源身份与版本可核验','适用期间和业务范围可对齐',
                '必要条件及全部标签有来源支持或明确进入歧义裁定','不能从正式答案解释反写独立来源'],
            next_trigger='Matching original source or explicit scope/adjudication evidence arrives',
            ready_for_authoring=False))
    report=dict(date='2026-09-22',scope='Nine B/D cases re-reviewed; all unresolved SC dependencies inventoried from prior metadata',
        input_sha256={str(p):sha(p) for p in (ledger_path,previous_path,snapshot)},
        total_sc_errors=32,retained_status_counts=dict(Counter(r['status'] for r in rows)),
        deeply_reviewed_items=9,deeply_reviewed_capability_groups=8,
        unresolved_items=sum(x['items'] for x in dependencies),unresolved_groups=len(dependencies),
        dependencies=dependencies,source_upgrades=0,source_qualified_sc_groups=1,
        stage_b_qualified_groups=0,stage_c_passed=False,
        web_evidence=[
            dict(id='sa_rate_history',url='https://zatca.gov.sa/en/MediaCenter/News/Pages/News-320.aspx',
                 status='body_read',locator='body announcement, effective 2020-07-01',
                 use='Historical rate change; does not establish the benchmark transaction date or treatment'),
            dict(id='uae_transport_scope',url='https://tax.gov.ae/DataFolder/Files/Pdf/AR/07-%20Public%20Transportation.pdf',
                 status='pdf_text_read',locator='VATP007 physical pages 1-2 and 6-7',
                 use='Vehicle supply and transport services have distinct scope; not a complete current-law audit'),
            dict(id='sa_professional_guide',url='https://www.zatca.gov.sa/en/HelpCenter/guidelines/Documents/VAT_Professional_Services_Guideline_English.pdf',
                 status='redirect_to_not_found',use='Search snippet excluded from qualification'),
            dict(id='gs1_label',url='https://www.gs1.org/standards/gs1-logistic-label-guideline/1-3',
                 status='body_fetch_failed_403',use='Search snippet only; no supported mapping to target templates')],
        new_model_calls=0,new_training_runs=0,new_tasks=0,
        original_scores_and_adjudications_unchanged=True,semantic_review_independent=False,
        decision='Stop unchanged public-source searches; resume on matching source/context or explicitly registered scope change')
    out=Path('docs/cpt_sc_source_readiness_20260922.safe.json')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('deeply_reviewed_items','unresolved_items','unresolved_groups','source_upgrades','stage_c_passed')},ensure_ascii=False))


if __name__=='__main__':
    main()
