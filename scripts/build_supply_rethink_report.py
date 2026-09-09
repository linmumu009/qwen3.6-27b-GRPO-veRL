"""Reader report from aggregate paired evidence; no question text."""
import json
import sqlite3
from pathlib import Path


def main():
    docs=Path(__file__).resolve().parents[1]/'docs'
    a=json.loads((docs/'supply_chain_rethink_20260909.safe.json').read_text())
    for p in [a['pairs']['all'],a['pairs']['both_stable'],*a['pairs']['dataset'].values(),*a['training_value'].values(),a['general']['paired']]:
        assert p['after']-p['before']==p['gains']-p['losses']
    assert a['pairs']['all']['n']==1672 and a['pairs']['all']['gains']==12 and a['pairs']['all']['losses']==14
    assert sum(x['n'] for x in a['pairs']['category'].values())==1672
    source={'id':'audit','label':'新SFT同配置逐题配对核算','path':'docs/supply_chain_rethink_20260909.safe.json'}
    old={'id':'history','label':'已有96题干预及证据复核','path':'docs/logistics_strategy_diagnostic_20260908.md'}
    title='供应链模型：下一步该验证什么'
    blocks=[]
    def md(i,t,s=None):
        b=dict(id=i,type='markdown',body=t)
        if s:b['sourceId']=s
        blocks.append(b)
    md('title','# '+title)
    md('summary','## Executive Summary\n\n- **暂停追加训练与扩量。** 新SFT没有证明整体收益，不替换当前模型。\n- **先检验干预，再决定数据。** 训练前匹配题已全会，增加同类题不等于补齐长候选列表能力。\n- **下一步只补一个识别缺口。** 复用旧诊断题，把核查指引和输出预算拆开；不训练、不向百炼发送评测。')
    md('metric','## 统一口径后，改善与退步互相抵消\n\n2026年9月9日，同配置三轮严格答案多数票，1672题，无多数计错。纯CPT1371→新SFT1369：改对12、改错14。两模型三轮各自完全稳定的1646题内，改对7、改错9。少量重复不能证明统计等价或系统退化，但当前不支持晋级。\n\nSC知识子集189→191/226，Logistika1182→1178/1446。图按题库类别展示净变化，不是原因分解；SC合为一个类别，其余类别来自Logistika。类别大小不同，条形不是错误率。','audit')
    rows=[]
    for cat,label in [('transport','运输'),('warehousing','仓储'),('material_handling','物料搬运'),('procurement','采购'),('supply_chain','供应链')]:
        p=a['pairs']['category'][cat];rows.append(dict(category=label,**p,change=p['after']-p['before']))
    p=a['pairs']['dataset']['SC-bench-knowledge'];rows.append(dict(category='SC知识',**p,change=p['after']-p['before']))
    blocks.append(dict(id='change',type='chart',chartId='change'))
    md('mechanism','## 不是所有目标子任务都没改善\n\n305道长候选题213→215，其他1367题1158→1154。物料搬运整体净增1，运输和仓储各净减2。**长候选略有改善和全局无收益同时成立。** 不能因为总分失败就断言匹配训练完全无效，也不能用一个子组上涨替代整体交付目标。','audit')
    md('value','## 数据正确，但难度和学习价值没有对齐\n\n402道训练客观题364→369；25道匹配题在训练前已全部答对，187道单选178题答对且训练后不变，新增净收益全来自190道多选161→166（7改对、2改错）。开发74客观题67→65（0改对、2改错）。\n\n这支持重新检查数据难度与任务跨度，不支持把“模型完全不学习”当结论。此检查仅单轮、只覆盖客观题；不能外推到全部591条训练样本，也不能据此证明学习率最优或SFT无用。','audit')
    md('prior','## 已有诊断支持多机制，尚未识别主因\n\n旧96题病例对照中，纯CPT原协议48题正确、重排53、核查加长输出57。核查条件同时改变提示和96→2048输出预算，输出约76倍；不是已隔离的纯推理收益。\n\n12题证据复核发现原检索不足；3题有支持释义后，CPT只救回1题。小样本与定向证据不能估计知识不足比例。已有证据足以排除简单二分，但不足以决定继续灌知识或直接强化推理。','history')
    md('next','## 下一步：拆开两个因素，停止无依据扩量\n\n1. 纯CPT、原96题不换题：原指令/核查指引 × 96/2048输出预算，各三轮，共1152次本地回答。\n2. 分层比较配对得失、解析失败、截断和输出成本；不挑最好轮次，不把病例对照收益外推到全库。\n3. 短预算指引若有效，先考虑低成本策略验证；只有长预算组合有效，则权衡成本。无稳定收益则停止该提示主线。\n4. 只有独立来源覆盖了可验证缺口、模型确实存在该缺口，并有独立迁移检查时，才讨论新训练。')
    md('open','## 仍需回答的问题\n\n残余稳定错误中，哪些有充分独立资料支持？任务内容、回答格式与训练强度各自贡献多少？直接Step120做同数据SFT是否优于CPT前置？这些没有实测答案，不以推测填补。此次四条件实验也不会回答全部问题。')
    md('caveats','## 交付边界不变\n\n通用314题281→279（4改对、6改错），仅覆盖数学与知识，Agent回归仍未执行。新SFT不晋级，不能声称原Agent能力保持。官方题已被反复观察，不再是未触碰的独立测试集；仍不得进入训练或外部API。判读需要区分知识缺失、调用失败、题库质量与运行波动。','audit')
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE paired_groups(category TEXT,n INTEGER,before INTEGER,after INTEGER,gains INTEGER,losses INTEGER)')
    db.executemany('INSERT INTO paired_groups VALUES(?,?,?,?,?,?)',[(r['category'],r['n'],r['before'],r['after'],r['gains'],r['losses']) for r in rows])
    sql='SELECT category,n,before,after,gains,losses,after-before AS change FROM paired_groups ORDER BY change,category'
    rows=[dict(r) for r in db.execute(sql)];db.close()
    chartsource=dict(id='chart_audit',label='配对计数净变化',path=source['path'],query=dict(engine='SQLite',sql=sql,tables_used=['paired_groups'],description='安全配对计数载入内存表后核算净变化；原始预测配对由audit_supply_sft_rethink.py执行。',metric_definitions={'change':'SFT多数票正确数减CPT多数票正确数，等于gains-losses'},filters=['SC合并为知识子集；Logistika按五类，4道未分类且无变化不绘图']))
    artifact=dict(surface='report',manifest=dict(version=1,title=title,blocks=blocks,sources=[source,old,chartsource],charts=[dict(id='change',type='bar',title='CPT后SFT各组净变化',description='单位：答对题数变化；多数票配对，各组分母见数据详情；未绘制4道无变化的未分类题',showDescription=True,dataset='changes',sourceId='chart_audit',source=chartsource,encodings=dict(x=dict(field='category',type='nominal'),y=dict(field='change',type='quantitative')),options=dict(orientation='vertical'),palette=dict(kind='sequential',name='blue'))]),snapshot=dict(version=1,status='ready',datasets=dict(changes=rows)),sources=[source,old,chartsource])
    (docs/'supply_chain_rethink_20260909.artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Pair arithmetic and denominators validated; report artifact saved.')


if __name__=='__main__':main()
