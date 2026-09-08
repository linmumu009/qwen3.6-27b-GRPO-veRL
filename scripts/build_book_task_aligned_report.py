"""Build safe source-quality handoff and reproducible checks for candidate data."""
import json
from pathlib import Path
import sqlite3


def main():
    root=Path(__file__).resolve().parents[1];docs=root/'docs'
    def read(name):return json.loads((docs/name).read_text(encoding='utf-8'))
    final=read('book_task_aligned_result_20260908.safe.json');quality=read('book_task_aligned_quality_20260908.safe.json');first=read('book_task_aligned_initial_20260908.safe.json')
    assert sum(final['statuses'].values())==final['completed_groups']
    names={'quality_pass':'组内审核通过','audit_reject':'来源审核未通过','solver_disagreement':'独立求答不一致','solver_unresolved':'求答未完成验证','structural_reject':'结构不合要求','overlap_quarantine':'词面重合隔离','source_abstain':'来源不足放弃','api_failure':'API失败','request_or_parse_failure':'请求或解析失败'}
    conn=sqlite3.connect(':memory:');conn.row_factory=sqlite3.Row
    conn.execute('CREATE TABLE candidate_outcomes(status TEXT,label TEXT,groups INTEGER,total INTEGER)')
    conn.executemany('INSERT INTO candidate_outcomes VALUES(?,?,?,?)',[(s,names.get(s,s),n,final['completed_groups']) for s,n in final['statuses'].items()])
    sql='SELECT status,label,SUM(groups) AS groups,MAX(total) AS total,1.0*SUM(groups)/MAX(total) AS share FROM candidate_outcomes GROUP BY status,label ORDER BY groups DESC'
    data=[dict(r) for r in conn.execute(sql)];conn.close()
    source={'id':'outcomes','label':'教材四任务构造安全汇总','path':'docs/book_task_aligned_result_20260908.safe.json','query':{'engine':'SQLite','sql':sql,'tables_used':['candidate_outcomes'],'description':'将最终批次互斥状态计数载入内存SQLite后聚合；包括复用的12组预检，不重复计算。','metric_definitions':{'groups':'最终状态对应的概念组数；每个完整组四题。','share':'该状态组数/最终批次已处理组数'},'filters':['仅修正后最终批次；初次失败尝试另列，不混入当前分母']}}
    sources=[source,{'id':'quality','label':'来源、切分与问题清单检查','path':'docs/book_task_aligned_quality_20260908.safe.json'},{'id':'first','label':'初次失败尝试，未丢弃记录','path':'docs/book_task_aligned_initial_20260908.safe.json'},{'id':'manifest','label':'预先冻结的来源和切分协议','path':'docs/book_task_aligned_manifest_20260908.safe.json'}]
    title='教材四任务数据首批审核';blocks=[]
    def md(key,text):blocks.append({'id':key,'type':'markdown','body':text})
    kept=final.get('retained_groups',0);qs=final.get('retained_questions',0);splits=final.get('retained_split_groups',{})
    md('title','# '+title)
    md('summary',f'## Executive Summary\n\n- **已完成教材数据构造和来源审核，不是训练完成。** 修正后处理{final["completed_groups"]}组，组内审核通过{final["quality_pass_groups"]}组，语义去重后保留{kept}组、{qs}道候选题。\n- **新增了概念辨析和证据选择任务。** 每组四题：直接问答、概念辨析、情境应用、带资料选择；整组任一题未通过便不进入本批候选。\n- **可靠性通过仍不等于有训练收益。** 两个起点模型尚未预答，当前不能称为学习机会池或直接启动SFT。')
    md('scope','## 先按来源隔离，再观察模型表现\n\n本轮仅使用现有授权教材，由5号机百炼qwen3.8-max生成和复核，并发上限64。120组尝试预先固定为90训练侧、30开发侧；相同章节不跨两侧，没有把评测题、答案或本轮定向证据发送百炼。\n\n开发章节只对这轮SFT保留，整本书已经用于CPT，因此不能称作未见知识测试。自动词面隔离也不是语义零泄露证明。')
    md('outcome',f'## 保留完整概念组，不用零散通过题凑数\n\n下图的每根柱是修正后批次的一个互斥最终状态，分母为{final["completed_groups"]}组，不是API调用数或题目数。组内审核通过后，另移除{final.get("removed_duplicate_groups",0)}个语义重复组，最终训练侧{splits.get("train",0)}组、开发侧{splits.get("dev",0)}组。\n\n审核失败表示当前证据或表达没有通过既定检查，不自动等于教材缺少该知识。只有保留组进入下一步模型预答。')
    blocks.append({'id':'status-chart','type':'chart','chartId':'status'})
    md('quality',f'## 索引一致性由程序检查，事实由来源审核\n\n求答请求看不到生成答案；选择题先重排选项，求答结果再由程序映射为选项文本。随后审核教材依据、语义一致性、条件、评分点及每个干扰项的排除理由。不得以“教材未提到”充当“该选项必然错误”。\n\n保留清单共有{quality.get("questions",0)}题；精确重复题组{quality.get("exact_duplicate_question_groups",0)}个，训练/开发来源块交集{quality.get("source_overlap_count",0)}个。引用编号能定位教材窗口，但自动审阅仍可能遗漏事实错误，尤其生成与审核使用同一模型系列。')
    md('corrections','## 初次零通过包含程序契约问题，不能归因于教材\n\n初次120组尝试没有完整通过组，记录全部保留。复核发现题干重复选项会破坏重排；审核提示也存在歧义，索引映射还会被审核器误读。修正后先跑12组，随后定位到选择题求答多余文字字段的校验问题。\n\n复用已有12组的生成和求答，仅补做缺失审核；达到至少3组完整通过的条件后才继续其余108组。未将原失败记录改写为通过，也未放宽事实、来源、完整组选项或语义审核要求。初次与最终批次不是受控质量提升实验。')
    total=first['usage']['total_tokens']+final['usage']['total_tokens']+final.get('semantic_usage',{}).get('total_tokens',0)
    calls=first['api_calls']+final['api_calls']+final.get('semantic_api_calls',0)
    md('cost',f'## 费用包含失败尝试，不重复计算复用请求\n\n本轮累计{calls}次API请求，成功返回的用量共{total:,} tokens；其中初次失败构造使用{first["usage"]["total_tokens"]:,} tokens。最终批次统计已包含复用的12组请求，不再叠加预检批次。以上是接口返回用量，不是人民币账单。\n\n没有本地模型生成训练QA，没有更新任何模型权重。')
    md('next','## 下一步先测数据的学习价值\n\n1. 冻结保留组，让Step120和纯教材CPT在不见参考答案的条件下作答；选择题检查原序与重排，开放题按预先冻结的必要语义点复核。\n2. 区分候选学习机会、能力保持题和仍有歧义的隔离题；开发数据不因为模型答对或答错而移动到训练侧。\n3. 确认存在足够学习机会及开发覆盖后，再组织同数据、双起点的短程SFT；保持Agent与开源回归约束。')
    md('questions','## 还不能回答的问题\n\n这些题对两个起点是否已经过于容易？带证据的辨析能力能否改善闭卷问答？目前均未测量。首批选择题提示要求六候选、两正确项，只是构造模板，后续不能把固定答案数量当作目标能力。')
    md('limits','## 使用边界\n\n当前只获得同系列API自动审核后的候选数据，不是人工金标，也没有证明训练收益。教材对企业规则的缺口仍保留，不生成无依据制度；本次来源授权记录来自用户确认，不是独立法律审查。私有题目、答案、教材和API响应仅保留在5号机，仓库只提交代码、汇总和检查结果。')
    chart={'id':'status','type':'bar','title':'概念组审核状态','description':f'修正后批次，单位：组；共{final["completed_groups"]}组','showDescription':True,'dataset':'outcomes','sourceId':'outcomes','source':source,'encodings':{'x':{'field':'label','type':'nominal'},'y':{'field':'groups','type':'quantitative'}},'options':{'orientation':'vertical'},'palette':{'kind':'sequential','name':'blue'}}
    artifact={'surface':'report','manifest':{'version':1,'title':title,'blocks':blocks,'charts':[chart],'sources':sources},'snapshot':{'version':1,'status':'ready','datasets':{'outcomes':data}},'sources':sources}
    (docs/'book_task_aligned_20260908.artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
    cells=[{'cell_type':'markdown','id':'scope','metadata':{},'source':['# 教材四任务候选数据检查\n仅读取安全汇总，不读取私有教材或问答。最终批次包含复用预检请求，不能重复加总。']}]
    code="""import json
from pathlib import Path
root=Path.cwd(); root=root.parent if root.name=='docs' else root
def read(name): return json.loads((root/'docs'/name).read_text(encoding='utf-8'))
a=read('book_task_aligned_result_20260908.safe.json')
q=read('book_task_aligned_quality_20260908.safe.json')
m=read('book_task_aligned_manifest_20260908.safe.json')
assert sum(a['statuses'].values())==a['completed_groups']==120
assert a['retained_questions']==4*a['retained_groups']==q['questions']
assert sum(a['retained_split_groups'].values())==a['retained_groups']
assert sum(q['by_kind'].values())==q['questions']
assert len(set(q['by_kind'].values()))==1
assert q['source_overlap_count']==q['exact_duplicate_question_groups']==0
assert sum(t['split']=='dev' for t in m['tasks'])==30
assert sum(t['split']=='train' for t in m['tasks'])==90
assert not a['training_ready'] and not a['target_model_probe_completed']
assert a['inherited_api_calls']==27
print('PASS: counts, source split, completeness, duplicates, cost inheritance and no-training status')
"""
    cells.append({'cell_type':'code','id':'checks','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)})
    (docs/'book_task_aligned_checks_20260908.ipynb').write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
