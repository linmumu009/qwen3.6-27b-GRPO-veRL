"""Recompute safe screening summaries and an inspectable companion notebook."""
from collections import Counter
import json
from pathlib import Path


def main():
    docs=Path(__file__).resolve().parents[1]/'docs'
    path=docs/'book_task_aligned_value_result_20260908.safe.json';a=json.loads(path.read_text(encoding='utf-8'))
    assert len(a['decisions'])==408 and len(a['group_assessment'])==51
    assert sum(a['grade_consensus'].values())==153
    assert sum(p['groups'] for p in a['pools'].values())==51
    rows=[]
    for model in ('step120','cpt'):
        for kind in ('definition','discrimination','application','evidence_selection'):
            subset=[r for r in a['table'] if r['model']==model and r['kind']==kind]
            keys=('items','original_correct','permuted_correct','answers_changed','invalid_responses') if kind=='evidence_selection' else ('items','agreed_items','closed_correct','evidence_correct')
            record=dict(model=model,kind=kind,**{k:sum(r[k] for r in subset) for k in keys})
            c=Counter()
            for r in subset:c.update(r['decisions'])
            record['decisions']=dict(c);rows.append(record)
    context=json.loads((docs/'book_task_aligned_value_context_20260908.safe.json').read_text(encoding='utf-8'))
    summary=dict(pre_context_aggregate=rows,pre_context_pools=a['pools'],pools=context['pools'],context_clean_aggregate=context['clean_aggregate'],
      context_flagged_questions=context['flagged_questions'],grade_consensus=a['grade_consensus'],judge_api_calls=a['judge_api_calls'],judge_usage=a['judge_usage'],training=False)
    (docs/'book_task_aligned_value_analysis_20260908.safe.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    code="""import json
from collections import Counter
from pathlib import Path
root=Path.cwd();root=root.parent if root.name=='docs' else root
def read(n):return json.loads((root/'docs'/n).read_text(encoding='utf-8'))
a=read('book_task_aligned_value_result_20260908.safe.json')
p=read('book_task_aligned_value_protocol_20260908.safe.json')
assert a['items']==p['items']==204
assert a['generations']==p['generations_total']==816
assert len(a['decisions'])==408 and len({(r['id'],r['model']) for r in a['decisions']})==408
assert sum(a['grade_consensus'].values())==153
for row in a['table']:
    ds=[r for r in a['decisions'] if all(r[k]==row[k] for k in ('split','model','kind'))]
    assert len(ds)==row['items'] and dict(Counter(r['decision'] for r in ds))==row['decisions']
    if row['kind']=='evidence_selection':
        for k in ('original_correct','permuted_correct','answers_changed','invalid_responses'):assert sum(r[k] for r in ds)==row[k]
    else:
        ss=[r for r in a['open_scores'] if all(r[k]==row[k] for k in ('split','model','kind'))]
        assert len(ss)==row['agreed_items']
        assert sum(r['closed_score']==2 for r in ss)==row['closed_correct']
        assert sum(r['evidence_score']==2 for r in ss)==row['evidence_correct']
assert sum(r['groups'] for r in a['pools'].values())==51
assert sum(r['questions'] for r in a['pools'].values())==204
assert a['pools']['development']['groups']==8
assert not a['training'] and not a['training_ready']
c=read('book_task_aligned_value_context_20260908.safe.json')
bad={r['id'] for r in c['flags']}
assert len(bad)==c['flagged_questions']==17
clean=[r for r in a['open_scores'] if r['id'] not in bad]
assert len({r['id'] for r in clean})==c['clean_agreed_questions']==110
for row in c['clean_aggregate']:
    rs=[r for r in clean if r['model']==row['model']]
    assert len(rs)==row['items']
    assert sum(r['closed_score']==2 for r in rs)==row['closed_correct']
    assert sum(r['evidence_score']==2 for r in rs)==row['evidence_correct']
assert sum(r['groups'] for r in c['pools'].values())==51
assert c['pools']['learning_candidate']=={'groups':8,'questions':32}
assert c['pools']['quarantine']=={'groups':35,'questions':140}
assert c['pools']['development']=={'groups':8,'questions':32}
i=read('book_task_aligned_value_pool_integrity_20260908.safe.json')
assert i['groups']==51 and i['questions']==204
assert i['duplicate_groups']==i['duplicate_questions']==i['changed_questions']==i['split_migrations']==0
for pool,counts in c['pools'].items():
    assert all(i['pools'][pool][k]==v for k,v in counts.items())
for name in ('step120','cpt'):
    s=read('book_task_aligned_value_'+name+'_20260908.safe.json')
    assert s['responses']==408 and s['cases_sha256']==p['cases_sha256']
print('PASS: 16 cells recomputed, context-filtered 110 paired items, final 8 learning/35 quarantine/8 dev groups, 816 generations')
"""
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':[
      {'cell_type':'markdown','id':'scope','metadata':{},'source':['# 教材候选学习价值复核\n只读取安全摘要，不读取题目、教材、答案或模型输出。两轮匿名评分不同的题保持隔离；已筛选一致子集成绩不是204题总体准确率。']},
      {'cell_type':'code','id':'checks','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    (docs/'book_task_aligned_value_checks_20260908.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
