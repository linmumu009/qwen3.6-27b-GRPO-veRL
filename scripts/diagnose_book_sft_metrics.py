"""Reconcile benchmark transitions and training/export evidence without question text."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3


def transitions(before,after):
    counts=Counter()
    for key in before:
        a=before[key]['correct']; b=after[key]['correct']
        counts['both_correct' if a and b else 'regressed' if a else 'improved' if b else 'both_wrong']+=1
    return dict(counts)


def main():
    p=argparse.ArgumentParser()
    for k in ('root','run','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    paths={'cpt':a.root/'runs/logistics-cpt-exposure-curve-diagnostics-20260904/safe/public_eval/cpt_4x.majority.safe.json',
           'sft':a.run/'majority.safe.json'}
    reports={k:json.loads(v.read_text()) for k,v in paths.items()}
    rows={k:{r['item_hash']:r for r in v['rows']} for k,v in reports.items()}
    if set(rows['cpt'])!=set(rows['sft']) or len(rows['cpt'])!=1672:raise ValueError('Population mismatch')
    for k in reports:
        if len(rows[k])!=len(reports[k]['rows']) or sum(r['correct'] for r in rows[k].values())!=reports[k]['correct']:raise ValueError('Count mismatch')
    groups=defaultdict(list)
    for key,r in rows['cpt'].items():
        s=rows['sft'][key]
        if any(r[f]!=s[f] for f in ('dataset','category','question_type')):raise ValueError('Grouping mismatch')
        groups[(r['dataset'],r['category'])].append(key)
    segments=[]
    for (dataset,category),keys in groups.items():
        t=transitions({k:rows['cpt'][k] for k in keys},{k:rows['sft'][k] for k in keys})
        segments.append(dict(dataset=dataset,category=category,items=len(keys),**t,net=t.get('improved',0)-t.get('regressed',0)))
    counts=transitions(rows['cpt'],rows['sft']); n=counts.get('improved',0)+counts.get('regressed',0)
    db=sqlite3.connect(':memory:')
    for label in ('cpt','sft'):
        db.execute(f'CREATE TABLE {label}_results(item_hash TEXT PRIMARY KEY,dataset TEXT,category TEXT,correct INTEGER)')
        db.executemany(f'INSERT INTO {label}_results VALUES(?,?,?,?)',[(k,r['dataset'],r['category'],int(r['correct'])) for k,r in rows[label].items()])
    sql='''SELECT c.dataset,c.category,COUNT(*) AS items,
 SUM(CASE WHEN c.correct=0 AND s.correct=1 THEN 1 ELSE 0 END) AS improved,
 SUM(CASE WHEN c.correct=1 AND s.correct=0 THEN 1 ELSE 0 END) AS regressed,
 SUM(s.correct-c.correct) AS net
 FROM cpt_results c JOIN sft_results s ON c.item_hash=s.item_hash
 GROUP BY c.dataset,c.category ORDER BY net,c.dataset,c.category'''
    sql_rows=[dict(zip(('dataset','category','items','improved','regressed','net'),r)) for r in db.execute(sql)]
    if sum(r['net'] for r in sql_rows)!=counts.get('improved',0)-counts.get('regressed',0):raise ValueError('SQL reconciliation')
    pvalue=min(1.,2*sum(math.comb(n,k) for k in range(min(counts.get('improved',0),counts.get('regressed',0))+1))/2**n) if n else 1.
    train=[]; validation=[]
    pattern=re.compile(r'step:(\d+).*?train/loss:([\d.eE+-]+).*?train/grad_norm:([\d.eE+-]+).*?train/lr:([\d.eE+-]+)')
    for f in a.run.glob('torchrun_logs/**/stdout.log'):
        for line in f.read_text(errors='replace').splitlines():
            m=pattern.search(line)
            if m:train.append({'step':int(m[1]),'loss':float(m[2]),'grad_norm':float(m[3]),'lr':float(m[4])})
            m=re.search(r'step:(\d+) - val/loss:([\d.eE+-]+)',line)
            if m:validation.append({'step':int(m[1]),'loss':float(m[2])})
    train=sorted({r['step']:r for r in train}.values(),key=lambda r:r['step'])
    export=json.loads((a.run/'hf_export/llin_export_manifest.json').read_text())
    safe={'baseline_correct':reports['cpt']['correct'],'sft_correct':reports['sft']['correct'],'items':1672,
        'transitions':counts,'mcnemar_exact_two_sided_p':pvalue,'segments':sorted(segments,key=lambda r:r['net']),
        'input_hashes':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
        'benchmark_case_hashes':{k:v.get('input_sha256') for k,v in reports.items()},
        'reported_requests':{k:v.get('request') for k,v in reports.items()},
        'parse_failures':{k:v['parse_failures'] for k,v in reports.items()},
        'training':{'logged_steps':len(train),'first':train[:3],'last':train[-3:],
            'first20_mean_loss':sum(r['loss'] for r in train[:20])/20,'last20_mean_loss':sum(r['loss'] for r in train[-20:])/20,
            'nonfinite':sum(not all(math.isfinite(r[k]) for k in ('loss','grad_norm','lr')) for r in train)},
        'validation':validation,'export_manifest':export,'source_content_included':False,
        'chart_sql':sql,'chart_rows':sql_rows,'sql_engine':'SQLite in-memory; inputs loaded from the two frozen safe row reports'}
    with a.output.open('x') as f:json.dump(safe,f,indent=2)
    print(json.dumps({k:v for k,v in safe.items() if k!='export_manifest'},indent=2))


if __name__=='__main__':main()
