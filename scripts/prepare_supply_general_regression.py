"""Freeze modest math and broad-knowledge regression subsets on the data host."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata


def fp(s):
    return hashlib.sha256(re.sub(r'[^\w]+','',unicodedata.normalize('NFKC',s).casefold()).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for k in ('raw','training','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();os.umask(0o077)
    training=[json.loads(x) for x in a.training.read_text().splitlines()]
    excluded={r['prompt_fingerprint'] for r in training}|{fp(r['prompt']) for r in training}
    sources={};selected=[];overlap=0
    specs=[('MATH-500',a.raw/'MATH-500/default__test.jsonl',100),('GSM8K',a.raw/'GSM8K/main__test.jsonl',100)]
    specs += [('MMLU',f,2) for f in sorted((a.raw/'mmlu').glob('*__test.jsonl')) if f.name!='all__test.jsonl']
    assert len(specs)==59, 'Expected 57 MMLU subjects'
    seen=set()
    for dataset,path,n in specs:
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        pool=[]
        for i,line in enumerate(path.read_text().split('\n')):
            if not line.strip():continue
            r=json.loads(line);question=r.get('problem',r.get('question'));key=fp(question)
            if key in excluded:overlap+=1;continue
            if key in seen:continue
            seen.add(key)
            if dataset=='GSM8K':
                assert '####' in r['answer'];answer=r['answer'].split('####')[-1].strip();typ='numeric'
            elif dataset=='MATH-500':answer=r['answer'];typ='math'
            else:
                assert type(r['answer']) is int and 0<=r['answer']<4 and len(r['choices'])==4
                answer='ABCD'[r['answer']];typ='choice'
                question+='\n\n'+'\n'.join(f'{"ABCD"[j]}. {v}' for j,v in enumerate(r['choices']))
            pool.append(dict(id=hashlib.sha256((dataset+path.name+str(i)).encode()).hexdigest(),dataset=dataset,question=question,ground_truth=dict(answer=answer,answer_type=typ),prompt_fingerprint=key,subject=r.get('subject','unknown')))
        pool.sort(key=lambda r:hashlib.sha256(('supply-regression-20260909:'+r['id']).encode()).hexdigest())
        assert len(pool)>=n;selected.extend(pool[:n])
    assert len(selected)==314 and len({r['prompt_fingerprint'] for r in selected})==314
    a.out.mkdir(parents=True,exist_ok=False)
    cases=a.out/'cases.private.jsonl';cases.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    summary=dict(count=314,datasets=dict(Counter(r['dataset'] for r in selected)),source_hashes=sources,cases_sha256=hashlib.sha256(cases.read_bytes()).hexdigest(),training_manifest_sha256=hashlib.sha256(a.training.read_bytes()).hexdigest(),excluded_training_overlaps=overlap,selection='100 MATH-500,100 GSM8K test,2 per 57 MMLU test subjects; deterministic hash, before predictions',limitation='Limited regression coverage, not proof of all general capabilities or pretraining absence',training=False)
    (a.out/'manifest.safe.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},indent=2))


if __name__=='__main__':main()
