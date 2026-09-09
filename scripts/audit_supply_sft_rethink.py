"""Read-only prediction audit; export aggregate evidence, never benchmark bodies."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import hashlib


def load(p):return [json.loads(x) for x in p.read_text().split('\n') if x.strip()]


def collapse(repeats):
    maps=[{r['item_hash']:r for r in rows} for rows in repeats]
    assert all(len(m)==1672 and set(m)==set(maps[0]) for m in maps)
    out={}
    for key,r in maps[0].items():
        rr=[m[key] for m in maps]
        assert all((x['expected'],x['options'],x['question'])==(r['expected'],r['options'],r['question']) for x in rr)
        answers=[tuple(x['parsed']) if x['parse_ok'] else None for x in rr]
        votes=Counter(answers);best,n=votes.most_common(1)[0]
        out[key]=dict(correct=n>=2 and best is not None and list(best)==r['expected'],stable=len(votes)==1,
            dataset=r['dataset'],category=r['category'],kind=r['question_type'],long=len(r['options'])>=100,
            nomajority=n<2,answer=best if n>=2 else None)
    return out


def paired(a,b,keys):
    keys=list(keys)
    return dict(n=len(keys),before=sum(a[k]['correct'] for k in keys),after=sum(b[k]['correct'] for k in keys),
        gains=sum(not a[k]['correct'] and b[k]['correct'] for k in keys),losses=sum(a[k]['correct'] and not b[k]['correct'] for k in keys))


def main():
    root=Path('/workspace/llin-verl-grpo/runs');p=root/'supply-chain-grounded-sft-validation-20260909-01'
    models={};result={'sources':{},'models':{},'pairs':{},'training_value':{}}
    for name in ('step120','cpt116','sft73'):
        files=[p/(name+'_official.private'+s+'.jsonl') for s in ('','.repeat2','.repeat3')]
        repeats=[load(f) for f in files];models[name]=collapse(repeats)
        for f in files:result['sources'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
        result['models'][name]=dict(repeat_correct=[sum(r['correct'] for r in rows) for rows in repeats],majority_correct=sum(r['correct'] for r in models[name].values()),unstable=sum(not r['stable'] for r in models[name].values()),no_majority=sum(r['nomajority'] for r in models[name].values()))
    a=models['cpt116'];b=models['sft73']
    result['pairs']['all']=paired(a,b,a)
    result['pairs']['both_stable']=paired(a,b,[k for k in a if a[k]['stable'] and b[k]['stable']])
    for field in ('dataset','category','kind','long'):
        result['pairs'][field]={str(v):paired(a,b,[k for k in a if a[k][field]==v]) for v in {r[field] for r in a.values()}}
    result['cpt_stable_wrong']=sum(r['stable'] and not r['correct'] for r in a.values())
    before=load(root/'supply-chain-task-baselines-20260909-01/cpt116.private.jsonl')
    after=load(p/'sft73_tasks.private.jsonl')
    amap={r['source_id']:r for r in before};bmap={r['source_id']:r for r in after}
    assert set(amap)==set(bmap) and len(amap)==476
    for dataset in {r['dataset'] for r in before}:
        keys=[k for k in amap if amap[k]['dataset']==dataset]
        result['training_value'][dataset]=paired(amap,bmap,keys)
        result['training_value'][dataset]['by_kind']={v:paired(amap,bmap,[k for k in keys if amap[k]['question_type']==v]) for v in {amap[k]['question_type'] for k in keys}}
    result['general']={}
    gg={m:{r['id']:r for r in json.loads((p/(m+'_general/summary.safe.json')).read_text())['items']} for m in models}
    assert all(set(x)==set(gg['cpt116']) for x in gg.values())
    result['general']['paired']=paired(gg['cpt116'],gg['sft73'],gg['cpt116'])
    result['general']['counts']={m:sum(r['correct'] for r in x.values()) for m,x in gg.items()}
    result['limitations']=['No benchmark text or IDs exported','Three repeats are not independent items','Task learning checks are single-repeat and only objective subset','Agent regression not performed','No causal assignment of knowledge versus reasoning']
    out=root/'supply-chain-rethink-20260909-01';out.mkdir(exist_ok=False)
    (out/'audit.safe.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
