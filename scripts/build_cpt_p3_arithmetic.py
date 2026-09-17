"""Source-free deterministic P3 arithmetic construction; no evaluation input."""
import argparse
import hashlib
import json
from pathlib import Path
import random

OPS=('add3','div_exact','products_sum','sub2')

def generate(seed, split, forbidden=()):
    rng=random.Random(seed);seen=set(forbidden);rows=[]
    def number(signed=False):
        value=rng.randint(2,99)
        return value*rng.choice((-1,1)) if signed else value
    for op in OPS:
        count=0
        while count<15:
            if op=='add3':
                a,b,c=[number() for _ in range(3)]
                expression=f'({a}) + ({b}) + ({c})';answer=a+b+c
                key=(op,*sorted((a,b,c)))
            elif op=='sub2':
                a,b,c=[number() for _ in range(3)]
                expression=f'({a}) - ({b}) - ({c})';answer=a-b-c
                key=(op,a,*sorted((b,c)))
            elif op=='products_sum':
                a,b,c,d=[number(True) for _ in range(4)]
                expression=f'({a}) * ({b}) + ({c}) * ({d})';answer=a*b+c*d
                key=(op,*sorted((tuple(sorted((a,b))),tuple(sorted((c,d))))))
            else:
                b,q=number(True),number(True);a=b*q
                expression=f'({a}) / ({b})';answer=q;key=(op,a,b)
            normalized=json.dumps(key,separators=(',',':'))
            if normalized in seen:continue
            seen.add(normalized);wrong=set();bound=max(200,abs(answer)*2)
            while len(wrong)<3:
                value=rng.randint(-bound,bound)
                if value!=answer:wrong.add(value)
            options=sorted(wrong)+[answer];rng.shuffle(options)
            rows.append(dict(id=f'p3-{split}-{op}-{count:02d}',operation=op,
                question=expression,options=[str(v) for v in options],correct_indices=[options.index(answer)],
                normalized=normalized,answer=answer,split=split));count+=1
    return rows,seen

def build(out):
    train,seen=generate(71437,'train');check,_=generate(71438,'check',seen)
    out.mkdir(exist_ok=False)
    for name,rows in [('train.private.jsonl',train),('check.private.jsonl',check)]:
        (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    cases=[dict(dataset='p3_arithmetic_check',source_id=x['id'],category=x['operation'],
        question_type='single_choice',question=x['question'],options=x['options'],expected=x['correct_indices']) for x in check]
    (out/'cases.private.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in cases),encoding='utf-8')
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    manifest=dict(seeds=[71437,71438],train=60,check=60,files=files,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_queries=0)
    (out/'manifest.safe.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);build(p.parse_args().out)
