"""Private lexical overlap leads for root semantic review; never releases a task."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(text):
    return ' '.join(re.findall(r'[a-z]+|#',re.sub(r'\d+(?:\.\d+)?','#',text.casefold())))


def shingles(text):
    tokens=norm(text).split()
    return {' '.join(tokens[i:i+5]) for i in range(len(tokens)-4)}


def collect(value, origin, corpus):
    if isinstance(value,dict):
        question=value.get('question')
        if isinstance(question,str):
            opts=value.get('options',[])
            if isinstance(opts,dict):opts=list(opts.values())
            opts=[str(o) for o in opts] if isinstance(opts,list) else []
            corpus.append(dict(source=origin,question=question,options=opts))
        for message in value.get('messages',[]) if isinstance(value.get('messages'),list) else []:
            if message.get('role')=='user' and isinstance(message.get('content'),str):
                corpus.append(dict(source=origin,question=message['content'],options=[]))
        for nested in value.values():collect(nested,origin,corpus)
    elif isinstance(value,list):
        for nested in value:collect(nested,origin,corpus)


def main(base, corpus_only):
    tree=ast.parse(Path('scripts/verify_cpt_premise_tasks.py').read_text(encoding='utf-8'))
    old=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='EXCLUSION_FILES' for t in n.targets))
    paths=list(old)+[
        'llin-transfer-audit-20260915-01/frozen_cases.private.jsonl',
        'llin-transfer-p4-data-20260918-01/train.messages.private.jsonl',
        'llin-stage-b-20260920-01/run04/calls.private.jsonl',
        'llin-stage-b-small-20260920-01/blind_tasks.private.json',
        'llin-source-recall-20260922-01/probes.private.json',
        'llin-context-bridge-20260922-01/cases.private.json',
        'llin-rule-disambiguation-20260922-01/cases.private.json']
    corpus=[];inputs=[]
    for name in paths:
        path=Path('CPT_resources')/name
        if not path.is_file():raise ValueError('Required exclusion input missing: '+name)
        content=path.read_text(encoding='utf-8')
        rows=[json.loads(s) for s in content.splitlines() if s.strip()] if path.suffix=='.jsonl' else json.loads(content)
        if name.endswith('run04/calls.private.jsonl'):
            decoded=[]
            for row in rows:
                if row['stage']=='author':
                    text=re.sub(r'\s*```$','',re.sub(r'^```(?:json)?\s*','',row['text'].strip()))
                    decoded.append(json.loads(text))
            rows=decoded
        before=len(corpus);collect(rows,name,corpus)
        inputs.append(dict(path=name,sha256=sha(path),extracted_records=len(corpus)-before))
    # Include options in signatures and retain exact originals privately for semantic review.
    for row in corpus:
        row['text']=row['question']+' '+' '.join(row['options'])
    out=base/'overlap_corpus.private.json'
    out.write_text(json.dumps(dict(inputs=inputs,records=corpus),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if corpus_only:
        print(json.dumps(dict(files=len(inputs),records=len(corpus),sha256=sha(out))))
        return
    tasks=json.loads((base/'reviewer_input.private.json').read_text(encoding='utf-8'))['records']
    signatures=[(r,norm(r['question']),shingles(r['question']),shingles(r['text'])) for r in corpus]
    results=[]
    for task in tasks:
        if not task['authored']:
            results.append(dict(id=task['id'],skipped=True,hits=[]));continue
        q=task['question'];full=q+' '+' '.join(task['options']);qs=shingles(q);fs=shingles(full);hits=[]
        for row,normalized,oldq,oldfull in signatures:
            exact=norm(q)==normalized
            qr=len(qs&oldq)/max(1,min(len(qs),len(oldq)))
            fr=len(fs&oldfull)/max(1,min(len(fs),len(oldfull)))
            if exact or max(qr,fr)>=.3:
                hits.append(dict(source=row['source'],question=row['question'],options=row['options'],
                                 exact=exact,question_ratio=qr,question_options_ratio=fr))
        results.append(dict(id=task['id'],hits=sorted(hits,key=lambda h:-max(h['question_ratio'],h['question_options_ratio']))))
    result=dict(inputs=inputs,corpus_sha256=sha(out),records=results,threshold=.3,
        limitation='Enumerated corpus only; lexical leads do not certify semantic novelty. Full source and task-operation review required.',
        training_allowed=False,diagnostic_released=False)
    (base/'overlap.private.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(corpus=len(corpus),tasks=len(tasks),with_lexical_hits=sum(bool(r['hits']) for r in results))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path('CPT_resources/llin-logistika-independent-20260922-01'))
    p.add_argument('--corpus-only',action='store_true');a=p.parse_args();main(a.base,a.corpus_only)
