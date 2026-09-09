"""Repair exact source spans only; review previously unaudited units once."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
from extract_book_learning_units import BOOK_SHA, DEV, AUDIT, FIELDS, sha, valid_unit, call, unpack


def repair(u, full):
    v = deepcopy(u)
    source = {f'S{i//800+1:03}': full[i:i+800] for i in range(0,len(full),800)}
    if not isinstance(v.get('source_ids'), list) or not isinstance(v.get('source_quotes'), list):
        return None
    ids = [s for s in v['source_ids'] if isinstance(s,str) and s in source]
    for q in v['source_quotes']:
        if not isinstance(q,dict) or not isinstance(q.get('quote'),str) or not 15 <= len(q['quote']) <= 240:
            return None
        text=q['quote']; sid=q.get('source_id')
        if sid in source and text in source[sid]:
            if sid not in ids:ids.append(sid)
            continue
        start=full.find(text)
        if start < 0 or full.find(text,start+1) >= 0:
            return None
        lo=max(0,start-300); hi=min(len(full),start+len(text)+300)
        sid=f'R{lo}_{hi}'
        source[sid]=full[lo:hi]
        q['source_id']=sid
        q['span_start']=start; q['span_end']=start+len(text)
        if sid not in ids:ids.append(sid)
    v['source_ids']=ids
    if not valid_unit(v,source):return None
    return v,{k:source[k] for k in ids}


def verdict_map(response):
    rows=unpack(response).get('units')
    if not isinstance(rows,list) or any(not isinstance(x,dict) or not isinstance(x.get('id'),str) for x in rows):
        raise ValueError('Audit schema')
    if len({x['id'] for x in rows})!=len(rows):raise ValueError('Duplicate audit IDs')
    return {x['id']:x for x in rows}


def passes(v):
    return bool(v) and all(v.get(k) is True for k in FIELDS) and v.get('issues')==[]


def main():
    import fcntl
    p=argparse.ArgumentParser()
    for k in ('root','out','api-config'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    old=a.root/'runs/supply-chain-learning-units-20260909-01'
    book=a.root/'runs/logistics-cpt-20260903/private/handbook8e_cpt_4096.jsonl'
    assert sha(book)==BOOK_SHA
    chapters=defaultdict(list)
    for r in map(json.loads,book.read_text().splitlines()):
        if r.get('chapter'):chapters[r['chapter']].append(r['text'])
    fulls={c:'\n\n'.join(x) for c,x in chapters.items()}
    retained=[]; jobs=[]; counters=Counter()
    for r in map(json.loads,(old/'chapters.private.jsonl').read_text().splitlines()):
        ch=r['chapter']; proposed=unpack(r['generation_response'])['units']
        originalsource={f'S{i//800+1:03}':fulls[ch][i:i+800] for i in range(0,len(fulls[ch]),800)}
        sent={u['id'] for u in proposed if valid_unit(u,originalsource)}
        verdicts=verdict_map(r['audit_response']) if 'audit_response' in r else {}
        batch=[]
        for u in proposed:
            fixed=repair(u,fulls[ch])
            if fixed is None:counters['unrepairable']+=1;continue
            v,refs=fixed
            unit=dict(v,id=f'C{ch:02}-'+v['id'],chapter=ch,split='dev' if ch in DEV else 'train',reference_units=refs)
            if u['id'] in sent and u['id'] in verdicts:
                if passes(verdicts[u['id']]):
                    retained.append(unit);counters['existing_positive_retained']+=1
                else:counters['existing_negative_not_rejudged']+=1
            else:
                batch.append((v,unit));counters['requires_new_audit']+=1
        if batch:jobs.append((ch,batch))
    assert len(jobs)<=44
    protocol=dict(max_calls=len(jobs),hard_cap_calls=44,workers=44,max_output_tokens_per_call=8000,retries=0,
                  model='qwen3.8-max',input_book_sha256=BOOK_SHA,input_responses_sha256=sha(old/'chapters.private.jsonl'),
                  script_sha256=sha(Path(__file__)),helper_sha256=sha(Path(__file__).with_name('extract_book_learning_units.py')),
                  counts=dict(counters),semantic_content_modified=False,training=False,benchmark_loaded=False)
    if a.prepare_only:print(json.dumps(protocol,indent=2));return
    os.umask(0o077)
    with (a.root/'runs/.book-task-aligned-generation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a.out.mkdir(parents=True,exist_ok=False)
        def save(name,v):(a.out/name).write_text(json.dumps(v,indent=2))
        save('protocol.safe.json',protocol);config=json.loads(a.api_config.read_text());results=[]
        def process(job):
            ch,batch=job;result=dict(chapter=ch,requested=len(batch),accepted=[],missing_ids=0)
            try:
                source={f'S{i//800+1:03}':fulls[ch][i:i+800] for i in range(0,len(fulls[ch]),800)}
                for _,unit in batch:source.update(unit['reference_units'])
                result['response']=call(config,AUDIT,{'units':[v for v,_ in batch],'numbered_chapter':source})
                vm=verdict_map(result['response']);wanted={v['id'] for v,_ in batch}
                if set(vm)-wanted:raise ValueError('Unknown audit IDs')
                for v,unit in batch:
                    if v['id'] not in vm:result['missing_ids']+=1
                    elif passes(vm[v['id']]):result['accepted'].append(unit)
                result['status']='complete'
            except Exception as e:result.update(status='failed',error_type=type(e).__name__)
            return result
        def summary(status):
            all_units=retained+[u for r in results for u in r['accepted']]
            return dict(status=status,planned_calls=len(jobs),completed_calls=len(results),failed_calls=sum(r['status']=='failed' for r in results),
                        missing_audit_ids=sum(r['missing_ids'] for r in results),units=len(all_units),
                        split_counts=dict(Counter(u['split'] for u in all_units)),domain_counts=dict(Counter(u['domain'] for u in all_units)),
                        usage={k:sum(r.get('response',{}).get('usage',{}).get(k,0) for r in results) for k in ('prompt_tokens','completion_tokens','total_tokens')},
                        training=False,training_ready=False)
        save('summary.safe.json',summary('running'))
        with ThreadPoolExecutor(max_workers=44) as pool:
            for f in as_completed([pool.submit(process,j) for j in jobs]):
                r=f.result();results.append(r)
                with (a.out/'audits.private.jsonl').open('a') as out:out.write(json.dumps(r)+'\n')
                save('summary.safe.json',summary('running'))
        all_units=retained+[u for r in results for u in r['accepted']]
        assert len(all_units)==len({u['id'] for u in all_units})
        with (a.out/'units.private.jsonl').open('x') as out:
            for u in sorted(all_units,key=lambda u:u['id']):out.write(json.dumps(u)+'\n')
        save('summary.safe.json',summary('audit_complete_needs_dedup_and_tasks'))
        print(json.dumps(summary('audit_complete_needs_dedup_and_tasks'),indent=2))


if __name__=='__main__':main()
